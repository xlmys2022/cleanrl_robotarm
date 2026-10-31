#!/usr/bin/env python3
"""验证真实物理链路：桌面 cube -> Reach(训练好的策略) -> 夹爪闭合 -> bilateral grasp。

用法:
    python scripts/check_reach_then_grasp.py [--model-path <Reach checkpoint>]

说明:
    - 环境是 PandaGrasp-v0（task="grasp"）, observation(34)/action(8) 与 Reach 完全一致;
    - approach 阶段只用 Reach policy 的 action[:7], action[7] 强制 +1（夹爪全开）,
      把"Reach 能力"和"gripper 行为"隔离开;
    - cube 不 teleport、不改 qpos、不改 goal, 全程真实接触物理;
    - 测试脚本不修改 ever_grasped / success_hold_count / is_success, 由环境自己判定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from eval_reach import build_agent, load_checkpoint, policy_action  # noqa: E402

DEFAULT_REACH_CKPT = (
    PROJECT_ROOT / "runs/PandaReach-v0__train_robot__1__1793349790/train_robot.checkpoint.pt"
)

REACH_DIST = 0.04          # 与 envs/panda_pick_place_env.py 的 REACH_DIST 一致
REACH_HOLD = 5             # Reach 判定需要的连续步数（本脚本自用）
MAX_APPROACH_STEPS = 100
MAX_CLOSE_STEPS = 50
PRINT_EVERY = 10


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reach -> close -> grasp 物理可行性验证")
    parser.add_argument("--model-path", default=str(DEFAULT_REACH_CKPT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Reach 动作用采样（默认 actor mean）")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    mean, var = norm["mean"].numpy(), norm["var"].numpy()
    eps = float(norm["epsilon"])
    clip_lo, clip_hi = float(norm["clip_low"]), float(norm["clip_high"])

    def normalize(obs):
        return np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps),
                       clip_lo, clip_hi).astype(np.float32)

    env = gym.make("PandaGrasp-v0")
    obs, info = env.reset(seed=args.seed)
    u = env.unwrapped
    cube_init = u.cube_position.copy()
    tcp_init = u._get_tcp_position()

    rule("0. 初始状态")
    print(f"  checkpoint : {args.model_path}")
    print(f"  env        : PandaGrasp-v0   obs={obs.shape} action={env.action_space.shape}")
    print(f"  TCP        : {np.array2string(tcp_init, precision=4)}")
    print(f"  cube       : {np.array2string(cube_init, precision=4)}")
    print(f"  初始 distance = {info['tcp_cube_distance']:.4f} m, "
          f"finger={info['finger_mean_position']:.4f}, is_grasped={info['is_grasped']}")

    # ------------------------------------------------------------------ approach
    rule(f"1. Approach（Reach actor 的 action[:7], 夹爪强制全开, 最多 {MAX_APPROACH_STEPS} 步）")
    reach_steps = 0
    reach_done = False
    first_left_approach = first_right_approach = None
    close_count = 0
    for step in range(1, MAX_APPROACH_STEPS + 1):
        action = np.zeros(8, dtype=np.float32)
        action[:7] = policy_action(agent, normalize(obs), deterministic=not args.stochastic)[:7]
        action[7] = 1.0                      # fully open
        obs, reward, terminated, truncated, info = env.step(action)
        reach_steps = step
        if info["left_contact"] and first_left_approach is None:
            first_left_approach = step
        if info["right_contact"] and first_right_approach is None:
            first_right_approach = step

        close_count = close_count + 1 if info["tcp_cube_distance"] < REACH_DIST else 0
        if step % PRINT_EVERY == 0 or close_count >= REACH_HOLD:
            print(f"  step {step:3d} dist={info['tcp_cube_distance']:.4f} "
                  f"finger={info['finger_mean_position']:.4f} "
                  f"cube={np.array2string(info['cube_position'], precision=4)} "
                  f"L={info['left_contact']} R={info['right_contact']} "
                  f"grasped={info['is_grasped']}")
        if close_count >= REACH_HOLD:
            reach_done = True
            break
        if terminated or truncated:
            print(f"  [warn] approach 阶段意外结束: terminated={terminated} truncated={truncated}")
            break

    dist_before_close = float(info["tcp_cube_distance"])
    tcp_before_close = u._get_tcp_position().copy()
    cube_before_close = u.cube_position.copy()
    print(f"\n  Reach completed: {reach_done} (step={reach_steps}, "
          f"distance={dist_before_close:.4f} m)")
    print(f"  夹爪全开时是否有 contact: L={info['left_contact']} R={info['right_contact']} "
          f"(approach 期间首次: L@{first_left_approach}, R@{first_right_approach})")

    # ------------------------------------------------------------------ close
    rule(f"2. Close（action[:7]=0 保持关节目标, action[7]=-1, 最多 {MAX_CLOSE_STEPS} 步）")
    first_left = first_right = first_bilateral = None
    bilateral_count = success_step = None
    max_left_force = max_right_force = 0.0
    success = False

    if reach_done:
        for step in range(1, MAX_CLOSE_STEPS + 1):
            action = np.zeros(8, dtype=np.float32)
            action[7] = -1.0
            obs, reward, terminated, truncated, info = env.step(action)

            if info["left_contact"] and first_left is None:
                first_left = step
            if info["right_contact"] and first_right is None:
                first_right = step
            if info["left_contact"] and info["right_contact"] and first_bilateral is None:
                first_bilateral = step
            bilateral = info["left_contact"] and info["right_contact"]
            bilateral_count = bilateral_count + 1 if bilateral else 0
            max_left_force = max(max_left_force, info["left_contact_force"])
            max_right_force = max(max_right_force, info["right_contact_force"])

            print(f"  step {step:2d} finger={info['finger_mean_position']:.4f} "
                  f"L={info['left_contact']}({info['left_contact_force']:.2f}N) "
                  f"R={info['right_contact']}({info['right_contact_force']:.2f}N) "
                  f"grasped={info['is_grasped']} hold={bilateral_count} "
                  f"cube={np.array2string(info['cube_position'], precision=4)}")

            if info["is_success"]:
                success = True
                success_step = step
                break
            if terminated or truncated:
                print(f"  [warn] close 阶段提前结束: terminated={terminated} truncated={truncated}")
                break

    # ------------------------------------------------------------------ 报告
    final_cube = u.cube_position.copy()
    # 位移只统计水平方向的推移（z 方向的差一部分来自初始悬空下落, 容易误判）
    cube_xy_drift = float(np.linalg.norm(final_cube[:2] - cube_init[:2]))
    cube_dz_vs_rest = float(final_cube[2] - u._cube_rest_z)
    rule("3. 结果")
    print(f"  Reach completed step      : {reach_steps if reach_done else 'FAILED'}")
    print(f"  distance when closing     : {dist_before_close:.4f} m")
    print(f"  TCP  (reach 结束时)       : {np.array2string(tcp_before_close, precision=4)}")
    print(f"  cube (reach 结束时)       : {np.array2string(cube_before_close, precision=4)}")
    print(f"  TCP-cube 相对位置         : "
          f"{np.array2string(tcp_before_close - cube_before_close, precision=4)}")
    print(f"  first left contact step   : {first_left}")
    print(f"  first right contact step  : {first_right}")
    print(f"  first bilateral step      : {first_bilateral}")
    print(f"  success step              : {success_step}")
    print(f"  is_success                : {success}")
    print(f"  left  contact force (max) : {max_left_force:.3f} N")
    print(f"  right contact force (max) : {max_right_force:.3f} N")
    print(f"  finger qpos (final)       : "
          f"{np.array2string(u.data.qpos[u._finger_qpos_adr], precision=4)}")
    print(f"  cube position (final)     : {np.array2string(final_cube, precision=4)}")
    print(f"  cube 水平位移 (xy drift)  : {cube_xy_drift:.4f} m "
          f"(起始 xy={np.array2string(cube_init[:2], precision=4)})")
    print(f"  cube z 相对静止高度       : {cube_dz_vs_rest:+.4f} m "
          f"(rest_z={u._cube_rest_z:.4f})")
    print(f"  ever_grasped              : {info.get('ever_grasped')}")

    if not success:
        print("\n  失败分类:")
        if not reach_done:
            print("    A: 100 步内没有稳定到达 cube 附近 (TCP 与 cube 几何关系仍不对)")
        elif first_bilateral is None:
            if cube_xy_drift > 0.02:
                print(f"    C: 闭合过程中把 cube 推走 (xy drift={cube_xy_drift:.4f} m)")
            elif first_left is not None and first_right is None:
                print("    D: 只有 left finger contact")
            elif first_right is not None and first_left is None:
                print("    E: 只有 right finger contact")
            else:
                rel = tcp_before_close - cube_before_close
                print(f"    B: 没有接触, 手指可能位于 cube 上方/下方 "
                      f"(TCP-cube dz={rel[2]:+.4f} m)")
        else:
            print(f"    F: 出现双侧 contact 但无法保持 5 步 (max hold={bilateral_count})")

    env.close()
    print("\n" + ("Reach -> Grasp 物理链路验证通过。" if success
                  else "Reach -> Grasp 物理链路验证失败，按上面分类处理。"))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

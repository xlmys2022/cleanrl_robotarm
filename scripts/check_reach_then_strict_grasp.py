#!/usr/bin/env python3
"""Reach checkpoint -> 真实接近 -> scripted close -> strict valid side grasp (PandaGrasp-v1)。

用法:
    python scripts/check_reach_then_strict_grasp.py [--model-path <Reach checkpoint>]

流程（禁止 teleport cube / 修改 qpos）:
    1. Reach actor 的 action[:7] 接近 cube, 夹爪强制全开 (action[7]=+1);
    2. 当 TCP-cube 距离首次 < 0.05 m 时停止 approach, 记录当前 TCP 作为 hold target;
    3. close 阶段: Jacobian + DLS 把 TCP **保持在 hold target**（否则关节目标会继续往下压,
       手指会压到桌面）, 同时 action[7]=-1 闭爪;
    4. 由 PandaGrasp-v1 自己判定 strict success（valid_grasp 连续 5 步）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
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

APPROACH_STOP_DIST = 0.05   # [m] 首次进入该距离就停止 approach（避免继续下压把 cube 压进桌面）
MAX_APPROACH_STEPS = 100
MAX_CLOSE_STEPS = 80
DLS_DAMPING = 0.02
MAX_HOLD_STEP = 0.002       # [m] 每个 control step 的 TCP 修正上限


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def tcp_hold_action(env, target: np.ndarray, gripper: float) -> np.ndarray:
    u = env.unwrapped
    jacp = np.zeros((3, u.model.nv))
    jacr = np.zeros((3, u.model.nv))
    mujoco.mj_jac(u.model, u.data, jacp, jacr, u._get_tcp_position(), u._hand_body_id)
    jac = jacp[:, u._arm_dof_adr]
    dx = np.clip(target - u._get_tcp_position(), -MAX_HOLD_STEP, MAX_HOLD_STEP)
    dq = jac.T @ np.linalg.solve(jac @ jac.T + (DLS_DAMPING ** 2) * np.eye(3), dx)
    action = np.zeros(8, dtype=np.float32)
    action[:7] = np.clip(dq / u.max_joint_delta, -1.0, 1.0)
    action[7] = gripper
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Reach -> strict grasp 物理可行性")
    parser.add_argument("--model-path", default=str(DEFAULT_REACH_CKPT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", default="PandaGrasp-v1")
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

    env = gym.make(args.env_id)
    obs, info = env.reset(seed=args.seed)
    u = env.unwrapped
    print(f"checkpoint : {args.model_path}")
    print(f"env        : {args.env_id}  task_version={info['task_version']}  "
          f"obs={obs.shape} action={env.action_space.shape}")
    print(f"初始: TCP={np.array2string(u._get_tcp_position(), precision=4)} "
          f"cube={np.array2string(u.cube_position, precision=4)} dist={info['tcp_cube_distance']:.4f}")

    # ------------------------------------------------------------------ approach
    rule(f"1. Approach（Reach actor action[:7], 夹爪全开, 首次 dist<{APPROACH_STOP_DIST} 即停止）")
    approach_step = 0
    for step in range(1, MAX_APPROACH_STEPS + 1):
        action = np.zeros(8, dtype=np.float32)
        action[:7] = policy_action(agent, normalize(obs), deterministic=True)[:7]
        action[7] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        approach_step = step
        if step <= 3 or step % 2 == 0 or info["tcp_cube_distance"] < APPROACH_STOP_DIST:
            print(f"  step {step:3d} dist={info['tcp_cube_distance']:.4f} "
                  f"tcp_z={info['tcp_position'][2]:.4f} cube_z={info['cube_position'][2]:.4f} "
                  f"cube_table_pen={info['cube_table_penetration'] * 1000:.1f}mm "
                  f"finger_table_force={info['finger_table_force']:.0f}N")
        if info["tcp_cube_distance"] < APPROACH_STOP_DIST:
            break
        if terminated or truncated:
            break

    hold_target = u._get_tcp_position().copy()
    print(f"\n  approach 结束: step={approach_step}, dist={info['tcp_cube_distance']:.4f} m")
    print(f"  hold target (TCP) = {np.array2string(hold_target, precision=4)}")
    print(f"  cube = {np.array2string(u.cube_position, precision=4)}, "
          f"cube_table_penetration={info['cube_table_penetration'] * 1000:.1f} mm")
    print(f"  finger qpos = {np.array2string(np.asarray(info['finger_joint_qpos']), precision=4)}")

    # ------------------------------------------------------------------ close
    rule(f"2. Close（DLS 保持 TCP 在 hold target + action[7]=-1, 最多 {MAX_CLOSE_STEPS} 步）")
    success_step = None
    first_bilateral = first_valid = None
    max_force = 0.0
    for step in range(1, MAX_CLOSE_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(
            tcp_hold_action(env, hold_target, -1.0)
        )
        max_force = max(max_force, info["left_contact_force"], info["right_contact_force"])
        if info["bilateral_contact"] and first_bilateral is None:
            first_bilateral = step
        if info["grasp_quality_valid"] and first_valid is None:
            first_valid = step
        if step % 5 == 0 or info["is_success"]:
            print(f"  step {step:3d} finger={info['finger_mean_position']:.4f} "
                  f"L/R={info['left_contact_force']:5.2f}/{info['right_contact_force']:5.2f}N "
                  f"bilat={info['bilateral_contact']} valid={info['grasp_quality_valid']} "
                  f"reason={info['grasp_invalid_reason']} hold={info['success_hold_count']} "
                  f"tcp_z={info['tcp_position'][2]:.4f}")
        if info["is_success"]:
            success_step = step
            break
        if terminated or truncated:
            print(f"  [warn] 提前结束 terminated={terminated} truncated={truncated}")
            break

    rule("3. 结果")
    print(f"  approach step            : {approach_step}")
    print(f"  first bilateral step     : {first_bilateral}")
    print(f"  first valid_grasp step   : {first_valid}")
    print(f"  strict success step      : {success_step}")
    print(f"  is_success               : {info['is_success']}")
    print(f"  finger qpos              : "
          f"{np.array2string(np.asarray(info['finger_joint_qpos']), precision=4)}")
    print(f"  finger-cube penetration  : {info['max_finger_cube_penetration'] * 1000:.2f} mm")
    print(f"  cube-table penetration   : {info['cube_table_penetration'] * 1000:.2f} mm")
    print(f"  finger-table contacts    : {info['finger_table_contact_count']} "
          f"({info['finger_table_force']:.1f} N)")
    print(f"  contact force L/R (max)  : {max_force:.2f} N")
    print(f"  normal opposition        : {info['normal_opposition']:+.3f}")
    print(f"  cube position            : {np.array2string(info['cube_position'], precision=4)}")

    env.close()
    if success_step is None:
        print("\n[FAIL] Reach + scripted close 无法形成 strict valid grasp, 需重新检查 "
              "approach pose / closing timing。")
        raise SystemExit(1)
    print("\nReach -> strict grasp 物理链路验证通过（未 teleport cube, 由环境自己判定 success）。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""构造 Curriculum Stage A 的 pre-grasp reference state 并保存成 configs/panda_grasp_stage_a.json。

用法:
    python scripts/build_stage_a_pregrasp.py

流程（复用已经验证成功的 check_reach_then_strict_grasp.py 逻辑）:
    Reach actor approach -> DLS 保持 TCP -> scripted close -> strict valid grasp
    -> 记录此时的 arm qpos / TCP / cube
    -> 把 finger 重新打开到 0.04, gripper ctrl=255, arm ctrl=qpos, cube 放回桌面静止高度
    -> 校验 "open pre-grasp"（无接触、无 penetration、未 grasp）
"""

from __future__ import annotations

import argparse
import json
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
from check_reach_then_strict_grasp import tcp_hold_action  # noqa: E402

DEFAULT_REACH_CKPT = (
    PROJECT_ROOT / "runs/PandaReach-v0__train_robot__1__1793349790/train_robot.checkpoint.pt"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "panda_grasp_stage_a.json"

TASK_VERSION = "grasp_curriculum_stage_a_v1"
APPROACH_STOP_DIST = 0.05
MAX_APPROACH_STEPS = 100
MAX_CLOSE_STEPS = 80
FINGER_OPEN = 0.04
GRIPPER_OPEN_CTRL = 255.0
MAX_CUBE_TABLE_PEN = 0.003


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 Stage A pre-grasp reference")
    parser.add_argument("--model-path", default=str(DEFAULT_REACH_CKPT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(CONFIG_PATH))
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    mean, var = norm["mean"].numpy(), norm["var"].numpy()
    eps = float(norm["epsilon"])
    lo, hi = float(norm["clip_low"]), float(norm["clip_high"])

    def normalize(obs):
        return np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps),
                       lo, hi).astype(np.float32)

    env = gym.make("PandaGrasp-v1")
    obs, info = env.reset(seed=args.seed)
    u = env.unwrapped

    # ---------------------------------------------------------------- 1. reach + close
    rule("1. Reach actor approach -> DLS hold -> scripted close")
    for step in range(1, MAX_APPROACH_STEPS + 1):
        action = np.zeros(8, dtype=np.float32)
        action[:7] = policy_action(agent, normalize(obs), deterministic=True)[:7]
        action[7] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        if info["tcp_cube_distance"] < APPROACH_STOP_DIST:
            break
    hold_target = u._get_tcp_position().copy()
    print(f"  approach step={step}, dist={info['tcp_cube_distance']:.4f}, "
          f"TCP={np.array2string(hold_target, precision=4)}")

    success_step = None
    for step in range(1, MAX_CLOSE_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(tcp_hold_action(env, hold_target, -1.0))
        if info["is_success"]:
            success_step = step
            break
        if terminated or truncated:
            break
    if success_step is None:
        print("[FAIL] 未能通过 scripted close 形成 strict valid grasp, 停止。")
        env.close()
        raise SystemExit(1)
    print(f"  scripted close: valid grasp + success at step {success_step}")

    arm_qpos = u.data.qpos[u._arm_qpos_adr].copy()
    arm_ctrl_grasp = u.data.ctrl[u._arm_act_ids].copy()
    tcp_at_grasp = u._get_tcp_position().copy()
    cube_at_grasp = u.cube_position.copy()
    finger_qpos_grasp = u.data.qpos[u._finger_qpos_adr].copy()
    grasp_metrics = {
        "finger_cube_penetration_m": float(info["max_finger_cube_penetration"]),
        "cube_table_penetration_m": float(info["cube_table_penetration"]),
        "finger_table_force_N": float(info["finger_table_force"]),
        "left_contact_force_N": float(info["left_contact_force"]),
        "right_contact_force_N": float(info["right_contact_force"]),
        "normal_opposition": float(info["normal_opposition"]),
    }
    print(f"  arm qpos      = {np.array2string(arm_qpos, precision=4)}")
    print(f"  arm ctrl      = {np.array2string(arm_ctrl_grasp, precision=4)}")
    print(f"  TCP           = {np.array2string(tcp_at_grasp, precision=4)}")
    print(f"  cube          = {np.array2string(cube_at_grasp, precision=4)}")
    print(f"  finger qpos   = {np.array2string(finger_qpos_grasp, precision=4)}")
    print(f"  grasp metrics = {grasp_metrics}")

    # ---------------------------------------------------------------- 2. build pre-grasp
    rule("2. 构造 open pre-grasp（手指打开 / arm ctrl=qpos / cube 静止在桌面）")
    cube_position = np.array([tcp_at_grasp[0], tcp_at_grasp[1], u._cube_rest_z])
    cube_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    u.data.qpos[u._arm_qpos_adr] = arm_qpos
    u.data.ctrl[u._arm_act_ids] = arm_qpos          # 目标 == 当前姿态, 机械臂不会继续往下压
    u.data.qpos[u._finger_qpos_adr] = FINGER_OPEN
    u.data.ctrl[u._gripper_act_id] = GRIPPER_OPEN_CTRL
    u.data.qpos[u._cube_qpos_adr: u._cube_qpos_adr + 3] = cube_position
    u.data.qpos[u._cube_qpos_adr + 3: u._cube_qpos_adr + 7] = cube_quaternion
    u.data.qvel[:] = 0.0
    mujoco.mj_forward(u.model, u.data)

    print(f"  cube_position   = {np.array2string(cube_position, precision=4)} "
          f"(cube_rest_z={u._cube_rest_z:.4f})")
    print(f"  finger_open     = {FINGER_OPEN}, gripper ctrl = {GRIPPER_OPEN_CTRL}")

    # ---------------------------------------------------------------- 3. verify pre-grasp
    rule("3. 校验 reset 后确实是 open pre-grasp")
    u._reset_task_state()      # 清掉抓取过程中残留的 task flag（等价于 env.reset 的初始化）
    u._init_potentials()
    m = u._task_metrics()
    tcp = u._get_tcp_position()
    print(f"  TCP                  = {np.array2string(tcp, precision=4)}")
    print(f"  cube                 = {np.array2string(m['cube'], precision=4)}")
    print(f"  TCP-cube relative    = {np.array2string(tcp - m['cube'], precision=4)}")
    print(f"  finger qpos          = {np.array2string(np.asarray(m['finger_qpos']), precision=4)}")
    print(f"  bilateral_contact    = {m['bilateral_contact']}")
    print(f"  grasp_quality_valid  = {m['grasp_quality_valid']} ({m['grasp_invalid_reason']})")
    print(f"  finger-table contacts= {m['finger_table_contact_count']} "
          f"({m['finger_table_force']:.2f} N)")
    print(f"  finger-cube penetr.  = {m['max_finger_cube_penetration'] * 1000:.2f} mm")
    print(f"  cube-table penetr.   = {m['cube_table_penetration'] * 1000:.2f} mm")
    print(f"  ncon                 = {u.data.ncon}")
    print(f"  保存的 arm ctrl = arm qpos（目标==当前姿态, 不会继续下降）；"
          f"grasp 时 DLS ctrl 与 qpos 相差 {np.max(np.abs(arm_qpos - arm_ctrl_grasp)):.4f} rad")

    checks = {
        "open_gripper": float(np.mean(m["finger_qpos"])) > 0.03,
        "no_bilateral": not m["bilateral_contact"],
        "not_grasped": not u._is_grasped,
        "no_finger_table": m["finger_table_contact_count"] == 0,
        "no_penetration": m["max_finger_cube_penetration"] < 0.002,
        "cube_on_table": m["cube_table_penetration"] <= MAX_CUBE_TABLE_PEN,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        env.close()
        raise SystemExit(1)

    # ---------------------------------------------------------------- 4. save
    rule("4. 保存 reference")
    config = {
        "task_version": TASK_VERSION,
        "source": {
            "reach_checkpoint": str(Path(args.model_path).relative_to(PROJECT_ROOT)),
            "procedure": "reach approach (dist<0.05) -> DLS hold -> scripted close -> valid grasp",
            "grasp_success_step": int(success_step),
            "seed": int(args.seed),
        },
        "arm_qpos": [float(x) for x in arm_qpos],
        "arm_ctrl": [float(x) for x in arm_qpos],
        "cube_position": [float(x) for x in cube_position],
        "cube_quaternion": [float(x) for x in cube_quaternion],
        "finger_open_position": FINGER_OPEN,
        "gripper_open_ctrl": GRIPPER_OPEN_CTRL,
        "reference_tcp_position": [float(x) for x in tcp],
        "measured_grasp_state": grasp_metrics,
        "measured_at_grasp": {
            "tcp": [float(x) for x in tcp_at_grasp],
            "cube": [float(x) for x in cube_at_grasp],
            "finger_qpos": [float(x) for x in finger_qpos_grasp],
        },
        "derived": {
            "cube_rest_z": float(u._cube_rest_z),
            "table_top_z": float(u._table_top_z),
            "cube_half_height": float(u._cube_half_height),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"  已写入 {output}")
    env.close()


if __name__ == "__main__":
    main()

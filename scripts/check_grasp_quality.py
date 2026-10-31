#!/usr/bin/env python3
"""标定 grasp quality 指标: 对比 "learned bad grasp" 与 "proper side pinch"。

用法:
    python scripts/check_grasp_quality.py

Case A: PandaGrasp-v0 + 旧 Grasp checkpoint（deterministic）跑到旧版 GRASP SUCCESS
Case B: 同一个环境里构造 proper side pinch（测试脚本允许注入, 仅用于标定）
输出统一对照表 + 当前 strict 阈值, 用于决定 PandaGrasp-v1 的 gate。
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
from check_grasp_contacts import setup_cube_in_gripper  # noqa: E402
from envs.panda_pick_place_env import (  # noqa: E402
    MAX_CUBE_TABLE_PENETRATION,
    MAX_FINGER_CUBE_PENETRATION,
    MAX_FINGER_TABLE_FORCE,
)

DEFAULT_GRASP_CKPT = (
    PROJECT_ROOT / "runs/PandaGrasp-v0__train_robot__1__1793353158/train_robot.checkpoint.pt"
)

METRIC_ROWS = (
    ("bilateral contact", lambda m: m["bilateral_contact"], "{}"),
    ("grasp_quality_valid", lambda m: m["grasp_quality_valid"], "{}"),
    ("invalid reason", lambda m: m["grasp_invalid_reason"], "{}"),
    ("finger-table contacts", lambda m: m["finger_table_contact_count"], "{}"),
    ("finger-table force [N]", lambda m: m["finger_table_force"], "{:.2f}"),
    ("finger-cube penetration [mm]", lambda m: m["max_finger_cube_penetration"] * 1000, "{:.2f}"),
    ("cube-table penetration [mm]", lambda m: m["cube_table_penetration"] * 1000, "{:.2f}"),
    ("finger qpos (mean) [m]", lambda m: float(np.mean(m["finger_qpos"])), "{:.4f}"),
    ("contact force L [N]", lambda m: m["left_contact_force"], "{:.2f}"),
    ("contact force R [N]", lambda m: m["right_contact_force"], "{:.2f}"),
    ("normal opposition", lambda m: m["normal_opposition"], "{:+.3f}"),
    ("TCP-cube z diff [mm]", lambda m: (m["tcp"][2] - m["cube"][2]) * 1000, "{:+.1f}"),
)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def case_bad_grasp(model_path: Path) -> dict:
    """Case A: 旧 checkpoint 在 legacy 环境里形成的"下压式"抓取。"""
    checkpoint = load_checkpoint(model_path)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    mean, var = norm["mean"].numpy(), norm["var"].numpy()
    eps = float(norm["epsilon"])
    clip_lo, clip_hi = float(norm["clip_low"]), float(norm["clip_high"])

    env = gym.make("PandaGrasp-v0")
    obs, info = env.reset(seed=0)
    u = env.unwrapped
    for step in range(1, 61):
        obs_norm = np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps),
                           clip_lo, clip_hi).astype(np.float32)
        action = policy_action(agent, obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["is_success"]:
            print(f"  旧环境判定 GRASP SUCCESS: step={step}, "
                  f"dist={info['tcp_cube_distance']:.4f} m")
            break
        if terminated or truncated:
            break
    metrics = u._task_metrics()
    env.close()
    return metrics


def case_proper_pinch() -> dict:
    """Case B: cube 置于两指之间 -> 真实闭爪 -> 侧面接触（只用于标定）。"""
    env = gym.make("PandaGrasp-v0")
    env.reset(seed=0)
    u = env.unwrapped
    setup_cube_in_gripper(env)
    action = np.zeros(8, dtype=np.float32)
    action[7] = -1.0
    info = {}
    for step in range(1, 21):
        _, _, terminated, truncated, info = env.step(action)
        if info["is_success"] or terminated or truncated:
            print(f"  proper pinch: 第 {step} 步 is_grasped={info['is_grasped']}")
            break
    metrics = u._task_metrics()
    env.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="grasp quality 指标标定")
    parser.add_argument("--model-path", default=str(DEFAULT_GRASP_CKPT))
    args = parser.parse_args()

    rule("Case A: learned bad grasp（PandaGrasp-v0 + 旧 checkpoint）")
    bad = case_bad_grasp(Path(args.model_path))

    rule("Case B: proper side pinch（测试脚本构造）")
    good = case_proper_pinch()

    rule("Good vs Bad 对照表")
    print(f"  {'metric':<32}{'bad RL grasp':>18}{'proper side pinch':>22}")
    print("  " + "-" * 70)
    for label, fn, fmt in METRIC_ROWS:
        try:
            left = fmt.format(fn(bad))
        except (TypeError, ValueError):
            left = str(fn(bad))
        try:
            right = fmt.format(fn(good))
        except (TypeError, ValueError):
            right = str(fn(good))
        print(f"  {label:<32}{left:>18}{right:>22}")

    rule("当前 strict 阈值（envs/panda_pick_place_env.py）")
    print(f"  MAX_FINGER_CUBE_PENETRATION = {MAX_FINGER_CUBE_PENETRATION * 1000:.1f} mm")
    print(f"  MAX_CUBE_TABLE_PENETRATION  = {MAX_CUBE_TABLE_PENETRATION * 1000:.1f} mm")
    print(f"  MAX_FINGER_TABLE_FORCE      = {MAX_FINGER_TABLE_FORCE:.2f} N")
    print(f"\n  判定: bad grasp valid={bad['grasp_quality_valid']} "
          f"({bad['grasp_invalid_reason']}), "
          f"proper pinch valid={good['grasp_quality_valid']} "
          f"({good['grasp_invalid_reason']})")


if __name__ == "__main__":
    main()

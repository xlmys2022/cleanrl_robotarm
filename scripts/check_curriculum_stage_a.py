#!/usr/bin/env python3
"""Curriculum Stage A 的 scripted sanity check：只闭爪就应该 valid grasp + success。

用法:
    python scripts/check_curriculum_stage_a.py [--episodes 5]

每个 episode:
    reset Stage A -> 校验 reset 后是 open pre-grasp（未 grasp、无接触、无穿透）
    -> action[:7]=0（保持姿态）, action[7]=-1（真实闭爪）
    -> 由 PandaGraspCurriculum-A-v0 自己判定 valid_grasp / success

禁止: teleport cube、改 contact flag、改 success_hold_count、手工设置 is_grasped。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401

ENV_ID = "PandaGraspCurriculum-A-v0"
MAX_STEPS = 100

RESULTS: list[tuple[str, bool]] = []


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(label: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    RESULTS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  [{detail}]" if detail else ""))


CLOSE = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)


def run_episode(env, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    u = env.unwrapped
    initial = {
        "obs_shape": obs.shape,
        "action_shape": env.action_space.shape,
        "bilateral": info["bilateral_contact"],
        "is_grasped": info["is_grasped"],
        "valid": info["grasp_quality_valid"],
        "finger_table_contacts": info["finger_table_contact_count"],
        "finger_table_force": info["finger_table_force"],
        "finger_penetration": info["max_finger_cube_penetration"],
        "cube_table_penetration": info["cube_table_penetration"],
        "finger_qpos": np.asarray(info["finger_joint_qpos"]).copy(),
        "tcp": info["tcp_position"].copy(),
        "cube": info["cube_position"].copy(),
        "dist": info["tcp_cube_distance"],
        "task_version": info["task_version"],
    }

    first_bilateral = first_valid = success_step = None
    max_left = max_right = 0.0
    for step in range(1, MAX_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(CLOSE)
        max_left = max(max_left, info["left_contact_force"])
        max_right = max(max_right, info["right_contact_force"])
        if info["bilateral_contact"] and first_bilateral is None:
            first_bilateral = step
        if info["grasp_quality_valid"] and first_valid is None:
            first_valid = step
        if info["is_success"]:
            success_step = step
        if terminated or truncated:
            break

    return {
        "initial": initial,
        "steps": step,
        "first_bilateral": first_bilateral,
        "first_valid": first_valid,
        "success_step": success_step,
        "is_success": bool(info["is_success"]),
        "finger_qpos": np.asarray(info["finger_joint_qpos"]).copy(),
        "left_force": max_left,
        "right_force": max_right,
        "penetration": info["max_finger_cube_penetration"],
        "cube_table_penetration": info["cube_table_penetration"],
        "finger_table_force": info["finger_table_force"],
        "cube_drift": float(np.linalg.norm(np.asarray(info["cube_position"])[:2]
                                           - initial["cube"][:2])),
        "normal_opposition": info["normal_opposition"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum Stage A scripted close sanity")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(ENV_ID)
    u = env.unwrapped
    rule(f"配置: {ENV_ID}")
    print(f"  task_version        = {u.task_version}")
    print(f"  initialization_mode = {u.initialization_mode}")
    print(f"  max_episode_steps   = {u.max_episode_steps} "
          f"({u.max_episode_steps * u.control_dt:.1f} s)")
    print(f"  observation / action= {env.observation_space.shape} / {env.action_space.shape}")

    episodes = []
    for ep in range(args.episodes):
        result = run_episode(env, args.seed + ep)
        episodes.append(result)
        init = result["initial"]
        print(f"\n  episode {ep + 1}:")
        print(f"    reset: bilateral={init['bilateral']} is_grasped={init['is_grasped']} "
              f"valid={init['valid']} finger={np.round(init['finger_qpos'], 4)} "
              f"dist={init['dist']:.4f}")
        print(f"           finger_table={init['finger_table_contacts']}"
              f"({init['finger_table_force']:.2f}N) "
              f"penetration={init['finger_penetration'] * 1000:.2f}mm "
              f"cube_table_pen={init['cube_table_penetration'] * 1000:.2f}mm")
        print(f"    close -> first bilateral @{result['first_bilateral']}, "
              f"first valid @{result['first_valid']}, success @{result['success_step']} "
              f"(steps={result['steps']})")
        print(f"    final: finger={np.round(result['finger_qpos'], 4)} "
              f"force L/R={result['left_force']:.2f}/{result['right_force']:.2f}N "
              f"penetration={result['penetration'] * 1000:.2f}mm "
              f"cube_table_pen={result['cube_table_penetration'] * 1000:.2f}mm "
              f"drift={result['cube_drift'] * 1000:.2f}mm "
              f"normal_opp={result['normal_opposition']:+.3f}")

    rule("验收")
    for i, e in enumerate(episodes):
        init = e["initial"]
        check(f"ep{i + 1}: reset 是 open pre-grasp（未 grasp / 无 finger-table / 无穿透）",
              (not init["bilateral"]) and (not init["is_grasped"]) and (not init["valid"])
              and init["finger_table_contacts"] == 0
              and init["finger_penetration"] < 0.002
              and init["cube_table_penetration"] <= 0.003)
        check(f"ep{i + 1}: 闭爪后 valid grasp 且 success", e["is_success"] is True,
              f"success@{e['success_step']}")
    passed = sum(ok for _, ok in RESULTS)
    print(f"\n  scripted close: {sum(1 for e in episodes if e['is_success'])}/"
          f"{len(episodes)} success, {passed}/{len(RESULTS)} checks passed")
    env.close()
    if not all(ok for _, ok in RESULTS):
        raise SystemExit(1)
    print("\nCurriculum Stage A scripted sanity passed.")


if __name__ == "__main__":
    main()

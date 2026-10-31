#!/usr/bin/env python3
"""Stage B calibration：候选 XY 半范围 -> scripted oracle 可行性 + Stage A zero-shot 成功率。

用法:
    python scripts/calibrate_curriculum_stage_b.py

输出并写入 configs/panda_grasp_stage_b.json：
    - 每个候选 range 的 scripted 9/9 grid 成功率
    - 每个候选 range 的 Stage A checkpoint zero-shot（deterministic, 25 random + 9 grid）
    - 按规则选出的 xy_half_range_m
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
from envs.panda_pick_place_env import (  # noqa: E402
    CONFIG_DIR,
    PandaPickPlaceEnv,
    STAGE_A_CONFIG_PATH,
    TASK_VERSION_CURRICULUM_B,
)

CANDIDATE_RANGES = (0.005, 0.010, 0.015, 0.020)
STAGE_A_CKPT = (PROJECT_ROOT
                / "runs/PandaGraspCurriculum-A-v0__train_robot__1__1793362519"
                / "train_robot.checkpoint.pt")
OUTPUT = CONFIG_DIR / "panda_grasp_stage_b.json"

ZERO_SHOT_EPISODES = 25
DLS_DAMPING = 0.02
MAX_MOVE_STEP = 0.002
MAX_CLOSE_STEPS = 60


def rule(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def make_env(half_range: float) -> PandaPickPlaceEnv:
    return PandaPickPlaceEnv(
        task="grasp",
        grasp_quality_mode="strict",
        initialization_mode="pregrasp_stage_b",
        stage_b_half_range_m=half_range,
        max_episode_steps=100,
    )


def dls_action(u, dx: np.ndarray, gripper: float) -> np.ndarray:
    jacp = np.zeros((3, u.model.nv))
    jacr = np.zeros((3, u.model.nv))
    mujoco.mj_jac(u.model, u.data, jacp, jacr, u._get_tcp_position(), u._hand_body_id)
    jac = jacp[:, u._arm_dof_adr]
    dq = jac.T @ np.linalg.solve(jac @ jac.T + (DLS_DAMPING ** 2) * np.eye(3), dx)
    action = np.zeros(8, dtype=np.float32)
    action[:7] = np.clip(dq / u.max_joint_delta, -1.0, 1.0)
    action[7] = gripper
    return action


def scripted_oracle(env, dx: float, dy: float, target_offset: np.ndarray) -> tuple[bool, int | None]:
    """TCP -> cube + Stage A 的 TCP-cube 相对偏移 -> 保持 -> 闭爪。"""
    env.reset(seed=0, options={"cube_xy_offset": [dx, dy]})
    u = env.unwrapped
    target = u.cube_position + target_offset

    for _ in range(60):
        err = target - u._get_tcp_position()
        if np.linalg.norm(err) < 5e-4:
            break
        _, _, terminated, truncated, info = env.step(
            dls_action(u, np.clip(err, -MAX_MOVE_STEP, MAX_MOVE_STEP), gripper=1.0)
        )
        if terminated or truncated:
            return False, None

    for step in range(1, MAX_CLOSE_STEPS + 1):
        err = np.clip(target - u._get_tcp_position(), -MAX_MOVE_STEP, MAX_MOVE_STEP)
        _, _, terminated, truncated, info = env.step(dls_action(u, err, gripper=-1.0))
        if info["is_success"]:
            return True, step
        if terminated or truncated:
            break
    return False, None


def policy_episode(env, agent, norm, seed: int,
                   offset: tuple[float, float] | None) -> dict:
    mean, var, eps_n, clip_lo, clip_hi = norm
    options = None if offset is None else {"cube_xy_offset": list(offset)}
    obs, info = env.reset(seed=seed, options=options)
    offset_used = tuple(np.asarray(info["initial_cube_xy_offset"], dtype=float))
    success_step = None
    step = 0
    while True:
        step += 1
        obs_norm = np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps_n),
                           clip_lo, clip_hi).astype(np.float32)
        action = policy_action(agent, obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["is_success"] and success_step is None:
            success_step = step
        if terminated or truncated:
            break
    return {
        "success": bool(info["is_success"]),
        "valid": bool(info["grasp_quality_valid"]),
        "offset": offset_used,
        "success_step": success_step,
        "steps": step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B calibration")
    parser.add_argument("--model-path", default=str(STAGE_A_CKPT))
    args = parser.parse_args()

    stage_a = json.loads(STAGE_A_CONFIG_PATH.read_text())
    target_offset = (np.asarray(stage_a["reference_tcp_position"], dtype=float)
                     - np.asarray(stage_a["cube_position"], dtype=float))

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    norm_block = checkpoint["observation_normalization"]
    norm = (
        norm_block["mean"].numpy(),
        norm_block["var"].numpy(),
        float(norm_block["epsilon"]),
        float(norm_block["clip_low"]),
        float(norm_block["clip_high"]),
    )
    print(f"Stage A checkpoint: {args.model_path}")
    print(f"target TCP-cube offset (from config): {np.array2string(target_offset, precision=5)}")

    grid = [(0.0, 0.0),
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

    results = {}
    for half_range in CANDIDATE_RANGES:
        rule(f"candidate range ±{half_range * 1000:.0f} mm")
        env = make_env(half_range)

        scripted_ok = 0
        for gx, gy in grid:
            ok, step = scripted_oracle(env, gx * half_range, gy * half_range, target_offset)
            scripted_ok += int(ok)
            print(f"  scripted ({gx:+.0f},{gy:+.0f})*r: success={ok} step={step}")

        zero_shot_grid = 0
        for gx, gy in grid:
            r = policy_episode(env, agent, norm, seed=0,
                               offset=(gx * half_range, gy * half_range))
            zero_shot_grid += int(r["success"])
        zero_shot_random = sum(
            policy_episode(env, agent, norm, seed=s, offset=None)["success"]
            for s in range(ZERO_SHOT_EPISODES)
        )
        env.close()

        results[half_range] = {
            "scripted_grid": f"{scripted_ok}/9",
            "scripted_grid_success": scripted_ok,
            "zero_shot_grid": f"{zero_shot_grid}/9",
            "zero_shot_random": f"{zero_shot_random}/{ZERO_SHOT_EPISODES}",
            "zero_shot_rate": zero_shot_random / ZERO_SHOT_EPISODES,
        }
        print(f"  scripted grid = {scripted_ok}/9, "
              f"zero-shot grid = {zero_shot_grid}/9, "
              f"zero-shot random = {zero_shot_random}/{ZERO_SHOT_EPISODES}")

    rule("Calibration 汇总")
    print(f"  {'range':>8} | {'scripted':>9} | {'zero-shot(25)':>13} | zero-shot grid")
    for half_range, r in results.items():
        print(f"  {half_range * 1000:6.0f}mm | {r['scripted_grid']:>9} | "
              f"{r['zero_shot_random']:>13} | {r['zero_shot_grid']}")

    feasible = [hr for hr, r in results.items() if r["scripted_grid_success"] == 9]
    if not feasible:
        print("\n[FAIL] 所有候选 range 的 scripted oracle 都做不到 9/9，停止。")
        raise SystemExit(1)

    band = [hr for hr in feasible if 0.20 <= results[hr]["zero_shot_rate"] < 0.90]
    if band:
        selected = min(band)
        reason = ("最小的、zero-shot 落在 [20%, 90%) 且 scripted 9/9 的候选范围"
                  "（留出学习空间又不至于完全不会）")
    elif all(results[hr]["zero_shot_rate"] >= 0.90 for hr in feasible):
        selected = max(feasible)
        reason = "所有候选 zero-shot 都 >= 90%，按规则取最大范围；说明 Stage A 本身已很鲁棒"
    else:
        selected = min(feasible)
        reason = "最小候选的 zero-shot 已 < 20%，但 scripted 9/9，按要求选最小范围"

    rule("选择结果")
    print(f"  selected half range = ±{selected * 1000:.0f} mm")
    print(f"  reason: {reason}")

    config = {
        "task_version": TASK_VERSION_CURRICULUM_B,
        "base_reference_config": "configs/panda_grasp_stage_a.json",
        "stage_a_checkpoint": str(Path(args.model_path).relative_to(PROJECT_ROOT)),
        "xy_half_range_m": float(selected),
        "selected_range": float(selected),
        "candidate_ranges_tested": [float(r) for r in CANDIDATE_RANGES],
        "scripted_grid_success": {f"{hr * 1000:.0f}mm": r["scripted_grid"]
                                  for hr, r in results.items()},
        "stage_a_zero_shot_success": {
            f"{hr * 1000:.0f}mm": {
                "random_25": r["zero_shot_random"],
                "grid_9": r["zero_shot_grid"],
            } for hr, r in results.items()
        },
        "selection_reason": reason,
        "grid_definition": "{(0,0),(±r,0),(0,±r),(±r,±r)}",
        "created_by": "scripts/calibrate_curriculum_stage_b.py",
    }
    OUTPUT.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"  已写入 {OUTPUT}")


if __name__ == "__main__":
    main()

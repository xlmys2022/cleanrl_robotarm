#!/usr/bin/env python3
"""Stage B.8 约束 curriculum 标定：为 "持续高力压桌 => failure" 选第一个 hold_steps。

用法:
    python scripts/calibrate_table_constraint_curriculum.py

对每个候选 hold_steps ∈ {15,12,10,8,7,5,3}:
    Test A clean scripted 9 boundary offsets  -> 必须 9/9 success, 0 constraint failure
    Test B parent deterministic 50             -> 期望 success <= 20%
    Test C parent stochastic 200               -> 必须 success > 0（否则探索信号被切断）

选择规则: 满足 A+B+C 的最大 hold_steps（最宽松的第一阶段）。
结果写入 configs/panda_grasp_constraint_curriculum.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from envs.panda_pick_place_env import STAGE_A_CONFIG_PATH  # noqa: E402
from eval_reach import (  # noqa: E402
    FrozenNormalizeObservation,
    build_agent,
    load_checkpoint,
    policy_action,
)

BASE_ENV_ID = "PandaGraspCurriculum-B-v3"
PARENT_CHECKPOINT = ("runs/PandaGraspCurriculum-B-v0__train_robot__1__1793364127/"
                     "train_robot.checkpoint.pt")
CONFIG_PATH = PROJECT_ROOT / "configs" / "panda_grasp_constraint_curriculum.json"

CANDIDATE_HOLD_STEPS = (15, 12, 10, 8, 7, 5, 3)
DET_EPISODES = 50
STOCH_EPISODES = 200
SEED = 0
FORCE_THRESHOLD = 100.0
DLS_DAMPING = 0.02
MAX_MOVE_STEP = 0.002
MAX_CLOSE_STEPS = 60
DET_SUCCESS_MAX = 0.20


def dls_action(u, delta: np.ndarray, gripper: float) -> np.ndarray:
    jacp = np.zeros((3, u.model.nv))
    jacr = np.zeros((3, u.model.nv))
    mujoco.mj_jac(u.model, u.data, jacp, jacr, u._get_tcp_position(), u._hand_body_id)
    jac = jacp[:, u._arm_dof_adr]
    dq = jac.T @ np.linalg.solve(jac @ jac.T + (DLS_DAMPING ** 2) * np.eye(3), delta)
    action = np.zeros(8, dtype=np.float32)
    action[:7] = np.clip(dq / u.max_joint_delta, -1.0, 1.0)
    action[7] = gripper
    return action


def make_env(hold_steps: int, checkpoint: dict | None):
    env = gym.make(BASE_ENV_ID, table_force_failure_hold_steps=int(hold_steps))
    if checkpoint is not None:
        norm = checkpoint["observation_normalization"]
        env = FrozenNormalizeObservation(
            env,
            mean=norm["mean"].numpy(),
            var=norm["var"].numpy(),
            epsilon=norm["epsilon"],
            clip_low=norm.get("clip_low", -10.0),
            clip_high=norm.get("clip_high", 10.0),
        )
    return env


def scripted_clean_test(env, offsets: list[tuple[float, float]],
                        target_offset: np.ndarray) -> dict:
    successes = failures = 0
    for dx, dy in offsets:
        env.reset(seed=0, options={"cube_xy_offset": [dx, dy]})
        u = env.unwrapped
        target = u.cube_position + target_offset
        ok = False
        terminated = truncated = False
        info = {"failure_reason": None}
        for _ in range(60):
            err = target - u._get_tcp_position()
            if np.linalg.norm(err) < 5e-4:
                break
            _, _, terminated, truncated, info = env.step(
                dls_action(u, np.clip(err, -MAX_MOVE_STEP, MAX_MOVE_STEP), gripper=1.0)
            )
            if terminated or truncated:
                break
        if not (terminated or truncated):
            for _ in range(MAX_CLOSE_STEPS):
                err = np.clip(target - u._get_tcp_position(), -MAX_MOVE_STEP, MAX_MOVE_STEP)
                _, _, terminated, truncated, info = env.step(dls_action(u, err, gripper=-1.0))
                if info["is_success"]:
                    ok = True
                    break
                if terminated or truncated:
                    break
        successes += int(ok)
        failures += int(info["failure_reason"] == "sustained_finger_table_force")
    return {"success": successes, "constraint_failures": failures, "total": len(offsets)}


def run_policy_episode(env, agent, seed: int, deterministic: bool) -> dict:
    obs, info = env.reset(seed=seed)
    u = env.unwrapped
    start_xy = np.array(u.cube_position[:2])
    first_high = first_table = None
    high_steps = 0
    segments: list[int] = []
    current_segment = 0
    max_force = 0.0
    finger_table_steps = 0
    success_step = None
    steps = 0
    while True:
        action = policy_action(agent, obs, deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        force = float(info["finger_table_force"])
        max_force = max(max_force, force)
        if info["finger_table_contact_count"] > 0:
            finger_table_steps += 1
            if first_table is None:
                first_table = steps
        if force > FORCE_THRESHOLD:
            high_steps += 1
            current_segment += 1
            if first_high is None:
                first_high = steps
        elif current_segment > 0:
            segments.append(current_segment)
            current_segment = 0
        if info["is_success"] and success_step is None:
            success_step = steps
        if terminated or truncated:
            break
    if current_segment > 0:
        segments.append(current_segment)
    failure_reason = info["failure_reason"]
    constraint_failed = failure_reason == "sustained_finger_table_force"
    return {
        "success": bool(info["is_success"]),
        "constraint_failed": constraint_failed,
        "valid": bool(info["grasp_quality_valid"]),
        "length": steps,
        "failure_step": steps if constraint_failed else None,
        "success_step": success_step,
        "first_high_step": first_high,
        "first_table_step": first_table,
        "high_force_steps": high_steps,
        "max_consecutive_high_force": max(segments) if segments else 0,
        "high_force_segments": len(segments),
        "max_table_force": max_force,
        "finger_table_steps": finger_table_steps,
        "cube_drift": float(np.linalg.norm(np.array(u.cube_position[:2]) - start_xy)),
    }


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    failures = [e for e in episodes if e["constraint_failed"]]
    successes = [e for e in episodes if e["success"]]
    return {
        "episodes": n,
        "success": len(successes),
        "success_rate": len(successes) / n,
        "constraint_failure": len(failures),
        "constraint_failure_rate": len(failures) / n,
        "valid_grasp": sum(e["valid"] for e in episodes),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "mean_failure_step": float(np.mean([e["failure_step"] for e in failures]))
        if failures else None,
        "mean_first_high_step": float(np.mean([e["first_high_step"] for e in episodes
                                               if e["first_high_step"]]))
        if any(e["first_high_step"] for e in episodes) else None,
        "mean_first_table_step": float(np.mean([e["first_table_step"] for e in episodes
                                                if e["first_table_step"]]))
        if any(e["first_table_step"] for e in episodes) else None,
        "success_detail": [
            {
                "success_step": e["success_step"],
                "high_force_steps": e["high_force_steps"],
                "max_consecutive_high_force": e["max_consecutive_high_force"],
                "high_force_segments": e["high_force_segments"],
                "max_table_force": e["max_table_force"],
                "finger_table_steps": e["finger_table_steps"],
                "cube_drift": e["cube_drift"],
            }
            for e in successes[:20]
        ],
    }


def calibrate_hold(hold_steps: int, checkpoint: dict, agent, offsets, target_offset) -> dict:
    env = make_env(hold_steps, checkpoint)
    clean = scripted_clean_test(env, offsets, target_offset)

    torch.manual_seed(SEED)
    det = [run_policy_episode(env, agent, SEED + ep, True) for ep in range(DET_EPISODES)]
    det_summary = summarize(det)

    torch.manual_seed(SEED)
    stoch = [run_policy_episode(env, agent, SEED + ep, False) for ep in range(STOCH_EPISODES)]
    stoch_summary = summarize(stoch)
    env.close()

    return {
        "hold_steps": hold_steps,
        "clean_scripted": clean,
        "deterministic": det_summary,
        "stochastic": stoch_summary,
    }


def main() -> None:
    stage_a = json.loads(STAGE_A_CONFIG_PATH.read_text())
    target_offset = (np.asarray(stage_a["reference_tcp_position"], dtype=float)
                     - np.asarray(stage_a["cube_position"], dtype=float))
    half_range = 0.02
    offsets = [(0.0, 0.0), (half_range, 0.0), (-half_range, 0.0), (0.0, half_range),
               (0.0, -half_range), (half_range, half_range), (half_range, -half_range),
               (-half_range, half_range), (-half_range, -half_range)]

    checkpoint = load_checkpoint(PROJECT_ROOT / PARENT_CHECKPOINT)
    agent = build_agent(checkpoint)
    print(f"parent checkpoint: {PARENT_CHECKPOINT}")
    print(f"candidates: {list(CANDIDATE_HOLD_STEPS)}")
    print(f"det episodes={DET_EPISODES}, stochastic episodes={STOCH_EPISODES}, "
          f"threshold={FORCE_THRESHOLD} N")

    results = []
    for hold in CANDIDATE_HOLD_STEPS:
        print(f"\n--- hold_steps={hold} ({hold * 0.02:.2f}s) ---", flush=True)
        r = calibrate_hold(hold, checkpoint, agent, offsets, target_offset)
        d, s = r["deterministic"], r["stochastic"]
        print(f"  clean scripted : {r['clean_scripted']['success']}/9 success, "
              f"{r['clean_scripted']['constraint_failures']} constraint failures")
        print(f"  det  50        : success {d['success']}/{d['episodes']}, "
              f"failure {d['constraint_failure_rate'] * 100:.0f}%, "
              f"mean fail step {d['mean_failure_step']}, "
              f"mean len {d['mean_length']:.1f}, valid {d['valid_grasp']}")
        print(f"  stoch 200      : success {s['success']}/{s['episodes']}, "
              f"failure {s['constraint_failure_rate'] * 100:.0f}%, "
              f"mean fail step {s['mean_failure_step']}, "
              f"mean len {s['mean_length']:.1f}, valid {s['valid_grasp']}, "
              f"first>100N {s['mean_first_high_step']}")
        for detail in s["success_detail"][:5]:
            print(f"    success ep: step={detail['success_step']} "
                  f">100N={detail['high_force_steps']} "
                  f"max_consec={detail['max_consecutive_high_force']} "
                  f"segments={detail['high_force_segments']} "
                  f"maxF={detail['max_table_force']:.0f}N "
                  f"drift={detail['cube_drift'] * 100:.2f}cm")
        results.append(r)

    print()
    print("=" * 100)
    print("calibration table")
    print("=" * 100)
    print(f"{'hold':>4} | {'clean':>7} | {'det succ':>8} | {'det fail':>8} | "
          f"{'stoch succ':>10} | {'stoch fail':>10} | {'mean fail step':>14} | eligible")
    for r in results:
        d, s = r["deterministic"], r["stochastic"]
        eligible = (r["clean_scripted"]["success"] == 9
                    and d["success_rate"] <= DET_SUCCESS_MAX
                    and s["success"] > 0)
        mean_fail = s["mean_failure_step"]
        print(f"{r['hold_steps']:>4} | "
              f"{r['clean_scripted']['success']:>3}/9   | "
              f"{d['success']:>3}/50   | {d['constraint_failure_rate'] * 100:>6.0f}%  | "
              f"{s['success']:>4}/200   | {s['constraint_failure_rate'] * 100:>6.0f}%    | "
              f"{(f'{mean_fail:.1f}' if mean_fail is not None else 'n/a'):>14} | "
              f"{'YES' if eligible else 'no'}")

    eligible_results = [r for r in results
                        if r["clean_scripted"]["success"] == 9
                        and r["deterministic"]["success_rate"] <= DET_SUCCESS_MAX
                        and r["stochastic"]["success"] > 0]
    if not eligible_results:
        print("\n没有候选同时满足 clean 9/9 + det success<=20% + stochastic success>0。")
        print("按规范 §十三 停止：仅修改 hold_steps 无法构造平滑 curriculum。")
        sys.exit(2)

    selected = max(eligible_results, key=lambda r: r["hold_steps"])
    print(f"\nselected hold_steps = {selected['hold_steps']} "
          f"(largest eligible / most permissive first stage)")

    config = {
        "task_version": "grasp_constraint_curriculum_v1",
        "force_threshold": FORCE_THRESHOLD,
        "candidate_hold_steps": list(CANDIDATE_HOLD_STEPS),
        "selected_hold_steps": int(selected["hold_steps"]),
        "parent_checkpoint": PARENT_CHECKPOINT,
        "deterministic_success_max": DET_SUCCESS_MAX,
        "calibration_results": {
            str(r["hold_steps"]): {
                "clean_scripted_success": f"{r['clean_scripted']['success']}/9",
                "clean_scripted_constraint_failures": r["clean_scripted"]["constraint_failures"],
                "det_success": f"{r['deterministic']['success']}/50",
                "det_constraint_failure_rate": r["deterministic"]["constraint_failure_rate"],
                "stoch_success": f"{r['stochastic']['success']}/200",
                "stoch_constraint_failure_rate": r["stochastic"]["constraint_failure_rate"],
                "stoch_mean_failure_step": r["stochastic"]["mean_failure_step"],
                "stoch_mean_first_high_force_step": r["stochastic"]["mean_first_high_step"],
                "stoch_success_detail": r["stochastic"]["success_detail"],
            }
            for r in results
        },
        "selection_reason": ("满足 clean=9/9 且 det success<=20% 且 stoch success>0 的候选中，"
                             "取最大 hold_steps（最宽松的第一阶段）"),
        "created_by": "scripts/calibrate_table_constraint_curriculum.py",
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"配置已写入 {CONFIG_PATH}")


if __name__ == "__main__":
    main()

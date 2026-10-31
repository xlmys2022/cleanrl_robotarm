#!/usr/bin/env python3
"""Stage B warm-start sanity（§二十六）：Stage A checkpoint 在 Stage B 环境 offset=[0,0] 下必须仍然成功。

用法:
    python scripts/check_warm_start_stage_b.py

覆盖:
    1. Stage A checkpoint 完整（model_state_dict + observation_normalization）
    2. Stage B(offset=[0,0]) 与 Stage A 的 reset 状态逐元素一致（qpos / ctrl / obs）
    3. Stage A policy deterministic 在 Stage B(offset=[0,0]) 上 5/5 success + valid grasp
    4. 该轨迹与 Stage A 环境上的轨迹一致（同 seed, 同 success step / return）
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from eval_reach import (  # noqa: E402
    FrozenNormalizeObservation,
    build_agent,
    load_checkpoint,
    policy_action,
)

STAGE_B_ENV_ID = "PandaGraspCurriculum-B-v0"
STAGE_A_ENV_ID = "PandaGraspCurriculum-A-v0"
STAGE_A_CHECKPOINT = ("runs/PandaGraspCurriculum-A-v0__train_robot__1__1793362519/"
                      "train_robot.checkpoint.pt")
EPISODES = 5
SEED = 0

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


def make_env(env_id: str, checkpoint: dict | None, seed: int):
    env = gym.make(env_id)
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
    env.reset(seed=seed)
    return env


def run_episode(env, agent, seed: int, offset: list[float] | None) -> dict:
    options = None if offset is None else {"cube_xy_offset": offset}
    obs, info = env.reset(seed=seed, options=options)
    u = env.unwrapped
    start_cube = u.cube_position.copy()
    start_tcp = u._get_tcp_position().copy()

    ep_return, steps = 0.0, 0
    success_step = None
    max_finger_table_force = 0.0
    max_finger_cube_pen = 0.0
    max_cube_table_pen = 0.0
    valid_steps = 0
    arm_action_sq = 0.0
    while True:
        action = policy_action(agent, obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        steps += 1
        arm_action_sq += float(np.sum(action[:7] ** 2))
        max_finger_table_force = max(max_finger_table_force, info["finger_table_force"])
        max_finger_cube_pen = max(max_finger_cube_pen, info["max_finger_cube_penetration"])
        max_cube_table_pen = max(max_cube_table_pen, info["cube_table_penetration"])
        if info["grasp_quality_valid"]:
            valid_steps += 1
        if info["is_success"] and success_step is None:
            success_step = steps
        if terminated or truncated:
            break

    return {
        "success": bool(info["is_success"]),
        "success_step": success_step,
        "return": ep_return,
        "steps": steps,
        "valid_steps": valid_steps,
        "max_finger_table_force": max_finger_table_force,
        "max_finger_cube_pen": max_finger_cube_pen,
        "max_cube_table_pen": max_cube_table_pen,
        "arm_action_norm": float(np.sqrt(arm_action_sq / max(steps, 1))),
        "cube_drift": float(np.linalg.norm(u.cube_position - start_cube)),
        "tcp_travel": float(np.linalg.norm(u._get_tcp_position() - start_tcp)),
    }


def main() -> None:
    rule("0. 加载 Stage A checkpoint")
    checkpoint_path = PROJECT_ROOT / STAGE_A_CHECKPOINT
    check("Stage A checkpoint 存在", checkpoint_path.is_file(), str(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    print(f"  env_id={checkpoint['env_id']}  task_version={checkpoint.get('task_version')}")
    print(f"  obs norm count={norm['count']:.0f}  epsilon={norm['epsilon']}  "
          f"clip=[{norm.get('clip_low')}, {norm.get('clip_high')}]")
    print(f"  actor std = {agent.actor_logstd.exp().mean().item():.4f}")

    rule("1. Stage B(offset=[0,0]) reset 状态 vs Stage A")
    env_a = make_env(STAGE_A_ENV_ID, checkpoint, SEED)
    env_b = make_env(STAGE_B_ENV_ID, checkpoint, SEED)
    obs_a, _ = env_a.reset(seed=SEED)
    obs_b, info_b = env_b.reset(seed=SEED, options={"cube_xy_offset": [0.0, 0.0]})
    ua, ub = env_a.unwrapped, env_b.unwrapped
    print(f"  Stage B initial_cube_xy_offset = {info_b['initial_cube_xy_offset']}")
    check("qpos 完全一致", np.allclose(ua.data.qpos, ub.data.qpos, atol=0.0),
          f"max|dq|={np.max(np.abs(ua.data.qpos - ub.data.qpos)):.3e}")
    check("ctrl 完全一致", np.allclose(ua.data.ctrl, ub.data.ctrl, atol=0.0),
          f"max|dctrl|={np.max(np.abs(ua.data.ctrl - ub.data.ctrl)):.3e}")
    check("observation 完全一致", np.allclose(obs_a, obs_b, atol=0.0),
          f"max|dobs|={np.max(np.abs(obs_a - obs_b)):.3e}")

    rule(f"2. Stage A policy (deterministic) on Stage B, offset=[0,0], {EPISODES} episodes")
    stage_b_runs = []
    for ep in range(EPISODES):
        r = run_episode(env_b, agent, SEED + ep, [0.0, 0.0])
        stage_b_runs.append(r)
        print(f"  episode {ep + 1}: success={str(r['success']):<5} step={r['success_step']} "
              f"return={r['return']:+7.3f} valid_steps={r['valid_steps']:2d} "
              f"finger_table_force={r['max_finger_table_force']:6.2f}N "
              f"finger_cube_pen={r['max_finger_cube_pen'] * 1000:5.2f}mm "
              f"cube_table_pen={r['max_cube_table_pen'] * 1000:5.2f}mm "
              f"cube_drift={r['cube_drift'] * 100:5.2f}cm")

    success_count = sum(r["success"] for r in stage_b_runs)
    check(f"offset=[0,0] success {success_count}/{EPISODES}", success_count == EPISODES)
    check("每个成功 episode 都有 strict valid grasp",
          all(r["valid_steps"] > 0 for r in stage_b_runs if r["success"]))

    rule("3. 与 Stage A 环境上的同 seed 轨迹对比")
    identical = True
    for ep in range(EPISODES):
        r_a = run_episode(env_a, agent, SEED + ep, None)
        r_b = stage_b_runs[ep]
        same = (r_a["success"] == r_b["success"]
                and r_a["success_step"] == r_b["success_step"]
                and abs(r_a["return"] - r_b["return"]) < 1e-9
                and abs(r_a["cube_drift"] - r_b["cube_drift"]) < 1e-12)
        identical = identical and same
        print(f"  episode {ep + 1}: stageA(step={r_a['success_step']}, ret={r_a['return']:+.3f}) "
              f"vs stageB(step={r_b['success_step']}, ret={r_b['return']:+.3f}) "
              f"-> {'same' if same else 'DIFF'}")
    check("Stage A / Stage B(offset=0) 轨迹一致", identical)

    env_a.close()
    env_b.close()

    rule("结果")
    failed = [label for label, ok in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for label in failed:
            print(f"  FAILED: {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()

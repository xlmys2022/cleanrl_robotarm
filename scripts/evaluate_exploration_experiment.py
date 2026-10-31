#!/usr/bin/env python3
"""统一评估 exploration 实验（det 50 / stoch 200 / boundary grid 9 / 5x5）。

用法:
    python scripts/evaluate_exploration_experiment.py \
        --model-path runs/<run>/train_robot.checkpoint.pt \
        --label ppo_ent001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from diagnose_ppo_exploration import (  # noqa: E402
    ClipStats,
    build_agent,
    make_env,
    run_episode,
    summarize,
)
from eval_reach import load_checkpoint  # noqa: E402

DET_EPISODES = 50
STOCH_EPISODES = 200
GRID_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def run_block(env, agent, episodes: int, deterministic: bool, seed: int,
              offsets=None) -> tuple[dict, dict]:
    clip_stats = ClipStats(dim=8)
    torch.manual_seed(seed)
    rows = []
    for ep in range(episodes):
        offset = None if offsets is None else offsets[ep]
        rows.append(run_episode(env, agent, seed + ep, clip_stats,
                                deterministic=deterministic, offset=offset))
    return summarize(rows), clip_stats.summary()


def fmt(summary: dict, clip: dict) -> str:
    return (f"success {summary['success']}/{summary['n']} "
            f"({summary['success_rate'] * 100:.0f}%), "
            f"clean_cand {summary['clean_candidate']}, "
            f"<=15 {summary['low_press_15']}, <=10 {summary['low_press_10']}, "
            f"<=5 {summary['low_press_5']}, zero {summary['zero_high_force']}, "
            f"maxconsec {summary['mean_max_consecutive']:.1f}, "
            f"total>100N {summary['mean_total_high']:.1f}, "
            f"clip {clip['p_clipped'].mean() * 100:.1f}%, "
            f"drift {summary['mean_cube_drift'] * 100:.2f}cm, "
            f"maxF {summary['max_force']:.0f}N, "
            f"minTCPz {summary['mean_min_tcp_z']:.3f}, "
            f"TCPdisp {summary['mean_tcp_displacement'] * 100:.2f}cm, "
            f"ret {summary['mean_return']:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="exploration 实验统一评估")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--det-episodes", type=int, default=DET_EPISODES)
    parser.add_argument("--stoch-episodes", type=int, default=STOCH_EPISODES)
    parser.add_argument("--half-range", type=float, default=0.02)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    env = make_env(checkpoint)      # diagnose_ppo_exploration.ENV_ID = B-v2
    print(f"env : {env.spec.id}")

    std = agent.actor_logstd.exp().mean().item()
    print(f"=== {args.label} ===")
    print(f"checkpoint : {args.model_path}")
    print(f"  env_id={checkpoint['env_id']}  reward_version={checkpoint.get('reward_version')}  "
          f"algorithm={checkpoint.get('algorithm', 'PPO')}  "
          f"ent_coef={checkpoint.get('ent_coef')}  "
          f"init_logstd_offset={checkpoint.get('init_logstd_offset')}")
    print(f"  global_step={checkpoint.get('global_step')}  actor_std(mean)={std:.4f}")

    det_summary, det_clip = run_block(env, agent, args.det_episodes, True, 0)
    print(f"\ndeterministic {args.det_episodes} : {fmt(det_summary, det_clip)}")

    stoch_summary, stoch_clip = run_block(env, agent, args.stoch_episodes, False, 0)
    print(f"stochastic {args.stoch_episodes}   : {fmt(stoch_summary, stoch_clip)}")
    print(f"  (stochastic clip: arm {stoch_clip['p_clipped'][:7].mean() * 100:.1f}%, "
          f"grip {stoch_clip['p_clipped'][7] * 100:.1f}%; "
          f"P(|a|>2) arm {stoch_clip['p_gt2'][:7].mean() * 100:.1f}%)")

    r = args.half_range
    boundary = [(0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                (r, r), (r, -r), (-r, r), (-r, -r)]
    grid_summary, grid_clip = run_block(env, agent, len(boundary), True, 1000, boundary)
    print(f"boundary grid 9      : {fmt(grid_summary, grid_clip)}")

    values = [v * r for v in GRID_VALUES]
    offsets = [(dx, dy) for dy in values for dx in values]
    full_summary, full_clip = run_block(env, agent, len(offsets), True, 2000, offsets)
    print(f"5x5 grid 25          : {fmt(full_summary, full_clip)}")
    env.close()

    print()
    print(json.dumps({
        "label": args.label,
        "model_path": args.model_path,
        "actor_std": std,
        "deterministic": det_summary,
        "stochastic": stoch_summary,
        "boundary_grid": grid_summary,
        "grid_5x5": full_summary,
    }, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

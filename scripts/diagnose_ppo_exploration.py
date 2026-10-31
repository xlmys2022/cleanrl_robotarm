#!/usr/bin/env python3
"""PPO exploration 诊断（不训练）：

1. Frozen-policy logstd sweep：固定 Stage B-v0 checkpoint，只在内存里放大 actor_logstd
2. Action clipping 一致性：raw action / env action / logprob 对应哪个
3. 逐维 action 饱和统计（arm 0:7 / gripper 7）

用法:
    python scripts/diagnose_ppo_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.normal import Normal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from eval_reach import FrozenNormalizeObservation, load_checkpoint  # noqa: E402
from ppo_continuous_action import Agent  # noqa: E402

ENV_ID = "PandaGraspCurriculum-B-v2"
PARENT_CHECKPOINT = ("runs/PandaGraspCurriculum-B-v0__train_robot__1__1793364127/"
                     "train_robot.checkpoint.pt")
LOGSTD_OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPISODES = 500
SEED = 0


def build_agent(checkpoint: dict) -> Agent:
    dummy_envs = type("E", (), {
        "single_observation_space": gym.spaces.Box(-np.inf, np.inf,
                                                   (checkpoint["observation_dim"],), np.float32),
        "single_action_space": gym.spaces.Box(-1.0, 1.0,
                                              (checkpoint["action_dim"],), np.float32),
    })()
    agent = Agent(dummy_envs)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent


def make_env(checkpoint: dict):
    norm = checkpoint["observation_normalization"]
    env = gym.make(ENV_ID)
    return FrozenNormalizeObservation(
        env,
        mean=norm["mean"].numpy(),
        var=norm["var"].numpy(),
        epsilon=norm["epsilon"],
        clip_low=norm.get("clip_low", -10.0),
        clip_high=norm.get("clip_high", 10.0),
    )


def sample_action(agent: Agent, obs: np.ndarray, deterministic: bool = False):
    """返回 (raw_action, clipped_action, logprob_raw, logprob_clipped, mean, std)。"""
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean = agent.actor_mean(x)
        std = agent.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = mean if deterministic else dist.sample()
        logprob_raw = dist.log_prob(raw).sum(1)
        clipped_t = torch.clamp(raw, -1.0, 1.0)
        logprob_clipped = dist.log_prob(clipped_t).sum(1)
    return (raw.squeeze(0).numpy(), clipped_t.squeeze(0).numpy(),
            float(logprob_raw.item()), float(logprob_clipped.item()),
            mean.squeeze(0).numpy(), std.squeeze(0).numpy())


class ClipStats:
    def __init__(self, dim: int) -> None:
        self.n = 0
        self.total = np.zeros(dim)
        self.total_sq = np.zeros(dim)
        self.gt1 = np.zeros(dim)
        self.gt2 = np.zeros(dim)
        self.clipped = np.zeros(dim)

    def add(self, raw: np.ndarray, clipped: np.ndarray) -> None:
        self.n += 1
        self.total += raw
        self.total_sq += raw ** 2
        self.gt1 += np.abs(raw) > 1.0
        self.gt2 += np.abs(raw) > 2.0
        self.clipped += np.isclose(np.abs(clipped), 1.0)

    def summary(self) -> dict:
        mean = self.total / self.n
        var = np.maximum(self.total_sq / self.n - mean ** 2, 0.0)
        return {
            "mean": mean,
            "std": np.sqrt(var),
            "p_gt1": self.gt1 / self.n,
            "p_gt2": self.gt2 / self.n,
            "p_clipped": self.clipped / self.n,
        }


def run_episode(env, agent, seed: int, clip_stats: ClipStats,
                trace: list[dict] | None = None, deterministic: bool = False,
                offset: list[float] | None = None) -> dict:
    options = None if offset is None else {"cube_xy_offset": [float(offset[0]), float(offset[1])]}
    obs, _ = env.reset(seed=seed, options=options)
    u = env.unwrapped
    start_xy = np.array(u.cube_position[:2])
    start_tcp = u._get_tcp_position().copy()
    min_tcp_z = float(start_tcp[2])

    steps = 0
    success_step = None
    max_force = 0.0
    finger_table_steps = 0
    first_table = first_high = None
    high_steps = 0
    segments: list[int] = []
    current_segment = 0
    ep_return = 0.0
    while True:
        raw, clipped, lp_raw, lp_clip, _, _ = sample_action(agent, obs, deterministic)
        clip_stats.add(raw, clipped)
        if trace is not None and len(trace) < 20:
            trace.append({
                "step": steps + 1,
                "raw": raw.copy(),
                "env": clipped.copy(),
                "logprob_raw": lp_raw,
                "logprob_clipped": lp_clip,
            })
        obs, reward, terminated, truncated, info = env.step(clipped)
        steps += 1
        ep_return += float(reward)
        tcp = u._get_tcp_position()
        min_tcp_z = min(min_tcp_z, float(tcp[2]))
        force = float(info["finger_table_force"])
        max_force = max(max_force, force)
        if info["finger_table_contact_count"] > 0:
            finger_table_steps += 1
            if first_table is None:
                first_table = steps
        if force > 100.0:
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
    return {
        "success": bool(info["is_success"]),
        "valid": bool(info["grasp_quality_valid"]),
        "success_step": success_step,
        "return": ep_return,
        "length": steps,
        "max_consecutive": max(segments) if segments else 0,
        "total_high": high_steps,
        "segments": len(segments),
        "mean_segment": float(np.mean(segments)) if segments else 0.0,
        "finger_table_steps": finger_table_steps,
        "max_force": max_force,
        "first_table": first_table,
        "first_high": first_high,
        "cube_drift": float(np.linalg.norm(np.array(u.cube_position[:2]) - start_xy)),
        "min_tcp_z": min_tcp_z,
        "tcp_displacement": float(np.linalg.norm(u._get_tcp_position() - start_tcp)),
    }


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    success = [e for e in episodes if e["success"]]
    return {
        "n": n,
        "success": len(success),
        "success_rate": len(success) / n,
        "clean_candidate": sum(1 for e in success if e["max_consecutive"] <= 5),
        "low_press_15": sum(1 for e in success if e["max_consecutive"] <= 15),
        "low_press_10": sum(1 for e in success if e["max_consecutive"] <= 10),
        "low_press_5": sum(1 for e in success if e["max_consecutive"] <= 5),
        "zero_high_force": sum(1 for e in episodes if e["total_high"] == 0),
        "mean_max_consecutive": float(np.mean([e["max_consecutive"] for e in episodes])),
        "mean_total_high": float(np.mean([e["total_high"] for e in episodes])),
        "mean_cube_drift": float(np.mean([e["cube_drift"] for e in episodes])),
        "max_force": float(np.max([e["max_force"] for e in episodes])),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "mean_min_tcp_z": float(np.mean([e["min_tcp_z"] for e in episodes])),
        "mean_tcp_displacement": float(np.mean([e["tcp_displacement"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
    }


def main() -> None:
    checkpoint = load_checkpoint(PROJECT_ROOT / PARENT_CHECKPOINT)
    agent = build_agent(checkpoint)
    original_logstd = agent.actor_logstd.detach().clone()
    print(f"parent checkpoint : {PARENT_CHECKPOINT}")
    print(f"env               : {ENV_ID}")
    print(f"episodes / offset : {EPISODES}, seeds {SEED}..{SEED + EPISODES - 1}")
    print(f"actor_logstd(original) = "
          f"{np.array2string(original_logstd.numpy().reshape(-1), precision=4)}")
    print(f"actor std(original)    = "
          f"{np.array2string(np.exp(original_logstd.numpy().reshape(-1)), precision=4)}")

    env = make_env(checkpoint)
    results = []
    trace = None
    for offset in LOGSTD_OFFSETS:
        with torch.no_grad():
            agent.actor_logstd.copy_(original_logstd + offset)
        clip_stats = ClipStats(dim=8)
        torch.manual_seed(SEED)
        episodes = []
        for ep in range(EPISODES):
            collect = trace is None and ep == 0
            ep_trace = [] if collect else None
            episodes.append(run_episode(env, agent, SEED + ep, clip_stats, ep_trace))
            if collect:
                trace = ep_trace
        summary = summarize(episodes)
        summary["offset"] = offset
        summary["std"] = np.exp(original_logstd.numpy().reshape(-1) + offset)
        summary["clip"] = clip_stats.summary()
        results.append(summary)
        print(f"\n[offset {offset:+.2f}] std range "
              f"[{summary['std'].min():.3f}, {summary['std'].max():.3f}]  "
              f"success {summary['success']}/{EPISODES}  "
              f"clean_candidate={summary['clean_candidate']}  "
              f"low15={summary['low_press_15']} low10={summary['low_press_10']} "
              f"low5={summary['low_press_5']} zero={summary['zero_high_force']}  "
              f"max_consec={summary['mean_max_consecutive']:.1f}  "
              f"clip={summary['clip']['p_clipped'].mean() * 100:.1f}%")
    env.close()
    with torch.no_grad():
        agent.actor_logstd.copy_(original_logstd)

    print()
    print("=" * 118)
    print("frozen logstd sweep")
    print("=" * 118)
    print(f"{'offset':>6} | {'std mult':>8} | {'success':>7} | {'<=15':>5} | {'<=10':>5} | "
          f"{'<=5':>4} | {'zero':>5} | {'mean maxcons':>12} | {'clip frac':>9} | "
          f"{'drift':>6} | {'maxF':>6} | {'return':>8}")
    for r in results:
        mult = float(np.mean(np.exp(r["offset"])))
        print(f"{r['offset']:>+6.2f} | {mult:>8.2f} | "
              f"{r['success']:>3}/{EPISODES:<3} | {r['low_press_15']:>5} | "
              f"{r['low_press_10']:>5} | {r['low_press_5']:>4} | {r['zero_high_force']:>5} | "
              f"{r['mean_max_consecutive']:>12.1f} | "
              f"{r['clip']['p_clipped'].mean() * 100:>8.1f}% | "
              f"{r['mean_cube_drift'] * 100:>5.2f}cm | {r['max_force']:>5.0f}N | "
              f"{r['mean_return']:>+8.2f}")

    print()
    print("=" * 118)
    print("action clipping / saturation（整体）")
    print("=" * 118)
    print(f"{'offset':>6} | {'arm |a|>1':>9} | {'arm |a|>2':>9} | {'arm clipped':>11} | "
          f"{'grip |a|>1':>10} | {'grip |a|>2':>10} | {'grip clipped':>12}")
    for r in results:
        c = r["clip"]
        print(f"{r['offset']:>+6.2f} | {c['p_gt1'][:7].mean() * 100:>8.1f}% | "
              f"{c['p_gt2'][:7].mean() * 100:>8.1f}% | "
              f"{c['p_clipped'][:7].mean() * 100:>10.1f}% | "
              f"{c['p_gt1'][7] * 100:>9.1f}% | {c['p_gt2'][7] * 100:>9.1f}% | "
              f"{c['p_clipped'][7] * 100:>11.1f}%")

    for r in results:
        c = r["clip"]
        print(f"\nper-dim (offset {r['offset']:+.2f})  "
              f"{'dim':>3} | {'raw mean':>9} | {'raw std':>8} | {'P(|a|>1)':>9} | {'P(clip)':>8}")
        for dim in range(8):
            label = f"arm{dim}" if dim < 7 else "grip"
            print(f"{label:>6} | {c['mean'][dim]:>+9.3f} | {c['std'][dim]:>8.3f} | "
                  f"{c['p_gt1'][dim] * 100:>8.1f}% | {c['p_clipped'][dim] * 100:>7.1f}%")

    print()
    print("=" * 118)
    print("action clipping 一致性（offset 0，episode 0，前 20 步）")
    print("=" * 118)
    print(f"{'step':>4} | {'raw a7':>8} | {'env a7':>8} | {'max|raw arm|':>12} | "
          f"{'logp(raw)':>11} | {'logp(clipped)':>14} | same?")
    for row in trace:
        same = abs(row["logprob_raw"] - row["logprob_clipped"]) < 1e-9
        print(f"{row['step']:>4} | {row['raw'][7]:>+8.3f} | {row['env'][7]:>+8.3f} | "
              f"{np.abs(row['raw'][:7]).max():>12.3f} | {row['logprob_raw']:>+11.4f} | "
              f"{row['logprob_clipped']:>+14.4f} | {same}")

    print()
    print("=" * 118)
    print("gate 判定")
    print("=" * 118)
    best = max(results, key=lambda r: (r["clean_candidate"], r["low_press_10"]))
    print(f"max clean_candidate = {max(r['clean_candidate'] for r in results)} "
          f"(offset {best['offset']:+.2f})")
    print(f"max low_press_10    = {max(r['low_press_10'] for r in results)}")
    print(f"max zero_high_force = {max(r['zero_high_force'] for r in results)}")
    print(f"mean max-consecutive @offset0 = {results[0]['mean_max_consecutive']:.1f}")
    print(f"max mean max-consecutive      = "
          f"{max(r['mean_max_consecutive'] for r in results):.1f}")


if __name__ == "__main__":
    main()

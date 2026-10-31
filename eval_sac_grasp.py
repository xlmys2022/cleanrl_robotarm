"""评估 SAC checkpoint（raw 34D observation, 无 NormalizeObservation）。

用法:
    python eval_sac_grasp.py --model-path runs/<run>/train_sac.checkpoint.pt --episodes 50
    python eval_sac_grasp.py --model-path ... --stochastic --episodes 200
    python eval_sac_grasp.py --random --episodes 50

deterministic 使用 squashed mean action（tanh(mean)*scale+bias），stochastic 使用 sampled action。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from sac_continuous_action import Actor  # noqa: E402

GRID_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("algorithm") != "SAC":
        raise SystemExit("该 checkpoint 不是 SAC (algorithm != 'SAC')")
    return checkpoint


def build_actor(checkpoint: dict, env_id: str) -> Actor:
    env = gym.make(env_id)
    spaces = SimpleNamespace(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    env.close()
    actor = Actor(spaces)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def act(actor: Actor, obs: np.ndarray, stochastic: bool) -> np.ndarray:
    with torch.no_grad():
        action, _, mean = actor.get_action(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    out = (action if stochastic else mean).squeeze(0).numpy()
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def run_episode(env, actor, seed: int, stochastic: bool, random_policy: bool,
                offset=None) -> dict:
    options = None if offset is None else {"cube_xy_offset": [float(offset[0]), float(offset[1])]}
    obs, info = env.reset(seed=seed, options=options)
    u = env.unwrapped
    start_xy = np.array(u.cube_position[:2])
    start_tcp = u._get_tcp_position().copy()
    cube_xy0 = u.cube_position[:2].copy()

    steps = 0
    ep_return = 0.0
    success_step = None
    first_table = first_high = None
    high_steps = 0
    current_segment = 0
    segments: list[int] = []
    finger_table_steps = 0
    max_force = 0.0
    max_pen = 0.0
    max_cube_pen = 0.0
    arm_norm_sum = arm_norm_max = 0.0
    near_095 = near_099 = 0
    n_action_entries = 0
    first_bilateral = None

    while True:
        if random_policy:
            action = env.action_space.sample()
        else:
            action = act(actor, obs, stochastic)
        raw_abs = np.abs(np.asarray(action))
        near_095 += int(np.sum(raw_abs > 0.95))
        near_099 += int(np.sum(raw_abs > 0.99))
        n_action_entries += raw_abs.size

        arm_norm = float(np.linalg.norm(np.asarray(action)[:7]))
        arm_norm_sum += arm_norm
        arm_norm_max = max(arm_norm_max, arm_norm)

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        ep_return += float(reward)
        force = float(info["finger_table_force"])
        max_force = max(max_force, force)
        max_pen = max(max_pen, float(info["max_finger_cube_penetration"]))
        max_cube_pen = max(max_cube_pen, float(info["cube_table_penetration"]))
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
        if info["bilateral_contact"] and first_bilateral is None:
            first_bilateral = steps
        if info["is_success"] and success_step is None:
            success_step = steps
        if terminated or truncated:
            break
    if current_segment > 0:
        segments.append(current_segment)

    return {
        "success": bool(info["is_success"]),
        "valid": bool(info["grasp_quality_valid"]),
        "bilateral": first_bilateral is not None,
        "success_step": success_step,
        "return": ep_return,
        "length": steps,
        "finger_table_steps": finger_table_steps,
        "high_force_steps": high_steps,
        "max_consecutive": max(segments) if segments else 0,
        "high_force_segments": len(segments),
        "max_table_force": max_force,
        "max_finger_cube_penetration": max_pen,
        "max_cube_table_penetration": max_cube_pen,
        "cube_xy_drift": float(np.linalg.norm(np.array(u.cube_position[:2]) - cube_xy0)),
        "tcp_displacement": float(np.linalg.norm(u._get_tcp_position() - start_tcp)),
        "arm_action_norm": arm_norm_sum / max(steps, 1),
        "max_arm_action_norm": arm_norm_max,
        "near_boundary_095": near_095 / max(n_action_entries, 1),
        "near_boundary_099": near_099 / max(n_action_entries, 1),
        "offset": info.get("initial_cube_xy_offset"),
    }


def summarize(episodes: list[dict], title: str) -> None:
    n = len(episodes)
    success = [e for e in episodes if e["success"]]
    print()
    print("=" * 78)
    print(f"{title}  ({n} episodes)")
    print("=" * 78)
    print(f"  success rate                   : {100.0 * len(success) / n:.1f}% "
          f"({len(success)}/{n})")
    print(f"  valid grasp episodes           : "
          f"{100.0 * sum(e['valid'] for e in episodes) / n:.1f}%")
    print(f"  bilateral contact episodes     : "
          f"{100.0 * sum(e['bilateral'] for e in episodes) / n:.1f}%")
    print(f"  mean success step              : "
          f"{np.mean([e['success_step'] for e in success]):.1f}" if success
          else "  mean success step              : n/a")
    print(f"  mean finger-table steps        : "
          f"{np.mean([e['finger_table_steps'] for e in episodes]):.1f}")
    print(f"  mean >100N steps               : "
          f"{np.mean([e['high_force_steps'] for e in episodes]):.1f}")
    print(f"  mean/max consecutive >100N     : "
          f"{np.mean([e['max_consecutive'] for e in episodes]):.1f} / "
          f"{np.max([e['max_consecutive'] for e in episodes]):.0f}")
    print(f"  mean high-force segments       : "
          f"{np.mean([e['high_force_segments'] for e in episodes]):.1f} 段/episode")
    print(f"  mean/max finger-table force    : "
          f"{np.mean([e['max_table_force'] for e in episodes]):.1f} / "
          f"{np.max([e['max_table_force'] for e in episodes]):.1f} N")
    print(f"  mean/max finger-cube penetration: "
          f"{np.mean([e['max_finger_cube_penetration'] for e in episodes]) * 1000:.2f} / "
          f"{np.max([e['max_finger_cube_penetration'] for e in episodes]) * 1000:.2f} mm")
    print(f"  mean max cube-table penetration: "
          f"{np.mean([e['max_cube_table_penetration'] for e in episodes]) * 1000:.2f} mm")
    print(f"  mean cube xy drift             : "
          f"{np.mean([e['cube_xy_drift'] for e in episodes]):.4f} m")
    print(f"  mean TCP displacement          : "
          f"{np.mean([e['tcp_displacement'] for e in episodes]):.4f} m")
    print(f"  mean arm action norm           : "
          f"{np.mean([e['arm_action_norm'] for e in episodes]):.4f}")
    print(f"  mean action |a|>0.95 fraction  : "
          f"{np.mean([e['near_boundary_095'] for e in episodes]) * 100:.1f}%")
    print(f"  mean action |a|>0.99 fraction  : "
          f"{np.mean([e['near_boundary_099'] for e in episodes]) * 100:.1f}%")
    print(f"  mean episode return            : "
          f"{np.mean([e['return'] for e in episodes]):+.3f}")
    clean = [e for e in success if e["max_consecutive"] <= 5]
    print(f"  clean_candidate (success & max_consec<=5) : {len(clean)}/{n}")


def run_grid(env, actor, half_range: float, stochastic: bool) -> None:
    values = [v * half_range for v in GRID_VALUES]
    print()
    print(f"5x5 offset grid (±{half_range * 1000:.1f} mm), "
          f"{'stochastic' if stochastic else 'deterministic'}")
    print("  dy\\dx " + "".join(f"{v * 1000:+7.1f}" for v in values))
    successes = 0
    for i, dy in enumerate(values):
        row = f"  {dy * 1000:+6.1f} "
        for j, dx in enumerate(values):
            ep = run_episode(env, actor, 3000 + i * len(values) + j, stochastic, False, [dx, dy])
            successes += int(ep["success"])
            row += f"  {'OK':<5}  " if ep["success"] else f"  {'FAIL':<5}  "
        print(row)
    total = len(values) ** 2
    print(f"  grid success = {successes}/{total} ({100.0 * successes / total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 SAC grasp checkpoint / 随机 baseline")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--env-id", default="PandaGraspCurriculum-B-v2")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--offset-grid", action="store_true")
    parser.add_argument("--half-range", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    env = gym.make(args.env_id)
    if args.random:
        episodes = [run_episode(env, None, args.seed + ep, False, True)
                    for ep in range(args.episodes)]
        summarize(episodes, f"random policy baseline | {args.env_id}")
        env.close()
        return

    if args.model_path is None:
        raise SystemExit("需要 --model-path（或用 --random）")
    checkpoint = load_checkpoint(args.model_path)
    if checkpoint["env_id"] != args.env_id:
        print(f"WARN: checkpoint env_id={checkpoint['env_id']} != eval env_id={args.env_id}")
    actor = build_actor(checkpoint, args.env_id)
    print(f"checkpoint : {args.model_path}")
    print(f"  algorithm=SAC  env_id={checkpoint['env_id']}  "
          f"reward_version={checkpoint['reward_version']}  "
          f"global_step={checkpoint['global_step']}  "
          f"alpha_final={checkpoint['alpha_final']:.4f}")
    print(f"  observation_preprocessing={checkpoint['observation_preprocessing']}  "
          f"action_distribution={checkpoint['action_distribution']}")

    episodes = [run_episode(env, actor, args.seed + ep, args.stochastic, False)
                for ep in range(args.episodes)]
    for i, ep in enumerate(episodes[:10]):
        print(f"  ep{i + 1:3d}: success={str(ep['success']):<5} steps={ep['length']:3d} "
              f"valid={'Y' if ep['valid'] else 'N'} succ@{ep['success_step']} "
              f">100N={ep['high_force_steps']:2d} maxconsec={ep['max_consecutive']:2d} "
              f"drift={ep['cube_xy_drift'] * 100:.2f}cm "
              f"a>0.95={ep['near_boundary_095'] * 100:.0f}%")
    summarize(episodes, f"{args.env_id} {'stochastic' if args.stochastic else 'deterministic'}")
    if args.offset_grid:
        run_grid(env, actor, args.half_range, args.stochastic)
    env.close()


if __name__ == "__main__":
    main()

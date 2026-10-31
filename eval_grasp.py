"""评估 PandaGrasp-v0 checkpoint（冻结 observation normalization）/ 随机 baseline。

用法:
    python eval_grasp.py --model-path runs/<run>/train_robot.checkpoint.pt --episodes 5
    python eval_grasp.py --model-path ... --stochastic --episodes 50
    python eval_grasp.py --random --episodes 20

与 eval_reach.py 使用完全相同的归一化逻辑（FrozenNormalizeObservation）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from envs.panda_pick_place_env import STAGE_B_CONFIG_PATH  # noqa: E402
from eval_reach import (  # noqa: E402
    build_agent,
    load_checkpoint,
    make_env,
    policy_action,
)

ENV_ID = "PandaGrasp-v0"
GRID_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def run_episode(env, agent, deterministic: bool, random_policy: bool, seed: int,
                offset: list[float] | None = None) -> dict:
    options = None if offset is None else {"cube_xy_offset": [float(offset[0]), float(offset[1])]}
    obs, info = env.reset(seed=seed, options=options)
    u = env.unwrapped
    cube_xy0 = u.cube_position[:2].copy()
    applied_offset = info.get("initial_cube_xy_offset")

    ep_return, steps, min_dist = 0.0, 0, float("inf")
    first = {"left": None, "right": None, "bilateral": None}
    max_force = {"left": 0.0, "right": 0.0}
    success_step = None
    saw_valid = False
    invalid_reasons: set[str] = set()
    bilateral_steps = valid_steps = 0
    max_penetration = 0.0
    max_cube_table_pen = 0.0
    finger_table_steps = 0
    finger_table_high_force_steps = 0
    max_finger_table_force = 0.0
    reward_finger_table_sum = 0.0
    reward_cube_drift_sum = 0.0
    arm_norm_sum = arm_norm_max = 0.0
    first_table_contact_step = None
    first_high_force_step = None
    current_high_segment = 0
    high_force_segments: list[int] = []
    tcp0 = u._get_tcp_position().copy()
    arm_qpos0 = u.data.qpos[u._arm_qpos_adr].copy()
    trace: list[dict] = []

    while True:
        if random_policy:
            action = env.action_space.sample()
        else:
            action = policy_action(agent, obs, deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        steps += 1
        reward_finger_table_sum += float(info.get("reward_finger_table", 0.0))
        reward_cube_drift_sum += float(info.get("reward_cube_drift", 0.0))
        min_dist = min(min_dist, float(info["tcp_cube_distance"]))

        arm_norm = float(np.linalg.norm(np.asarray(action)[:7]))
        arm_norm_sum += arm_norm
        arm_norm_max = max(arm_norm_max, arm_norm)
        if len(trace) < 20:
            trace.append({
                "step": steps,
                "action_gripper": float(np.asarray(action)[7]),
                "finger": float(info["finger_mean_position"]),
                "valid": bool(info["grasp_quality_valid"]),
                "dist": float(info["tcp_cube_distance"]),
            })

        if info["left_contact"] and first["left"] is None:
            first["left"] = steps
        if info["right_contact"] and first["right"] is None:
            first["right"] = steps
        if info["bilateral_contact"]:
            if first["bilateral"] is None:
                first["bilateral"] = steps
            bilateral_steps += 1
            if not info["grasp_quality_valid"]:
                invalid_reasons.add(str(info["grasp_invalid_reason"]))
        # 这里统计的是"质量合格"（strict 定义）, 与 env 是否接受(legacy/strict)无关
        if info["grasp_quality_valid"]:
            valid_steps += 1
            saw_valid = True
        if info["finger_table_contact_count"] > 0:
            finger_table_steps += 1
            if first_table_contact_step is None:
                first_table_contact_step = steps
        max_finger_table_force = max(max_finger_table_force, float(info["finger_table_force"]))
        if float(info["finger_table_force"]) > 100.0:
            finger_table_high_force_steps += 1
            current_high_segment += 1
            if first_high_force_step is None:
                first_high_force_step = steps
        elif current_high_segment > 0:
            high_force_segments.append(current_high_segment)
            current_high_segment = 0
        max_penetration = max(max_penetration, float(info["max_finger_cube_penetration"]))
        max_cube_table_pen = max(max_cube_table_pen, float(info["cube_table_penetration"]))
        max_force["left"] = max(max_force["left"], float(info["left_contact_force"]))
        max_force["right"] = max(max_force["right"], float(info["right_contact_force"]))

        if info["is_success"] and success_step is None:
            success_step = steps
        if terminated or truncated:
            break

    if current_high_segment > 0:
        high_force_segments.append(current_high_segment)
    cube_xy_drift = float(np.linalg.norm(u.cube_position[:2] - cube_xy0))
    failure_reason = info.get("failure_reason")
    constraint_failed = failure_reason == "sustained_finger_table_force"
    return {
        "success": bool(info["is_success"]),
        "offset": applied_offset,
        "failure_reason": failure_reason,
        "constraint_failed": constraint_failed,
        "failure_step": steps if constraint_failed else None,
        "first_table_contact_step": first_table_contact_step,
        "first_high_force_step": first_high_force_step,
        "return": ep_return,
        "length": steps,
        "min_distance": min_dist,
        "final_distance": float(info["tcp_cube_distance"]),
        "first_left": first["left"],
        "first_right": first["right"],
        "first_bilateral": first["bilateral"],
        "success_step": success_step,
        "max_left_force": max_force["left"],
        "max_right_force": max_force["right"],
        "cube_xy_drift": cube_xy_drift,
        "final_finger": float(info["finger_mean_position"]),
        "saw_valid_grasp": saw_valid,
        "bilateral_steps": bilateral_steps,
        "valid_steps": valid_steps,
        "invalid_reasons": sorted(invalid_reasons),
        "max_penetration": max_penetration,
        "max_cube_table_penetration": max_cube_table_pen,
        "finger_table_steps": finger_table_steps,
        "finger_table_high_force_steps": finger_table_high_force_steps,
        "max_consecutive_high_force": max(high_force_segments) if high_force_segments else 0,
        "high_force_segments": len(high_force_segments),
        "mean_high_force_segment_length": (float(np.mean(high_force_segments))
                                           if high_force_segments else 0.0),
        "max_finger_table_force": max_finger_table_force,
        "reward_finger_table": reward_finger_table_sum,
        "reward_cube_drift": reward_cube_drift_sum,
        "task_version": info.get("task_version"),
        "mean_arm_action_norm": arm_norm_sum / max(steps, 1),
        "max_arm_action_norm": arm_norm_max,
        "arm_joint_displacement": float(
            np.max(np.abs(u.data.qpos[u._arm_qpos_adr] - arm_qpos0))
        ),
        "tcp_displacement": float(np.linalg.norm(u._get_tcp_position() - tcp0)),
        "trace": trace,
    }


def print_trace(episode: dict) -> None:
    print("\n  deterministic trace (前 20 步):")
    print("    step | action[7] | finger | valid | dist")
    for row in episode["trace"]:
        print(f"    {row['step']:4d} | {row['action_gripper']:+.3f}   | "
              f"{row['finger']:.4f} | {str(row['valid']):<5} | {row['dist']:.4f}")


def summarize(episodes: list[dict]) -> None:
    n = len(episodes)
    bilateral = [e for e in episodes if e["first_bilateral"] is not None]
    success = [e for e in episodes if e["success"]]
    valid = [e for e in episodes if e.get("saw_valid_grasp")]
    invalid_bilateral = [e for e in bilateral if not e.get("saw_valid_grasp")]
    finger_table = [e for e in episodes if e.get("finger_table_steps", 0) > 0]
    print()
    print("=" * 74)
    print(f"summary over {n} episodes  (task_version={episodes[0].get('task_version')})")
    print("=" * 74)
    print(f"  success rate                   : {100.0 * len(success) / n:.1f}% "
          f"({len(success)}/{n})")
    constraint = [e for e in episodes if e.get("constraint_failed")]
    print(f"  table constraint failures      : {100.0 * len(constraint) / n:.1f}% "
          f"({len(constraint)}/{n})")
    if constraint:
        print(f"  mean constraint failure step   : "
              f"{np.mean([e['failure_step'] for e in constraint]):.1f}")
    first_table = [e["first_table_contact_step"] for e in episodes
                   if e.get("first_table_contact_step")]
    first_high = [e["first_high_force_step"] for e in episodes
                  if e.get("first_high_force_step")]
    print(f"  mean first table-contact step  : "
          f"{np.mean(first_table):.1f}" if first_table else "  mean first table-contact step  : n/a")
    print(f"  mean first >100N step          : "
          f"{np.mean(first_high):.1f}" if first_high else "  mean first >100N step          : n/a")
    print(f"  bilateral contact episodes     : {100.0 * len(bilateral) / n:.1f}% "
          f"({len(bilateral)}/{n})")
    print(f"  valid grasp episodes           : {100.0 * len(valid) / n:.1f}% "
          f"({len(valid)}/{n})")
    print(f"  invalid bilateral episodes     : {100.0 * len(invalid_bilateral) / n:.1f}% "
          f"({len(invalid_bilateral)}/{n})")
    print(f"  finger-table contact episodes  : {100.0 * len(finger_table) / n:.1f}% "
          f"({len(finger_table)}/{n})")
    print(f"  mean episode return            : "
          f"{np.mean([e['return'] for e in episodes]):+.3f}")
    print(f"  mean episode length            : {np.mean([e['length'] for e in episodes]):.1f}")
    print(f"  mean min tcp_cube_distance     : "
          f"{np.mean([e['min_distance'] for e in episodes]):.4f} m")
    if bilateral:
        print(f"  mean first bilateral step     : "
              f"{np.mean([e['first_bilateral'] for e in bilateral]):.1f} "
              f"(有 bilateral 的 {len(bilateral)} 个 episode)")
    else:
        print("  mean first bilateral step     : n/a（全程没有双侧接触）")
    if success:
        print(f"  mean success step              : "
              f"{np.mean([e['success_step'] for e in success]):.1f}")
    else:
        print("  mean success step              : n/a（没有成功 episode）")
    print(f"  mean/max finger-cube penetration: "
          f"{np.mean([e['max_penetration'] for e in episodes]) * 1000:.2f} / "
          f"{np.max([e['max_penetration'] for e in episodes]) * 1000:.2f} mm")
    print(f"  mean max cube-table penetration: "
          f"{np.mean([e['max_cube_table_penetration'] for e in episodes]) * 1000:.2f} mm")
    print(f"  mean/max finger-table force    : "
          f"{np.mean([e['max_finger_table_force'] for e in episodes]):.2f} / "
          f"{np.max([e['max_finger_table_force'] for e in episodes]):.2f} N")
    print(f"  mean finger-table steps        : "
          f"{np.mean([e['finger_table_steps'] for e in episodes]):.1f} "
          f"(其中 force>100N 的步数 "
          f"{np.mean([e['finger_table_high_force_steps'] for e in episodes]):.1f})")
    print(f"  mean/max consecutive >100N     : "
          f"{np.mean([e['mean_high_force_segment_length'] for e in episodes]):.1f} / "
          f"{np.max([e['max_consecutive_high_force'] for e in episodes]):.0f} steps")
    print(f"  mean high-force segments       : "
          f"{np.mean([e['high_force_segments'] for e in episodes]):.1f} 段/episode")
    print(f"  mean reward_finger_table       : "
          f"{np.mean([e['reward_finger_table'] for e in episodes]):+.4f} "
          f"(每 step {np.mean([e['reward_finger_table'] / max(e['length'], 1) for e in episodes]):+.4f})")
    print(f"  mean reward_cube_drift         : "
          f"{np.mean([e['reward_cube_drift'] for e in episodes]):+.4f} "
          f"(每 step {np.mean([e['reward_cube_drift'] / max(e['length'], 1) for e in episodes]):+.4f})")
    print(f"  mean max left contact force    : "
          f"{np.mean([e['max_left_force'] for e in episodes]):.3f} N")
    print(f"  mean max right contact force   : "
          f"{np.mean([e['max_right_force'] for e in episodes]):.3f} N")
    print(f"  mean cube xy drift             : "
          f"{np.mean([e['cube_xy_drift'] for e in episodes]):.4f} m")
    print(f"  mean arm action norm           : "
          f"{np.mean([e['mean_arm_action_norm'] for e in episodes]):.4f}")
    print(f"  max  arm action norm           : "
          f"{np.max([e['max_arm_action_norm'] for e in episodes]):.4f}")
    print(f"  mean arm joint displacement    : "
          f"{np.mean([e['arm_joint_displacement'] for e in episodes]):.4f} rad")
    print(f"  mean TCP displacement          : "
          f"{np.mean([e['tcp_displacement'] for e in episodes]):.4f} m")
    reasons: dict[str, int] = {}
    for e in episodes:
        for r in e.get("invalid_reasons", []):
            reasons[r] = reasons.get(r, 0) + 1
    print(f"  invalid reason distribution    : {reasons if reasons else 'none'}")
    failure_counts: dict[str, int] = {}
    for e in episodes:
        reason = e.get("failure_reason")
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    print(f"  failure reason distribution     : {failure_counts if failure_counts else 'none'}")


def run_offset_grid(env, agent, deterministic: bool, seed: int, half_range: float) -> None:
    """Stage B offset 泛化分析: dy 行 / dx 列的 5x5 success 表。"""
    values = [v * half_range for v in GRID_VALUES]
    print()
    print(f"offset grid (±{half_range * 1000:.1f} mm), "
          f"{'deterministic' if deterministic else 'stochastic'}")
    print("  dy\\dx " + "".join(f"{v * 1000:+7.1f}" for v in values))
    successes = 0
    for i, dy in enumerate(values):
        row = f"  {dy * 1000:+6.1f} "
        for j, dx in enumerate(values):
            metrics = run_episode(env, agent, deterministic, False,
                                  seed + 1000 + i * len(values) + j, [dx, dy])
            ok = metrics["success"]
            successes += int(ok)
            row += f"  {'OK':<5}  " if ok else f"  {'FAIL':<5}  "
        print(row)
    total = len(values) ** 2
    print(f"  grid success = {successes}/{total} ({100.0 * successes / total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 PandaGrasp（v0 legacy / v1 strict）")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--env-id", default=ENV_ID,
                        help="PandaGrasp-v0 (legacy, 默认) 或 PandaGrasp-v1 (strict)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true",
                        help="打印第 1 个 episode 前 20 步的 action[7]/finger/valid_grasp")
    parser.add_argument("--cube-xy-offset", nargs=2, type=float, default=None,
                        metavar=("DX", "DY"),
                        help="固定 Stage B cube XY 偏移 [m]（不传则用环境默认：Stage B 随机采样）")
    parser.add_argument("--offset-grid", action="store_true",
                        help="Stage B offset 泛化表（5x5, 覆盖 ±half_range, deterministic）")
    parser.add_argument("--offset-half-range", type=float, default=None,
                        help="offset grid 的半范围 [m]，默认读 configs/panda_grasp_stage_b.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)   # stochastic 评估可复现

    if args.random:
        env = make_env(None, args.env_id, args.seed)
        print(f"随机策略 baseline: {args.env_id}, {args.episodes} episodes")
        episodes = []
        for ep in range(args.episodes):
            metrics = run_episode(env, None, True, True, args.seed + ep, args.cube_xy_offset)
            episodes.append(metrics)
            print(f"  episode {ep + 1:3d}: success={str(metrics['success']):<5} "
                  f"steps={metrics['length']:3d} return={metrics['return']:+8.3f} "
                  f"min_dist={metrics['min_distance']:.4f} "
                  f"bi@{metrics['first_bilateral']} "
                  f"valid={'Y' if metrics['saw_valid_grasp'] else 'N'} "
                  f"pen={metrics['max_penetration'] * 1000:.2f}mm "
                  f"drift={metrics['cube_xy_drift']:.4f}")
        env.close()
        summarize(episodes)
        return

    if args.model_path is None:
        raise SystemExit("需要 --model-path（或用 --random）")

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    mode = "stochastic" if args.stochastic else "deterministic"
    std = agent.actor_logstd.exp().mean().item()
    print(f"checkpoint: {args.model_path}")
    print(f"  env_id={checkpoint['env_id']} reward_version={checkpoint.get('reward_version')} "
          f"global_step={checkpoint.get('global_step')} mode={mode} actor_std={std:.4f}")

    env = make_env(checkpoint, args.env_id, args.seed)
    episodes = []
    for ep in range(args.episodes):
        metrics = run_episode(env, agent, not args.stochastic, False, args.seed + ep,
                              args.cube_xy_offset)
        episodes.append(metrics)
        offset_text = ("" if metrics["offset"] is None
                       else f" dx/dy=[{metrics['offset'][0] * 1000:+.1f},"
                            f"{metrics['offset'][1] * 1000:+.1f}]mm")
        failure_text = ("" if not metrics.get("failure_reason")
                        else f" failure={metrics['failure_reason']}@{metrics['length']}")
        print(f"  episode {ep + 1:3d}: success={str(metrics['success']):<5} "
              f"steps={metrics['length']:3d} return={metrics['return']:+8.3f} "
              f"min_dist={metrics['min_distance']:.4f} "
              f"bi@{metrics['first_bilateral']} valid={'Y' if metrics['saw_valid_grasp'] else 'N'} "
              f"succ@{metrics['success_step']} "
              f"pen={metrics['max_penetration'] * 1000:.2f}mm "
              f"cube_pen={metrics['max_cube_table_penetration'] * 1000:.1f}mm "
              f"force(L/R)={metrics['max_left_force']:.2f}/{metrics['max_right_force']:.2f} "
              f"drift={metrics['cube_xy_drift']:.4f}{offset_text}{failure_text}")
    if args.trace:
        print_trace(episodes[0])
    summarize(episodes)

    if args.offset_grid:
        half_range = args.offset_half_range
        if half_range is None:
            if not STAGE_B_CONFIG_PATH.is_file():
                raise SystemExit(f"缺少 {STAGE_B_CONFIG_PATH}, 请显式传 --offset-half-range")
            half_range = float(json.loads(STAGE_B_CONFIG_PATH.read_text())["xy_half_range_m"])
        run_offset_grid(env, agent, not args.stochastic, args.seed, half_range)
    env.close()


if __name__ == "__main__":
    main()

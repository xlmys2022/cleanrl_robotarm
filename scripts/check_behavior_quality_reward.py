#!/usr/bin/env python3
"""Stage B.5 / B.6 behavior-quality reward 回归（不训练）。

用法:
    python scripts/check_behavior_quality_reward.py

同时验证:
    PandaGraspCurriculum-B-v1  (w_table = 0.05, potential_v2_quality)
    PandaGraspCurriculum-B-v2  (w_table = 0.30, potential_v3_quality_table_strong)

覆盖（每个环境各跑一遍）:
    Test 1 clean pre-grasp stationary     -> table / drift penalty ≈ 0
    Test 2 finger-table force penalty     -> 饱和于 -w_table/step, 不爆炸
    Test 3 stationary table press         -> 连续 10 步每步都被惩罚
    Test 4 cube 不动                      -> drift penalty ≈ 0
    Test 5 cube 向外移动 5mm / 10mm       -> 两步都是负 reward
    Test 6 cube 从 10mm 回到 5mm          -> drift penalty == 0（不奖励推回来）
    Test 7 bad policy vs scripted clean   -> clean return > press-policy return
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from envs.panda_pick_place_env import STAGE_A_CONFIG_PATH  # noqa: E402
from eval_reach import build_agent, load_checkpoint, make_env, policy_action  # noqa: E402

STAGE_B_CHECKPOINT = ("runs/PandaGraspCurriculum-B-v0__train_robot__1__1793364127/"
                      "train_robot.checkpoint.pt")
DLS_DAMPING = 0.02
MAX_MOVE_STEP = 0.002
MAX_CLOSE_STEPS = 60

# (env_id, 期望 w_table, 期望 reward_version, 期望 task_version, clean-vs-press 最小差距)
ENV_CONFIGS = (
    ("PandaGraspCurriculum-B-v1", 0.05, "potential_v2_quality",
     "grasp_curriculum_stage_b_quality_v1", 0.0),
    ("PandaGraspCurriculum-B-v2", 0.30, "potential_v3_quality_table_strong",
     "grasp_curriculum_stage_b_quality_v2", 4.0),
)

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


def hold_action() -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)


def test_1_clean_stationary(env) -> None:
    rule("Test 1: clean pre-grasp stationary")
    env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
    before = np.array(env.unwrapped.cube_position[:2])
    obs, reward, terminated, truncated, info = env.step(hold_action())
    moved = float(np.linalg.norm(np.array(env.unwrapped.cube_position[:2]) - before))
    print(f"  table_force={info['finger_table_force']:.3f} N  table_count="
          f"{info['finger_table_contact_count']}  cube_moved={moved * 1000:.4f} mm")
    print(f"  reward_finger_table={info['reward_finger_table']:+.6f}  "
          f"reward_cube_drift={info['reward_cube_drift']:+.6f}")
    check("无 table contact", info["finger_table_contact_count"] == 0)
    check("reward_finger_table ≈ 0", abs(info["reward_finger_table"]) < 1e-9,
          f"{info['reward_finger_table']:+.6f}")
    check("reward_cube_drift ≈ 0", abs(info["reward_cube_drift"]) < 1e-9,
          f"{info['reward_cube_drift']:+.6f}")
    check("reward_total == 各分项之和",
          abs(info["reward_total"] - sum(
              v for k, v in info.items()
              if k.startswith("reward_") and k not in ("reward_total", "reward_version")
          )) < 1e-9, f"{info['reward_total']:+.6f}")


def test_2_table_force(env) -> None:
    rule("Test 2: finger-table force penalty（真实物理 + 解析检查）")
    u = env.unwrapped
    weight = float(u.finger_table_penalty_weight)
    env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
    forces, penalties = [], []
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(
            dls_action(u, np.array([0.0, 0.0, -0.004]), gripper=1.0)
        )
        forces.append(float(info["finger_table_force"]))
        penalties.append(float(info["reward_finger_table"]))
        if terminated or truncated:
            break
    print(f"  真实物理: max table force={max(forces):.1f} N, "
          f"min penalty={min(penalties):+.4f}")
    check("真实物理能产生 finger-table contact", max(forces) > 0.0)
    check("有 contact 的步惩罚 < 0", all(p < 0.0 for f, p in zip(forces, penalties) if f > 0.0))

    # 解析: 用当前 metrics 替换 force, 检查单调 + 饱和
    m = u._task_metrics()
    u._prev_cube_xy_drift = 0.0
    u._initial_cube_xy = np.array(u.cube_position[:2])
    analytic = []
    for f in (0.0, 1.0, 20.0, 100.0, 400.0, 4000.0):
        m["finger_table_force"] = f
        table_penalty, drift_penalty = u._quality_penalties(m)
        analytic.append((f, table_penalty, drift_penalty))
        print(f"  F={f:7.1f} N -> reward_finger_table={table_penalty:+.5f} "
              f"(drift={drift_penalty:+.5f})")
    check("penalty 全部 <= 0", all(p <= 0.0 for _, p, _ in analytic))
    check("penalty 随 force 单调不增",
          all(analytic[i][1] >= analytic[i + 1][1] - 1e-12 for i in range(len(analytic) - 1)))
    check(f"penalty 饱和于 -{weight:.2f}（不超过）",
          min(p for _, p, _ in analytic) >= -weight - 1e-9,
          f"min={min(p for _, p, _ in analytic):+.5f}")
    check(f"F=20N 时 ≈ -{weight:.2f}·tanh(1)（尺度正确）",
          abs(analytic[2][1] - (-weight * np.tanh(1.0))) < 1e-9,
          f"{analytic[2][1]:+.5f}")
    check("单 step 最差 >= -0.30（不出现 -3/-30）",
          min(p for _, p, _ in analytic) >= -0.30 - 1e-9)


def test_3_sustained_press(env) -> None:
    rule("Test 3: stationary table press（连续 10 步都要被惩罚）")
    u = env.unwrapped
    env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
    for _ in range(8):      # 先压到桌面
        _, _, terminated, truncated, info = env.step(
            dls_action(u, np.array([0.0, 0.0, -0.004]), gripper=1.0)
        )
        if info["finger_table_contact_count"] > 0:
            break
    check("进入 table contact", info["finger_table_contact_count"] > 0,
          f"force={info['finger_table_force']:.1f} N")
    all_negative = True
    for i in range(10):
        _, _, terminated, truncated, info = env.step(hold_action())
        negative = info["reward_finger_table"] < 0.0
        all_negative = all_negative and negative
        print(f"  press step {i + 1:2d}: force={info['finger_table_force']:7.1f} N "
              f"reward_finger_table={info['reward_finger_table']:+.5f}")
        if terminated or truncated:
            break
    check("持续压桌期间每一步 reward_finger_table < 0", all_negative)


def _set_cube_xy(u, xy: np.ndarray) -> None:
    adr = u._cube_qpos_adr
    u.data.qpos[adr: adr + 2] = xy
    u.data.qvel[:] = 0.0
    mujoco.mj_forward(u.model, u.data)


def test_4_6_cube_drift(env) -> None:
    rule("Test 4/5/6: cube drift penalty（不动 / 向外 / 推回）")
    u = env.unwrapped
    w_drift = float(u.cube_drift_penalty_weight)
    env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
    start = np.array(u.cube_position[:2])

    terms = u._compute_reward_from_metrics(np.zeros(8), u._task_metrics())
    print(f"  Test4 cube 不动: reward_cube_drift={terms['reward_cube_drift']:+.6f}")
    check("Test4 cube 不动 -> 0", abs(terms["reward_cube_drift"]) < 1e-12)

    _set_cube_xy(u, start + np.array([0.005, 0.0]))
    terms5 = u._compute_reward_from_metrics(np.zeros(8), u._task_metrics())
    _set_cube_xy(u, start + np.array([0.010, 0.0]))
    terms10 = u._compute_reward_from_metrics(np.zeros(8), u._task_metrics())
    print(f"  Test5 0->5mm : reward_cube_drift={terms5['reward_cube_drift']:+.6f}")
    print(f"  Test5 5->10mm: reward_cube_drift={terms10['reward_cube_drift']:+.6f}")
    check("Test5 向外移动两阶段都为负",
          terms5["reward_cube_drift"] < 0.0 and terms10["reward_cube_drift"] < 0.0)
    check("Test5 5mm 增量 ≈ -0.5 * w_d",
          abs(terms5["reward_cube_drift"] + 0.5 * w_drift) < 1e-9)

    _set_cube_xy(u, start + np.array([0.005, 0.0]))
    terms_back = u._compute_reward_from_metrics(np.zeros(8), u._task_metrics())
    print(f"  Test6 10->5mm: reward_cube_drift={terms_back['reward_cube_drift']:+.6f}")
    check("Test6 推回来不给奖励（== 0）", abs(terms_back["reward_cube_drift"]) < 1e-12)


def run_policy_episode(env, agent, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    u = env.unwrapped
    start = np.array(u.cube_position[:2])
    ep_return = 0.0
    table_sum = drift_sum = 0.0
    high_force_steps = table_steps = 0
    success_step = None
    steps = 0
    while True:
        action = policy_action(agent, obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        ep_return += float(reward)
        table_sum += float(info["reward_finger_table"])
        drift_sum += float(info["reward_cube_drift"])
        if info["finger_table_contact_count"] > 0:
            table_steps += 1
        if info["finger_table_force"] > 100.0:
            high_force_steps += 1
        if info["is_success"] and success_step is None:
            success_step = steps
        if terminated or truncated:
            break
    return {
        "success": bool(info["is_success"]),
        "return": ep_return,
        "table": table_sum,
        "drift": drift_sum,
        "table_steps": table_steps,
        "high_force_steps": high_force_steps,
        "cube_drift": float(np.linalg.norm(np.array(u.cube_position[:2]) - start)),
        "success_step": success_step,
        "steps": steps,
    }


def run_scripted_episode(env, dx: float, dy: float, target_offset: np.ndarray) -> dict:
    env.reset(seed=0, options={"cube_xy_offset": [dx, dy]})
    u = env.unwrapped
    start = np.array(u.cube_position[:2])
    target = u.cube_position + target_offset
    ep_return = table_sum = drift_sum = 0.0
    high_force_steps = table_steps = steps = 0
    success_step = None

    def record(info, reward):
        nonlocal ep_return, table_sum, drift_sum, high_force_steps, table_steps, steps
        steps += 1
        ep_return += float(reward)
        table_sum += float(info["reward_finger_table"])
        drift_sum += float(info["reward_cube_drift"])
        if info["finger_table_contact_count"] > 0:
            table_steps += 1
        if info["finger_table_force"] > 100.0:
            high_force_steps += 1

    for _ in range(60):
        err = target - u._get_tcp_position()
        if np.linalg.norm(err) < 5e-4:
            break
        action = dls_action(u, np.clip(err, -MAX_MOVE_STEP, MAX_MOVE_STEP), gripper=1.0)
        _, reward, terminated, truncated, info = env.step(action)
        record(info, reward)
        if terminated or truncated:
            break
    for _ in range(MAX_CLOSE_STEPS):
        err = np.clip(target - u._get_tcp_position(), -MAX_MOVE_STEP, MAX_MOVE_STEP)
        _, reward, terminated, truncated, info = env.step(dls_action(u, err, gripper=-1.0))
        record(info, reward)
        if info["is_success"] and success_step is None:
            success_step = steps
        if terminated or truncated:
            break
    return {
        "success": bool(info["is_success"]),
        "return": ep_return,
        "table": table_sum,
        "drift": drift_sum,
        "table_steps": table_steps,
        "high_force_steps": high_force_steps,
        "cube_drift": float(np.linalg.norm(np.array(u.cube_position[:2]) - start)),
        "success_step": success_step,
        "steps": steps,
    }


def test_7_bad_vs_clean(env, min_gap: float) -> None:
    rule("Test 7: Stage B v0 policy vs scripted clean grasp（同一 reward）")
    checkpoint = load_checkpoint(PROJECT_ROOT / STAGE_B_CHECKPOINT)
    agent = build_agent(checkpoint)
    policy_env = make_env(checkpoint, env.spec.id, 0)
    bad = [run_policy_episode(policy_env, agent, seed) for seed in range(5)]
    policy_env.close()

    stage_a = json.loads(STAGE_A_CONFIG_PATH.read_text())
    target_offset = (np.asarray(stage_a["reference_tcp_position"], dtype=float)
                     - np.asarray(stage_a["cube_position"], dtype=float))
    clean = [run_scripted_episode(env, dx, dy, target_offset)
             for dx, dy in ((0.0, 0.0), (0.02, 0.0), (0.0, 0.02), (0.02, 0.02))]

    print("  bad policy (Stage B v0, deterministic 5 episodes):")
    for i, r in enumerate(bad):
        print(f"    ep{i + 1}: success={str(r['success']):<5} return={r['return']:+7.3f} "
              f"table={r['table']:+7.3f} drift={r['drift']:+6.3f} "
              f"table_steps={r['table_steps']:2d} >100N={r['high_force_steps']:2d} "
              f"cube_drift={r['cube_drift'] * 100:5.2f}cm")
    print("  scripted clean grasp:")
    for (dx, dy), r in zip(((0.0, 0.0), (0.02, 0.0), (0.0, 0.02), (0.02, 0.02)), clean):
        print(f"    dx={dx * 1000:+5.1f} dy={dy * 1000:+5.1f}: success={str(r['success']):<5} "
              f"return={r['return']:+7.3f} table={r['table']:+7.3f} drift={r['drift']:+6.3f} "
              f"table_steps={r['table_steps']:2d} >100N={r['high_force_steps']:2d} "
              f"cube_drift={r['cube_drift'] * 100:5.2f}cm")

    bad_success = [r for r in bad if r["success"]]
    clean_success = [r for r in clean if r["success"]]
    check("bad policy 仍然能 success（success 定义未改）", len(bad_success) >= 4,
          f"{len(bad_success)}/{len(bad)}")
    check("scripted clean 全部 success", len(clean_success) == len(clean),
          f"{len(clean_success)}/{len(clean)}")
    if bad_success and clean_success:
        bad_return = float(np.mean([r["return"] for r in bad_success]))
        clean_return = float(np.mean([r["return"] for r in clean_success]))
        bad_table = float(np.mean([r["table"] for r in bad_success]))
        clean_table = float(np.mean([r["table"] for r in clean_success]))
        bad_drift = float(np.mean([r["drift"] for r in bad_success]))
        clean_drift = float(np.mean([r["drift"] for r in clean_success]))
        gap = clean_return - bad_return
        print(f"\n  mean return: press-policy {bad_return:+.3f} vs clean {clean_return:+.3f} "
              f"(gap {gap:+.3f})")
        print(f"  table penalty: {bad_table:+.3f} vs {clean_table:+.3f}")
        print(f"  drift penalty: {bad_drift:+.3f} vs {clean_drift:+.3f}")
        check("clean return > press-policy return", gap > 0.0)
        check(f"return gap >= {min_gap}（sanity target）", gap >= min_gap, f"{gap:+.3f}")
        check("press-policy table penalty 明显为负", bad_table < -0.3, f"{bad_table:+.3f}")
        check("clean table penalty 更小（更接近 0）", clean_table > bad_table)


def run_env_checks(env_id: str, weight: float, reward_version: str, task_version: str,
                   min_gap: float) -> None:
    env = gym.make(env_id)
    u = env.unwrapped
    rule(f"环境 {env_id}")
    print(f"  task_version   = {u.task_version}")
    print(f"  reward_version = {u.reward_version}")
    print(f"  half range     = ±{u._stage_b['xy_half_range_m'] * 1000:.1f} mm")
    print(f"  weights        = w_table={u.finger_table_penalty_weight}, "
          f"F0={u.finger_table_force_scale} N, w_drift={u.cube_drift_penalty_weight}, "
          f"norm={u.cube_drift_norm}")
    check("reward_version 正确", u.reward_version == reward_version, u.reward_version)
    check("task_version 正确", u.task_version == task_version, u.task_version)
    check("w_table 正确", abs(u.finger_table_penalty_weight - weight) < 1e-12,
          f"{u.finger_table_penalty_weight}")

    test_1_clean_stationary(env)
    test_2_table_force(env)
    test_3_sustained_press(env)
    test_4_6_cube_drift(env)
    test_7_bad_vs_clean(env, min_gap)
    env.close()


def main() -> None:
    for env_id, weight, reward_version, task_version, min_gap in ENV_CONFIGS:
        run_env_checks(env_id, weight, reward_version, task_version, min_gap)

    rule("结果")
    failed = [label for label, ok in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for label in failed:
            print(f"  FAILED: {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()

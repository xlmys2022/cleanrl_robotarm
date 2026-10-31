#!/usr/bin/env python3
"""Curriculum Stage B 检查：reset 规则 / 可复现性 / 物理可行性 / smoke test。

用法:
    python scripts/check_curriculum_stage_b.py

覆盖:
    1. reset offset 落在 ±half_range 内
    2. seed 可复现（同 seed 同 offset, 不同 seed 不同 offset）
    3. options={"cube_xy_offset": [dx, dy]} 精确生效, 超范围报错
    4. initial finger 完全打开, 未 grasp, 无 finger-table, 无穿透
    5. z / yaw / arm 不随机（只有 cube XY 变）
    6. scripted DLS + close 的 9 边界 grid 全部 success
    7. observation=(34,) / action=(8,) / 1000 random steps 无 NaN
"""

from __future__ import annotations

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
from envs.panda_pick_place_env import STAGE_A_CONFIG_PATH, STAGE_B_CONFIG_PATH  # noqa: E402

ENV_ID = "PandaGraspCurriculum-B-v0"
DLS_DAMPING = 0.02
MAX_MOVE_STEP = 0.002
MAX_CLOSE_STEPS = 60

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
    env.reset(seed=0, options={"cube_xy_offset": [dx, dy]})
    u = env.unwrapped
    target = u.cube_position + target_offset
    for _ in range(60):
        err = target - u._get_tcp_position()
        if np.linalg.norm(err) < 5e-4:
            break
        _, _, terminated, truncated, _ = env.step(
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


def main() -> None:
    config = json.loads(STAGE_B_CONFIG_PATH.read_text())
    stage_a = json.loads(STAGE_A_CONFIG_PATH.read_text())
    half_range = float(config["xy_half_range_m"])
    target_offset = (np.asarray(stage_a["reference_tcp_position"], dtype=float)
                     - np.asarray(stage_a["cube_position"], dtype=float))

    env = gym.make(ENV_ID)
    u = env.unwrapped

    rule("0. 配置")
    print(f"  task_version      = {u.task_version}")
    print(f"  initialization    = {u.initialization_mode}")
    print(f"  half range        = ±{half_range * 1000:.1f} mm")
    print(f"  max_episode_steps = {u.max_episode_steps}")
    print(f"  obs / action      = {env.observation_space.shape} / {env.action_space.shape}")

    rule("1. reset offset 范围 / 可复现性 / 显式 offset")
    offsets = []
    for seed in range(30):
        _, info = env.reset(seed=seed)
        offsets.append(np.asarray(info["initial_cube_xy_offset"], dtype=float))
    offsets = np.asarray(offsets)
    print(f"  30 次 random reset: dx∈[{offsets[:, 0].min():+.4f}, {offsets[:, 0].max():+.4f}], "
          f"dy∈[{offsets[:, 1].min():+.4f}, {offsets[:, 1].max():+.4f}]")
    check("offset 都在 ±half_range 内", bool(np.all(np.abs(offsets) <= half_range + 1e-9)))

    _, info_a1 = env.reset(seed=123)
    _, info_a2 = env.reset(seed=123)
    _, info_b = env.reset(seed=124)
    off_a1 = np.asarray(info_a1["initial_cube_xy_offset"])
    off_a2 = np.asarray(info_a2["initial_cube_xy_offset"])
    off_b = np.asarray(info_b["initial_cube_xy_offset"])
    print(f"  seed=123 -> {np.round(off_a1, 5)}, 再来一次 -> {np.round(off_a2, 5)}, "
          f"seed=124 -> {np.round(off_b, 5)}")
    check("seed=123 两次 offset 相同", bool(np.allclose(off_a1, off_a2)))
    check("seed=124 与 123 不同", not bool(np.allclose(off_a1, off_b)))

    _, info = env.reset(seed=1, options={"cube_xy_offset": [0.01, -0.01]})
    print(f"  explicit options offset -> {np.round(info['initial_cube_xy_offset'], 5)}, "
          f"info half_range={info['stage_b_half_range_m']}")
    check("显式 offset 精确生效",
          np.allclose(np.asarray(info["initial_cube_xy_offset"]), [0.01, -0.01]))
    try:
        env.reset(seed=1, options={"cube_xy_offset": [half_range + 0.01, 0.0]})
        raised = False
    except ValueError:
        raised = True
    check("超出 half_range 的 offset 会报错", raised)

    rule("2. reset 后是 open pre-grasp（只有 cube XY 变）")
    _, info = env.reset(seed=7)
    m = u._task_metrics()
    print(f"  finger qpos        = {np.round(np.asarray(m['finger_qpos']), 4)}")
    print(f"  bilateral={info['bilateral_contact']} valid={info['grasp_quality_valid']} "
          f"is_grasped={info['is_grasped']}")
    print(f"  finger-table={info['finger_table_contact_count']} "
          f"penetration={info['max_finger_cube_penetration'] * 1000:.2f}mm "
          f"cube_table_pen={info['cube_table_penetration'] * 1000:.2f}mm")
    print(f"  cube z={info['cube_position'][2]:.4f} (rest {info['cube_rest_z']:.4f}), "
          f"cube quat={np.round(u.data.qpos[u._cube_qpos_adr + 3: u._cube_qpos_adr + 7], 4)}")
    print(f"  TCP-cube rel={np.round(info['initial_tcp_cube_relative_position'], 4)}")
    check("finger 完全打开", bool(np.all(np.asarray(m["finger_qpos"]) > 0.039)))
    check("未 bilateral / 未 valid / 未 grasp",
          (not info["bilateral_contact"]) and (not info["grasp_quality_valid"])
          and (not info["is_grasped"]))
    check("无 finger-table 接触", info["finger_table_contact_count"] == 0)
    check("无 finger-cube 穿透", info["max_finger_cube_penetration"] < 1e-6)
    check("cube z 不随机（= cube_rest_z）", abs(info["cube_position"][2] - info["cube_rest_z"]) < 1e-9)
    check("cube yaw 不随机（quat=[1,0,0,0]）",
          bool(np.allclose(u.data.qpos[u._cube_qpos_adr + 3: u._cube_qpos_adr + 7],
                           [1, 0, 0, 0])))
    arm_qpos_1 = u.data.qpos[u._arm_qpos_adr].copy()
    env.reset(seed=99)
    check("arm pose 不随机", bool(np.allclose(u.data.qpos[u._arm_qpos_adr], arm_qpos_1)))

    rule(f"3. scripted DLS + close 边界 grid（±{half_range * 1000:.0f} mm）")
    grid = [(0.0, 0.0),
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    ok_count = 0
    for gx, gy in grid:
        dx, dy = gx * half_range, gy * half_range
        ok, step = scripted_oracle(env, dx, dy, target_offset)
        ok_count += int(ok)
        print(f"  dx={dx * 1000:+6.1f}mm dy={dy * 1000:+6.1f}mm -> success={ok} step={step}")
    check("scripted grid 9/9", ok_count == 9, f"{ok_count}/9")

    rule("4. smoke test（1000 random steps）")
    env.reset(seed=0)
    finite = True
    ep_returns = []
    ep_return = 0.0
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ep_return += float(reward)
        finite = finite and bool(np.isfinite(obs).all()) and bool(np.isfinite(reward))
        if terminated or truncated:
            ep_returns.append(ep_return)
            ep_return = 0.0
            env.reset()
    print(f"  obs finite={finite}, episodes finished={len(ep_returns)}, "
          f"obs.shape={obs.shape}, action.shape={env.action_space.shape}")
    check("1000 步无 NaN/Inf", finite)
    check("observation=34 / action=8", obs.shape == (34,) and env.action_space.shape == (8,))

    rule("总结")
    for label, ok in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = sum(ok for _, ok in RESULTS)
    print(f"\n  {passed}/{len(RESULTS)} checks passed")
    env.close()
    if passed != len(RESULTS):
        raise SystemExit(1)
    print("\nCurriculum Stage B check passed.")


if __name__ == "__main__":
    main()

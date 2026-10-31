#!/usr/bin/env python3
"""验证 Pick & Place 任务逻辑（reward gating / phase / success / termination），不训练。

用法:
    python scripts/check_pick_place_task.py

包含:
    Test 1  far state        —— 初始状态没有 grasp / lift / transport
    Test 2  reach            —— TCP 靠近 cube 时 reward_reach 增大, lift/transport 仍为 0
    Test 3  lift gating      —— cube 被抬高但没有双侧接触 -> reward_lift = 0, ever_lifted = False
    Test 4  transport gating —— cube 在 goal 附近但没有 grasp/lift 历史 -> reward_transport = 0
    Test 5  place gating     —— cube 静置在 goal 上但没有 history -> is_success = False
    Test 6  success path     —— 真实抓住 -> 抬起 -> 移动到 goal 附近 -> 释放 -> 稳定放置
    Part 7  smoke test       —— 四个环境各 1000 step 随机动作 + reward 量级统计

注意: 测试脚本里允许直接写 data.qpos / data.qvel 构造测试状态;
正式环境 step() 里没有任何这种代码。
"""

from __future__ import annotations

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
from envs.panda_pick_place_env import (  # noqa: E402
    LIFT_HEIGHT,
    PLACE_XY_TOL,
    REACH_DIST,
    REWARD_TERM_KEYS,
    SUCCESS_HOLD_STEPS,
    TRANSPORT_NEAR_GOAL_XY,
)
from check_grasp_contacts import setup_cube_in_gripper  # noqa: E402

RESULTS: list[tuple[str, bool]] = []


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(label: str, condition: bool, detail: str = "") -> bool:
    ok = bool(condition)
    RESULTS.append((label, ok))
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{suffix}")
    return ok


def print_terms(info: dict) -> None:
    print("    reward terms: "
          + "  ".join(f"{k.replace('reward_', '')}={info[k]:+.4f}" for k in REWARD_TERM_KEYS))
    print(f"    phase={info['task_phase']:<9} grasped={info['is_grasped']} "
          f"ever_grasped={info['ever_grasped']} ever_lifted={info['ever_lifted']} "
          f"near_goal={info['ever_transport_near_goal']} success={info['is_success']}")


# --------------------------------------------------------------------------- 状态注入工具
def set_cube(env, pos, quat=(1.0, 0.0, 0.0, 0.0), zero_vel: bool = True) -> None:
    u = env.unwrapped
    adr = u._cube_qpos_adr
    u.data.qpos[adr: adr + 3] = pos
    u.data.qpos[adr + 3: adr + 7] = quat
    if zero_vel:
        u.data.qvel[u._cube_dof_adr: u._cube_dof_adr + 6] = 0.0
    mujoco.mj_forward(u.model, u.data)


def set_goal_xy(env, xy) -> None:
    u = env.unwrapped
    u.model.site_pos[u._goal_site_id][0] = xy[0]
    u.model.site_pos[u._goal_site_id][1] = xy[1]
    mujoco.mj_forward(u.model, u.data)


def step_hold(env, n: int = 1, gripper: float = 0.0):
    """机械臂保持当前目标, 只按 gripper 分量动作, 推进 n 个 control step。"""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[7] = gripper
    info = {}
    for _ in range(n):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


# --------------------------------------------------------------------------- Test 1
def test_far_state() -> None:
    rule("Test 1: far state（reset 后什么都没发生）")
    env = gym.make("PandaPickPlace-v0")
    obs, info = env.reset(seed=0)
    info = step_hold(env, 1)
    print(f"  tcp_cube_distance = {info['tcp_cube_distance']:.4f} m, "
          f"cube_goal_xy = {info['cube_goal_xy_distance']:.4f} m")
    print(f"  cube_z = {info['cube_position'][2]:.4f}, rest_z = {info['cube_rest_z']:.4f}, "
          f"lift_threshold = {info['lift_threshold']:.4f}")
    print_terms(info)
    check("is_grasped == False", info["is_grasped"] is False)
    check("ever_grasped == False", info["ever_grasped"] is False)
    check("ever_lifted == False", info["ever_lifted"] is False)
    check("is_success == False", info["is_success"] is False)
    check("task_phase == 'reach'", info["task_phase"] == "reach")
    check("observation 仍是 34 维", obs.shape == (34,), str(obs.shape))
    env.close()


# --------------------------------------------------------------------------- Test 2
def test_reach() -> None:
    rule("Test 2: reach（cube 靠近 TCP, reward_reach 应增大）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    info_far = step_hold(env, 1)
    r_far = info_far["reward_reach"]

    u = env.unwrapped
    tcp = u._get_tcp_position()
    set_cube(env, tcp - np.array([0.0, 0.0, 0.02]))
    info_near = step_hold(env, 1)
    print(f"  tcp_cube_distance: {info_far['tcp_cube_distance']:.4f} -> "
          f"{info_near['tcp_cube_distance']:.4f} m")
    print(f"  reward_reach     : {r_far:+.4f} -> {info_near['reward_reach']:+.4f}")
    print_terms(info_near)
    check("reward_reach 增大", info_near["reward_reach"] > r_far,
          f"{r_far:+.3f} -> {info_near['reward_reach']:+.3f}")
    check("未抓住时 reward_lift == 0", info_near["reward_lift"] == 0.0)
    check("未抓住时 reward_transport == 0", info_near["reward_transport"] == 0.0)
    check("未抓住时 reward_grasp == 0", info_near["reward_grasp"] == 0.0)
    env.close()


# --------------------------------------------------------------------------- Test 3
def test_lift_gating() -> None:
    rule("Test 3: lift gating（cube 被人为抬高, 但没有双侧接触）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped
    set_cube(env, [0.62, 0.0, u._lift_threshold + 0.05])
    info = step_hold(env, 1)
    print(f"  cube_z = {info['cube_position'][2]:.4f} (lift_threshold = "
          f"{info['lift_threshold']:.4f}, LIFT_HEIGHT = {LIFT_HEIGHT})")
    print_terms(info)
    check("cube_z 高于 lift_threshold", info["cube_position"][2] > info["lift_threshold"])
    check("没有双侧接触", info["is_grasped"] is False)
    check("reward_lift == 0（未抓住不给 lift 奖励）", info["reward_lift"] == 0.0,
          f"{info['reward_lift']:+.4f}")
    check("ever_lifted == False（cube 自己飞起来不算 lift）", info["ever_lifted"] is False)
    env.close()


# --------------------------------------------------------------------------- Test 4
def test_transport_gating() -> None:
    rule("Test 4: transport gating（cube 已在 goal 附近, 但没有 grasp/lift 历史）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped
    set_cube(env, [u.goal_position[0], u.goal_position[1], u._cube_rest_z])
    info = step_hold(env, 3, gripper=+1.0)
    print(f"  cube_goal_xy_distance = {info['cube_goal_xy_distance']:.4f} m "
          f"(TRANSPORT_NEAR_GOAL_XY = {TRANSPORT_NEAR_GOAL_XY})")
    print_terms(info)
    check("cube 确实在 goal 附近", info["cube_goal_xy_distance"] < TRANSPORT_NEAR_GOAL_XY)
    check("reward_transport == 0", info["reward_transport"] == 0.0,
          f"{info['reward_transport']:+.4f}")
    check("ever_transport_near_goal == False", info["ever_transport_near_goal"] is False)
    check("is_success == False", info["is_success"] is False)
    env.close()


# --------------------------------------------------------------------------- Test 5
def test_place_gating() -> None:
    rule("Test 5: place gating（cube 静止放在 goal 上, 但没有 lift 历史）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped
    set_cube(env, [u.goal_position[0], u.goal_position[1], u._cube_rest_z])
    info = step_hold(env, 20, gripper=+1.0)  # 夹爪张开, 保持 20 步
    print(f"  cube_goal_xy_distance = {info['cube_goal_xy_distance']:.4f} m (< "
          f"PLACE_XY_TOL = {PLACE_XY_TOL})")
    print(f"  cube_z = {info['cube_position'][2]:.4f}, rest_z = {info['cube_rest_z']:.4f}")
    print(f"  cube 速度 = {info['cube_linear_speed']:.4f} m/s, "
          f"角速度 = {info['cube_angular_speed']:.4f} rad/s")
    print_terms(info)
    check("物理条件虽然满足, 但 ever_lifted == False", info["ever_lifted"] is False)
    check("success_hold_count == 0（history 不满足）", info["success_hold_count"] == 0,
          str(info["success_hold_count"]))
    check("is_success == False（直接推过去不算成功）", info["is_success"] is False)
    env.close()


# --------------------------------------------------------------------------- Test 6
def test_success_path() -> None:
    rule("Test 6: success path（grasp -> lift -> transport -> release -> stable place）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped

    # --- 1) 真实夹持: cube 放入两指之间, 闭合夹爪（真实 contact） ---
    tcp = setup_cube_in_gripper(env)
    close_action = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    grasp_info = None
    for _ in range(5):  # 夹爪闭合需要几个 control step 才压到 cube
        _, _, _, _, info = env.step(close_action)
        if info["is_grasped"]:
            grasp_info = info
            break
    assert grasp_info is not None, "测试未能形成真实夹持"
    print(f"  1) grasp: cube={np.array2string(grasp_info['cube_position'], precision=4)} "
          f"L={grasp_info['left_contact_force']:.2f}N "
          f"R={grasp_info['right_contact_force']:.2f}N grasped={grasp_info['is_grasped']} "
          f"grasp_event={grasp_info['reward_grasp_event']:+.2f}")
    check("真实双侧接触 -> is_grasped", grasp_info["is_grasped"] is True)
    check("ever_grasped == True", grasp_info["ever_grasped"] is True)
    check("reward_grasp_event 一次性发放", grasp_info["reward_grasp_event"] > 0.0,
          f"{grasp_info['reward_grasp_event']:+.2f}")
    info = step_hold(env, 1, gripper=-1.0)
    check("grasp event 不会每步重复发放", info["reward_grasp_event"] == 0.0,
          f"{info['reward_grasp_event']:+.2f}")

    # --- 2) lift: home 姿态下 TCP 在 z=0.52, 已经高于 lift_threshold, 抓着即满足 ---
    info = step_hold(env, 1, gripper=-1.0)
    print(f"  2) lift : cube_z={info['cube_position'][2]:.4f} "
          f">= lift_threshold={info['lift_threshold']:.4f}  "
          f"reward_lift={info['reward_lift']:+.4f} (potential 已饱和, 无增量)")
    check("ever_lifted == True", info["ever_lifted"] is True)
    check("抓住前 cube 已在 lift 高度 -> 不给虚假 lift 增量",
          abs(info["reward_lift"]) < 1e-6, f"{info['reward_lift']:+.4f}")

    # --- 3) transport: 把 goal 标记移到当前 cube 正上方（goal 只是标记, 不是物理体） ---
    cube_xy = u.cube_position[:2]
    set_goal_xy(env, cube_xy)
    info = step_hold(env, 1, gripper=-1.0)   # 这一步 goal 距离突然变近 -> 一次性 positive progress
    print(f"  3) transport: cube_goal_xy={info['cube_goal_xy_distance']:.4f} m "
          f"reward_transport={info['reward_transport']:+.4f}")
    check("ever_transport_near_goal == True", info["ever_transport_near_goal"] is True)
    check("抓着到达 goal 附近 -> reward_transport > 0", info["reward_transport"] > 0.0,
          f"{info['reward_transport']:+.3f}")
    info = step_hold(env, 1, gripper=-1.0)
    check("transport 停住后不再持续给奖励", abs(info["reward_transport"]) < 0.05,
          f"{info['reward_transport']:+.4f}")

    # --- 4) release: 张开夹爪, 让 cube 落回桌面并静止 ---
    hold_history: list[int] = []
    success_step = None
    last_info = info
    for i in range(300):
        _, reward, terminated, truncated, last_info = env.step(
            np.array([0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        )
        if last_info["success_hold_count"] > 0:
            hold_history.append(last_info["success_hold_count"])
        if terminated or truncated:
            success_step = i
            break
    print(f"  4) release: 释放后 {success_step} 步终止; "
          f"cube={np.array2string(last_info['cube_position'], precision=4)} "
          f"v={last_info['cube_linear_speed']:.4f} m/s "
          f"omega={last_info['cube_angular_speed']:.4f} rad/s")
    print(f"     success_hold_count 序列 = {hold_history[:15]}"
          f"{' ...' if len(hold_history) > 15 else ''}")
    print_terms(last_info)
    check("success_hold_count 逐步累加而不是一步成功",
          len(hold_history) >= SUCCESS_HOLD_STEPS["pick_place"])
    check("is_success == True", last_info["is_success"] is True)
    check("terminated == True", success_step is not None)
    check("reward_success 只发一次 +20", last_info["reward_success"] == 20.0,
          f"{last_info['reward_success']:+.1f}")
    check("failure_reason is None", last_info["failure_reason"] is None,
          str(last_info["failure_reason"]))
    env.close()


# --------------------------------------------------------------------------- Test 6b
def test_per_task_success() -> None:
    rule("Test 6b: Reach / Grasp / Lift 各自的 success 与终止")
    zero = np.zeros(8, dtype=np.float32)

    # reach: 每一步把 cube 重新放到 TCP（测试专用注入, 否则 cube 会自由下落）
    env = gym.make("PandaReach-v0")
    env.reset(seed=0)
    u = env.unwrapped
    success_step = None
    hold = 0
    for i in range(10):
        set_cube(env, u._get_tcp_position())
        _, _, terminated, truncated, info = env.step(zero)
        hold = info["success_hold_count"]
        if terminated or truncated:
            success_step = i
            break
    print(f"  reach : cube 逐步放在 TCP, success 于 step {success_step}, "
          f"hold={hold}, is_success={info['is_success']}, phase={info['task_phase']}")
    check("reach success 需要连续 5 步", success_step is not None and success_step >= 4,
          f"step={success_step}")
    check("reach: terminated 且 is_success", bool(info["is_success"]))
    env.close()

    # grasp / lift: 真实夹持后继续保持, 应在 5 步后 success
    for env_id in ("PandaGrasp-v0", "PandaLift-v0"):
        env = gym.make(env_id)
        env.reset(seed=0)
        setup_cube_in_gripper(env)
        close_action = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        success_step = None
        for i in range(10):
            _, _, terminated, truncated, info = env.step(close_action)
            if terminated or truncated:
                success_step = i
                break
        print(f"  {env_id:<15} success 于 step {success_step}, hold={info['success_hold_count']}, "
              f"is_success={info['is_success']}, phase={info['task_phase']}")
        check(f"{env_id}: terminated 且 is_success", bool(info["is_success"]))
        env.close()


# --------------------------------------------------------------------------- Part 7
def rollout_stats(env_id: str, steps: int = 1000, scale: float = 0.2, seed: int = 0) -> None:
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    assert obs.shape == (34,), f"{env_id}: obs.shape = {obs.shape}"
    assert env.action_space.shape == (8,)

    rewards = np.zeros(steps)
    terms = {k: np.zeros(steps) for k in REWARD_TERM_KEYS}
    n_term = n_trunc = n_success = 0
    for i in range(steps):
        action = env.action_space.sample() * scale
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert env.observation_space.contains(obs)
        rewards[i] = reward
        for k in REWARD_TERM_KEYS:
            terms[k][i] = info[k]
        if terminated:
            n_term += 1
            n_success += int(info["is_success"])
        if truncated:
            n_trunc += 1
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    print(f"\n  {env_id}  ({steps} steps, action×{scale})")
    print(f"    reward: mean={rewards.mean():+.4f}  min={rewards.min():+.4f}  "
          f"max={rewards.max():+.4f}  (sum={rewards.sum():+.2f})")
    print(f"    terminated={n_term} (success={n_success})  truncated={n_trunc}")
    print("    term means: " + "  ".join(
        f"{k.replace('reward_', '')}={terms[k].mean():+.4f}" for k in REWARD_TERM_KEYS))
    print(f"    |action penalty| mean = {abs(terms['reward_action'].mean()):.4f}, "
          f"|smooth penalty| mean = {abs(terms['reward_smooth'].mean()):.4f}, "
          f"time penalty mean = {terms['reward_time'].mean():.4f}, "
          f"reach(progress) mean = {terms['reward_reach'].mean():+.4f}")
    # 惩罚项必须远小于任务奖励量级（success=+20, lift/transport ≤ 2/3）
    assert abs(terms["reward_action"].mean()) < 0.01
    assert abs(terms["reward_smooth"].mean()) < 0.01
    assert abs(terms["reward_action"].mean()) + abs(terms["reward_smooth"].mean()) < 0.1 * 2.0


def main() -> None:
    test_far_state()
    test_reach()
    test_lift_gating()
    test_transport_gating()
    test_place_gating()
    test_success_path()
    test_per_task_success()

    rule("Part 7: 四个环境 1000 step 随机 rollout + reward 量级")
    for env_id in envs.TASK_ENV_IDS:
        rollout_stats(env_id)

    rule("总结")
    passed = sum(ok for _, ok in RESULTS)
    for label, ok in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  {passed}/{len(RESULTS)} checks passed")
    if passed != len(RESULTS):
        raise SystemExit(1)
    print("\nTask logic check passed.")


if __name__ == "__main__":
    main()

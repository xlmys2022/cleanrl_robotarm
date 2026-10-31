#!/usr/bin/env python3
"""验证 reward 不会鼓励"故意不成功"（anti-stall / progress 方向），不训练。

用法:
    python scripts/check_reward_incentives.py

背景:
    reward 已改成 potential difference:  r = w * (phi_t - phi_{t-1})。
    旧写法 r_reach = 1 - tanh(k*d) 每步都给正值, 因此"停在 success 阈值外侧"可以一直刷分;
    改成差分后, 停在原地每步 ≈ 0, 只剩 time penalty。

测试:
    A. reach 停在 d≈0.041 m: 第 1 步之后 reward_reach ≈ 0, 总 reward 为负
    B. 快速 success 的 return > 一直 hover 的 return（本阶段最重要）
    C. lift 停在固定高度: 第 2 步之后 reward_lift ≈ 0
    D. transport 抓住 cube 停在 goal 附近: 第 2 步之后 reward_transport ≈ 0
    E. progress 符号: 靠近 -> 正, 远离 -> 负
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import envs  # noqa: E402,F401
from check_grasp_contacts import setup_cube_in_gripper  # noqa: E402
from check_pick_place_task import set_cube, set_goal_xy  # noqa: E402

ZERO_ACTION = np.zeros(8, dtype=np.float32)
GRIP_CLOSE = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)

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


def fake_metrics(env, tcp_cube_distance: float = 0.09,
                 cube_z: float | None = None, cube_goal_xy: float = 0.15) -> dict:
    """reward/potential 只用这三个量, 用于 reward 函数单元测试（注入 metrics）。"""
    u = env.unwrapped
    if cube_z is None:
        cube_z = u._cube_rest_z
    return {
        "tcp_cube_distance": float(tcp_cube_distance),
        "cube_z": float(cube_z),
        "cube_goal_xy_distance": float(cube_goal_xy),
    }


# --------------------------------------------------------------------------- Test A
def test_reach_stationary() -> None:
    rule("Test A: reach 停在 d≈0.041 m（success 阈值 0.04 外侧）")
    env = gym.make("PandaReach-v0")
    env.reset(seed=0)
    u = env.unwrapped
    hover_offset = np.array([0.0, 0.0, 0.041])
    # 先用一步 warm-up 进入 hover 状态（注入本身的距离跃迁不计入统计）
    set_cube(env, u._get_tcp_position() - hover_offset)
    env.step(ZERO_ACTION)
    rewards, reach_terms = [], []
    for _ in range(10):
        # 测试专用状态注入: 让 cube 保持在 TCP 下方 0.041 m, 模拟"hover 在阈值外"
        set_cube(env, u._get_tcp_position() - hover_offset)
        _, r, _, _, info = env.step(ZERO_ACTION)
        rewards.append(r)
        reach_terms.append(info["reward_reach"])
    print(f"  reward_reach 逐步 = {np.array2string(np.array(reach_terms), precision=4)}")
    print(f"  10 步 return = {sum(rewards):+.4f}  (time penalty -0.01/step)")
    check("第 1 步之后 reward_reach ≈ 0（不再每步 +0.6）",
          max(abs(x) for x in reach_terms[1:]) < 0.05,
          f"max|r_reach|={max(abs(x) for x in reach_terms[1:]):.4f}")
    check("hover 的总 return 为负", sum(rewards) < 0.0, f"{sum(rewards):+.3f}")
    env.close()


# --------------------------------------------------------------------------- Test B
def test_fast_success_beats_hover() -> None:
    rule("Test B: 快速 success 的 return > 一直 hover 的 return")
    # Policy A: 快速进入 0.04 m 并保持 5 步 -> success
    env = gym.make("PandaReach-v0")
    env.reset(seed=0)
    u = env.unwrapped
    ret_a, steps_a = 0.0, 0
    for i in range(20):
        set_cube(env, u._get_tcp_position())
        _, r, terminated, truncated, info = env.step(ZERO_ACTION)
        ret_a += r
        steps_a = i + 1
        if terminated or truncated:
            break
    success_a = info["is_success"]
    env.close()

    # Policy B: 一直停在 0.041 m 直到 500 步截断
    env = gym.make("PandaReach-v0")
    env.reset(seed=0)
    u = env.unwrapped
    ret_b, steps_b = 0.0, 0
    for i in range(500):
        set_cube(env, u._get_tcp_position() - np.array([0.0, 0.0, 0.041]))
        _, r, terminated, truncated, info = env.step(ZERO_ACTION)
        ret_b += r
        steps_b = i + 1
        if terminated or truncated:
            break
    env.close()

    print(f"  Policy A (快速 success): step={steps_a}, success={success_a}, return={ret_a:+.3f}")
    print(f"  Policy B (hover 到截断): step={steps_b}, return={ret_b:+.3f}")
    check("Policy A 真的成功了", success_a is True)
    check("Return(快速 success) > Return(hover)", ret_a > ret_b,
          f"{ret_a:+.2f} > {ret_b:+.2f}")


# --------------------------------------------------------------------------- Test C
def test_lift_stationary() -> None:
    rule("Test C: 已 grasp, cube 停在 lift_threshold 下方固定高度")
    env = gym.make("PandaLift-v0")
    env.reset(seed=0)
    u = env.unwrapped
    # reward 函数单元测试: 注入 metrics 与 grasp 状态（不依赖能否在空中共抓）
    u._is_grasped = True
    u._ever_grasped = True
    u._ever_lifted = False
    u._prev_pot = u._potentials(fake_metrics(env, cube_z=u._cube_rest_z))
    stationary_z = u._cube_rest_z + 0.03      # 比 lift 目标低, 但已抬起 3 cm
    lift_terms = []
    for _ in range(6):
        terms = u._compute_reward_from_metrics(
            ZERO_ACTION, fake_metrics(env, cube_z=stationary_z)
        )
        lift_terms.append(terms["reward_lift"])
    print(f"  reward_lift 逐步 = {np.array2string(np.array(lift_terms), precision=4)}")
    check("第 1 步有正 progress（0 -> 0.03 m 抬升）", lift_terms[0] > 0.5,
          f"{lift_terms[0]:+.3f}")
    check("第 2 步之后 reward_lift ≈ 0（不能每步 +1.9）",
          max(abs(x) for x in lift_terms[1:]) < 1e-6,
          f"max|r_lift|={max(abs(x) for x in lift_terms[1:]):.2e}")
    env.close()


# --------------------------------------------------------------------------- Test D
def test_transport_stationary() -> None:
    rule("Test D: 抓住 cube 停在 goal 附近（真实物理）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped
    setup_cube_in_gripper(env)
    for _ in range(3):
        _, _, _, _, info = env.step(GRIP_CLOSE)     # 闭合夹爪, 形成双侧接触
    assert info["is_grasped"], "测试未能形成真实夹持"
    set_goal_xy(env, u.cube_position[:2])           # goal 标记移到 cube 正上方
    _, _, _, _, info = env.step(GRIP_CLOSE)         # 产生一次 goal 距离跃迁
    transport_terms = [info["reward_transport"]]
    for _ in range(10):
        _, _, _, _, info = env.step(GRIP_CLOSE)
        transport_terms.append(info["reward_transport"])
    print(f"  reward_transport 逐步 = "
          f"{np.array2string(np.array(transport_terms), precision=4)}")
    print(f"  ever_lifted={info['ever_lifted']}, is_grasped={info['is_grasped']}, "
          f"cube_goal_xy={info['cube_goal_xy_distance']:.4f} m")
    check("第 2 步之后 reward_transport ≈ 0（不能每步 +2.9）",
          max(abs(x) for x in transport_terms[1:]) < 0.05,
          f"max|r_transport|={max(abs(x) for x in transport_terms[1:]):.4f}")
    env.close()


# --------------------------------------------------------------------------- Test E
def test_progress_sign() -> None:
    rule("Test E: progress 符号（靠近为正, 远离为负）")

    # --- reach: 靠近 / 远离 ---
    env = gym.make("PandaReach-v0")
    env.reset(seed=0)
    u = env.unwrapped
    set_cube(env, u._get_tcp_position() - np.array([0.0, 0.0, 0.08]))
    _, _, _, _, _ = env.step(ZERO_ACTION)            # 建立 prev potential (d≈0.08)
    set_cube(env, u._get_tcp_position() - np.array([0.0, 0.0, 0.05]))
    _, _, _, _, info = env.step(ZERO_ACTION)
    r_closer = info["reward_reach"]
    set_cube(env, u._get_tcp_position() - np.array([0.0, 0.0, 0.09]))
    _, _, _, _, info = env.step(ZERO_ACTION)
    r_farther = info["reward_reach"]
    print(f"  reach : 靠近 d 0.08->0.05  r_reach={r_closer:+.4f};  "
          f"远离 d 0.05->0.09  r_reach={r_farther:+.4f}")
    check("靠近 cube -> reward_reach > 0", r_closer > 0.0, f"{r_closer:+.4f}")
    check("远离 cube -> reward_reach < 0", r_farther < 0.0, f"{r_farther:+.4f}")
    env.close()

    # --- lift: 抬高 / 降低（reward 函数单元测试） ---
    env = gym.make("PandaLift-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u._is_grasped = True
    u._ever_grasped = True
    u._ever_lifted = False
    u._prev_pot = u._potentials(fake_metrics(env, cube_z=u._cube_rest_z + 0.02))
    r_up = u._compute_reward_from_metrics(
        ZERO_ACTION, fake_metrics(env, cube_z=u._cube_rest_z + 0.04)
    )["reward_lift"]
    r_down = u._compute_reward_from_metrics(
        ZERO_ACTION, fake_metrics(env, cube_z=u._cube_rest_z + 0.03)
    )["reward_lift"]
    print(f"  lift  : 抬高 0.02->0.04 m  r_lift={r_up:+.4f};  降低 0.04->0.03 m  "
          f"r_lift={r_down:+.4f}")
    check("抬高 cube -> reward_lift > 0", r_up > 0.0, f"{r_up:+.4f}")
    check("放低 cube -> reward_lift < 0", r_down < 0.0, f"{r_down:+.4f}")
    env.close()

    # --- transport: 向 goal 移动（真实物理） ---
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped
    setup_cube_in_gripper(env)
    for _ in range(3):
        _, _, _, _, info = env.step(GRIP_CLOSE)
    assert info["is_grasped"], "测试未能形成真实夹持"
    cube_xy = u.cube_position[:2].copy()
    # 先拉开到 0.30 m（比初始 0.164 m 更远）, 再移回 cube 正上方
    set_goal_xy(env, cube_xy + np.array([0.30, 0.0]))

    # 这一跳是 goal 变远 -> transport potential 下降（负）
    _, _, _, _, info = env.step(GRIP_CLOSE)
    r_away = info["reward_transport"]
    set_goal_xy(env, cube_xy)
    _, _, _, _, info = env.step(GRIP_CLOSE)
    r_toward = info["reward_transport"]
    print(f"  transport: 远离 goal r_transport={r_away:+.4f};  "
          f"靠近 goal r_transport={r_toward:+.4f}")
    check("向 goal 移动 -> reward_transport > 0", r_toward > 0.0, f"{r_toward:+.4f}")
    check("远离 goal -> reward_transport < 0", r_away < 0.0, f"{r_away:+.4f}")
    env.close()


# --------------------------------------------------------------------------- Test F
def test_no_sustained_rewards() -> None:
    rule("Test F: 没有持续正奖励（grasp / time）")
    env = gym.make("PandaPickPlace-v0")
    env.reset(seed=0)
    u = env.unwrapped
    setup_cube_in_gripper(env)
    info = {}
    for _ in range(5):
        _, _, _, _, info = env.step(GRIP_CLOSE)
    print(f"  is_grasped={info['is_grasped']}  reward_grasp={info['reward_grasp']:+.4f}  "
          f"reward_grasp_event={info['reward_grasp_event']:+.4f}  "
          f"reward_time={info['reward_time']:+.4f}")
    check("抓住期间 reward_grasp == 0（不再每步 +0.5）", info["reward_grasp"] == 0.0)
    check("grasp event 仍然只发一次", info["reward_grasp_event"] == 0.0)
    check("reward_time 为恒定负值", abs(info["reward_time"] + 0.01) < 1e-9,
          f"{info['reward_time']:+.4f}")
    env.close()


def main() -> None:
    test_reach_stationary()
    test_fast_success_beats_hover()
    test_lift_stationary()
    test_transport_stationary()
    test_progress_sign()
    test_no_sustained_rewards()

    rule("总结")
    for label, ok in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = sum(ok for _, ok in RESULTS)
    print(f"\n  {passed}/{len(RESULTS)} checks passed")
    if passed != len(RESULTS):
        raise SystemExit(1)
    print("\nReward incentive check passed.")


if __name__ == "__main__":
    main()

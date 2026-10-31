#!/usr/bin/env python3
"""验证 PandaGrasp-v1 的 strict grasp gate。

用法:
    python scripts/check_grasp_quality_gate.py

Test 1  bad learned grasp（旧 checkpoint 的下压式接触）在 v1 里必须被拒绝
Test 2  proper side pinch 在 v1 里必须直接 success
Test 3  raw bilateral 但深穿透 -> is_grasped False（含合成 metrics 单元测试）
Test 4  finger<->table 承力 -> is_grasped False
Test 5  cube 被压进桌面 -> is_grasped False
Test 6  hold_count 必须重置（3 步 valid + 1 步 invalid + 2 步 valid 不能算 success）
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
from eval_reach import build_agent, load_checkpoint, policy_action  # noqa: E402
from check_grasp_contacts import setup_cube_in_gripper  # noqa: E402
from envs.panda_pick_place_env import (  # noqa: E402
    MAX_CUBE_TABLE_PENETRATION,
    MAX_FINGER_CUBE_PENETRATION,
    MAX_FINGER_TABLE_FORCE,
)

BAD_CKPT = (PROJECT_ROOT
            / "runs/PandaGrasp-v0__train_robot__1__1793353158/train_robot.checkpoint.pt")
CLOSE = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
OPEN = np.array([0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)

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


def fake_metrics(env, **overrides) -> dict:
    """从真实 metrics 出发, 覆盖某个质量字段, 用于 gate 单元测试。"""
    m = env.unwrapped._task_metrics()
    m.update(overrides)
    return m


# --------------------------------------------------------------------------- Test 1
def test_bad_grasp_rejected() -> None:
    rule("Test 1: legacy 下压式抓取在 PandaGrasp-v1 中必须被拒绝")
    checkpoint = load_checkpoint(BAD_CKPT)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    mean, var = norm["mean"].numpy(), norm["var"].numpy()
    eps = float(norm["epsilon"])
    lo, hi = float(norm["clip_low"]), float(norm["clip_high"])

    env = gym.make("PandaGrasp-v1")
    obs, info = env.reset(seed=0)
    saw_bilateral = False
    saw_accepted = False
    reasons: list[str] = []
    worst_force = 0.0
    for step in range(1, 21):
        obs_norm = np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps),
                           lo, hi).astype(np.float32)
        action = policy_action(agent, obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        worst_force = max(worst_force, info["left_contact_force"], info["right_contact_force"])
        if info["bilateral_contact"]:
            saw_bilateral = True
            reasons.append(str(info["grasp_invalid_reason"]))
        saw_accepted = saw_accepted or info["is_grasped"]
        if terminated or truncated:
            break

    print(f"  step={step} bilateral_seen={saw_bilateral} is_grasped_seen={saw_accepted}")
    print(f"  finger_table_contacts={info['finger_table_contact_count']}, "
          f"max_penetration={info['max_finger_cube_penetration'] * 1000:.2f} mm, "
          f"cube_table_penetration={info['cube_table_penetration'] * 1000:.2f} mm, "
          f"max force={worst_force:.1f} N")
    check("旧策略确实产生了 bilateral contact", saw_bilateral is True)
    check("v1 从未把它当作 is_grasped", saw_accepted is False)
    check("v1 没有 success", info["is_success"] is False)
    check("invalid reason 明确（不是 none）",
          "multiple" in reasons or any(r in ("finger_table_contact",
                                             "deep_finger_cube_penetration",
                                             "cube_pressed_into_table") for r in reasons),
          str(sorted(set(reasons))))
    env.close()


# --------------------------------------------------------------------------- Test 2
def test_proper_pinch_accepted() -> None:
    rule("Test 2: proper side pinch 必须通过并被 v1 判定 success")
    env = gym.make("PandaGrasp-v1")
    env.reset(seed=0)
    setup_cube_in_gripper(env)
    info = {}
    success_step = None
    for step in range(1, 21):
        _, _, terminated, truncated, info = env.step(CLOSE)
        if info["is_success"]:
            success_step = step
            break
        if terminated or truncated:
            break
    print(f"  success_step={success_step}, hold={info['success_hold_count']}, "
          f"bilateral={info['bilateral_contact']}, valid={info['grasp_quality_valid']}")
    print(f"  finger_table_contacts={info['finger_table_contact_count']}, "
          f"penetration={info['max_finger_cube_penetration'] * 1000:.2f} mm, "
          f"cube_table_penetration={info['cube_table_penetration'] * 1000:.2f} mm")
    check("valid grasp", info["grasp_quality_valid"] is True)
    check("is_grasped", info["is_grasped"] is True)
    check("v1 success", info["is_success"] is True, f"step={success_step}")
    env.close()


# --------------------------------------------------------------------------- Test 3
def test_bilateral_alone_insufficient() -> None:
    rule("Test 3: raw bilateral + 深穿透 => invalid")
    env = gym.make("PandaGrasp-v1")
    env.reset(seed=0)
    setup_cube_in_gripper(env)
    for _ in range(8):
        _, _, terminated, truncated, info = env.step(CLOSE)
        if terminated or truncated:
            break
    u = env.unwrapped
    m = fake_metrics(
        env,
        bilateral_contact=True,
        left_contact=True,
        right_contact=True,
        max_finger_cube_penetration=MAX_FINGER_CUBE_PENETRATION + 0.005,
    )
    valid, reason = u._evaluate_grasp_quality(m)
    print(f"  合成 metrics: bilateral=True, penetration="
          f"{m['max_finger_cube_penetration'] * 1000:.1f} mm -> valid={valid}, reason={reason}")
    check("深穿透时 valid=False", valid is False)
    check("reason == deep_finger_cube_penetration",
          reason == "deep_finger_cube_penetration", str(reason))
    env.close()


# --------------------------------------------------------------------------- Test 4
def test_finger_table_contact() -> None:
    rule("Test 4: finger<->table 承力 => invalid")
    env = gym.make("PandaGrasp-v1")
    env.reset(seed=0)
    u = env.unwrapped
    m = fake_metrics(
        env,
        bilateral_contact=True,
        finger_table_contact_count=6,
        finger_table_force=MAX_FINGER_TABLE_FORCE + 10.0,
    )
    valid, reason = u._evaluate_grasp_quality(m)
    print(f"  合成 metrics: finger_table_force={m['finger_table_force']:.1f} N "
          f"-> valid={valid}, reason={reason}")
    check("finger-table 承力时 valid=False", valid is False)
    check("reason == finger_table_contact", reason == "finger_table_contact", str(reason))
    env.close()


# --------------------------------------------------------------------------- Test 5
def test_cube_pressed_into_table() -> None:
    rule("Test 5: cube 被压进桌面 => invalid")
    env = gym.make("PandaGrasp-v1")
    env.reset(seed=0)
    u = env.unwrapped
    m = fake_metrics(
        env,
        bilateral_contact=True,
        cube_table_penetration=MAX_CUBE_TABLE_PENETRATION + 0.010,
    )
    valid, reason = u._evaluate_grasp_quality(m)
    print(f"  合成 metrics: cube_table_penetration="
          f"{m['cube_table_penetration'] * 1000:.1f} mm -> valid={valid}, reason={reason}")
    check("cube 被压进桌面时 valid=False", valid is False)
    check("reason == cube_pressed_into_table", reason == "cube_pressed_into_table", str(reason))
    env.close()


# --------------------------------------------------------------------------- Test 6
def test_hold_count_reset() -> None:
    rule("Test 6: hold_count 必须在 invalid 后重置（3 + 2 不能算 success）")
    env = gym.make("PandaGrasp-v1")
    env.reset(seed=0)
    setup_cube_in_gripper(env)

    hold = 0
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(CLOSE)
        hold = info["success_hold_count"]
        if terminated or truncated or hold >= 3:
            break
    print(f"  valid 连续 {hold} 步 (expect 3, 未 success: {not info['is_success']})")
    check("valid 3 步时还没 success", hold == 3 and not info["is_success"], f"hold={hold}")

    _, _, terminated, truncated, info = env.step(OPEN)   # 破坏 valid grasp
    print(f"  张开一步后: bilateral={info['bilateral_contact']} valid={info['grasp_quality_valid']} "
          f"is_grasped={info['is_grasped']} hold={info['success_hold_count']}")
    check("invalid 一步后 hold_count 归零", info["success_hold_count"] == 0,
          f"hold={info['success_hold_count']}")
    check("没有因为 3 步而 success", info["is_success"] is False)

    counts = []
    for _ in range(10):                      # 重新闭合需要几步才能再形成 valid grasp
        _, _, terminated, truncated, info = env.step(CLOSE)
        if info["success_hold_count"] > 0:
            counts.append(info["success_hold_count"])
            break
    if counts:                               # 再走一步, 应该严格 +1
        _, _, terminated, truncated, info = env.step(CLOSE)
        counts.append(info["success_hold_count"])
    print(f"  重新 valid 后 hold 序列 = {counts}, success={info['is_success']}")
    check("重新计数从 1 开始（不是 4/5）", counts == [1, 2], str(counts))
    check("2 步时仍未 success", info["is_success"] is False)

    for _ in range(6):
        _, _, terminated, truncated, info = env.step(CLOSE)
        if terminated or truncated:
            break
    check("继续到 5 步才 success", info["is_success"] is True,
          f"hold={info['success_hold_count']}")
    env.close()


def main() -> None:
    test_bad_grasp_rejected()
    test_proper_pinch_accepted()
    test_bilateral_alone_insufficient()
    test_finger_table_contact()
    test_cube_pressed_into_table()
    test_hold_count_reset()

    rule("总结")
    for label, ok in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = sum(ok for _, ok in RESULTS)
    print(f"\n  {passed}/{len(RESULTS)} checks passed")
    if passed != len(RESULTS):
        raise SystemExit(1)
    print("\nGrasp quality gate check passed.")


if __name__ == "__main__":
    main()

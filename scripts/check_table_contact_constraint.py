#!/usr/bin/env python3
"""Stage B.7 table-contact failure constraint 回归（不训练）。

用法:
    python scripts/check_table_contact_constraint.py

覆盖:
    Test 1  无接触 10 步                  -> counter=0, 不失败
    Test 2  单步 150N 冲击                -> 1 -> 0, 不失败
    Test 3  连续两步 150/160N             -> 1 -> 2 -> 0, 不失败
    Test 4  连续三步 150/160/170N         -> 第 3 步 terminated + failure
    Test 4b success 与 failure 同一步     -> failure 优先
    Test 5  150,150,0,150,150             -> 计数必须连续, 不失败
    Test 6  真实物理压桌                  -> 环境自己 terminated
    Test 7  clean scripted 9 boundary grid -> 9/9 success, 0 constraint failure
    Test 8  旧环境默认 disabled            -> B-v0 行为不受影响
"""

from __future__ import annotations

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

ENV_ID = "PandaGraspCurriculum-B-v3"
DLS_DAMPING = 0.02
MAX_MOVE_STEP = 0.002
MAX_CLOSE_STEPS = 60
HOLD_ACTION = np.array([0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)

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


class ForcedForce:
    """把 finger_table_force 替换成指定值, 走完整 step() 路径（不绕过 priority/termination）。"""

    def __init__(self, u, value: float = 0.0) -> None:
        self.u = u
        self.value = float(value)
        self._orig = u._task_metrics
        u._task_metrics = self._patched

    def _patched(self) -> dict:
        m = self._orig()
        m["finger_table_force"] = self.value
        return m

    def set(self, value: float) -> None:
        self.value = float(value)

    def restore(self) -> None:
        self.u._task_metrics = self._orig


def run_force_sequence(env, forces: list[float], reset_seed: int = 0) -> list[dict]:
    """按给定 force 序列逐步 step, 返回每步 info 快照。"""
    u = env.unwrapped
    u._reset_task_state()
    env.reset(seed=reset_seed, options={"cube_xy_offset": [0.0, 0.0]})
    forced = ForcedForce(u)
    rows = []
    try:
        for f in forces:
            forced.set(f)
            _, reward, terminated, truncated, info = env.step(HOLD_ACTION)
            rows.append({
                "force": f,
                "count": int(info["table_high_force_hold_count"]),
                "triggered": bool(info["table_contact_failure_triggered"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "is_success": bool(info["is_success"]),
                "failure_reason": info["failure_reason"],
                "reward_success": float(info["reward_success"]),
                "reward_total": float(info["reward_total"]),
            })
            if terminated or truncated:
                break
    finally:
        forced.restore()
    return rows


def test_1_no_contact(env) -> None:
    rule("Test 1: 无接触 10 步")
    rows = run_force_sequence(env, [0.0] * 10)
    print("  " + "  ".join(f"c={r['count']}" for r in rows))
    check("10 步 counter 始终 0", all(r["count"] == 0 for r in rows))
    check("10 步都不 triggered", all(not r["triggered"] for r in rows))
    check("没有提前 terminated", all(not r["terminated"] for r in rows))


def test_2_single_impact(env) -> None:
    rule("Test 2: 单步 150N 冲击")
    rows = run_force_sequence(env, [150.0, 0.0])
    print(f"  force=150 -> count={rows[0]['count']}, force=0 -> count={rows[1]['count']}")
    check("第 1 步 count == 1", rows[0]["count"] == 1)
    check("第 2 步 count 归零", rows[1]["count"] == 0)
    check("不失败", all(not r["triggered"] and not r["terminated"] for r in rows))


def test_3_two_impacts(env) -> None:
    rule("Test 3: 两步高力 150/160/0")
    rows = run_force_sequence(env, [150.0, 160.0, 0.0])
    print(f"  counts = {[r['count'] for r in rows]}")
    check("count 序列 == [1, 2, 0]", [r["count"] for r in rows] == [1, 2, 0])
    check("不失败", all(not r["triggered"] and not r["terminated"] for r in rows))


def test_4_three_sustained(env) -> None:
    rule("Test 4: 连续三步高力 150/160/170")
    rows = run_force_sequence(env, [150.0, 160.0, 170.0])
    print(f"  counts = {[r['count'] for r in rows]}, "
          f"terminated = {[r['terminated'] for r in rows]}")
    last = rows[-1]
    check("第 3 步 counter == 3", len(rows) == 3 and last["count"] == 3)
    check("第 3 步 terminated", last["terminated"])
    check("is_success == False", not last["is_success"])
    check("failure_reason == sustained_finger_table_force",
          last["failure_reason"] == "sustained_finger_table_force")
    check("拿不到 success bonus", last["reward_success"] == 0.0)


def test_4b_failure_priority(env) -> None:
    rule("Test 4b: success 条件与 failure 同一步 -> failure 优先")
    u = env.unwrapped
    u._reset_task_state()
    env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
    forced = ForcedForce(u)
    orig_success_condition = u._success_condition
    u._success_condition = lambda m: True     # 强制 success 条件成立
    try:
        rows = []
        for i in range(3):
            forced.set(150.0)
            if i == 2:
                u._success_hold_count = 4          # 这一步本该达成 success
            _, _, terminated, truncated, info = env.step(HOLD_ACTION)
            rows.append(info)
            if terminated or truncated:
                break
    finally:
        forced.restore()
        u._success_condition = orig_success_condition
    info = rows[-1]
    print(f"  第 3 步: hold_count={info['success_hold_count']}, "
          f"is_success={info['is_success']}, failure_reason={info['failure_reason']}, "
          f"terminated={terminated}")
    check("同一步 failure 优先，is_success 仍为 False", not info["is_success"])
    check("failure_reason 是 table constraint",
          info["failure_reason"] == "sustained_finger_table_force")
    check("success hold 被清零", info["success_hold_count"] == 0)
    check("reward_success == 0", info["reward_success"] == 0.0)


def test_5_counter_must_be_consecutive(env) -> None:
    rule("Test 5: 计数必须连续 150/150/0/150/150")
    rows = run_force_sequence(env, [150.0, 150.0, 0.0, 150.0, 150.0])
    print(f"  counts = {[r['count'] for r in rows]}")
    check("count 序列 == [1, 2, 0, 1, 2]", [r["count"] for r in rows] == [1, 2, 0, 1, 2])
    check("始终不失败", all(not r["triggered"] and not r["terminated"] for r in rows))


def test_6_real_physics(env) -> None:
    rule("Test 6: 真实物理压桌（环境自己 terminated）")
    u = env.unwrapped
    env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
    first_contact = first_high = None
    rows = []
    for step in range(1, 13):
        _, _, terminated, truncated, info = env.step(
            dls_action(u, np.array([0.0, 0.0, -0.004]), gripper=1.0)
        )
        if info["finger_table_contact_count"] > 0 and first_contact is None:
            first_contact = step
        if info["finger_table_force"] > 100.0 and first_high is None:
            first_high = step
        rows.append((step, float(info["finger_table_force"]),
                     int(info["table_high_force_hold_count"]), bool(terminated),
                     info["failure_reason"]))
        if terminated or truncated:
            break
    for step, f, c, term, reason in rows:
        print(f"  step {step}: force={f:7.1f} N  count={c}  terminated={term}  "
              f"reason={reason}")
    check("真实物理产生 >100N 接触", first_high is not None)
    check("环境自己 terminated", terminated)
    check("failure_reason == sustained_finger_table_force",
          info["failure_reason"] == "sustained_finger_table_force")
    check("failure step 在首个 >100N 步之后 ≤3 步",
          first_high is not None and rows[-1][0] - first_high <= 2,
          f"first_high={first_high}, fail={rows[-1][0]}")


def test_7_clean_scripted(env) -> None:
    rule("Test 7: clean scripted grasp（9 boundary grid）")
    import json
    stage_a = json.loads(STAGE_A_CONFIG_PATH.read_text())
    target_offset = (np.asarray(stage_a["reference_tcp_position"], dtype=float)
                     - np.asarray(stage_a["cube_position"], dtype=float))
    r = env.unwrapped._stage_b["xy_half_range_m"]
    offsets = [(0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
               (r, r), (r, -r), (-r, r), (-r, -r)]
    successes = 0
    failures = 0
    for dx, dy in offsets:
        env.reset(seed=0, options={"cube_xy_offset": [dx, dy]})
        u = env.unwrapped
        target = u.cube_position + target_offset
        ok = False
        terminated = truncated = False
        info = {"failure_reason": None, "is_success": False}
        for _ in range(60):
            err = target - u._get_tcp_position()
            if np.linalg.norm(err) < 5e-4:
                break
            _, _, terminated, truncated, info = env.step(
                dls_action(u, np.clip(err, -MAX_MOVE_STEP, MAX_MOVE_STEP), gripper=1.0)
            )
            if terminated or truncated:
                break
        if not (terminated or truncated):
            for _ in range(MAX_CLOSE_STEPS):
                err = np.clip(target - u._get_tcp_position(), -MAX_MOVE_STEP, MAX_MOVE_STEP)
                _, _, terminated, truncated, info = env.step(
                    dls_action(u, err, gripper=-1.0)
                )
                if info["is_success"]:
                    ok = True
                    break
                if terminated or truncated:
                    break
        successes += int(ok)
        failures += int(info["failure_reason"] == "sustained_finger_table_force")
        print(f"  dx={dx * 1000:+6.1f} dy={dy * 1000:+6.1f} success={str(ok):<5} "
              f"failure={info['failure_reason']}")
    check("scripted clean 9/9 success", successes == 9, f"{successes}/9")
    check("0 table constraint failures", failures == 0, f"{failures}")


def test_8_old_env_disabled() -> None:
    rule("Test 8: 旧环境默认 disabled")
    for env_id in ("PandaGraspCurriculum-B-v0", "PandaGraspCurriculum-B-v1",
                   "PandaGraspCurriculum-B-v2"):
        env = gym.make(env_id)
        u = env.unwrapped
        env.reset(seed=0, options={"cube_xy_offset": [0.0, 0.0]})
        _, _, terminated, truncated, info = env.step(
            dls_action(u, np.array([0.0, 0.0, -0.004]), gripper=1.0)
        )
        print(f"  {env_id}: mode={u.table_contact_failure_mode}, "
              f"count={info['table_high_force_hold_count']}, terminated={terminated}")
        check(f"{env_id} constraint disabled", u.table_contact_failure_mode == "disabled")
        check(f"{env_id} 不会被 table 约束终止", not terminated)
        env.close()


def main() -> None:
    env = gym.make(ENV_ID)
    u = env.unwrapped
    rule("0. 环境 / 参数")
    print(f"  env_id          = {ENV_ID}")
    print(f"  task_version    = {u.task_version}")
    print(f"  reward_version  = {u.reward_version}")
    print(f"  mode            = {u.table_contact_failure_mode}")
    print(f"  threshold       = {u.table_force_failure_threshold} N")
    print(f"  hold_steps      = {u.table_force_failure_hold_steps}")
    print(f"  half range      = ±{u._stage_b['xy_half_range_m'] * 1000:.1f} mm")
    check("task_version 正确",
          u.task_version == "grasp_curriculum_stage_b_table_constraint_v1")
    check("reward_version == potential_v1（无 quality penalty）",
          u.reward_version == "potential_v1")
    check("threshold == 100 N", u.table_force_failure_threshold == 100.0)
    check("hold_steps == 3", u.table_force_failure_hold_steps == 3)

    test_1_no_contact(env)
    test_2_single_impact(env)
    test_3_two_impacts(env)
    test_4_three_sustained(env)
    test_4b_failure_priority(env)
    test_5_counter_must_be_consecutive(env)
    test_6_real_physics(env)
    test_7_clean_scripted(env)
    env.close()
    test_8_old_env_disabled()

    rule("结果")
    failed = [label for label, ok in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for label in failed:
            print(f"  FAILED: {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""验证物理链路：Grasp checkpoint 抓稳 -> scripted Cartesian lift (+Z) -> cube 是否真的抬起来。

用法:
    python scripts/check_grasp_then_lift.py [--model-path <Grasp checkpoint>]

设计:
    1. 用 Grasp checkpoint（deterministic actor mean）真实抓取 cube, 直到双侧接触连续 5 步;
    2. 记录抓取时刻的全部状态（TCP/cube/finger/ctrl/力/joint）;
    3. lift 阶段不再用策略, 而是 Jacobian + DLS 让 TCP 沿世界 +Z 缓慢上升,
       再把关节增量除以 max_joint_delta 映射回环境原来的 action[:7];
       夹爪保持"抓取成功那一刻"的 ctrl, 不重新全闭也不张开;
    4. 由 PandaLift-v0 自己判定 success（is_grasped AND cube_z >= threshold 连续 5 步）;
    5. lift 结束后再做 2 s hold test（100 control steps 不动）, 测量 cube 下滑速度。

测试脚本只做状态读取与动作注入, 不修改 is_grasped / ever_grasped / is_success。
"""

from __future__ import annotations

import argparse
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
from eval_reach import build_agent, load_checkpoint, policy_action  # noqa: E402

DEFAULT_GRASP_CKPT = (
    PROJECT_ROOT / "runs/PandaGrasp-v0__train_robot__1__1793353158/train_robot.checkpoint.pt"
)

GRASP_HOLD_STEPS = 5          # 双侧接触连续 5 步视为抓稳
MAX_GRASP_STEPS = 60
MAX_LIFT_STEPS = 160          # 0.8 mm/step -> 160 步最多 +12.8 cm
LIFT_DZ_PER_STEP = 0.0008     # 40 mm/s
DLS_DAMPING = 0.02
HOLD_STEPS = 100              # 2 s @ 50 Hz
HIGH_FORCE_WARNING = 100.0    # 只告警, 不判失败


def rule(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def tcp_jacobian_arm(u) -> np.ndarray:
    """TCP 世界点的平移 Jacobian, 只保留 7 个 arm joint 对应的列 (3x7)。"""
    jacp = np.zeros((3, u.model.nv))
    jacr = np.zeros((3, u.model.nv))
    mujoco.mj_jac(u.model, u.data, jacp, jacr, u._get_tcp_position(), u._hand_body_id)
    return jacp[:, u._arm_dof_adr]


def lift_action(u, dz: float, gripper_action: float) -> np.ndarray:
    """Cartesian +Z 增量 -> 关节增量(DLS) -> 归一化 action[:7]（走环境原有 action 接口）。"""
    jac = tcp_jacobian_arm(u)
    dx = np.array([0.0, 0.0, dz])
    dq = jac.T @ np.linalg.solve(jac @ jac.T + (DLS_DAMPING ** 2) * np.eye(3), dx)
    action = np.zeros(8, dtype=np.float32)
    action[:7] = np.clip(dq / u.max_joint_delta, -1.0, 1.0)
    action[7] = gripper_action
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Grasp -> scripted lift 物理可行性验证")
    parser.add_argument("--model-path", default=str(DEFAULT_GRASP_CKPT))
    parser.add_argument("--env-id", default="PandaLift-v0",
                        help="默认 legacy lift 环境；也可传 PandaGrasp-v1 做 grasp 阶段对比")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    mean, var = norm["mean"].numpy(), norm["var"].numpy()
    eps = float(norm["epsilon"])
    clip_lo, clip_hi = float(norm["clip_low"]), float(norm["clip_high"])

    def normalize(obs):
        return np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps),
                       clip_lo, clip_hi).astype(np.float32)

    env = gym.make(args.env_id)
    obs, info = env.reset(seed=args.seed)
    u = env.unwrapped
    print(f"checkpoint : {args.model_path}")
    print(f"env        : {args.env_id}  obs={obs.shape} action={env.action_space.shape}")
    print(f"control_dt : {u.control_dt:.3f}s  max_joint_delta={u.max_joint_delta}"
          f"  lift_threshold={u._lift_threshold:.4f} m  cube_rest_z={u._cube_rest_z:.4f} m")

    # ------------------------------------------------------------------ 1. grasp
    rule(f"1. Grasp（Grasp checkpoint, deterministic, 双侧接触连续 {GRASP_HOLD_STEPS} 步）")
    max_left_force = max_right_force = 0.0
    bilateral_run = 0
    grasp_step = None
    for step in range(1, MAX_GRASP_STEPS + 1):
        action = policy_action(agent, normalize(obs), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        max_left_force = max(max_left_force, info["left_contact_force"])
        max_right_force = max(max_right_force, info["right_contact_force"])
        bilateral_run = bilateral_run + 1 if (info["left_contact"] and info["right_contact"]) else 0
        if step % 4 == 0 or bilateral_run == GRASP_HOLD_STEPS:
            print(f"  step {step:3d} dist={info['tcp_cube_distance']:.4f} "
                  f"finger={info['finger_mean_position']:.4f} "
                  f"L/R={info['left_contact_force']:6.2f}/{info['right_contact_force']:6.2f}N "
                  f"grasped={info['is_grasped']} run={bilateral_run}")
        if bilateral_run >= GRASP_HOLD_STEPS:
            grasp_step = step
            break
        if terminated or truncated:
            break

    if grasp_step is None:
        print("\n[FAIL] 未能在 Lift 环境里稳定抓取, 停止。")
        env.close()
        raise SystemExit(1)

    tcp_z_grasp = float(u._get_tcp_position()[2])
    cube_z_grasp = float(u.cube_position[2])
    gripper_ctrl_grasp = float(u.data.ctrl[u._gripper_act_id])
    gripper_action_hold = float(
        2.0 * (gripper_ctrl_grasp - u._gripper_ctrl_lo) / (u._gripper_ctrl_hi - u._gripper_ctrl_lo) - 1.0
    )
    gripper_action_hold = float(np.clip(gripper_action_hold, -1.0, 1.0))
    print(f"\n  grasp 成功: step={grasp_step}, TCP_z={tcp_z_grasp:.4f}, cube_z={cube_z_grasp:.4f}, "
          f"dist={info['tcp_cube_distance']:.4f}")
    print(f"  finger qpos={np.array2string(u.data.qpos[u._finger_qpos_adr], precision=4)}")
    print(f"  gripper ctrl={gripper_ctrl_grasp:.2f} -> hold action[7]={gripper_action_hold:+.3f}")
    print(f"  arm joint qpos={np.array2string(u.data.qpos[u._arm_qpos_adr], precision=4)}")
    print(f"  arm joint ctrl={np.array2string(u.data.ctrl[u._arm_act_ids], precision=4)}")
    print(f"  grasp 阶段 max force: L={max_left_force:.2f} N, R={max_right_force:.2f} N")
    if max(max_left_force, max_right_force) > HIGH_FORCE_WARNING:
        print(f"  WARNING: high contact force (> {HIGH_FORCE_WARNING} N)")

    # 抓取几何诊断: 手指是否压在桌面上 / 手指与 cube 的穿透深度
    finger_ids = u._left_finger_geom_ids | u._right_finger_geom_ids
    finger_table_contacts = 0
    deepest_penetration = 0.0
    for i in range(u.data.ncon):
        c = u.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if u._table_geom_id in (g1, g2) and (g1 in finger_ids or g2 in finger_ids):
            finger_table_contacts += 1
        if u._cube_geom_id in (g1, g2) and (g1 in finger_ids or g2 in finger_ids):
            deepest_penetration = min(deepest_penetration, float(c.dist))
    print(f"  诊断: finger<->table contact 数 = {finger_table_contacts}, "
          f"finger<->cube 最深穿透 = {deepest_penetration * 1000:+.2f} mm")
    print(f"  诊断: TCP-cube z 差 = {tcp_z_grasp - cube_z_grasp:+.4f} m, "
          f"finger qpos={np.array2string(u.data.qpos[u._finger_qpos_adr], precision=4)} "
          f"(cube 边长 0.045 m)")

    # ------------------------------------------------------------------ 2. lift
    rule(f"2. Scripted lift（TCP +Z, {LIFT_DZ_PER_STEP * 1000:.1f} mm/step = "
         f"{LIFT_DZ_PER_STEP / u.control_dt * 1000:.0f} mm/s, DLS damping={DLS_DAMPING}）")
    threshold_reached_step = None
    success_step = None
    bilateral_duration = 0
    lift_max_left_force = lift_max_right_force = 0.0
    tcp_z = tcp_z_grasp
    rel_offset_start = tcp_z_grasp - cube_z_grasp

    for step in range(1, MAX_LIFT_STEPS + 1):
        action = lift_action(u, LIFT_DZ_PER_STEP, gripper_action_hold)
        obs, reward, terminated, truncated, info = env.step(action)
        tcp_z = float(u._get_tcp_position()[2])
        cube_z = float(info["cube_position"][2])
        lift_max_left_force = max(lift_max_left_force, info["left_contact_force"])
        lift_max_right_force = max(lift_max_right_force, info["right_contact_force"])
        bilateral_duration = bilateral_duration + 1 if info["is_grasped"] else 0
        if threshold_reached_step is None and cube_z >= u._lift_threshold:
            threshold_reached_step = step

        if step % 5 == 0 or info["is_success"]:
            print(f"  step {step:3d} TCP_z={tcp_z:.4f} cube_z={cube_z:.4f} "
                  f"(cube dz={cube_z - cube_z_grasp:+.4f}) "
                  f"rel={tcp_z - cube_z:+.4f} L/R={info['left_contact_force']:6.2f}/"
                  f"{info['right_contact_force']:6.2f}N grasped={info['is_grasped']}")
        if info["is_success"]:
            success_step = step
            break
        if terminated or truncated:
            print(f"  [warn] lift 阶段提前结束: terminated={terminated} truncated={truncated}")
            break

    tcp_dz = tcp_z - tcp_z_grasp
    cube_dz = float(u.cube_position[2]) - cube_z_grasp
    tracking_ratio = cube_dz / tcp_dz if abs(tcp_dz) > 1e-6 else float("nan")

    rule("3. Lift 结果")
    print(f"  grasp success step        : {grasp_step}")
    print(f"  lift start cube z         : {cube_z_grasp:.4f} m")
    print(f"  lift start TCP z          : {tcp_z_grasp:.4f} m")
    print(f"  lift threshold            : {u._lift_threshold:.4f} m")
    print(f"  threshold reached step    : {threshold_reached_step}")
    print(f"  lift success step         : {success_step}")
    print(f"  TCP vertical displacement : {tcp_dz * 1000:+.1f} mm")
    print(f"  cube vertical displacement: {cube_dz * 1000:+.1f} mm")
    print(f"  lift tracking ratio       : {tracking_ratio:.3f}")
    print(f"  bilateral contact duration: {bilateral_duration} control steps "
          f"({bilateral_duration * u.control_dt:.2f} s)")
    print(f"  max contact force (lift)  : L={lift_max_left_force:.2f} N, "
          f"R={lift_max_right_force:.2f} N")
    print(f"  cube z at lift end        : {u.cube_position[2]:.4f} m")
    print(f"  relative TCP-cube offset  : start={rel_offset_start:+.4f} -> "
          f"now={tcp_z - u.cube_position[2]:+.4f} m")

    if max(lift_max_left_force, lift_max_right_force) > HIGH_FORCE_WARNING:
        print(f"  WARNING: high contact force (> {HIGH_FORCE_WARNING} N) 出现在 lift 阶段")

    # ------------------------------------------------------------------ 4. hold test
    rule(f"4. Hold test（保持 arm/gripper 目标 {HOLD_STEPS} 步 = "
         f"{HOLD_STEPS * u.control_dt:.1f} s, 测量 slip）")
    hold_action = np.zeros(8, dtype=np.float32)
    hold_action[7] = gripper_action_hold
    cube_z_hold_start = float(u.cube_position[2])
    tcp_z_hold_start = float(u._get_tcp_position()[2])
    hold_bilateral = 0
    hold_left_force = hold_right_force = 0.0
    for step in range(1, HOLD_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(hold_action)
        hold_bilateral = hold_bilateral + 1 if info["is_grasped"] else 0
        hold_left_force = max(hold_left_force, info["left_contact_force"])
        hold_right_force = max(hold_right_force, info["right_contact_force"])
        if step % 20 == 0:
            print(f"  step {step:3d} TCP_z={u._get_tcp_position()[2]:.4f} "
                  f"cube_z={u.cube_position[2]:.4f} "
                  f"rel={u._get_tcp_position()[2] - u.cube_position[2]:+.4f} "
                  f"grasped={info['is_grasped']}")

    cube_z_drop = cube_z_hold_start - float(u.cube_position[2])
    rel_drift = ((tcp_z_hold_start - cube_z_hold_start)
                 - (float(u._get_tcp_position()[2]) - float(u.cube_position[2])))
    hold_seconds = HOLD_STEPS * u.control_dt
    slip_mm_s = rel_drift / hold_seconds * 1000.0

    print(f"\n  cube z: {cube_z_hold_start:.4f} -> {u.cube_position[2]:.4f} m "
          f"(drop {cube_z_drop * 1000:+.2f} mm)")
    print(f"  relative TCP-cube drift   : {rel_drift * 1000:+.2f} mm")
    print(f"  estimated slip            : {slip_mm_s:+.3f} mm/s")
    print(f"  2 s 内最大连续 bilateral  : {hold_bilateral} steps "
          f"({hold_bilateral * u.control_dt:.2f} s)")
    print(f"  hold 阶段 max force       : L={hold_left_force:.2f} N, R={hold_right_force:.2f} N")
    if hold_bilateral == 0:
        print("  注意: hold 期间 cube 并没有被夹住, 这里的 slip 数字只反映自由 cube, "
              "不能当作抓取滑移率")

    env.close()

    # ------------------------------------------------------------------ 结论
    rule("5. 结论")
    ok = success_step is not None
    if ok:
        print("  Grasp checkpoint + scripted lift 能够在物理上把 cube 抬高 "
              f"{cube_dz * 1000:.1f} mm 并让 PandaLift-v0 判定 success。")
    else:
        print("  物理链路失败, 失败分类:")
        if finger_table_contacts > 0 and deepest_penetration < -0.005:
            print("    E/F: 学到的 grasp 是「下压式」而不是侧面夹持 —— "
                  f"finger<->table contact {finger_table_contacts} 个, "
                  f"finger<->cube 穿透 {deepest_penetration * 1000:.1f} mm。")
            print("         lift 一开始 cube 就回弹/被释放, 属于 grasp 姿态问题, 不是 friction 或力不足。")
        elif abs(tcp_dz) < 0.01:
            print("    D: Jacobian lift controller 没有让 TCP 上升")
        elif tracking_ratio < 0.2:
            print(f"    A/B: TCP 上升 {tcp_dz * 1000:.1f} mm 但 cube 只升 "
                  f"{cube_dz * 1000:.1f} mm (friction / gripping force 不足)")
        elif bilateral_duration < 3:
            print("    C/E: lift 过程中双侧接触很快消失 (grasp geometry / 初始姿态不适合 lift)")
        elif threshold_reached_step is None:
            print("    E: 抬升过程中 cube 没到 threshold (可能边升边滑)")
        else:
            print("    F: 达到 threshold 但未能保持 5 步 (high-impact grasp 导致不稳定)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

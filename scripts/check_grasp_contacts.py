#!/usr/bin/env python3
"""验证 Panda 双指对 cube 的真实 contact 检测（不是训练，也不是距离近似）。

用法:
    python scripts/check_grasp_contacts.py

做法:
1. 空夹爪闭合 —— 必须检测不到 finger<->cube contact（防误报）;
2. 把 cube 临时放到两个 fingertip pad 之间（预闭合夹爪），真实闭合夹爪;
3. 逐步读取 MuJoCo data.contact + mj_contactForce, 打印 contact pair / dist / 法向力;
4. 汇总双侧接触时 contact.dist 与 normal force 的实际范围, 用于确定
   envs/panda_pick_place_env.py 里的 CONTACT_FORCE_THRESHOLD。
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from envs.panda_pick_place_env import CONTACT_FORCE_THRESHOLD  # noqa: E402

# 预闭合开度: 两指 pad 内侧间隙略大于 cube 边长 0.045 m, 保证 cube 放入时几乎贴合
PREGRASP_FINGER = 0.0245
CUBE_SETUP_DZ = 0.006     # cube 初始略高于 pad 中心, 留一点下落余量


def gripper_ctrl_ratio(model: mujoco.MjModel, gripper_aid: int) -> float:
    """tendon 位置伺服: 目标 tendon_length = ratio * ctrl。"""
    return float(model.actuator_gainprm[gripper_aid][0] / -model.actuator_biasprm[gripper_aid][1])


def setup_cube_in_gripper(env, dz: float = CUBE_SETUP_DZ,
                          finger_open: float = PREGRASP_FINGER) -> np.ndarray:
    """把 cube 放到当前 TCP（两指 pad 中点）附近, 预闭合夹爪到 finger_open。

    只在测试脚本里使用的状态注入; 环境 step() 里没有任何这类代码。
    """
    u = env.unwrapped
    ratio = gripper_ctrl_ratio(u.model, u._gripper_act_id)
    tcp = u._get_tcp_position()

    u.data.qpos[u._finger_qpos_adr] = finger_open
    u.data.ctrl[u._gripper_act_id] = finger_open / ratio

    adr = u._cube_qpos_adr
    u.data.qpos[adr: adr + 3] = tcp + np.array([0.0, 0.0, dz])
    u.data.qpos[adr + 3: adr + 7] = [1.0, 0.0, 0.0, 0.0]
    u.data.qvel[:] = 0.0

    mujoco.mj_forward(u.model, u.data)
    return tcp


def finger_contact_forces(env) -> tuple[float, float]:
    """返回 (left 法向力合力, right 法向力合力) [N]，直接来自 data.contact。"""
    return env.unwrapped._read_grasp_contacts()


def cube_contact_pairs(env) -> list[tuple[int, int, float, float]]:
    """列出所有与 cube 相关的 contact: (geom1, geom2, dist, normal_force)。"""
    u = env.unwrapped
    out = []
    f6 = np.zeros(6)
    for i in range(u.data.ncon):
        c = u.data.contact[i]
        if int(c.geom1) != u._cube_geom_id and int(c.geom2) != u._cube_geom_id:
            continue
        mujoco.mj_contactForce(u.model, u.data, i, f6)
        out.append((int(c.geom1), int(c.geom2), float(c.dist), abs(float(f6[0]))))
    return out


def describe_geom(env, gid: int) -> str:
    u = env.unwrapped
    name = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if name:
        return f"{name}(id={gid})"
    if gid == u._cube_geom_id:
        return f"cube_geom(id={gid})"
    if gid in u._left_finger_geom_ids:
        return f"left_finger_geom(id={gid})"
    if gid in u._right_finger_geom_ids:
        return f"right_finger_geom(id={gid})"
    body = int(u.model.geom_bodyid[gid])
    bname = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_BODY, body)
    return f"unnamed(id={gid}, body={bname})"


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    env = gym.make("PandaGrasp-v0")
    u = env.unwrapped

    rule("0. 接触检测用到的 id（全部由 name / body 解析, 不硬编码）")
    print(f"  cube_geom_id            = {u._cube_geom_id}")
    print(f"  left_finger  geom ids   = {sorted(u._left_finger_geom_ids)}")
    print(f"  right_finger geom ids   = {sorted(u._right_finger_geom_ids)}")
    print(f"  CONTACT_FORCE_THRESHOLD = {CONTACT_FORCE_THRESHOLD} N")
    print(f"  gripper ctrl -> tendon_length 比例 = "
          f"{gripper_ctrl_ratio(u.model, u._gripper_act_id):.10g}")

    # ---------------------------------------------------------------- 空夹爪
    rule("1. 空夹爪闭合（cube 在桌面上, 夹爪内部为空）")
    env.reset(seed=0)
    max_force = 0.0
    for _ in range(150):
        action = np.zeros(8, dtype=np.float32)
        action[7] = -1.0  # 闭合
        _, _, _, _, info = env.step(action)
        max_force = max(max_force, info["left_contact_force"], info["right_contact_force"])
    print(f"  闭合后 finger 位置 = {info['finger_mean_position']:.4f} "
          f"(全闭 0.0, 全开 0.04)")
    print(f"  left_contact={info['left_contact']}  right_contact={info['right_contact']}  "
          f"is_grasped={info['is_grasped']}")
    print(f"  与 cube 相关的 contact 数 = {len(cube_contact_pairs(env))}, "
          f"最大法向力 = {max_force:.4f} N")
    assert not info["is_grasped"], "空夹爪不应判定为抓住"
    print("  -> PASS: 空夹爪不会误判为 grasp")

    # ---------------------------------------------------------------- 真实夹持
    rule("2. 真实夹持: cube 放入两指之间, 闭合夹爪")
    env.reset(seed=0)
    tcp = setup_cube_in_gripper(env)
    print(f"  TCP = {np.array2string(tcp, precision=4)}")
    print(f"  cube 初始位置 = {np.array2string(u.cube_position, precision=4)} "
          f"(预闭合 finger={PREGRASP_FINGER:.4f})")

    grasp_step = None
    dist_values: list[float] = []
    force_values: list[float] = []
    for step in range(200):
        action = np.zeros(8, dtype=np.float32)
        action[7] = -1.0  # 闭合
        _, _, _, _, info = env.step(action)
        pairs = cube_contact_pairs(env)
        for _, _, dist, force in pairs:
            dist_values.append(dist)
            force_values.append(force)

        if step < 4 or (info["is_grasped"] and grasp_step is None):
            print(f"\n  step {step:3d}: finger={info['finger_mean_position']:.4f} "
                  f"cube={np.array2string(info['cube_position'], precision=4)} "
                  f"L={info['left_contact']}({info['left_contact_force']:.3f}N) "
                  f"R={info['right_contact']}({info['right_contact_force']:.3f}N) "
                  f"grasped={info['is_grasped']}")
            for g1, g2, dist, force in pairs[:6]:
                print(f"      contact {describe_geom(env, g1)} <-> {describe_geom(env, g2)}"
                      f"  dist={dist:+.6f} m  normal_force={force:.4f} N")
            if len(pairs) > 6:
                print(f"      ... 共 {len(pairs)} 条 cube contact（其余省略）")

        if info["is_grasped"] and grasp_step is None:
            grasp_step = step

    print(f"\n  首次检测到双侧接触: step {grasp_step}  (50 Hz 下 = "
          f"{grasp_step * u.control_dt:.3f} s)")
    print(f"  left_contact detected  = {info['left_contact']}")
    print(f"  right_contact detected = {info['right_contact']}")
    print(f"  bilateral grasp detected = {info['is_grasped']}")

    if dist_values:
        print(f"\n  实测 contact.dist   : {min(dist_values):+.6f} ~ {max(dist_values):+.6f} m")
        print(f"  实测 normal force   : {min(force_values):.4f} ~ {max(force_values):.4f} N")
        print(f"  法向力中位数        : {np.median(force_values):.4f} N")
    print(f"  稳定夹持 200 步后 cube 位置 = "
          f"{np.array2string(info['cube_position'], precision=4)}")
    print(f"  is_grasped 仍为 {info['is_grasped']}, ever_grasped={info['ever_grasped']}")

    assert info["is_grasped"], "未能检测到双侧接触"
    env.close()
    print("\nGrasp contact check passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""检查自定义 pick&place 场景（只读, 不使用 Gymnasium）。

用法:
    python scripts/check_pick_place_scene.py

检查内容:
1. 场景能加载, 打印 nq/nv/nu/nbody/ngeom/nsite;
2. 关键对象能按名字查找到: hand / cube / cube_geom / goal / table;
3. 打印 hand / TCP / cube / goal 位置;
4. 推进仿真, 验证 cube 自然落到桌面并稳定、无 NaN、无穿透、Panda 不爆炸。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 场景加载器是纯 MuJoCo 的（不含 Gymnasium 逻辑）, 直接复用, 避免复制一份 keyframe 处理逻辑
from envs.scene_loader import SCENE_XML, load_pick_place_model  # noqa: E402

TABLE_TOP_Z = 0.40          # 场景里桌面顶面高度 [m]
CUBE_HALF_SIZE = 0.0225     # cube 边长 4.5 cm 的一半 [m]
SIM_SECONDS = 3.0


def rule(title: str = "") -> None:
    print()
    if title:
        print("=" * 70)
        print(title)
        print("=" * 70)


def name_of(model: mujoco.MjModel, objtype: int, objid: int) -> str:
    return mujoco.mj_id2name(model, objtype, objid) or f"<unnamed#{objid}>"


def main() -> None:
    model = load_pick_place_model()
    data = mujoco.MjData(model)

    rule("1. 场景总体信息")
    print(f"scene xml      : {SCENE_XML}")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  "
          f"ngeom={model.ngeom}  nsite={model.nsite}  nkey={model.nkey}")

    rule("2. 关键对象（name -> id）")
    checks = [
        ("hand", mujoco.mjtObj.mjOBJ_BODY),
        ("cube", mujoco.mjtObj.mjOBJ_BODY),
        ("cube_geom", mujoco.mjtObj.mjOBJ_GEOM),
        ("goal", mujoco.mjtObj.mjOBJ_SITE),
        ("table", mujoco.mjtObj.mjOBJ_GEOM),
        ("overview_camera", mujoco.mjtObj.mjOBJ_CAMERA),
        ("cube_freejoint", mujoco.mjtObj.mjOBJ_JOINT),
        ("home", mujoco.mjtObj.mjOBJ_KEY),
    ]
    all_found = True
    for name, objtype in checks:
        oid = mujoco.mj_name2id(model, objtype, name)
        print(f"  {name:<18} id = {oid}")
        all_found &= oid >= 0

    rule("3. reset 到 home 后的位置")
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qvel[:] = 0.0
    data.ctrl[:] = model.key_ctrl[home]
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")

    hand_pos = data.xpos[hand_id].copy()
    hand_rot = data.xmat[hand_id].reshape(3, 3)
    tcp_pos = hand_pos + hand_rot @ np.array([0.0, 0.0, 0.103])
    cube_pos = data.xpos[cube_id].copy()
    goal_pos = data.site_xpos[goal_id].copy()

    print(f"  hand position : {np.array2string(hand_pos, precision=4)}")
    print(f"  TCP position  : {np.array2string(tcp_pos, precision=4)}  "
          f"(hand xpos + hand xmat @ [0, 0, 0.103])")
    print(f"  cube position : {np.array2string(cube_pos, precision=4)}")
    print(f"  goal position : {np.array2string(goal_pos, precision=4)}")
    print(f"  home ctrl     : {np.array2string(data.ctrl, precision=3)}  "
          f"(最后一项 255 = 夹爪张开)")
    print(f"  home 时的接触数: {data.ncon}  (期望 0: Panda 与桌子/地面不干涉)")

    rule(f"4. 推进 {SIM_SECONDS}s 仿真（cube 从悬空自然落到桌面）")
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    cube_dadr = int(model.jnt_dofadr[cube_joint])
    n_steps = int(round(SIM_SECONDS / model.opt.timestep))
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    cube_settled = data.qpos[cube_qadr:cube_qadr + 3].copy()
    cube_vel = data.qvel[cube_dadr:cube_dadr + 3].copy()
    cube_bottom = float(cube_settled[2] - CUBE_HALF_SIZE)
    print(f"  cube 最终位置 : {np.array2string(cube_settled, precision=4)}")
    print(f"  cube 最终速度 : {np.array2string(cube_vel, precision=6)}")
    print(f"  cube 底面 z   : {cube_bottom:.4f}  (桌面 {TABLE_TOP_Z:.2f}, "
          f"理论静止中心 z = {TABLE_TOP_Z + CUBE_HALF_SIZE:.4f})")
    print(f"  最终接触数    : {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        print(f"    {name_of(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)} <-> "
              f"{name_of(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}  dist={c.dist:+.5f}")

    rule("5. 自动检查")
    results = {
        "名称全部可查": all_found,
        "qpos/qvel 有限": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "cube 落在桌面上(未掉穿)": cube_bottom > TABLE_TOP_Z - 0.002,
        "cube 静止(线速度 < 1e-3)": bool(np.linalg.norm(cube_vel) < 1e-3),
        "Panda 未爆炸(qpos 在范围内)": bool(
            np.all(np.abs(data.qpos[:7]) < 3.8) and np.all(np.abs(data.qvel[:7]) < 20)
        ),
        "goal 在桌面上方": bool(goal_pos[2] > TABLE_TOP_Z),
    }
    for label, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    if not all(results.values()):
        raise SystemExit(1)
    print("\nScene check passed.")


if __name__ == "__main__":
    main()

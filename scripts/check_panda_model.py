#!/usr/bin/env python3
"""Panda 底层模型结构检查（只读，不控制机器人）。

用法:
    python scripts/check_panda_model.py

设计约束:
- 只用 MuJoCo，不依赖 Gymnasium / CleanRL；
- 不修改官方 MJCF（panda.xml / hand.xml / scene.xml）；
- 所有 joint / actuator / body / geom 均通过 name -> id API 查找，
  不硬编码索引，只把解析出来的索引打印出来。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = PROJECT_ROOT / "assets" / "robots" / "franka_panda" / "scene.xml"

# MuJoCo 枚举 -> 可读名字（用 getattr 兼容不同版本）
JOINT_TYPE_NAMES: dict[int, str] = {}
for _key, _label in (
    ("mjJNT_FREE", "free"),
    ("mjJNT_BALL", "ball"),
    ("mjJNT_SLIDE", "slide"),
    ("mjJNT_HINGE", "hinge"),
):
    _val = getattr(mujoco.mjtJoint, _key, None)
    if _val is not None:
        JOINT_TYPE_NAMES[int(_val)] = _label

TRN_TYPE_NAMES: dict[int, str] = {}
for _key, _label in (
    ("mjTRN_JOINT", "joint"),
    ("mjTRN_JOINTINPARENT", "joint(parent)"),
    ("mjTRN_SLIDERCRANK", "slider-crank"),
    ("mjTRN_TENDON", "tendon"),
    ("mjTRN_SITE", "site"),
    ("mjTRN_BODY", "body"),
):
    _val = getattr(mujoco.mjtTrn, _key, None)
    if _val is not None:
        TRN_TYPE_NAMES.setdefault(int(_val), _label)

# 关注的关键字：用于在 body / geom / site 中寻找夹爪与末端执行器
KEYWORDS = ("hand", "finger", "left", "right", "gripper", "ee", "end", "tip")


def jname(model: mujoco.MjModel, objtype: int, objid: int) -> str:
    name = mujoco.mj_id2name(model, objtype, objid)
    return name if name is not None else f"<unnamed#{objid}>"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_model_overview(model: mujoco.MjModel) -> None:
    section("1. 模型总体信息")
    print(f"XML                : {SCENE_XML}")
    print(f"MuJoCo version     : {mujoco.__version__}")
    names = bytes(model.names)
    model_name = names.split(b"\x00", 1)[0].decode(errors="ignore") if names else ""
    print(f"model name         : {model_name}")
    print(f"nq   (qpos 维数)   : {model.nq}")
    print(f"nv   (qvel 维数)   : {model.nv}")
    print(f"nu   (actuator 数) : {model.nu}")
    print(f"njnt (joint 数)    : {model.njnt}")
    print(f"nbody              : {model.nbody}")
    print(f"ngeom              : {model.ngeom}")
    print(f"nsite              : {model.nsite}")
    print(f"ntendon            : {model.ntendon}")
    print(f"neq   (equality)   : {model.neq}", end="")
    if model.neq > 0:
        eq_types = {int(x) for x in model.eq_type}
        print(f"  types={sorted(eq_types)}", end="")
    print()
    print(f"nkey  (keyframe)   : {model.nkey}")
    print(f"opt.timestep       : {model.opt.timestep}")


def print_joints(model: mujoco.MjModel) -> None:
    section("2. Joint 信息（qpos / qvel 地址映射）")
    header = (
        f"{'ID':>3}  {'name':<14} {'type':<6} {'body':<13} "
        f"{'qposadr':>7} {'dofadr':>6}  {'range':<20} {'limited':>7}"
    )
    print(header)
    print("-" * len(header))
    for jid in range(model.njnt):
        jnt_type = JOINT_TYPE_NAMES.get(int(model.jnt_type[jid]), str(int(model.jnt_type[jid])))
        bid = int(model.jnt_bodyid[jid])
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        limited = bool(model.jnt_limited[jid])
        rng = model.jnt_range[jid]
        rng_str = f"[{rng[0]:+.4f}, {rng[1]:+.4f}]" if limited else "unlimited"
        print(
            f"{jid:>3}  {jname(model, mujoco.mjtObj.mjOBJ_JOINT, jid):<14} {jnt_type:<6} "
            f"{jname(model, mujoco.mjtObj.mjOBJ_BODY, bid):<13} "
            f"{qadr:>7} {dadr:>6}  {rng_str:<20} {str(limited):>7}"
        )

    # 反向确认：每个 qpos 分量属于哪个 joint（free joint 占多份，这里没有 free joint）
    print("\nqpos 分量 -> joint 反向映射:")
    for i in range(model.nq):
        for jid in range(model.njnt):
            qadr = int(model.jnt_qposadr[jid])
            width = 1
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                width = 7
            elif int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_BALL):
                width = 4
            if qadr <= i < qadr + width:
                name = jname(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                extra = "" if width == 1 else f" (分量 {i - qadr}/{width - 1})"
                print(f"  qpos[{i}] <- joint '{name}' (id={jid}){extra}")
                break

    print("\nqvel 分量 -> joint(dof) 反向映射:")
    for i in range(model.nv):
        for jid in range(model.njnt):
            dadr = int(model.jnt_dofadr[jid])
            width = 1
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                width = 6
            elif int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_BALL):
                width = 3
            if dadr <= i < dadr + width:
                name = jname(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                extra = "" if width == 1 else f" (分量 {i - dadr}/{width - 1})"
                print(f"  qvel[{i}] <- joint '{name}' (id={jid}){extra}")
                break


def transmission_desc(model: mujoco.MjModel, aid: int) -> tuple[str, str]:
    """返回 (transmission 描述, 目标对象名)。"""
    trn_type = int(model.actuator_trntype[aid])
    trn_id = int(model.actuator_trnid[aid, 0])
    label = TRN_TYPE_NAMES.get(trn_type, str(trn_type))
    if trn_type in (int(mujoco.mjtTrn.mjTRN_JOINT),):
        return f"{label} -> joint", jname(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)
    if trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
        return f"{label} -> tendon", jname(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)
    return f"{label} -> obj", jname(model, mujoco.mjtObj.mjOBJ_UNKNOWN, trn_id)


def servo_kind(model: mujoco.MjModel, aid: int) -> str:
    """根据 gainprm / biasprm 判断 actuator 的控制类型。

    general actuator 的标量力:
        force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length + biasprm[2]*velocity
    若 biasprm[1] == -gainprm[0] 且 biasprm[2] < 0，即为 PD 位置伺服:
        force = kp*(ctrl - length) - kd*velocity
    """
    gain = model.actuator_gainprm[aid]
    bias = model.actuator_biasprm[aid]
    biastype = int(model.actuator_biastype[aid])
    dyntype = int(model.actuator_dyntype[aid])
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE) if hasattr(mujoco, "mjtBias") else 1
    none_dyn = 0
    if biastype == affine and bias[1] < 0 and bias[2] < 0 and gain[0] != 0:
        kp, kd = -bias[1], -bias[2]
        ratio = gain[0] / kp  # ctrl -> 目标 length 的比例
        if abs(ratio - 1.0) < 1e-9:
            return (f"position servo (PD): kp={kp:g}, kd={kd:g}  "
                    f"[force = kp*(ctrl - length) - kd*velocity]")
        return (f"scaled position servo: kp={kp:g}, kd={kd:g}, ctrl->length 比例={ratio:.9g}  "
                f"[force = kp*(ratio*ctrl - length) - kd*velocity]")
    if dyntype != none_dyn:
        return f"其他 (dyntype={dyntype}, biastype={biastype})"
    return f"其他 (biastype={biastype}, gainprm[0]={gain[0]:g}, biasprm={np.array2string(bias[:3])})"


def print_actuators(model: mujoco.MjModel) -> None:
    section("3. Actuator 信息（data.ctrl 映射）")
    for aid in range(model.nu):
        name = jname(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        trn_desc, target = transmission_desc(model, aid)
        ctrl_limited = bool(model.actuator_ctrllimited[aid])
        ctrlrange = model.actuator_ctrlrange[aid]
        forcelimited = bool(model.actuator_forcelimited[aid])
        forcerange = model.actuator_forcerange[aid]
        gear = model.actuator_gear[aid, 0]
        print(f"--- actuator id={aid}  name='{name}'  (data.ctrl[{aid}]) ---")
        print(f"    transmission      : {trn_desc}  -> '{target}'")
        print(f"    gear              : {gear:g}")
        if ctrl_limited:
            print(f"    ctrlrange         : [{ctrlrange[0]:g}, {ctrlrange[1]:g}]  (limited=True)")
        else:
            print("    ctrlrange         : unlimited")
        if forcelimited:
            print(f"    forcerange        : [{forcerange[0]:g}, {forcerange[1]:g}]")
        else:
            print("    forcerange        : unlimited")
        print(f"    gainprm           : {np.array2string(model.actuator_gainprm[aid, :3], precision=6)}")
        print(f"    biasprm           : {np.array2string(model.actuator_biasprm[aid, :3], precision=6)}")
        print(f"    控制类型          : {servo_kind(model, aid)}")

    print("\n由此得到的 data.ctrl 语义:")
    for aid in range(model.nu):
        name = jname(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        _, target = transmission_desc(model, aid)
        kind = "位置目标" if "position servo" in servo_kind(model, aid) else "控制量"
        print(f"  data.ctrl[{aid}] ({name}) -> {kind} of '{target}'")


def print_gripper_mechanism(model: mujoco.MjModel) -> None:
    section("3b. 夹爪传动机制（tendon / equality）")
    print("Tendon:")
    for tid in range(model.ntendon):
        tname = jname(model, mujoco.mjtObj.mjOBJ_TENDON, tid)
        adr, num = int(model.tendon_adr[tid]), int(model.tendon_num[tid])
        print(f"  tendon id={tid} name='{tname}'  wrap 行数={num}  adr={adr}")
        for w in range(adr, adr + num):
            wt = int(model.wrap_type[w])
            oid = int(model.wrap_objid[w])
            # mjWRAP_JOINT 表示 tendon 直接对某个 joint 的 qpos 求和
            is_joint = wt == int(mujoco.mjtWrap.mjWRAP_JOINT)
            oname = jname(model, mujoco.mjtObj.mjOBJ_JOINT, oid) if is_joint else f"obj#{oid}"
            print(f"    wrap[{w}] type={wt} obj='{oname}' coef={model.wrap_prm[w]:g} "
                  f"(tendon_length = sum(coef * joint_qpos))")
        print("    -> 全部 wrap 都是 joint: 这是 fixed tendon")

    print("\nEquality constraint:")
    eq_joint_type = None
    for eid in range(model.neq):
        eq_type = int(model.eq_type[eid])
        obj1, obj2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
        print(f"  eq id={eid} type={eq_type} "
              f"obj1='{jname(model, mujoco.mjtObj.mjOBJ_JOINT, obj1)}' "
              f"obj2='{jname(model, mujoco.mjtObj.mjOBJ_JOINT, obj2)}' "
              f"data={np.array2string(model.eq_data[eid], precision=4)}")
        if eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
            eq_joint_type = "joint equality（qpos1 == qpos2）"
    if eq_joint_type:
        print(f"  含义: {eq_joint_type} -> 两个 finger joint 的位移被强制相等，因此同步开合")


def print_bodies_sites_geoms(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    section("4. Body / Site / Geom（夹爪与末端相关对象）")
    print("全部 body:")
    for bid in range(model.nbody):
        print(f"  body id={bid:>2} name='{jname(model, mujoco.mjtObj.mjOBJ_BODY, bid)}' "
              f"parent={int(model.body_parentid[bid])}")

    print("\n名字命中关键字的 body:")
    matched_bodies = []
    for bid in range(model.nbody):
        name = jname(model, mujoco.mjtObj.mjOBJ_BODY, bid).lower()
        if any(k in name for k in KEYWORDS):
            matched_bodies.append(bid)
            print(f"  body id={bid} name='{jname(model, mujoco.mjtObj.mjOBJ_BODY, bid)}'")
    if not matched_bodies:
        print("  (无)")

    print(f"\nSite: nsite={model.nsite}")
    if model.nsite == 0:
        print("  当前模型没有任何 site -> 没有现成的 TCP / 末端 reference site。")
    else:
        for sid in range(model.nsite):
            print(f"  site id={sid} name='{jname(model, mujoco.mjtObj.mjOBJ_SITE, sid)}' "
                  f"body='{jname(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[sid]))}' "
                  f"pos={np.array2string(data.site_xpos[sid], precision=4)}")

    print("\n名字命中关键字的 geom:")
    geom_hits = []
    for gid in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
        if any(k in name.lower() for k in KEYWORDS):
            geom_hits.append(gid)
            print(f"  geom id={gid} name='{name}'")
    if not geom_hits:
        print("  (无) —— 本模型的 geom 全部未命名（只有 mesh 名，不是 geom name）。")

    print("\nhand / left_finger / right_finger 三个 body 上的 geom（含无名碰撞 geom）:")
    for bname in ("hand", "left_finger", "right_finger"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            print(f"  body '{bname}' 不存在")
            continue
        adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
        print(f"  body '{bname}' (id={bid})  geom 数={num}")
        for gid in range(adr, adr + num):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            gtype = int(model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                meshid = int(model.geom_dataid[gid])
                meshname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, meshid)
                desc = f"mesh='{meshname}'"
            else:
                desc = (f"size={np.array2string(model.geom_size[gid], precision=4)} "
                        f"pos={np.array2string(model.geom_pos[gid], precision=4)}")
            print(f"    geom id={gid:>2} name={gname if gname else '<unnamed>':<12} "
                  f"type={gtype} {desc}")


def pad_center(model: mujoco.MjModel, data: mujoco.MjData, finger_body: str) -> np.ndarray | None:
    """求某个 finger body 上 fingertip_pad 碰撞 geom 的世界坐标中心。

    fingertip_pad_collision_* 是 box 类型的碰撞 geom，位于 finger body 局部坐标；
    这里取这些 box 中心的平均，作为该手指指尖（pad 中心）的世界位置。
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger_body)
    if bid < 0:
        return None
    adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
    centers = []
    for gid in range(adr, adr + num):
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            centers.append(np.array(data.geom_xpos[gid], dtype=float))
    if not centers:
        return None
    return np.mean(np.vstack(centers), axis=0)


def print_initial_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    section("5. 初始状态与末端位置")
    print("注: MjData 中 qpos 初值来自模型默认值（各 joint 的 ref），不是 keyframe。")
    print("初始 qpos:", np.array2string(data.qpos, precision=5))
    print("初始 qvel:", np.array2string(data.qvel, precision=5))
    print("初始 ctrl:", np.array2string(data.ctrl, precision=5))

    if model.nkey > 0:
        print("\nKeyframe:")
        for kid in range(model.nkey):
            kname = jname(model, mujoco.mjtObj.mjOBJ_KEY, kid)
            print(f"  key id={kid} name='{kname}'")
            print(f"    key_qpos = {np.array2string(model.key_qpos[kid], precision=5)}")
            print(f"    key_ctrl = {np.array2string(model.key_ctrl[kid], precision=5)}")
        print("  -> 可作为 reset / home pose 使用（qpos 与 ctrl 都已给出）。")

    # 用 home keyframe 再算一次末端位置（更接近实际起始构型）
    if model.nkey > 0:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            data.qpos[:] = model.key_qpos[kid]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print("\n以下位置基于 keyframe 'home'（mj_forward 之后）:")

    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand >= 0:
        print(f"  body 'hand'      xpos = {np.array2string(data.xpos[hand], precision=4)}")
        print(f"  body 'hand'      xquat= {np.array2string(data.xquat[hand], precision=4)}")
    for bname in ("left_finger", "right_finger"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            print(f"  body '{bname}' xpos = {np.array2string(data.xpos[bid], precision=4)}")

    left_pad = pad_center(model, data, "left_finger")
    right_pad = pad_center(model, data, "right_finger")
    if left_pad is not None:
        print(f"  left  fingertip pad 中心 = {np.array2string(left_pad, precision=4)}")
    if right_pad is not None:
        print(f"  right fingertip pad 中心 = {np.array2string(right_pad, precision=4)}")
    if left_pad is not None and right_pad is not None:
        tcp = 0.5 * (left_pad + right_pad)
        print(f"  两指 pad 中心的中点(TCP 近似) = {np.array2string(tcp, precision=4)}")
        print(f"  两指间距 = {np.linalg.norm(left_pad - right_pad):.4f} m")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print_model_overview(model)
    print_joints(model)
    print_actuators(model)
    print_gripper_mechanism(model)
    print_bodies_sites_geoms(model, data)
    print_initial_state(model, data)

    section("6. 结论提示")
    print("- 机械臂: joint1..joint7 -> actuator1..actuator7（均为 joint transmission 的位置伺服）")
    print("- 夹爪  : finger_joint1/keyframe 0.04 为张开；由 actuator8 通过 fixed tendon 'split' 驱动，")
    print("          再加 equality(joint) 约束保证两指位移相等")
    print("- End effector: 模型无 site，只能用 body 'hand' 作为参考，")
    print("          或在后续自有场景（不改官方 MJCF）中给 hand 挂一个 TCP site")


if __name__ == "__main__":
    main()

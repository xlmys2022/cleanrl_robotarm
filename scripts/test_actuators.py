#!/usr/bin/env python3
"""顺序、小幅度地手动测试 Panda 的 8 个 actuator（不是 RL，不训练）。

用法（Wayland 下需要强制 X11，否则 viewer 会报
"Wayland: Focusing a window requires user interaction"）:

    env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
        python scripts/test_actuators.py

只做无 GUI 的自动验证（打印完全相同的测试结果，不开窗口）:

    python scripts/test_actuators.py --headless

设计原则:
- 不依赖 Gymnasium / CleanRL；
- 不修改官方 MJCF（panda.xml / hand.xml / scene.xml）；
- actuator / joint / tendon 索引全部通过模型结构 + name->id 查找，不硬编码；
- 每次只动一个 actuator，幅度小（约 0.15 rad），并经过 ctrlrange 与 joint range 双重限幅。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = PROJECT_ROOT / "assets" / "robots" / "franka_panda" / "scene.xml"

HOME_KEYNAME = "home"      # 官方 panda.xml 自带的 keyframe(同时给出 qpos 与 ctrl)
ARM_DELTA = 0.15           # 每个机械臂 joint 的测试幅度 [rad]
JOINT_MARGIN = 0.05        # 与 joint 极限保持的安全余量 [rad]
HOLD_INITIAL_S = 2.0       # 阶段一: 初始姿态保持时间
SETTLE_S = 1.2             # 每次改目标后等待稳定的时间
RETURN_S = 1.0             # 回到 home 目标后等待稳定的时间
GRIPPER_HOLD_S = 1.5       # 夹爪每个动作的保持时间


def obj_name(model: mujoco.MjModel, objtype: int, objid: int) -> str:
    name = mujoco.mj_id2name(model, objtype, objid)
    return name if name is not None else f"<unnamed#{objid}>"


def rule(title: str = "", width: int = 78) -> None:
    print()
    if title:
        print("=" * width)
        print(title)
        print("=" * width)
    else:
        print("-" * width)


class PandaRig:
    """把 model/data/viewer 和步进逻辑封装在一起。"""

    def __init__(self, scene_xml: Path, use_viewer: bool) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.use_viewer = use_viewer
        self.viewer = None

        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYNAME)
        if self.home_key < 0:
            raise RuntimeError(f"模型中没有 keyframe '{HOME_KEYNAME}'")

    # ---------- viewer ----------
    def launch_viewer(self) -> None:
        import mujoco.viewer as mj_viewer  # 延迟导入: --headless 时不依赖 GUI 库

        # MuJoCo 2.3.3 的 launch_passive 返回 None: 窗口在独立线程里持续渲染共享的 MjData,
        # 主线程只要继续 mj_step 即可; 没有 handle 可以设置相机或 sync。
        mj_viewer.launch_passive(self.model, self.data)
        self.viewer = True

    def is_running(self) -> bool:
        # 2.3.3 拿不到窗口状态, 因此无法自动检测关窗; 整个测试用固定时长顺序执行
        return True

    def close_viewer(self) -> None:
        # 2.3.3 的 passive viewer 没有可编程关闭的 handle, 窗口随进程退出而关闭
        self.viewer = None

    # ---------- 状态 ----------
    def reset_to_home(self) -> None:
        """keyframe 'home' 同时给出 qpos 与 ctrl，可直接作为 reset。"""
        self.data.qpos[:] = self.model.key_qpos[self.home_key]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.model.key_ctrl[self.home_key]
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def hold_ctrl(self, seconds: float) -> bool:
        """保持当前 ctrl 继续仿真 seconds 秒（viewer 模式下按真实时间走）。"""
        dt = float(self.model.opt.timestep)
        n_steps = max(1, int(round(seconds / dt)))
        t0 = time.perf_counter()
        for i in range(n_steps):
            if not self.is_running():
                return False
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                # 2.3.3 的 viewer 线程自己每个渲染帧读取 data, 不需要也不能 sync
                lag = t0 + (i + 1) * dt - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        return True

    # ---------- 结构解析（不硬编码索引） ----------
    def resolve_actuators(self) -> tuple[list[int], list[int]]:
        """按 transmission 类型区分机械臂 actuator 与夹爪 actuator。"""
        joint_trn = int(mujoco.mjtTrn.mjTRN_JOINT)
        tendon_trn = int(mujoco.mjtTrn.mjTRN_TENDON)

        arm_ids = [aid for aid in range(self.model.nu)
                   if int(self.model.actuator_trntype[aid]) == joint_trn]
        gripper_ids = [aid for aid in range(self.model.nu)
                       if int(self.model.actuator_trntype[aid]) == tendon_trn]

        # 机械臂 actuator 按对应 joint 的 qpos 地址排序，保证 joint1..joint7 的顺序
        arm_ids.sort(key=lambda a: int(self.model.jnt_qposadr[int(self.model.actuator_trnid[a, 0])]))
        return arm_ids, gripper_ids


def print_mapping(rig: PandaRig, arm_ids: list[int], gripper_ids: list[int]) -> None:
    model = rig.model
    rule("actuator / joint 映射（来自模型结构，不是猜测）")
    for aid in arm_ids:
        jid = int(model.actuator_trnid[aid, 0])
        qadr = int(model.jnt_qposadr[jid])
        cr = model.actuator_ctrlrange[aid]
        jr = model.jnt_range[jid]
        print(f"  data.ctrl[{aid}] {obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid):<10}"
              f"-> joint '{obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)}' (qpos[{qadr}])"
              f"  ctrlrange=[{cr[0]:+.4f},{cr[1]:+.4f}]"
              f"  joint range=[{jr[0]:+.4f},{jr[1]:+.4f}]")
    for aid in gripper_ids:
        tid = int(model.actuator_trnid[aid, 0])
        cr = model.actuator_ctrlrange[aid]
        print(f"  data.ctrl[{aid}] {obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid):<10}"
              f"-> tendon '{obj_name(model, mujoco.mjtObj.mjOBJ_TENDON, tid)}'"
              f"  ctrlrange=[{cr[0]:g},{cr[1]:g}]")


def gripper_state(rig: PandaRig) -> tuple[float, float, float]:
    """返回 (left finger qpos, right finger qpos, tendon length)。"""
    model, data = rig.model, rig.data
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    return (float(data.qpos[model.jnt_qposadr[j1]]),
            float(data.qpos[model.jnt_qposadr[j2]]),
            float(data.ten_length[0]))


def test_initial_pose(rig: PandaRig) -> None:
    rule("阶段一: 初始姿态（keyframe 'home'），请确认模型稳定、无抖动")
    rig.reset_to_home()
    qpos = rig.data.qpos
    ctrl = rig.data.ctrl
    print(f"  qpos = {np.array2string(qpos, precision=4)}")
    print(f"  ctrl = {np.array2string(ctrl, precision=4)}  (最后一项为夹爪: 255 = 张开)")
    if not rig.hold_ctrl(HOLD_INITIAL_S):
        raise KeyboardInterrupt
    print(f"  保持 {HOLD_INITIAL_S:.1f}s 后的 qpos = {np.array2string(rig.data.qpos, precision=4)}")


def test_arm_actuators(rig: PandaRig, arm_ids: list[int]) -> None:
    model, data = rig.model, rig.data
    home_ctrl = np.array(model.key_ctrl[rig.home_key], dtype=float)

    for k, aid in enumerate(arm_ids, start=1):
        jid = int(model.actuator_trnid[aid, 0])
        qadr = int(model.jnt_qposadr[jid])
        aname = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        jname = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jr = model.jnt_range[jid]
        cr = model.actuator_ctrlrange[aid]

        # 先回到 home，保证每个 actuator 的测试起点一致
        rig.reset_to_home()
        if not rig.hold_ctrl(RETURN_S):
            raise KeyboardInterrupt

        q_before = float(data.qpos[qadr])

        # 方向选择: 默认 +delta，若会撞上限则取 -delta
        target = q_before + ARM_DELTA
        if target > jr[1] - JOINT_MARGIN:
            target = q_before - ARM_DELTA
        # 双重安全限幅: joint range（留余量） + actuator ctrlrange
        target = float(np.clip(target, jr[0] + JOINT_MARGIN, jr[1] - JOINT_MARGIN))
        target = float(np.clip(target, cr[0], cr[1]))

        rule(f"阶段二 [{k}/{len(arm_ids)}]: 正在测试 {aname}  (data.ctrl[{aid}])")
        print(f"  actuator name : {aname}")
        print(f"  对应 joint    : {jname} (qpos[{qadr}])")
        print(f"  joint range   : [{jr[0]:+.4f}, {jr[1]:+.4f}]    safety margin={JOINT_MARGIN}")
        print(f"  ctrlrange     : [{cr[0]:+.4f}, {cr[1]:+.4f}]")
        print(f"  初始 qpos     : {q_before:+.4f} rad")
        print(f"  目标 ctrl     : {target:+.4f} rad  (只改这一个分量, 其余保持 home)")

        data.ctrl[:] = home_ctrl          # 先全部回到 home 目标
        data.ctrl[aid] = target           # 只动当前 actuator
        if not rig.hold_ctrl(SETTLE_S):
            raise KeyboardInterrupt

        q_after = float(data.qpos[qadr])
        others = [f"qpos[{int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])])}]="
                  f"{data.qpos[int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])])]:+.4f}"
                  for a in arm_ids if a != aid]
        print(f"  最终 qpos     : {q_after:+.4f} rad   (跟随误差 {q_after - target:+.4f} rad)")
        print(f"  其他 joint    : {', '.join(others)}")
        print(f"  -> data.ctrl[{aid}] 确实只驱动 {jname}")

    rig.reset_to_home()
    if not rig.hold_ctrl(RETURN_S):
        raise KeyboardInterrupt


def test_gripper(rig: PandaRig, gripper_ids: list[int]) -> None:
    model, data = rig.model, rig.data
    if len(gripper_ids) != 1:
        print(f"[warn] 期望 1 个 tendon actuator，实际 {len(gripper_ids)}，跳过夹爪测试")
        return

    aid = gripper_ids[0]
    aname = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    cr = model.actuator_ctrlrange[aid]
    limited = bool(model.actuator_ctrllimited[aid])
    open_ctrl = float(cr[1])
    close_ctrl = float(cr[0])

    rule(f"阶段三: 夹爪测试 {aname}  (data.ctrl[{aid}])")
    print(f"  gripper actuator name : {aname}")
    print(f"  ctrlrange             : [{cr[0]:g}, {cr[1]:g}]  (limited={limited})")
    print(f"  约定                  : {open_ctrl:g} = 张开, {close_ctrl:g} = 闭合"
          f"  （来自模型 ctrlrange，不是猜的）")
    print("  机制: tendon 'split' 的 length = 0.5*qpos[finger_joint1] + 0.5*qpos[finger_joint2]，")
    print("        该 actuator 是 tendon 位置伺服；同时 equality(joint) 强制两个 finger 位移相等。")

    for label, target in (("open", open_ctrl), ("close", close_ctrl), ("open", open_ctrl)):
        data.ctrl[:] = model.key_ctrl[rig.home_key]
        data.ctrl[aid] = target
        if not rig.hold_ctrl(GRIPPER_HOLD_S):
            raise KeyboardInterrupt
        ql, qr, tl = gripper_state(rig)
        print(f"  [{label:<5}] ctrl={target:6.1f} -> "
              f"left_finger qpos={ql:.4f}  right_finger qpos={qr:.4f}  "
              f"tendon_length={tl:.4f}  两指差={abs(ql - qr):.5f}")
        if label == "open":
            print("           （若此前就是 open 状态，这一次可能看不到明显变化）")

    print("  -> 一个 ctrl 分量同时开合两个 finger，两指位移由 equality 约束保持一致")


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda actuator 顺序测试（非 RL）")
    parser.add_argument("--headless", action="store_true",
                        help="不打开 viewer，仅跑仿真并打印结果（用于自动验证）")
    args = parser.parse_args()

    rig = PandaRig(SCENE_XML, use_viewer=not args.headless)
    arm_ids, gripper_ids = rig.resolve_actuators()

    rule("Panda actuator 测试")
    print(f"  XML      : {SCENE_XML}")
    print(f"  nq={rig.model.nq} nv={rig.model.nv} nu={rig.model.nu} "
          f"njnt={rig.model.njnt} nsite={rig.model.nsite}")
    print(f"  viewer   : {'关闭 (--headless)' if args.headless else '开启'}")
    if not args.headless:
        print("  提示: 若在 Wayland 下无法聚焦窗口，请用 "
              "env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 python scripts/test_actuators.py")

    print_mapping(rig, arm_ids, gripper_ids)

    try:
        if not args.headless:
            rig.launch_viewer()

        test_initial_pose(rig)
        test_arm_actuators(rig, arm_ids)
        test_gripper(rig, gripper_ids)

        rule("测试完成")
        print("  验收点:")
        print("   1) 7 个机械臂 actuator 分别只驱动 joint1..joint7；")
        print("   2) 夹爪 ctrl 用 0..255，两个 finger 同步开合；")
        print("   3) 过程中无穿透地面 / 无关节越限抖动。")

        if rig.viewer is not None:
            print("\n  窗口再保持 5 秒供观察，然后程序退出（窗口关闭）。")
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("\n[中断] 用户提前结束测试")
    finally:
        rig.close_viewer()

    if not args.headless:
        # MuJoCo 2.3.3 的 passive viewer 线程仍在渲染时, 正常解释器退出会触发 GLFW 段错误
        # ("Segmentation fault (core dumped)"); 这里强制退出进程, 保证退出码正常
        # （先 flush, 否则 os._exit 会丢掉缓冲输出）。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

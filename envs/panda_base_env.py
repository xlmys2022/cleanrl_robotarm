"""Panda 基础 Gymnasium 环境（本阶段还不是完整 PickPlace 任务）。

职责边界:
- 只负责: 场景 / 8 维连续动作 / 低维状态 observation / reset-step-时序;
- 不做: reward 设计（只有 placeholder）、成功判定、contact 奖励、视觉 observation、课程学习。

observation 布局（共 34 维, 顺序与 _get_obs() 一致）:
    [ 0: 7]  机械臂 7 个 joint 的 qpos              [rad]
    [ 7:14]  机械臂 7 个 joint 的 qvel              [rad/s]
    [14:16]  finger_joint1 / finger_joint2 的 qpos  [m]
    [16:19]  TCP 世界坐标                            [m]
    [19:22]  cube 世界坐标（freejoint 平移部分）      [m]
    [22:25]  cube 线速度（freejoint qvel 前 3 个）    [m/s]
    [25:28]  goal site 世界坐标                      [m]
    [28:31]  TCP - cube                              [m]
    [31:34]  cube - goal                             [m]

action（8 维, [-1, 1]）:
    action[0:7] -> 机械臂 7 个 actuator 的增量位置目标
                   target = 上一次 ctrl + action * MAX_JOINT_DELTA, 再经 ctrlrange/joint range 限幅
    action[7]   -> 夹爪, -1 = 闭合, +1 = 张开, 线性映射到 actuator ctrlrange（0..255）
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .scene_loader import SCENE_XML, load_pick_place_model

# TCP: hand body 局部坐标系下、由 Menagerie Panda fingertip pad 中点估算得到的近似偏移 [m]。
# 官方模型没有 TCP site（nsite = 0）, 本阶段也约定不修改官方 XML, 因此统一在这里做换算。
TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.103], dtype=np.float64)

# 每个 control step 允许的最大关节目标增量 [rad]。0.04 rad @ 50Hz ≈ 2 rad/s 的目标位移速度。
MAX_JOINT_DELTA = 0.04

# MuJoCo 物理步长 0.002 s × 10 = 0.02 s -> 50 Hz 控制频率（运行时以 model.opt.timestep 为准）
DEFAULT_FRAME_SKIP = 10
DEFAULT_MAX_EPISODE_STEPS = 500  # 500 × 0.02 s = 10 s

# cube 固定初始位姿（与 assets/scenes/panda_pick_place.xml 的 keyframe 保持一致）
CUBE_INIT_POS = np.array([0.62, 0.0, 0.46], dtype=np.float64)
CUBE_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w x y z

OBS_DIM = 34


class PandaBaseEnv(gym.Env):
    """Panda + 桌子 + cube + goal 的通用机器人环境（非完整任务）。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_xml: Path | str = SCENE_XML,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        max_joint_delta: float = MAX_JOINT_DELTA,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.model = load_pick_place_model(Path(scene_xml))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.control_dt = float(self.model.opt.timestep * self.frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.max_joint_delta = float(max_joint_delta)
        self.render_mode = render_mode
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(round(1.0 / self.control_dt)),
        }

        self._resolve_model_indices()
        self._resolve_action_limits()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        assert self.model.nu == 8, f"本环境假设 8 个 actuator, 实际 nu={self.model.nu}"

        self._step_count = 0
        self._viewer = None
        self._renderer = None

    # ------------------------------------------------------------------ 索引解析
    def _resolve_model_indices(self) -> None:
        """全部通过 transmission 结构与 name->id 解析, 不假设 actuator/joint 的 id 顺序。"""
        model = self.model

        joint_trn = int(mujoco.mjtTrn.mjTRN_JOINT)
        tendon_trn = int(mujoco.mjtTrn.mjTRN_TENDON)

        arm_ids = [aid for aid in range(model.nu)
                   if int(model.actuator_trntype[aid]) == joint_trn]
        # 按对应 joint 的 qpos 地址排序, 保证 action[0:7] 依次是 joint1..joint7
        arm_ids.sort(key=lambda a: int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])]))
        gripper_ids = [aid for aid in range(model.nu)
                       if int(model.actuator_trntype[aid]) == tendon_trn]

        if len(arm_ids) != 7 or len(gripper_ids) != 1:
            raise RuntimeError(
                f"机械臂 actuator 数={len(arm_ids)}, 夹爪 actuator 数={len(gripper_ids)}, "
                "与预期 (7, 1) 不符"
            )

        self._arm_act_ids = np.array(arm_ids, dtype=np.int32)
        self._gripper_act_id = int(gripper_ids[0])

        self._arm_joint_ids = np.array(
            [int(model.actuator_trnid[aid, 0]) for aid in arm_ids], dtype=np.int32
        )
        self._arm_qpos_adr = np.array(
            [int(model.jnt_qposadr[j]) for j in self._arm_joint_ids], dtype=np.int32
        )
        self._arm_dof_adr = np.array(
            [int(model.jnt_dofadr[j]) for j in self._arm_joint_ids], dtype=np.int32
        )

        self._finger_joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
             mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")],
            dtype=np.int32,
        )
        self._finger_qpos_adr = np.array(
            [int(model.jnt_qposadr[j]) for j in self._finger_joint_ids], dtype=np.int32
        )

        self._hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self._cube_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint"
        )
        self._home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

        for name, oid in (
            ("hand body", self._hand_body_id),
            ("cube body", self._cube_body_id),
            ("goal site", self._goal_site_id),
            ("cube_freejoint", self._cube_joint_id),
            ("keyframe 'home'", self._home_key_id),
        ):
            if oid < 0:
                raise RuntimeError(f"场景里找不到 {name}")

        # cube 是 freejoint: qpos = 3 位置 + 4 四元数, qvel = 3 线速度 + 3 角速度
        self._cube_qpos_adr = int(model.jnt_qposadr[self._cube_joint_id])
        self._cube_dof_adr = int(model.jnt_dofadr[self._cube_joint_id])

    def _resolve_action_limits(self) -> None:
        """增量目标的双重限幅边界: joint range ∩ actuator ctrlrange。"""
        model = self.model
        ctrl = model.actuator_ctrlrange[self._arm_act_ids]
        jnt = model.jnt_range[self._arm_joint_ids]
        self._target_low = np.maximum(ctrl[:, 0], jnt[:, 0]).astype(np.float64)
        self._target_high = np.minimum(ctrl[:, 1], jnt[:, 1]).astype(np.float64)

        g_ctrl = model.actuator_ctrlrange[self._gripper_act_id]
        self._gripper_ctrl_lo = float(g_ctrl[0])
        self._gripper_ctrl_hi = float(g_ctrl[1])

    # ------------------------------------------------------------------ 状态读取
    @property
    def tcp_position(self) -> np.ndarray:
        body_pos = self.data.xpos[self._hand_body_id]
        body_rot = self.data.xmat[self._hand_body_id].reshape(3, 3)
        return body_pos + body_rot @ TCP_LOCAL_OFFSET

    def _get_tcp_position(self) -> np.ndarray:
        return self.tcp_position

    @property
    def cube_position(self) -> np.ndarray:
        a = self._cube_qpos_adr
        return self.data.qpos[a: a + 3].copy()

    @property
    def cube_linear_velocity(self) -> np.ndarray:
        a = self._cube_dof_adr
        return self.data.qvel[a: a + 3].copy()

    @property
    def goal_position(self) -> np.ndarray:
        return self.data.site_xpos[self._goal_site_id].copy()

    def _get_obs(self) -> np.ndarray:
        data = self.data
        tcp = self._get_tcp_position()
        cube_pos = self.cube_position
        cube_vel = self.cube_linear_velocity
        goal_pos = self.goal_position
        obs = np.concatenate([
            data.qpos[self._arm_qpos_adr],       # 7
            data.qvel[self._arm_dof_adr],       # 7
            data.qpos[self._finger_qpos_adr],   # 2
            tcp,                                # 3
            cube_pos,                           # 3
            cube_vel,                           # 3
            goal_pos,                           # 3
            tcp - cube_pos,                     # 3
            cube_pos - goal_pos,                # 3
        ])
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        tcp = self._get_tcp_position()
        cube_pos = self.cube_position
        goal_pos = self.goal_position
        return {
            "step_count": self._step_count,
            "control_dt": self.control_dt,
            "tcp_position": tcp,
            "cube_position": cube_pos,
            "goal_position": goal_pos,
            "tcp_cube_distance": float(np.linalg.norm(tcp - cube_pos)),
            "cube_goal_distance": float(np.linalg.norm(cube_pos - goal_pos)),
        }

    # ------------------------------------------------------------------ Gym 接口
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # keyframe 'home' 同时恢复 qpos 与 ctrl（夹爪 255 = 张开）
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.model.key_ctrl[self._home_key_id]

        # cube 固定初始位姿（与 keyframe 中的 7 维一致, 这里显式再写一次, 后续便于随机化）
        a = self._cube_qpos_adr
        self.data.qpos[a: a + 3] = CUBE_INIT_POS
        self.data.qpos[a + 3: a + 7] = CUBE_INIT_QUAT

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(self.action_space.shape), -1.0, 1.0
        )

        # 机械臂: 增量位置目标, 再经 joint range ∩ ctrlrange 限幅
        target = self.data.ctrl[self._arm_act_ids] + action[:7] * self.max_joint_delta
        self.data.ctrl[self._arm_act_ids] = np.clip(
            target, self._target_low, self._target_high
        )

        # 夹爪: [-1, 1] -> [ctrlrange_lo, ctrlrange_hi], -1 = 闭合, +1 = 张开
        ratio = 0.5 * (action[7] + 1.0)
        self.data.ctrl[self._gripper_act_id] = (
            self._gripper_ctrl_lo + ratio * (self._gripper_ctrl_hi - self._gripper_ctrl_lo)
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        # placeholder reward: 只用来让 step 返回合法数值, 不是 PickPlace 的 reward 设计
        reward = -info["tcp_cube_distance"]

        invalid_state = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        terminated = bool(invalid_state)  # 本阶段没有任务成功条件, 只有非法状态才终止
        truncated = self._step_count >= self.max_episode_steps

        if invalid_state:
            info["invalid_state"] = True
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ 渲染
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="overview_camera")
            return self._renderer.render()

        if self.render_mode == "human":
            # MuJoCo 2.3.3 的 launch_passive 返回 None: GUI 在自己的线程里持续渲染共享的
            # MjData, 主线程继续 mj_step 即可, 既不需要也不能 sync; 也没有 handle 可以用来
            # 设置相机（新版 MuJoCo 的 handle API 才有）。
            if self._viewer is None:
                import mujoco.viewer as mj_viewer

                mj_viewer.launch_passive(self.model, self.data)
                self._viewer = True
            return None

        return None

    def close(self):
        # MuJoCo 2.3.3 的 passive viewer 没有可编程关闭的 handle, 窗口随进程退出而关闭
        self._viewer = None
        if self._renderer is not None:
            # MuJoCo 2.3.x 的 Renderer 没有 close(), 释放引用后由 GC 回收 GL 资源
            self._renderer = None

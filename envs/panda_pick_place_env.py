"""Panda Pick & Place 任务环境（继承 PandaBaseEnv）。

本文件只负责任务层:
    contact 检测 / 阶段状态机 / reward / success / failure / task phase

机器人底层（模型加载、action->ctrl、frame_skip、TCP、cube freejoint、render、reset）
全部沿用 envs/panda_base_env.py 的 PandaBaseEnv, 这里不复制第二套控制代码。

四个 task mode 共用同一个类:
    task="reach"      -> PandaReach-v0
    task="grasp"      -> PandaGrasp-v0
    task="lift"       -> PandaLift-v0
    task="pick_place" -> PandaPickPlace-v0

observation / action 完全沿用父类: 34 维 / 8 维。contact、phase、reward term 只进 info。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .panda_base_env import PandaBaseEnv

# --------------------------------------------------------------------------- 任务阈值
# Reach: TCP 与 cube 中心的距离阈值 [m]。cube 边长 0.045、两指全开间隙约 0.077，
# 0.04 m 大致等于 cube 半对角线量级, 再小会要求 TCP 进入 cube 内部。
REACH_DIST = 0.04

# Lift: 相对 cube 静止高度的抬升量 [m]
LIFT_HEIGHT = 0.05

# Transport: cube 与 goal 的水平(XY)距离阈值 [m]（goal 在桌面上, 不比较 z）
TRANSPORT_NEAR_GOAL_XY = 0.06

# Place: 最终放置判定
PLACE_XY_TOL = 0.04          # [m]
PLACE_HEIGHT_TOL = 0.015     # |cube_z - cube_rest_z| [m]
PLACE_LIN_VEL_TOL = 0.10     # [m/s]
PLACE_ANG_VEL_TOL = 0.5      # [rad/s]
GRIPPER_OPEN_FINGER_POS = 0.03   # 两指平均开度 > 0.03 m 视为已张开（全开 0.04）

# 接触力阈值 [N]: 只有真实产生法向力才算 contact。
# 实测（scripts/check_grasp_contacts.py，真实夹持 cube）:
#   单条 pad<->cube contact 的法向力约 0.05 ~ 0.10 N，单侧 16 条 contact 合计约 1.1 N，
#   contact.dist 约 -0.0005 m；空夹爪闭合时 force = 0。取 0.01 N 作为"有真实接触"的下限。
CONTACT_FORCE_THRESHOLD = 0.01

# success 需要连续保持的 control step 数（50 Hz）
SUCCESS_HOLD_STEPS = {"reach": 5, "grasp": 5, "lift": 5, "pick_place": 10}

# --------------------------------------------------------------------------- grasp quality
# legacy  : PandaGrasp-v0, raw bilateral contact 就算抓住（保留旧语义用于复现）
# strict  : PandaGrasp-v1, 必须是真正的侧面夹持（quality validated）
TASK_VERSION_LEGACY = "legacy_v0"
TASK_VERSION_STRICT = "grasp_quality_v1"
GRASP_QUALITY_MODES = ("legacy", "strict")

# 阈值来自 scripts/check_grasp_quality.py 对 bad RL press / proper side pinch 的实测标定
MAX_FINGER_CUBE_PENETRATION = 0.002   # [m] finger geom 允许的最大穿透深度
MAX_CUBE_TABLE_PENETRATION = 0.003    # [m] cube 被压入桌面的最大深度
MAX_FINGER_TABLE_FORCE = 2.0          # [N] finger<->table 允许的残余接触力

# --------------------------------------------------------------------------- curriculum
# initialization_mode:
#   "default"            : 官方 home keyframe（PandaGrasp-v0 / v1 等）
#   "pregrasp_stage_a"   : Curriculum Stage A, 直接从一个已验证的 open pre-grasp pose 开始
#   "pregrasp_stage_b"   : Stage A + cube XY 小范围随机扰动（其它一切不变）
INITIALIZATION_MODES = ("default", "pregrasp_stage_a", "pregrasp_stage_b")
TASK_VERSION_CURRICULUM_A = "grasp_curriculum_stage_a_v1"
TASK_VERSION_CURRICULUM_B = "grasp_curriculum_stage_b_v1"
TASK_VERSION_CURRICULUM_B_QUALITY = "grasp_curriculum_stage_b_quality_v1"
TASK_VERSION_CURRICULUM_B_QUALITY_V2 = "grasp_curriculum_stage_b_quality_v2"
TASK_VERSION_CURRICULUM_B_CONSTRAINT = "grasp_curriculum_stage_b_table_constraint_v1"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
STAGE_A_CONFIG_PATH = CONFIG_DIR / "panda_grasp_stage_a.json"
STAGE_B_CONFIG_PATH = CONFIG_DIR / "panda_grasp_stage_b.json"

# Failure: cube 掉到桌面以下 (地面附近) / 离开桌面工作区太远
CUBE_FELL_Z = 0.10
WORKSPACE_MARGIN = 0.30

# --------------------------------------------------------------------------- behavior constraint
# "持续高力压桌" 直接判 failure（不是 reward penalty）。默认关闭, 旧环境语义不变。
TABLE_CONTACT_FAILURE_MODES = ("disabled", "sustained_force")
TABLE_CONTACT_FAILURE_DISABLED = "disabled"
TABLE_CONTACT_FAILURE_SUSTAINED = "sustained_force"
TABLE_FORCE_FAILURE_THRESHOLD = 100.0     # [N]
TABLE_FORCE_FAILURE_HOLD_STEPS = 3        # 连续 3 个 control step (60 ms) 才失败
TABLE_CONTACT_FAILURE_REASON = "sustained_finger_table_force"

# --------------------------------------------------------------------------- reward
# 所有 shaping reward 都是 potential difference: reward = w * (phi_t - phi_{t-1})。
# 这样"停在阈值附近"每步只能拿到 ~0, 只有真正改善状态才有奖励, 不能靠拖延刷分。
REWARD_VERSION = "potential_v1"
REWARD_VERSION_QUALITY = "potential_v2_quality"
REWARD_VERSION_QUALITY_STRONG = "potential_v3_quality_table_strong"
REWARD_VERSIONS = (REWARD_VERSION, REWARD_VERSION_QUALITY, REWARD_VERSION_QUALITY_STRONG)

REACH_WEIGHT = 1.0           # w_r
K_REACH = 10.0               # phi_reach = 1 - tanh(k_r * tcp_cube_distance)
GRASP_BONUS = 0.0            # 不再提供"抓住期间每步"的持续奖励（保留 key 兼容 info）
GRASP_EVENT_BONUS = 1.0      # 本 episode 第一次双侧 grasp 时一次性
LIFT_WEIGHT = 2.0            # w_l, phi_lift = clip((cube_z - cube_rest_z) / LIFT_HEIGHT, 0, 1)
K_TRANSPORT = 10.0           # phi_transport = 1 - tanh(k_t * cube_goal_xy_distance)
TRANSPORT_WEIGHT = 3.0       # w_t
SUCCESS_BONUS = 20.0         # 只在 success 真正触发那一步
TIME_PENALTY = 0.01          # 每 control step 恒定的时间惩罚
ACTION_PENALTY = 0.002
SMOOTH_PENALTY = 0.002

# ---- Stage B.5/B.6 quality penalty（只作用于 *_quality reward）----
# finger-table: 饱和惩罚, 避免 300~400 N 的瞬时接触力把 reward scale 打爆
FINGER_TABLE_PENALTY_WEIGHT = 0.05
FINGER_TABLE_FORCE_SCALE = 20.0      # [N]
# cube drift: 只惩罚相对 episode 初始位置的"向外增量", 推回来不给正奖励
CUBE_DRIFT_PENALTY_WEIGHT = 0.10
CUBE_DRIFT_NORM = 0.01               # [m]

# 每个 reward version 的 quality 参数默认值; 环境实例可用 kwargs 覆盖（Stage B.6 = 0.30）
QUALITY_REWARD_DEFAULTS = {
    REWARD_VERSION_QUALITY: {
        "finger_table_weight": FINGER_TABLE_PENALTY_WEIGHT,
        "finger_table_force_scale": FINGER_TABLE_FORCE_SCALE,
        "cube_drift_weight": CUBE_DRIFT_PENALTY_WEIGHT,
        "cube_drift_norm": CUBE_DRIFT_NORM,
    },
    REWARD_VERSION_QUALITY_STRONG: {
        "finger_table_weight": 0.30,
        "finger_table_force_scale": FINGER_TABLE_FORCE_SCALE,
        "cube_drift_weight": CUBE_DRIFT_PENALTY_WEIGHT,
        "cube_drift_norm": CUBE_DRIFT_NORM,
    },
}

REWARD_TERM_KEYS = (
    "reward_reach",
    "reward_grasp",
    "reward_grasp_event",
    "reward_lift",
    "reward_transport",
    "reward_success",
    "reward_time",
    "reward_action",
    "reward_smooth",
    "reward_finger_table",
    "reward_cube_drift",
)

QUALITY_REWARD_TERM_KEYS = ("reward_finger_table", "reward_cube_drift")

TASKS = ("reach", "grasp", "lift", "pick_place")


class PandaPickPlaceEnv(PandaBaseEnv):
    """Panda + table + cube + goal 的 Reach / Grasp / Lift / PickPlace 任务环境。"""

    def __init__(self, task: str = "pick_place", grasp_quality_mode: str = "legacy",
                 initialization_mode: str = "default",
                 stage_b_half_range_m: float | None = None,
                 reward_version: str = REWARD_VERSION,
                 finger_table_penalty_weight: float | None = None,
                 finger_table_force_scale: float | None = None,
                 cube_drift_penalty_weight: float | None = None,
                 cube_drift_norm: float | None = None,
                 table_contact_failure_mode: str = TABLE_CONTACT_FAILURE_DISABLED,
                 table_force_failure_threshold: float = TABLE_FORCE_FAILURE_THRESHOLD,
                 table_force_failure_hold_steps: int = TABLE_FORCE_FAILURE_HOLD_STEPS,
                 **kwargs) -> None:
        if task not in TASKS:
            raise ValueError(f"task 必须是 {TASKS} 之一, 收到 {task!r}")
        if grasp_quality_mode not in GRASP_QUALITY_MODES:
            raise ValueError(f"grasp_quality_mode 必须是 {GRASP_QUALITY_MODES} 之一, "
                             f"收到 {grasp_quality_mode!r}")
        if initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(f"initialization_mode 必须是 {INITIALIZATION_MODES} 之一, "
                             f"收到 {initialization_mode!r}")
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(f"reward_version 必须是 {REWARD_VERSIONS} 之一, "
                             f"收到 {reward_version!r}")
        if table_contact_failure_mode not in TABLE_CONTACT_FAILURE_MODES:
            raise ValueError(f"table_contact_failure_mode 必须是 "
                             f"{TABLE_CONTACT_FAILURE_MODES} 之一, "
                             f"收到 {table_contact_failure_mode!r}")
        if table_force_failure_threshold <= 0.0:
            raise ValueError("table_force_failure_threshold 必须 > 0")
        if int(table_force_failure_hold_steps) < 1:
            raise ValueError("table_force_failure_hold_steps 必须 >= 1")
        super().__init__(**kwargs)
        self.task = task
        self.grasp_quality_mode = grasp_quality_mode
        self.initialization_mode = initialization_mode
        self.reward_version = reward_version
        # quality penalty 参数按版本取默认值, 也可由环境配置显式覆盖（同一套公式, 不复制实现）
        quality_defaults = QUALITY_REWARD_DEFAULTS.get(reward_version, QUALITY_REWARD_DEFAULTS[
            REWARD_VERSION_QUALITY])
        self.finger_table_penalty_weight = float(
            quality_defaults["finger_table_weight"] if finger_table_penalty_weight is None
            else finger_table_penalty_weight
        )
        self.finger_table_force_scale = float(
            quality_defaults["finger_table_force_scale"] if finger_table_force_scale is None
            else finger_table_force_scale
        )
        self.cube_drift_penalty_weight = float(
            quality_defaults["cube_drift_weight"] if cube_drift_penalty_weight is None
            else cube_drift_penalty_weight
        )
        self.cube_drift_norm = float(
            quality_defaults["cube_drift_norm"] if cube_drift_norm is None
            else cube_drift_norm
        )
        if self.finger_table_force_scale <= 0.0 or self.cube_drift_norm <= 0.0:
            raise ValueError("finger_table_force_scale / cube_drift_norm 必须 > 0")
        self.table_contact_failure_mode = table_contact_failure_mode
        self.table_force_failure_threshold = float(table_force_failure_threshold)
        self.table_force_failure_hold_steps = int(table_force_failure_hold_steps)
        self._stage_a = None
        self._stage_b = None
        self._stage_info: dict = {}
        self._last_cube_xy_offset = (0.0, 0.0)
        if initialization_mode == "default":
            self.task_version = (TASK_VERSION_STRICT if grasp_quality_mode == "strict"
                                 else TASK_VERSION_LEGACY)
        elif initialization_mode == "pregrasp_stage_a":
            self.task_version = TASK_VERSION_CURRICULUM_A
            self._stage_a = self._load_json_config(STAGE_A_CONFIG_PATH,
                                                   "python scripts/build_stage_a_pregrasp.py")
        else:   # pregrasp_stage_b
            if table_contact_failure_mode != TABLE_CONTACT_FAILURE_DISABLED:
                self.task_version = TASK_VERSION_CURRICULUM_B_CONSTRAINT
            elif reward_version == REWARD_VERSION_QUALITY:
                self.task_version = TASK_VERSION_CURRICULUM_B_QUALITY
            elif reward_version == REWARD_VERSION_QUALITY_STRONG:
                self.task_version = TASK_VERSION_CURRICULUM_B_QUALITY_V2
            else:
                self.task_version = TASK_VERSION_CURRICULUM_B
            self._stage_a = self._load_json_config(STAGE_A_CONFIG_PATH,
                                                   "python scripts/build_stage_a_pregrasp.py")
            if stage_b_half_range_m is not None:
                half_range = float(stage_b_half_range_m)
            else:
                config = self._load_json_config(
                    STAGE_B_CONFIG_PATH,
                    "python scripts/calibrate_curriculum_stage_b.py",
                )
                half_range = float(config["xy_half_range_m"])
            if half_range <= 0.0:
                raise ValueError(f"stage_b_half_range_m 必须 > 0, 收到 {half_range}")
            self._stage_b = {"xy_half_range_m": half_range}
        if initialization_mode != "default" and grasp_quality_mode != "strict":
            raise ValueError(f"{initialization_mode} 只能和 grasp_quality_mode='strict' 一起使用")

        self._resolve_contact_geometry()
        self._reset_task_state()
        self._init_potentials()

    # ------------------------------------------------------------------ 结构解析
    def _resolve_contact_geometry(self) -> None:
        """用 body/geom name 解析接触检测需要的 id, 不硬编码 geom index。

        - left_finger / right_finger body 下所有 geom（fingertip pad 等碰撞 geom 都没有名字）
        - cube geom: name "cube_geom"
        - table geom: name "table"（用于从几何推出桌面高度, 而不是写死 0.4225）
        """
        model = self.model
        self._cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if self._cube_geom_id < 0 or self._table_geom_id < 0:
            raise RuntimeError("场景里找不到 cube_geom / table geom")

        self._left_finger_geom_ids = self._body_geom_ids("left_finger")
        self._right_finger_geom_ids = self._body_geom_ids("right_finger")

        # cube 半高直接取自 geom size; 桌面高度由 geom 世界位置 + 半厚度推出
        self._cube_half_height = float(model.geom_size[self._cube_geom_id][2])
        mujoco.mj_forward(model, self.data)
        self._table_top_z = float(
            self.data.geom_xpos[self._table_geom_id][2] + model.geom_size[self._table_geom_id][2]
        )
        self._cube_rest_z = self._table_top_z + self._cube_half_height
        self._lift_threshold = self._cube_rest_z + LIFT_HEIGHT

        table_center = self.data.geom_xpos[self._table_geom_id][:2].copy()
        table_half = model.geom_size[self._table_geom_id][:2].copy()
        self._workspace_low = table_center - table_half - WORKSPACE_MARGIN
        self._workspace_high = table_center + table_half + WORKSPACE_MARGIN

    def _body_geom_ids(self, body_name: str) -> set[int]:
        model = self.model
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise RuntimeError(f"场景里找不到 body '{body_name}'")
        adr = int(model.body_geomadr[bid])
        num = int(model.body_geomnum[bid])
        return set(range(adr, adr + num))

    # ------------------------------------------------------------------ 任务状态
    def _reset_task_state(self) -> None:
        self._ever_grasped = False
        self._ever_lifted = False
        self._ever_transport_near_goal = False
        self._success_hold_count = 0
        self._is_grasped = False
        self._is_success = False
        self._failure_reason: str | None = None
        self._phase = "reach"
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._reward_terms = {k: 0.0 for k in REWARD_TERM_KEYS}
        # potential-difference reward 需要"上一时刻的 potential"
        self._prev_pot = {"reach": 0.0, "lift": 0.0, "transport": 0.0}
        # Stage B.5: cube 相对 episode 初始位置的漂移（reset 后重新初始化, 不跨 episode）
        self._initial_cube_xy: np.ndarray | None = None
        self._prev_cube_xy_drift = 0.0
        # Stage B.7: 持续高力压桌计数（只算连续步, reset 归零）
        self._table_high_force_hold_count = 0
        self._table_contact_failure_triggered = False
        self._stage_info = {}

    def _init_potentials(self) -> None:
        """按 reset 后的真实状态初始化 previous potential（不能简单置 0）。"""
        self._prev_pot = self._potentials(self._task_metrics())

    # ------------------------------------------------------------------ curriculum
    @staticmethod
    def _load_json_config(path: Path, build_hint: str) -> dict:
        import json
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path}；请先运行 {build_hint}")
        return json.loads(path.read_text())

    def _apply_stage_a_state(self) -> None:
        """把 robot + cube 放到已验证的 open pre-grasp 状态（不 grasp, 夹爪全开）。"""
        cfg = self._stage_a
        data = self.data
        data.qpos[self._arm_qpos_adr] = cfg["arm_qpos"]
        data.ctrl[self._arm_act_ids] = cfg["arm_ctrl"]
        data.qpos[self._finger_qpos_adr] = cfg["finger_open_position"]
        data.ctrl[self._gripper_act_id] = cfg["gripper_open_ctrl"]
        cube_adr = self._cube_qpos_adr
        data.qpos[cube_adr: cube_adr + 3] = cfg["cube_position"]
        data.qpos[cube_adr + 3: cube_adr + 7] = cfg["cube_quaternion"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        self._last_cube_xy_offset = (0.0, 0.0)
        self._stage_info = {
            "curriculum_stage": "A",
            "initial_cube_xy_offset": [0.0, 0.0],
            "stage_b_half_range_m": None,
            "initial_tcp_cube_relative_position":
                (self._get_tcp_position() - self.cube_position).tolist(),
        }

    def _apply_stage_b_state(self, options: dict | None) -> None:
        """Stage A 的 pose（arm / finger / cube z / yaw 都不变），只给 cube XY 加小扰动。

        随机数必须来自 self.np_random（Gymnasium 的 RNG），保证 reset(seed=...) 可复现;
        测试时可用 options={"cube_xy_offset": [dx, dy]} 精确指定。
        """
        half_range = float(self._stage_b["xy_half_range_m"])
        requested = None if options is None else options.get("cube_xy_offset")
        if requested is None:
            dx, dy = self.np_random.uniform(-half_range, half_range, size=2)
            dx, dy = float(dx), float(dy)
        else:
            dx, dy = float(requested[0]), float(requested[1])
            if abs(dx) > half_range + 1e-9 or abs(dy) > half_range + 1e-9:
                raise ValueError(
                    f"cube_xy_offset=[{dx}, {dy}] 超出 Stage B 半范围 ±{half_range} m"
                )

        cfg = self._stage_a
        data = self.data
        data.qpos[self._arm_qpos_adr] = cfg["arm_qpos"]
        data.ctrl[self._arm_act_ids] = cfg["arm_ctrl"]
        data.qpos[self._finger_qpos_adr] = cfg["finger_open_position"]
        data.ctrl[self._gripper_act_id] = cfg["gripper_open_ctrl"]
        cube_adr = self._cube_qpos_adr
        base_cube = np.asarray(cfg["cube_position"], dtype=np.float64)
        data.qpos[cube_adr: cube_adr + 3] = base_cube + np.array([dx, dy, 0.0])
        data.qpos[cube_adr + 3: cube_adr + 7] = cfg["cube_quaternion"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        self._last_cube_xy_offset = (dx, dy)
        self._stage_info = {
            "curriculum_stage": "B",
            "initial_cube_xy_offset": [dx, dy],
            "stage_b_half_range_m": half_range,
            "initial_tcp_cube_relative_position":
                (self._get_tcp_position() - self.cube_position).tolist(),
        }

    # ------------------------------------------------------------------ 接触检测
    def _read_grasp_contacts(self) -> tuple[float, float]:
        """遍历 data.contact, 返回 (left finger 法向力, right finger 法向力) [N]。

        只有真实产生法向力（> CONTACT_FORCE_THRESHOLD）且 dist < 0 的 contact 才算数;
        夹爪闭合但没有物体时不会产生任何 finger<->cube contact。
        """
        left_force = 0.0
        right_force = 0.0
        force6 = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 == self._cube_geom_id:
                other = g2
            elif g2 == self._cube_geom_id:
                other = g1
            else:
                continue
            if float(contact.dist) > 0.0:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force6)
            normal_force = abs(float(force6[0]))
            if normal_force < CONTACT_FORCE_THRESHOLD:
                continue
            if other in self._left_finger_geom_ids:
                left_force += normal_force
            elif other in self._right_finger_geom_ids:
                right_force += normal_force
        return left_force, right_force

    # ------------------------------------------------------------------ 抓取质量
    def _read_contact_quality(self) -> dict:
        """grasp quality 指标（strict 模式用来区分"真侧夹"和"下压式伪抓取"）。

        全部来自真实 data.contact, 不硬编码 geom index:
            finger<->table  : 数量 / 单侧力 / 总力
            finger<->cube   : 单侧最深 penetration（dist<0 为穿透）
            cube<->table    : 由 cube_rest_z - cube_z 推出被压入深度
            contact normal  : cube->finger 方向, 双侧 dot 值（diagnostic, ≈ -1 才是对夹）
        """
        data = self.data
        left_ids = self._left_finger_geom_ids
        right_ids = self._right_finger_geom_ids
        table_id = self._table_geom_id
        cube_id = self._cube_geom_id

        left_table_force = 0.0
        right_table_force = 0.0
        table_contact_count = 0
        left_pen = 0.0    # <= 0, 负值越大表示穿透越深
        right_pen = 0.0
        left_normals: list[np.ndarray] = []
        right_normals: list[np.ndarray] = []
        force6 = np.zeros(6, dtype=np.float64)

        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            dist = float(contact.dist)

            # ---- finger <-> table ----
            if table_id in (g1, g2):
                other = g2 if g1 == table_id else g1
                if other in left_ids or other in right_ids:
                    mujoco.mj_contactForce(self.model, data, i, force6)
                    normal_force = abs(float(force6[0]))
                    if normal_force >= CONTACT_FORCE_THRESHOLD:
                        table_contact_count += 1
                        if other in left_ids:
                            left_table_force += normal_force
                        else:
                            right_table_force += normal_force

            # ---- finger <-> cube ----
            if cube_id in (g1, g2):
                other = g2 if g1 == cube_id else g1
                if other in left_ids or other in right_ids:
                    if other in left_ids:
                        left_pen = min(left_pen, dist)
                    else:
                        right_pen = min(right_pen, dist)
                    # MuJoCo 法线方向为 geom1 -> geom2; 统一成 cube -> finger
                    normal = np.array(contact.frame[:3], dtype=np.float64)
                    n_cube_to_finger = normal if g1 == cube_id else -normal
                    if other in left_ids:
                        left_normals.append(n_cube_to_finger)
                    else:
                        right_normals.append(n_cube_to_finger)

        normal_opposition = 0.0
        if left_normals and right_normals:
            n_left = np.mean(left_normals, axis=0)
            n_right = np.mean(right_normals, axis=0)
            denom = float(np.linalg.norm(n_left) * np.linalg.norm(n_right))
            if denom > 1e-9:
                normal_opposition = float(np.dot(n_left, n_right) / denom)

        cube_z = float(data.qpos[self._cube_qpos_adr + 2])
        return {
            "left_finger_table_contact": bool(left_table_force > 0.0),
            "right_finger_table_contact": bool(right_table_force > 0.0),
            "finger_table_contact_count": int(table_contact_count),
            "left_finger_table_force": float(left_table_force),
            "right_finger_table_force": float(right_table_force),
            "finger_table_force": float(left_table_force + right_table_force),
            "left_cube_deepest_penetration": float(left_pen),
            "right_cube_deepest_penetration": float(right_pen),
            "max_finger_cube_penetration": float(max(0.0, -min(left_pen, right_pen))),
            "cube_table_penetration": float(max(0.0, self._cube_rest_z - cube_z)),
            "finger_qpos": data.qpos[self._finger_qpos_adr].copy(),
            "normal_opposition": normal_opposition,
        }

    def _evaluate_grasp_quality(self, m: dict) -> tuple[bool, str | None]:
        """strict 质量判定: valid_grasp 与 invalid reason。"""
        reasons: list[str] = []
        if not m["bilateral_contact"]:
            reasons.append("no_bilateral_contact")
        if (m["finger_table_contact_count"] > 0
                and m["finger_table_force"] > MAX_FINGER_TABLE_FORCE):
            reasons.append("finger_table_contact")
        if m["max_finger_cube_penetration"] > MAX_FINGER_CUBE_PENETRATION:
            reasons.append("deep_finger_cube_penetration")
        if m["cube_table_penetration"] > MAX_CUBE_TABLE_PENETRATION:
            reasons.append("cube_pressed_into_table")
        if not reasons:
            return True, None
        return False, (reasons[0] if len(reasons) == 1 else "multiple")

    # ------------------------------------------------------------------ 度量
    def _task_metrics(self) -> dict:
        data = self.data
        tcp = self._get_tcp_position()
        cube = self.cube_position
        goal = self.goal_position
        cube_vel = self.cube_linear_velocity
        cube_ang_vel = data.qvel[self._cube_dof_adr + 3: self._cube_dof_adr + 6]
        left_force, right_force = self._read_grasp_contacts()
        finger_mean = float(np.mean(data.qpos[self._finger_qpos_adr]))
        delta_xy = cube[:2] - goal[:2]
        metrics = {
            "tcp": tcp,
            "cube": cube,
            "goal": goal,
            "tcp_cube_distance": float(np.linalg.norm(tcp - cube)),
            "cube_goal_xy_distance": float(np.linalg.norm(delta_xy)),
            "left_contact_force": float(left_force),
            "right_contact_force": float(right_force),
            "left_contact": bool(left_force > 0.0),
            "right_contact": bool(right_force > 0.0),
            "cube_z": float(cube[2]),
            "cube_height_above_table": float(cube[2] - self._table_top_z),
            "cube_linear_speed": float(np.linalg.norm(cube_vel)),
            "cube_angular_speed": float(np.linalg.norm(cube_ang_vel)),
            "finger_mean_position": finger_mean,
        }
        metrics["bilateral_contact"] = bool(metrics["left_contact"] and metrics["right_contact"])
        metrics.update(self._read_contact_quality())
        valid, reason = self._evaluate_grasp_quality(metrics)
        metrics["grasp_quality_valid"] = valid
        metrics["grasp_invalid_reason"] = reason
        return metrics

    # ------------------------------------------------------------------ reward
    def _potentials(self, m: dict) -> dict:
        """状态势函数 phi(s)，三者都在 [0, 1]，越大表示越接近目标。"""
        return {
            "reach": 1.0 - float(np.tanh(K_REACH * m["tcp_cube_distance"])),
            "lift": float(np.clip((m["cube_z"] - self._cube_rest_z) / LIFT_HEIGHT, 0.0, 1.0)),
            "transport": 1.0 - float(np.tanh(K_TRANSPORT * m["cube_goal_xy_distance"])),
        }

    def _compute_reward_terms(
        self, action: np.ndarray, m: dict, just_grasped: bool, phi: dict
    ) -> dict:
        """potential-difference reward: 每项 = w * (phi_t - phi_{t-1})，按 task 与阶段 gating。

        gating 用的是"更新历史 flag 之前"的状态（见 step()），因此阶段性切换那一步的
        potential 增量不会被吞掉。
        """
        terms = {k: 0.0 for k in REWARD_TERM_KEYS}

        # reach: 只在还没抓住时开启（抓住后 TCP 与 cube 一起动, 这个距离已无意义）
        reach_enabled = not (self.task == "pick_place" and self._ever_grasped)
        if reach_enabled:
            terms["reward_reach"] = REACH_WEIGHT * (phi["reach"] - self._prev_pot["reach"])

        # grasp: 不再提供每步持续奖励, 只保留 episode 内第一次双侧 grasp 的一次性奖励
        allow_grasp = self.task in ("grasp", "lift", "pick_place")
        if allow_grasp and just_grasped:
            terms["reward_grasp_event"] = GRASP_EVENT_BONUS

        # lift: 必须真的抓着, 且还没达到 lift 目标高度
        allow_lift = self.task in ("lift", "pick_place")
        if allow_lift and self._is_grasped and not self._ever_lifted:
            terms["reward_lift"] = LIFT_WEIGHT * (phi["lift"] - self._prev_pot["lift"])

        # transport: 必须真的抬起过并且还抓着
        allow_transport = self.task == "pick_place"
        if allow_transport and self._ever_lifted and self._is_grasped:
            terms["reward_transport"] = TRANSPORT_WEIGHT * (
                phi["transport"] - self._prev_pot["transport"]
            )

        terms["reward_time"] = -TIME_PENALTY
        terms["reward_action"] = -ACTION_PENALTY * float(np.sum(action ** 2))
        terms["reward_smooth"] = -SMOOTH_PENALTY * float(
            np.sum((action - self._previous_action) ** 2)
        )

        # Stage B.5/B.6: 只在 *_quality reward 下生效, 旧版本这两项恒为 0
        if self.reward_version in (REWARD_VERSION_QUALITY, REWARD_VERSION_QUALITY_STRONG):
            terms["reward_finger_table"], terms["reward_cube_drift"] = (
                self._quality_penalties(m)
            )
        return terms

    def _cube_xy_drift(self, m: dict) -> float:
        """cube 相对本 episode 初始 XY 的漂移 [m]（Stage B 的随机 offset 已被基准吸收）。"""
        if self._initial_cube_xy is None or "cube" not in m:
            return 0.0
        return float(np.linalg.norm(m["cube"][:2] - self._initial_cube_xy))

    def _quality_penalties(self, m: dict) -> tuple[float, float]:
        """行为质量惩罚: finger-table 力（饱和）+ cube 向外漂移增量（只罚不奖）。"""
        table_penalty = -self.finger_table_penalty_weight * float(
            np.tanh(float(m["finger_table_force"]) / self.finger_table_force_scale)
        )
        drift_increase = max(0.0, self._cube_xy_drift(m) - self._prev_cube_xy_drift)
        drift_penalty = -self.cube_drift_penalty_weight * (drift_increase / self.cube_drift_norm)
        return table_penalty, drift_penalty

    def _compute_reward_from_metrics(
        self, action: np.ndarray, m: dict, just_grasped: bool = False,
        update_potential: bool = True,
    ) -> dict:
        """按 metrics 计算（并默认更新）potential；单元测试也走这条路径。"""
        phi = self._potentials(m)
        terms = self._compute_reward_terms(action, m, just_grasped, phi)
        if update_potential:
            self._prev_pot = phi
            self._prev_cube_xy_drift = self._cube_xy_drift(m)
        return terms

    # ------------------------------------------------------------------ success
    def _place_condition(self, m: dict) -> bool:
        return (
            m["cube_goal_xy_distance"] < PLACE_XY_TOL
            and abs(m["cube_z"] - self._cube_rest_z) < PLACE_HEIGHT_TOL
            and m["cube_linear_speed"] < PLACE_LIN_VEL_TOL
            and m["cube_angular_speed"] < PLACE_ANG_VEL_TOL
            and m["finger_mean_position"] > GRIPPER_OPEN_FINGER_POS
            and not self._is_grasped
        )

    def _success_condition(self, m: dict) -> bool:
        if self.task == "reach":
            return m["tcp_cube_distance"] < REACH_DIST
        if self.task == "grasp":
            # legacy: raw bilateral; strict: valid_grasp —— 由 self._is_grasped 统一表示
            return self._is_grasped
        if self.task == "lift":
            return self._is_grasped and m["cube_z"] >= self._lift_threshold
        # pick_place: 必须真的抓过、真的抬起过、抓着移动到 goal 附近过, 再稳定放置
        return (
            self._ever_grasped
            and self._ever_lifted
            and self._ever_transport_near_goal
            and self._place_condition(m)
        )

    def _update_phase(self) -> None:
        if self._is_success:
            self._phase = "success"
        elif self._ever_transport_near_goal:
            self._phase = "place"
        elif self._ever_lifted:
            self._phase = "transport"
        elif self._ever_grasped:
            self._phase = "lift"
        elif self._is_grasped:
            self._phase = "grasp"
        else:
            self._phase = "reach"

    def _update_table_contact_failure(self, m: dict) -> None:
        """持续高力压桌 => episode failure（只算连续步; disabled 时永远不触发）。"""
        if self.table_contact_failure_mode == TABLE_CONTACT_FAILURE_DISABLED:
            self._table_high_force_hold_count = 0
            self._table_contact_failure_triggered = False
            return
        if float(m["finger_table_force"]) > self.table_force_failure_threshold:
            self._table_high_force_hold_count += 1
        else:
            self._table_high_force_hold_count = 0
        if self._table_high_force_hold_count >= self.table_force_failure_hold_steps:
            self._table_contact_failure_triggered = True

    def _failure_reason_for(self, m: dict) -> str | None:
        if self.task != "pick_place":
            return None
        if m["cube_z"] < CUBE_FELL_Z:
            return "cube_fell"
        xy = m["cube"][:2]
        if np.any(xy < self._workspace_low) or np.any(xy > self._workspace_high):
            return "cube_out_of_workspace"
        return None

    # ------------------------------------------------------------------ Gym 接口
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._reset_task_state()
        super().reset(seed=seed, options=options)      # 官方 home keyframe + cube 初始位置
        if self._stage_b is not None:
            self._apply_stage_b_state(options)          # Stage B: Stage A pose + cube XY 小扰动
        elif self._stage_a is not None:
            self._apply_stage_a_state()                # Curriculum Stage A: 覆盖成 open pre-grasp
        # cube 漂移基准必须取 reset 之后的真实位置（Stage B 的随机 offset 已生效）
        self._initial_cube_xy = self.cube_position[:2].copy()
        self._prev_cube_xy_drift = 0.0
        # 用 reset 后的真实状态初始化 previous potential, 避免第一步凭空得到大 shaping reward
        self._init_potentials()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(self.action_space.shape), -1.0, 1.0
        )
        # 底层: action -> ctrl -> frame_skip 次 mj_step（父类的 placeholder reward 在这里丢弃）
        obs, _, base_terminated, base_truncated, _ = super().step(action)

        m = self._task_metrics()
        bilateral = bool(m["bilateral_contact"])
        # legacy: raw bilateral 即抓住; strict: 必须是 quality-validated 的侧夹
        accepted = m["grasp_quality_valid"] if self.grasp_quality_mode == "strict" else bilateral
        just_grasped = accepted and not self._ever_grasped
        self._is_grasped = accepted

        # --- reward（gating 使用"更新历史 flag 之前"的状态, 阶段切换那一步的增量不会被吞掉）---
        self._reward_terms = self._compute_reward_from_metrics(action, m, just_grasped)

        # --- 任务状态机（历史 flag 只在对应条件真正成立时才置位）---
        if self._is_grasped:
            self._ever_grasped = True
        if self._is_grasped and m["cube_z"] >= self._lift_threshold:
            self._ever_lifted = True
        if (self._ever_lifted and self._is_grasped
                and m["cube_goal_xy_distance"] < TRANSPORT_NEAR_GOAL_XY):
            self._ever_transport_near_goal = True

        # --- behavior constraint: 持续高力压桌 => failure（优先级高于 success）---
        self._update_table_contact_failure(m)
        if self._table_contact_failure_triggered:
            # 不伪造 ever_grasped / valid_grasp / is_success, 只是终止 episode 且拿不到 +20
            self._success_hold_count = 0
            if self._failure_reason is None:
                self._failure_reason = TABLE_CONTACT_FAILURE_REASON
            terminated = True
        else:
            # --- success (必须连续保持 N 步) ---
            if self._success_condition(m):
                self._success_hold_count += 1
            else:
                self._success_hold_count = 0
            terminated = base_terminated
            if (not self._is_success
                    and self._success_hold_count >= SUCCESS_HOLD_STEPS[self.task]):
                self._is_success = True
                self._reward_terms["reward_success"] = SUCCESS_BONUS
                terminated = True

            # --- failure ---
            if not self._is_success and self._failure_reason is None:
                reason = self._failure_reason_for(m)
                if reason is not None:
                    self._failure_reason = reason
                    terminated = True
            if base_terminated and self._failure_reason is None:
                self._failure_reason = "invalid_state"

        self._update_phase()
        self._previous_action = action

        reward = float(sum(self._reward_terms.values()))
        info = self._get_info()
        return obs, reward, terminated, base_truncated, info

    def _get_info(self) -> dict:
        info = super()._get_info()
        m = self._task_metrics()
        info.update({
            "task": self.task,
            "task_version": self.task_version,
            "grasp_quality_mode": self.grasp_quality_mode,
            "reward_version": self.reward_version,
            "reward_total": float(sum(self._reward_terms.values())),
            "cube_xy_drift": self._cube_xy_drift(m),
            "task_phase": self._phase,
            "tcp_cube_distance": m["tcp_cube_distance"],
            "cube_goal_xy_distance": m["cube_goal_xy_distance"],
            "left_contact": m["left_contact"],
            "right_contact": m["right_contact"],
            "bilateral_contact": m["bilateral_contact"],
            "grasp_quality_valid": m["grasp_quality_valid"],
            "grasp_invalid_reason": m["grasp_invalid_reason"],
            "finger_table_contact_count": m["finger_table_contact_count"],
            "left_finger_table_force": m["left_finger_table_force"],
            "right_finger_table_force": m["right_finger_table_force"],
            "finger_table_force": m["finger_table_force"],
            "max_finger_cube_penetration": m["max_finger_cube_penetration"],
            "left_cube_deepest_penetration": m["left_cube_deepest_penetration"],
            "right_cube_deepest_penetration": m["right_cube_deepest_penetration"],
            "cube_table_penetration": m["cube_table_penetration"],
            "normal_opposition": m["normal_opposition"],
            "finger_joint_qpos": m["finger_qpos"],
            "left_contact_force": m["left_contact_force"],
            "right_contact_force": m["right_contact_force"],
            "is_grasped": self._is_grasped,
            "ever_grasped": self._ever_grasped,
            "ever_lifted": self._ever_lifted,
            "ever_transport_near_goal": self._ever_transport_near_goal,
            "cube_height_above_table": m["cube_height_above_table"],
            "cube_linear_speed": m["cube_linear_speed"],
            "cube_angular_speed": m["cube_angular_speed"],
            "finger_mean_position": m["finger_mean_position"],
            "success_hold_count": self._success_hold_count,
            "is_success": self._is_success,
            "failure_reason": self._failure_reason,
            "table_contact_failure_mode": self.table_contact_failure_mode,
            "table_high_force_hold_count": self._table_high_force_hold_count,
            "table_force_failure_threshold": self.table_force_failure_threshold,
            "table_force_failure_hold_steps": self.table_force_failure_hold_steps,
            "table_contact_failure_triggered": self._table_contact_failure_triggered,
            "cube_rest_z": self._cube_rest_z,
            "lift_threshold": self._lift_threshold,
            **self._stage_info,
            **self._reward_terms,
        })
        return info

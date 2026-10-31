"""myrobot_rl 的 Gymnasium 环境包。

用法（工作目录 = 项目根目录）:
    import gymnasium as gym
    import envs

    env = gym.make("PandaBase-v0")        # 基础环境（无任务）
    env = gym.make("PandaReach-v0")       # 任务: reach
    env = gym.make("PandaGrasp-v0")       # 任务: grasp（legacy raw bilateral contact）
    env = gym.make("PandaGrasp-v1")       # 任务: grasp（strict, quality-validated side grasp）
    env = gym.make("PandaLift-v0")        # 任务: lift
    env = gym.make("PandaPickPlace-v0")   # 任务: pick_place
"""

from gymnasium.envs.registration import register, registry

from .panda_base_env import PandaBaseEnv  # noqa: F401
from .panda_pick_place_env import PandaPickPlaceEnv  # noqa: F401

ENV_ID = "PandaBase-v0"

# 四个任务环境共享同一个 PandaPickPlaceEnv, 只是 task 不同（legacy grasp 语义）
TASK_ENV_IDS = {
    "PandaReach-v0": "reach",
    "PandaGrasp-v0": "grasp",
    "PandaLift-v0": "lift",
    "PandaPickPlace-v0": "pick_place",
}

# 任务语义升级版本: 同一个类, 通过 grasp_quality_mode 选择严格抓取判定
STRICT_ENV_KWARGS = {
    "PandaGrasp-v1": {"task": "grasp", "grasp_quality_mode": "strict"},
}

# Curriculum Stage A: 从已验证的 open pre-grasp pose 开始, episode 缩短到 100 step (2 s)
CURRICULUM_ENV_KWARGS = {
    "PandaGraspCurriculum-A-v0": {
        "task": "grasp",
        "grasp_quality_mode": "strict",
        "initialization_mode": "pregrasp_stage_a",
        "max_episode_steps": 100,
    },
    # Stage A + cube XY 小范围扰动（半范围读 configs/panda_grasp_stage_b.json）
    "PandaGraspCurriculum-B-v0": {
        "task": "grasp",
        "grasp_quality_mode": "strict",
        "initialization_mode": "pregrasp_stage_b",
        "max_episode_steps": 100,
    },
    # Stage B.5: 同一任务/同一 reset 分布, 只多了 behavior quality penalty
    "PandaGraspCurriculum-B-v1": {
        "task": "grasp",
        "grasp_quality_mode": "strict",
        "initialization_mode": "pregrasp_stage_b",
        "max_episode_steps": 100,
        "reward_version": "potential_v2_quality",
    },
    # Stage B.6: 与 B-v1 完全一致, 只把 finger-table 权重 0.05 -> 0.30（一次只改一个变量）
    "PandaGraspCurriculum-B-v2": {
        "task": "grasp",
        "grasp_quality_mode": "strict",
        "initialization_mode": "pregrasp_stage_b",
        "max_episode_steps": 100,
        "reward_version": "potential_v3_quality_table_strong",
        "finger_table_penalty_weight": 0.30,
    },
    # Stage B.7: 退回 potential_v1, 只新增"持续高力压桌 => failure"的硬约束
    "PandaGraspCurriculum-B-v3": {
        "task": "grasp",
        "grasp_quality_mode": "strict",
        "initialization_mode": "pregrasp_stage_b",
        "max_episode_steps": 100,
        "reward_version": "potential_v1",
        "table_contact_failure_mode": "sustained_force",
        "table_force_failure_threshold": 100.0,
        "table_force_failure_hold_steps": 3,
    },
}

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="envs.panda_base_env:PandaBaseEnv",
        # episode 长度由环境内部 step_count 控制并返回 truncated, 不额外套 TimeLimit
        max_episode_steps=None,
    )

for _env_id, _task in TASK_ENV_IDS.items():
    if _env_id not in registry:
        register(
            id=_env_id,
            entry_point="envs.panda_pick_place_env:PandaPickPlaceEnv",
            kwargs={"task": _task},
            max_episode_steps=None,
        )

for _env_id, _kwargs in STRICT_ENV_KWARGS.items():
    if _env_id not in registry:
        register(
            id=_env_id,
            entry_point="envs.panda_pick_place_env:PandaPickPlaceEnv",
            kwargs=_kwargs,
            max_episode_steps=None,
        )

for _env_id, _kwargs in CURRICULUM_ENV_KWARGS.items():
    if _env_id not in registry:
        register(
            id=_env_id,
            entry_point="envs.panda_pick_place_env:PandaPickPlaceEnv",
            kwargs=_kwargs,
            max_episode_steps=None,
        )

__all__ = ["PandaBaseEnv", "PandaPickPlaceEnv", "ENV_ID", "TASK_ENV_IDS",
           "STRICT_ENV_KWARGS", "CURRICULUM_ENV_KWARGS"]

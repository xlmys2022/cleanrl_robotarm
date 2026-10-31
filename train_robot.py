"""对 ppo_continuous_action.py 的最小侵入式训练入口（不修改 PPO、不复制算法）。

用法:
    python train_robot.py --env-id PandaReach-v0 --total-timesteps 50000 --seed 1 --no-cuda

职责:
    1. 在同一进程内先 import envs, 保证 PandaReach-v0 已注册, 再运行原始 PPO;
    2. 训练结束后把"网络权重 + NormalizeObservation 统计"一起保存成 checkpoint,
       推理时 Actor 必须使用同一套观测归一化。
"""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401  (注册 PandaReach-v0 等; 必须在 runpy 之前)
from envs.panda_pick_place_env import (  # noqa: E402
    REWARD_VERSION,
    REWARD_VERSION_QUALITY,
    REWARD_VERSION_QUALITY_STRONG,
    TASK_VERSION_CURRICULUM_A,
    TASK_VERSION_CURRICULUM_B,
    TASK_VERSION_CURRICULUM_B_QUALITY,
    TASK_VERSION_CURRICULUM_B_QUALITY_V2,
    TASK_VERSION_CURRICULUM_B_CONSTRAINT,
)
from ppo_continuous_action import Args  # noqa: E402

# 与 PPO 中 NormalizeObservation 之后的 np.clip(obs, -10, 10) 保持一致
OBS_CLIP = 10.0


def build_ppo_argv(cli: argparse.Namespace) -> list[str]:
    """只覆盖必要参数, 其余超参数完全沿用 PPO 默认值。"""
    argv = [
        "--env-id", cli.env_id,
        "--total-timesteps", str(cli.total_timesteps),
        "--seed", str(cli.seed),
        "--exp-name", cli.exp_name,
        "--no-save-model",      # 保存工作由本入口负责
        "--no-upload-model",
        "--no-capture-video",
        "--cuda" if cli.cuda else "--no-cuda",
    ]
    if cli.init_checkpoint:
        argv += ["--init-checkpoint", str(cli.init_checkpoint)]
    return argv


def find_task_env(vec_env):
    """去掉 gym wrapper 拿到 PandaPickPlaceEnv 本体。"""
    env = vec_env.envs[0]
    while isinstance(env, gym.Wrapper):
        env = env.env
    return env


def find_normalize_observation(vec_env) -> gym.wrappers.NormalizeObservation:
    env = vec_env.envs[0]
    while isinstance(env, gym.Wrapper):
        if isinstance(env, gym.wrappers.NormalizeObservation):
            return env
        env = env.env
    raise RuntimeError("PPO 的 env 链里找不到 NormalizeObservation, 无法保存推理所需统计量")


def find_task_version(vec_env) -> str | None:
    """从环境里读 task_version（PandaGrasp-v1 = grasp_quality_v1），写进 checkpoint metadata。"""
    return getattr(find_task_env(vec_env), "task_version", None)


def save_checkpoint(training: dict, cli: argparse.Namespace, warm_start: dict | None) -> Path:
    vec_env = training["envs"]
    normalizer = find_normalize_observation(vec_env)
    task_env = find_task_env(vec_env)
    task_version = find_task_version(vec_env)

    curriculum_stage = {
        TASK_VERSION_CURRICULUM_A: "A",
        TASK_VERSION_CURRICULUM_B: "B",
        TASK_VERSION_CURRICULUM_B_QUALITY: "B",
        TASK_VERSION_CURRICULUM_B_QUALITY_V2: "B",
        TASK_VERSION_CURRICULUM_B_CONSTRAINT: "B",
    }.get(task_version)
    stage_b_half_range = None
    if curriculum_stage == "B":
        stage_b_half_range = float(task_env._stage_b["xy_half_range_m"])

    reward_version = getattr(task_env, "reward_version", REWARD_VERSION)
    quality_reward = None
    if reward_version in (REWARD_VERSION_QUALITY, REWARD_VERSION_QUALITY_STRONG):
        quality_reward = {
            "finger_table_weight": float(task_env.finger_table_penalty_weight),
            "finger_table_force_scale": float(task_env.finger_table_force_scale),
            "cube_drift_weight": float(task_env.cube_drift_penalty_weight),
        }

    checkpoint = {
        "format_version": 1,
        "env_id": cli.env_id,
        "task_version": task_version,
        "reward_version": reward_version,
        "quality_reward": quality_reward,
        "table_contact_constraint": {
            "enabled": task_env.table_contact_failure_mode != "disabled",
            "mode": task_env.table_contact_failure_mode,
            "force_threshold": float(task_env.table_force_failure_threshold),
            "hold_steps": int(task_env.table_force_failure_hold_steps),
        },
        "seed": int(cli.seed),
        "total_timesteps": int(cli.total_timesteps),
        "global_step": int(training["global_step"]),
        "observation_dim": int(np.prod(vec_env.single_observation_space.shape)),
        "action_dim": int(np.prod(vec_env.single_action_space.shape)),
        "curriculum_stage": curriculum_stage,
        "stage_b_xy_half_range_m": stage_b_half_range,
        "warm_start": warm_start is not None,
        "warm_start_obs_normalization": warm_start is not None,
        "optimizer_resumed": False,
        "parent_checkpoint": None if warm_start is None else warm_start["parent_checkpoint"],
        "parent_env_id": None if warm_start is None else warm_start["parent_env_id"],
        "parent_task_version": None if warm_start is None else warm_start["parent_task_version"],
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in training["agent"].state_dict().items()
        },
        "observation_normalization": {
            "mean": torch.from_numpy(normalizer.obs_rms.mean.copy()),
            "var": torch.from_numpy(normalizer.obs_rms.var.copy()),
            "count": float(normalizer.obs_rms.count),
            "epsilon": float(normalizer.epsilon),
            "clip_low": -OBS_CLIP,
            "clip_high": OBS_CLIP,
        },
        "training_args": asdict(training["args"]),
    }

    output_dir = Path("runs") / training["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{cli.exp_name}.checkpoint.pt"
    temporary_path = model_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(model_path)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda 任务 PPO 训练入口（不修改 PPO）")
    parser.add_argument("--env-id", default="PandaReach-v0")
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-name", default="train_robot")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False,
                        help="默认 CPU（--no-cuda）")
    parser.add_argument("--init-checkpoint", default=None,
                        help="可选: 用已有 checkpoint warm-start（恢复网络权重 + obs 归一化统计, "
                             "NormalizeReward/optimizer 全部重置）")
    cli = parser.parse_args()

    known_env_ids = (*envs.TASK_ENV_IDS, *envs.STRICT_ENV_KWARGS, *envs.CURRICULUM_ENV_KWARGS)
    if cli.env_id not in known_env_ids:
        raise SystemExit(f"未知 env-id: {cli.env_id}, 可选 {list(known_env_ids)}")

    warm_start = None
    if cli.init_checkpoint:
        init_path = Path(cli.init_checkpoint)
        if not init_path.is_file():
            raise SystemExit(f"init-checkpoint 不存在: {init_path}")
        parent = torch.load(init_path, map_location="cpu", weights_only=False)
        warm_start = {
            "parent_checkpoint": str(init_path),
            "parent_env_id": parent.get("env_id"),
            "parent_task_version": parent.get("task_version"),
        }
        print(f"warm-start 来源: {init_path} "
              f"(env_id={warm_start['parent_env_id']}, "
              f"task_version={warm_start['parent_task_version']})", flush=True)

    ppo_path = PROJECT_ROOT / "ppo_continuous_action.py"
    default_args = Args()
    if cli.total_timesteps < default_args.num_steps:
        raise SystemExit("total-timesteps 必须至少覆盖一个完整 rollout (num-steps=2048)")

    original_argv = sys.argv
    try:
        sys.argv = [str(ppo_path), *build_ppo_argv(cli)]
        print(f"运行原始 PPO: {' '.join(sys.argv[1:])}", flush=True)
        training = runpy.run_path(str(ppo_path), run_name="__main__")
    finally:
        sys.argv = original_argv

    model_path = save_checkpoint(training, cli, warm_start)
    env_reward_version = getattr(find_task_env(training["envs"]), "reward_version", REWARD_VERSION)
    print(f"\ncheckpoint 已保存: {model_path}")
    print(f"  包含 model_state_dict + observation_normalization(mean/var/count/epsilon/clip)")
    print(f"  env_id={cli.env_id}, reward_version={env_reward_version}, "
          f"global_step={training['global_step']}")
    if warm_start is not None:
        print(f"  warm_start=True (parent={warm_start['parent_env_id']}), "
              f"optimizer_resumed=False")


if __name__ == "__main__":
    main()

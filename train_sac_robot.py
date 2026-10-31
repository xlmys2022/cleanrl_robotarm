"""SAC 训练入口（最小侵入，不复制 SAC 训练循环）。

用法:
    python train_sac_robot.py --env-id PandaGraspCurriculum-A-v0 \
        --total-timesteps 50000 --seed 1 --no-cuda --experiment-id sac_stage_a_50k

职责:
    1. 先 import envs, 保证 Panda env 已注册;
    2. 用 CleanRL 默认超参构造 SAC CLI（第一轮不调参）;
    3. runpy 运行 sac_continuous_action.py（算法只存在于该文件）;
    4. 保存 SAC checkpoint（actor / twin Q / target Q / alpha，raw observation）。
"""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from envs.panda_pick_place_env import (  # noqa: E402
    REWARD_VERSION,
    REWARD_VERSION_QUALITY,
    REWARD_VERSION_QUALITY_STRONG,
)
from ppo_continuous_action import Args as PPOArgs  # noqa: E402


def find_task_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def build_sac_argv(cli: argparse.Namespace) -> list[str]:
    argv = [
        "--env-id", cli.env_id,
        "--total-timesteps", str(cli.total_timesteps),
        "--seed", str(cli.seed),
        "--exp-name", cli.exp_name,
        "--buffer-size", str(cli.buffer_size),
        "--batch-size", str(cli.batch_size),
        "--learning-starts", str(cli.learning_starts),
        "--gamma", str(cli.gamma),
        "--tau", str(cli.tau),
        "--policy-lr", str(cli.policy_lr),
        "--q-lr", str(cli.q_lr),
        "--alpha", str(cli.alpha),
        "--policy-frequency", str(cli.policy_frequency),
        "--target-network-frequency", str(cli.target_network_frequency),
        "--autotune" if cli.autotune else "--no-autotune",
        "--cuda" if cli.cuda else "--no-cuda",
        "--no-capture-video",
    ]
    return argv


def _to_cpu_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return torch.as_tensor(value)


def _state_dict(module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def save_checkpoint(training: dict, cli: argparse.Namespace) -> Path:
    vec_env = training["envs"]
    task_env = find_task_env(vec_env)
    reward_version = getattr(task_env, "reward_version", REWARD_VERSION)
    quality_reward = None
    if reward_version in (REWARD_VERSION_QUALITY, REWARD_VERSION_QUALITY_STRONG):
        quality_reward = {
            "finger_table_weight": float(task_env.finger_table_penalty_weight),
            "finger_table_force_scale": float(task_env.finger_table_force_scale),
            "cube_drift_weight": float(task_env.cube_drift_penalty_weight),
        }
    log_alpha = training.get("log_alpha")
    checkpoint = {
        "format_version": 2,
        "algorithm": "SAC",
        "experiment_id": cli.experiment_id or cli.exp_name,
        "env_id": cli.env_id,
        "task_version": getattr(task_env, "task_version", None),
        "reward_version": reward_version,
        "quality_reward": quality_reward,
        "seed": int(cli.seed),
        "total_timesteps": int(cli.total_timesteps),
        "global_step": int(training.get("global_step", cli.total_timesteps - 1)),
        "observation_dim": int(np.prod(vec_env.single_observation_space.shape)),
        "action_dim": int(np.prod(vec_env.single_action_space.shape)),
        "observation_preprocessing": "raw",
        "observation_normalization": None,
        "action_distribution": "squashed_gaussian_state_dependent_std",
        "action_clip_required": False,
        "buffer_size": int(training["args"].buffer_size),
        "batch_size": int(training["args"].batch_size),
        "learning_starts": float(training["args"].learning_starts),
        "gamma": float(training["args"].gamma),
        "tau": float(training["args"].tau),
        "policy_lr": float(training["args"].policy_lr),
        "q_lr": float(training["args"].q_lr),
        "policy_frequency": int(training["args"].policy_frequency),
        "target_network_frequency": int(training["args"].target_network_frequency),
        "autotune": bool(training["args"].autotune),
        "alpha_final": float(training["alpha"]),
        "log_alpha": _to_cpu_tensor(log_alpha),
        "actor_state_dict": _state_dict(training["actor"]),
        "qf1_state_dict": _state_dict(training["qf1"]),
        "qf2_state_dict": _state_dict(training["qf2"]),
        "qf1_target_state_dict": _state_dict(training["qf1_target"]),
        "qf2_target_state_dict": _state_dict(training["qf2_target"]),
        "optimizer_resumed": False,
        "replay_buffer_resumed": False,
        "replay_buffer_saved": False,
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
    parser = argparse.ArgumentParser(description="Panda SAC 训练入口（不修改 SAC 算法）")
    parser.add_argument("--env-id", default="PandaGraspCurriculum-A-v0")
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-name", default="train_sac")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    # CleanRL SAC 默认值, 第一轮不调参（只透传, 便于以后独立实验）
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    cli = parser.parse_args()

    known_env_ids = (*envs.TASK_ENV_IDS, *envs.STRICT_ENV_KWARGS, *envs.CURRICULUM_ENV_KWARGS)
    if cli.env_id not in known_env_ids:
        raise SystemExit(f"未知 env-id: {cli.env_id}, 可选 {list(known_env_ids)}")

    sac_path = PROJECT_ROOT / "sac_continuous_action.py"
    default_args = PPOArgs()
    if cli.total_timesteps < default_args.num_steps:
        raise SystemExit("total-timesteps 过短")

    original_argv = sys.argv
    try:
        sys.argv = [str(sac_path), *build_sac_argv(cli)]
        print(f"运行原始 SAC: {' '.join(sys.argv[1:])}", flush=True)
        training = runpy.run_path(str(sac_path), run_name="__main__")
    finally:
        sys.argv = original_argv

    model_path = save_checkpoint(training, cli)
    print(f"\ncheckpoint 已保存: {model_path}")
    print(f"  algorithm=SAC, env_id={cli.env_id}, global_step={training.get('global_step')}, "
          f"alpha_final={training['alpha']:.4f}")
    print(f"  observation_preprocessing=raw, replay_buffer_saved=False, "
          f"optimizer_resumed=False")


if __name__ == "__main__":
    main()

"""评估 Panda 任务 checkpoint（冻结 observation normalization），不训练。

用法:
    # deterministic（Actor 均值, 最终 demo 用的方式）
    python eval_reach.py --model-path runs/<run>/train_robot.checkpoint.pt --episodes 5

    # stochastic（按 Normal(mean, std) 采样, 看策略稳定性）
    python eval_reach.py --model-path ... --stochastic --episodes 50

    # 随机策略 baseline（不需要 checkpoint）
    python eval_reach.py --random --episodes 20

注意:
    Actor 在训练时看到的是 NormalizeObservation + clip(-10, 10) 之后的观测,
    所以推理必须使用 checkpoint 里冻结的 mean/var/epsilon, 且不能更新统计量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from ppo_continuous_action import Agent  # noqa: E402  (只 import 网络定义, 不运行训练)


class FrozenNormalizeObservation(gym.ObservationWrapper):
    """用训练时冻结的统计量做观测归一化, 且 clip 与训练时一致; 绝不更新统计量。"""

    def __init__(self, env, mean, var, epsilon, clip_low, clip_high):
        super().__init__(env)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)
        self.epsilon = float(epsilon)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)

    def observation(self, obs):
        normalized = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )
        return np.clip(normalized, self.clip_low, self.clip_high).astype(np.float32)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "observation_normalization" not in checkpoint:
        raise SystemExit("checkpoint 缺少 model_state_dict 或 observation_normalization")
    return checkpoint


def build_agent(checkpoint: dict) -> Agent:
    dummy_envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -np.inf, np.inf, (checkpoint["observation_dim"],), np.float32
        ),
        single_action_space=gym.spaces.Box(
            -1.0, 1.0, (checkpoint["action_dim"],), np.float32
        ),
    )
    agent = Agent(dummy_envs)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent


def make_env(checkpoint: dict | None, env_id: str, seed: int):
    env = gym.make(env_id)
    if checkpoint is not None:
        norm = checkpoint["observation_normalization"]
        env = FrozenNormalizeObservation(
            env,
            mean=norm["mean"].numpy(),
            var=norm["var"].numpy(),
            epsilon=norm["epsilon"],
            clip_low=norm.get("clip_low", -10.0),
            clip_high=norm.get("clip_high", 10.0),
        )
        env.reset(seed=seed)
    return env


def policy_action(agent: Agent, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            action = agent.actor_mean(x)
        else:
            action, _, _, _ = agent.get_action_and_value(x)
    return np.clip(action.squeeze(0).numpy(), -1.0, 1.0).astype(np.float32)


def evaluate(env, agent: Agent | None, episodes: int, deterministic: bool,
             random_policy: bool, seed: int, verbose: bool = True) -> dict:
    returns, lengths, min_distances, success_steps, successes = [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return, steps, min_dist, success_step = 0.0, 0, float("inf"), None
        done = False
        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action = policy_action(agent, obs, deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            steps += 1
            min_dist = min(min_dist, float(info["tcp_cube_distance"]))
            if terminated and info["is_success"]:
                success_step = steps
            if info["is_success"]:
                success_step = success_step or steps
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(steps)
        min_distances.append(min_dist)
        successes.append(bool(info["is_success"]))
        success_steps.append(success_step)
        if verbose:
            print(f"  episode {ep + 1:3d}: success={str(info['is_success']):<5} "
                  f"steps={steps:3d} return={ep_return:+8.3f} "
                  f"min_dist={min_dist:.4f} final_dist={info['tcp_cube_distance']:.4f} "
                  f"success_step={success_step}")
    env.close()

    succeeded_steps = [s for s in success_steps if s is not None]
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_min_distance": float(np.mean(min_distances)),
        "mean_success_step": float(np.mean(succeeded_steps)) if succeeded_steps else float("nan"),
    }


def print_summary(summary: dict, title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"  success rate      : {summary['success_rate'] * 100:.1f}%")
    print(f"  mean return       : {summary['mean_return']:+.3f}")
    print(f"  mean episode len  : {summary['mean_length']:.1f}")
    print(f"  mean min distance : {summary['mean_min_distance']:.4f} m")
    print(f"  mean success step : {summary['mean_success_step']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Panda checkpoint / 随机 baseline")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--env-id", default="PandaReach-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true", help="按 Normal(mean,std) 采样")
    parser.add_argument("--random", action="store_true", help="随机策略 baseline（不需要模型）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.random:
        env = make_env(None, args.env_id, args.seed)
        summary = evaluate(env, None, args.episodes, True, True, args.seed)
        print_summary(summary, "Random policy baseline")
        return

    if args.model_path is None:
        raise SystemExit("需要 --model-path（或用 --random 跑随机 baseline）")

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    std = agent.actor_logstd.exp().mean().item()
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"checkpoint: {args.model_path}")
    print(f"  env_id={checkpoint['env_id']}  reward_version={checkpoint.get('reward_version')}  "
          f"global_step={checkpoint.get('global_step')}  seed={checkpoint.get('seed')}")
    print(f"  冻结的 obs normalization: mean/var count={checkpoint['observation_normalization']['count']:.0f}, "
          f"epsilon={checkpoint['observation_normalization']['epsilon']}, "
          f"clip=[{checkpoint['observation_normalization']['clip_low']}, "
          f"{checkpoint['observation_normalization']['clip_high']}]")
    print(f"  actor std (mean over dims) = {std:.4f}   mode={mode}")

    env = make_env(checkpoint, args.env_id, args.seed)
    summary = evaluate(env, agent, args.episodes, not args.stochastic, False, args.seed)
    print_summary(summary, f"{args.env_id} {mode} evaluation ({args.episodes} episodes)")


if __name__ == "__main__":
    main()

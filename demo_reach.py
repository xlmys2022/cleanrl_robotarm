"""GUI demo：加载 checkpoint, 用 deterministic Actor 均值跑 PandaReach-v0。

用法（Wayland 下需要强制 X11）:
    env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
    python demo_reach.py --model-path runs/<run>/train_robot.checkpoint.pt

观察:
    Panda 从 home 姿态出发, 逐步把 TCP 靠近桌面上的 cube, 进入 0.04 m 并保持 5 步后打印 SUCCESS。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from eval_reach import build_agent, load_checkpoint, policy_action  # noqa: E402

PRINT_EVERY = 25


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI demo（deterministic Actor）")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-id", default="PandaReach-v0")
    parser.add_argument("--stochastic", action="store_true", help="按 Normal(mean,std) 采样")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model_path)
    agent = build_agent(checkpoint)
    norm = checkpoint["observation_normalization"]
    mean, var = norm["mean"].numpy(), norm["var"].numpy()
    epsilon = float(norm["epsilon"])
    clip_low, clip_high = float(norm["clip_low"]), float(norm["clip_high"])

    import gymnasium as gym

    env = gym.make(args.env_id, render_mode="human")
    unwrapped = env.unwrapped
    control_dt = unwrapped.control_dt

    print(f"model: {args.model_path}")
    print(f"mode : {'stochastic' if args.stochastic else 'deterministic (actor mean)'}")
    if not args.stochastic:
        print(f"提示: 若窗口无法聚焦, 用 env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 运行")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_return, step, success = 0.0, 0, False
        while True:
            obs_norm = np.clip(
                (np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + epsilon),
                clip_low, clip_high,
            ).astype(np.float32)

            t0 = time.perf_counter()
            action = policy_action(agent, obs_norm, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            ep_return += float(reward)
            step += 1

            if step % PRINT_EVERY == 0:
                print(f"  step {step:4d} | tcp_cube={info['tcp_cube_distance']:.4f} | "
                      f"phase={info['task_phase']:<8} | "
                      f"action={np.array2string(action, precision=2, suppress_small=True)}")

            if info["is_success"] and not success:
                success = True
                print(f"\n  SUCCESS at step {step} ({(step * control_dt):.2f} s), "
                      f"final tcp_cube_distance = {info['tcp_cube_distance']:.4f} m\n")

            if terminated or truncated:
                break

            lag = control_dt - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)

        print(f"episode {ep + 1}: success={info['is_success']}  steps={step}  "
              f"return={ep_return:+.3f}  final_dist={info['tcp_cube_distance']:.4f} m")

    env.close()
    # MuJoCo 2.3.3 的 passive viewer 线程仍在渲染, 正常退出会触发 GLFW 段错误
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

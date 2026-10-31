#!/usr/bin/env python3
"""随机动作 rollout：只用来验证 Gymnasium 接口 / 数值稳定性, 不是策略。

用法:
    # 无 GUI 的自动验证
    python scripts/random_rollout.py --headless

    # GUI 观察（Wayland 下需要强制 X11）
    env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
        python scripts/random_rollout.py

说明:
    纯均匀随机动作会让机械臂快速乱动, 所以这里默认把 action 乘以 --action-scale (0.2),
    这是为了安全观察, 而不是任何策略。之后接 PPO 时由训练脚本输出真实动作。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402

import envs  # noqa: E402,F401  (注册 PandaBase-v0)


def run_headless(env_id: str, steps: int, action_scale: float, seed: int) -> None:
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    print(f"observation_space: {env.observation_space}")
    print(f"action_space     : {env.action_space}")
    print(f"obs.shape        : {obs.shape}  dtype={obs.dtype}")

    max_abs_qpos = 0.0
    min_cube_z = np.inf
    n_resets = 0

    for step in range(1, steps + 1):
        action = env.action_space.sample() * action_scale
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.isfinite(obs).all(), f"step {step}: observation 出现 NaN/inf"
        assert np.isfinite(reward), f"step {step}: reward 非有限"
        assert env.observation_space.contains(obs), f"step {step}: observation 越界"

        data = env.unwrapped.data
        max_abs_qpos = max(max_abs_qpos, float(np.abs(data.qpos[:7]).max()))
        min_cube_z = min(min_cube_z, float(info["cube_position"][2]))

        if terminated or truncated:
            n_resets += 1
            obs, info = env.reset()

    tcp_cube = info["tcp_cube_distance"]
    cube_goal = info["cube_goal_distance"]
    print(f"\n完成 {steps} 个 control step, resets={n_resets}")
    print(f"  max |arm qpos|   = {max_abs_qpos:.3f} rad")
    print(f"  cube 最低 z      = {min_cube_z:.4f} m  (桌面 0.40, 不允许掉穿)")
    print(f"  最终 TCP-cube    = {tcp_cube:.4f} m")
    print(f"  最终 cube-goal   = {cube_goal:.4f} m")
    print(f"  step_count       = {info['step_count']}  control_dt={info['control_dt']:.3f}s")

    if min_cube_z < 0.35:
        raise SystemExit("cube 掉穿桌面")
    if max_abs_qpos > 3.8:
        raise SystemExit("机械臂关节越界, 疑似爆炸")
    env.close()
    print(f"\nRandom rollout passed: {steps} steps")


def run_gui(env_id: str, steps: int, action_scale: float, seed: int) -> None:
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset(seed=seed)
    print(f"GUI 模式: 观察 {steps} 步随机动作 (action_scale={action_scale})")
    print("提示: 若 Wayland 下窗口无法聚焦, 用 "
          "env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 重新运行")

    control_dt = info["control_dt"]
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        # 机械臂 7 维动作放小, 只让夹爪（第 8 维）保留较大范围, 方便观察开合
        action = env.action_space.sample()
        action[:7] *= action_scale
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if step % 100 == 0:
            print(f"  step {step:4d} | TCP-cube={info['tcp_cube_distance']:.3f} "
                  f"| cube={np.array2string(info['cube_position'], precision=3)} "
                  f"| reward(placeholder)={reward:+.3f}")

        if terminated or truncated:
            print(f"  episode 结束于 step {step} (terminated={terminated}, truncated={truncated})")
            obs, info = env.reset()

        # 保持接近真实时间, 便于肉眼观察
        lag = control_dt - (time.perf_counter() - t0)
        if lag > 0:
            time.sleep(lag)

    env.close()
    print("\nGUI rollout 结束（只是接口验证, 动作是随机的, 不代表任何策略）")
    # MuJoCo 2.3.3 的 passive viewer 线程仍在渲染时, 正常解释器退出会触发 GLFW 段错误,
    # 这里强制退出进程, 保证退出码正常（先 flush, 否则 os._exit 会丢掉缓冲输出）。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="PandaBase-v0 随机 rollout（接口验证）")
    parser.add_argument("--headless", action="store_true", help="不开 viewer, 只跑自动验证")
    parser.add_argument("--steps", type=int, default=1000, help="control step 数")
    parser.add_argument("--action-scale", type=float, default=0.2,
                        help="随机动作缩放系数（安全观察用）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.headless:
        run_headless(envs.ENV_ID, args.steps, args.action_scale, args.seed)
    else:
        run_gui(envs.ENV_ID, args.steps, args.action_scale, args.seed)


if __name__ == "__main__":
    main()

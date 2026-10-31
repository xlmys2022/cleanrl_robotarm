#!/usr/bin/env python3
"""SAC ↔ Panda 环境集成检查（不训练）。

用法:
    python scripts/check_sac_integration.py

覆盖:
    Test 1  Environment interface（A-v0 / B-v2: obs 34, action 8, [-1,1]）
    Test 2  Actor bounds（1000 obs -> -1 < a < 1）
    Test 3  Finite logprob / mean / log_std
    Test 4  State-dependent log_std 路径存在（fc_logstd 有效 + 对 state 有梯度）
    Test 5  Twin Q 输出 shape / finite
    Test 6  一次 synthetic backward（Q / actor / alpha loss 无 NaN）
    Test 7  Panda 真实 random rollout 1000 steps（无 NaN / 无越界）
    Test 8  Raw 34D observation per-dim 统计（无 NaN/Inf, 尺度检查）
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs  # noqa: E402,F401
from sac_continuous_action import Actor, SoftQNetwork  # noqa: E402

RESULTS: list[tuple[str, bool]] = []


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check(label: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    RESULTS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  [{detail}]" if detail else ""))


def spaces_of(env_id: str):
    env = gym.make(env_id)
    spaces = SimpleNamespace(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    return env, spaces


def test_1_interface() -> None:
    rule("Test 1: Environment interface")
    for env_id in ("PandaGraspCurriculum-A-v0", "PandaGraspCurriculum-B-v2"):
        env, _ = spaces_of(env_id)
        print(f"  {env_id}: obs={env.observation_space.shape} "
              f"action={env.action_space.shape} "
              f"low={env.action_space.low[0]} high={env.action_space.high[0]}")
        check(f"{env_id} obs shape == (34,)", env.observation_space.shape == (34,))
        check(f"{env_id} action shape == (8,)", env.action_space.shape == (8,))
        check(f"{env_id} action bounds == [-1, 1]",
              np.allclose(env.action_space.low, -1.0) and np.allclose(env.action_space.high, 1.0))
        env.close()


def collect_random_rollout(env_id: str, steps: int = 1000):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    observations = [np.asarray(obs, dtype=np.float64)]
    actions = []
    rewards = []
    infos = []
    bad = False
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(np.asarray(action))
        rewards.append(float(reward))
        infos.append(info)
        observations.append(np.asarray(obs, dtype=np.float64))
        if not np.all(np.isfinite(obs)) or np.any(np.abs(action) > 1.0 + 1e-9):
            bad = True
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.array(observations[:steps]), np.array(actions), np.array(rewards), infos, bad


def test_2_3_4_5(observations: np.ndarray, actions_np: np.ndarray) -> None:
    env, spaces = spaces_of("PandaGraspCurriculum-B-v2")
    env.close()
    torch.manual_seed(0)
    actor = Actor(spaces)
    qf1 = SoftQNetwork(spaces)
    qf2 = SoftQNetwork(spaces)

    rule("Test 2: Actor bounds")
    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(obs_t), 250):
            a, _, _ = actor.get_action(obs_t[i:i + 250])
            if i == 0:
                all_a = [a]
            else:
                all_a.append(a)
        all_a = torch.cat(all_a)
    print(f"  min action = {all_a.min().item():+.6f}  max action = {all_a.max().item():+.6f}")
    check("1000 obs 的 action 全部有限", bool(torch.isfinite(all_a).all()))
    check("action 严格在 (-1, 1)", bool((all_a > -1.0).all() and (all_a < 1.0).all()))

    rule("Test 3: Finite logprob / mean / log_std")
    with torch.no_grad():
        a, log_prob, mean = actor.get_action(obs_t)
        mu, log_std = actor(obs_t)
    print(f"  action finite={bool(torch.isfinite(a).all())} "
          f"log_prob finite={bool(torch.isfinite(log_prob).all())} "
          f"mean finite={bool(torch.isfinite(mean).all())} "
          f"log_std finite={bool(torch.isfinite(log_std).all())}")
    print(f"  log_std range = [{log_std.min().item():.3f}, {log_std.max().item():.3f}]  "
          f"mean={log_std.mean().item():.3f}")
    check("log_prob finite", bool(torch.isfinite(log_prob).all()))
    check("mean / log_std finite",
          bool(torch.isfinite(mean).all()) and bool(torch.isfinite(log_std).all()))

    rule("Test 4: State-dependent log_std")
    w_norm = float(actor.fc_logstd.weight.norm())
    obs_a = obs_t[:1].clone().requires_grad_(True)
    _, ls_a = actor(obs_a)
    ls_a.sum().backward()
    grad_norm = float(obs_a.grad.norm())
    obs_b = obs_t[100:101]
    with torch.no_grad():
        _, ls_b = actor(obs_b)
    diff = float((ls_a - ls_b).abs().max())
    print(f"  fc_logstd.weight.norm = {w_norm:.4f}, d(log_std)/d(obs) norm = {grad_norm:.6f}")
    print(f"  |log_std(obs_a) - log_std(obs_b)|max = {diff:.6f}")
    check("fc_logstd 是逐 state 的 Linear 路径", w_norm > 0.0)
    check("log_std 对 observation 有非零梯度", grad_norm > 0.0)

    rule("Test 5: Twin Q networks")
    a_t = torch.as_tensor(actions_np, dtype=torch.float32)
    with torch.no_grad():
        q1 = qf1(obs_t, a_t)
        q2 = qf2(obs_t, a_t)
    print(f"  Q1 shape={tuple(q1.shape)} Q2 shape={tuple(q2.shape)} "
          f"Q1 mean={q1.mean().item():+.4f}")
    check("Q1/Q2 shape == (N, 1)", q1.shape == (len(obs_t), 1) and q2.shape == (len(obs_t), 1))
    check("Q1/Q2 finite", bool(torch.isfinite(q1).all()) and bool(torch.isfinite(q2).all()))


def test_6_backward(observations: np.ndarray, actions_np: np.ndarray) -> None:
    rule("Test 6: one synthetic backward step")
    env, spaces = spaces_of("PandaGraspCurriculum-B-v2")
    env.close()
    torch.manual_seed(1)
    actor = Actor(spaces)
    qf1 = SoftQNetwork(spaces)
    qf2 = SoftQNetwork(spaces)
    qf1_target = SoftQNetwork(spaces)
    qf2_target = SoftQNetwork(spaces)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha = log_alpha.exp().item()
    target_entropy = -float(spaces.single_action_space.shape[0])
    q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-3)
    a_opt = torch.optim.Adam(list(actor.parameters()), lr=3e-4)
    alpha_opt = torch.optim.Adam([log_alpha], lr=1e-3)

    n = 64
    obs = torch.as_tensor(observations[:n], dtype=torch.float32)
    next_obs = torch.as_tensor(observations[1:n + 1], dtype=torch.float32)
    actions = torch.as_tensor(actions_np[:n], dtype=torch.float32)
    rewards = torch.full((n, 1), 0.5)
    dones = torch.zeros((n, 1))

    with torch.no_grad():
        next_a, next_logp, _ = actor.get_action(next_obs)
        min_q_next = torch.min(qf1_target(next_obs, next_a), qf2_target(next_obs, next_a))
        target_q = rewards + 0.99 * (1 - dones) * (min_q_next - alpha * next_logp)
    qf1_loss = F.mse_loss(qf1(obs, actions), target_q)
    qf2_loss = F.mse_loss(qf2(obs, actions), target_q)
    q_loss = qf1_loss + qf2_loss
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()

    pi, log_pi, _ = actor.get_action(obs)
    min_q_pi = torch.min(qf1(obs, pi), qf2(obs, pi))
    actor_loss = (alpha * log_pi - min_q_pi).mean()
    a_opt.zero_grad()
    actor_loss.backward()
    a_opt.step()

    _, log_pi_alpha, _ = actor.get_action(obs)
    alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()
    alpha_new = log_alpha.exp().item()

    print(f"  qf1_loss={qf1_loss.item():.4f} qf2_loss={qf2_loss.item():.4f} "
          f"actor_loss={actor_loss.item():+.4f} alpha_loss={alpha_loss.item():+.4f} "
          f"alpha {alpha:.4f}->{alpha_new:.4f}")
    check("Q / actor / alpha loss 全部 finite",
          all(np.isfinite(x) for x in (qf1_loss.item(), qf2_loss.item(),
                                       actor_loss.item(), alpha_loss.item(), alpha_new)))
    check("alpha 保持 finite 且 > 0", np.isfinite(alpha_new) and alpha_new > 0.0)


def test_7_rollout(observations: np.ndarray, actions_np: np.ndarray, rewards: np.ndarray,
                   bad: bool) -> None:
    rule("Test 7: Panda random rollout 1000 steps")
    print(f"  steps={len(actions_np)}  reward mean={rewards.mean():+.4f}  "
          f"obs finite={bool(np.isfinite(observations).all())}")
    check("rollout 无 NaN/Inf", not bad)
    check("action 始终在 [-1, 1]", bool(np.all(np.abs(actions_np) <= 1.0 + 1e-9)))
    check("reward 全部 finite", bool(np.all(np.isfinite(rewards))))


def test_8_observation_stats(observations: np.ndarray) -> None:
    rule("Test 8: raw 34D observation 统计")
    obs = observations[:1000]
    mins = obs.min(axis=0)
    maxs = obs.max(axis=0)
    means = obs.mean(axis=0)
    stds = obs.std(axis=0)
    print(f"  {'dim':>3} | {'min':>10} | {'max':>10} | {'mean':>10} | {'std':>9}")
    for d in range(obs.shape[1]):
        print(f"  {d:>3} | {mins[d]:>+10.4f} | {maxs[d]:>+10.4f} | "
              f"{means[d]:>+10.4f} | {stds[d]:>9.4f}")
    absmax = float(np.abs(obs).max())
    print(f"  global |obs|max = {absmax:.4f}")
    check("observation 全部 finite", bool(np.isfinite(obs).all()))
    check("observation 尺度正常 (|obs|max < 1e4)", absmax < 1e4, f"{absmax:.4f}")


def main() -> None:
    test_1_interface()
    observations, actions_np, rewards, infos, bad = collect_random_rollout(
        "PandaGraspCurriculum-B-v2", steps=1000)
    test_2_3_4_5(observations, actions_np)
    test_6_backward(observations, actions_np)
    test_7_rollout(observations, actions_np, rewards, bad)
    test_8_observation_stats(observations)

    rule("结果")
    failed = [label for label, ok in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for label in failed:
            print(f"  FAILED: {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()

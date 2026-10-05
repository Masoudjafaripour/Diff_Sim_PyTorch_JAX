"""
JAX vs PyTorch: speed of an RL training loop.

Same task, same algorithm, same network in both frameworks:
  - Env:    CartPole (gym dynamics), written as batched array code, auto-reset
  - Agent:  REINFORCE, MLP 4 -> 64 -> 64 -> 2 (tanh), Adam
  - Loop:   each iteration = rollout T steps in N parallel envs -> returns -> one gradient step

Only the execution model differs:
  - PyTorch: eager mode, Python loop over time steps (the usual way to write it)
  - JAX:     whole iteration (rollout via lax.scan + loss + grad + Adam) is one jit-compiled function

Metric: env steps / second (N * T * iters / wall time), measured after warm-up.
JAX compile time is reported separately so it doesn't hide inside the steady-state number.

Usage:
  python rl_acc.py --backend jax   --device gpu --num-envs 1024    # one run
  python rl_acc.py --backend torch --device cpu --num-envs 1024
  python rl_acc.py --sweep                                        # full grid -> CSV + plot
"""

import argparse
import json
import os
import subprocess
import sys
import time

# CartPole constants (same as gym CartPole-v1)
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02
X_LIMIT = 2.4
THETA_LIMIT = 12 * 2 * 3.141592653589793 / 360
MAX_EP_STEPS = 500  # CartPole-v1 time limit

HIDDEN = 64
LR = 1e-3
GAMMA = 0.99


# ----------------------------------------------------------------------------- JAX
def run_jax(args):
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    N, T = args.num_envs, args.horizon

    def init_params(key):
        sizes = [4, HIDDEN, HIDDEN, 2]
        params = []
        for k, (i, o) in zip(jax.random.split(key, 3), zip(sizes[:-1], sizes[1:])):
            params.append((jax.random.normal(k, (i, o)) * jnp.sqrt(1.0 / i), jnp.zeros(o)))
        return params

    def policy(params, obs):
        h = obs
        for w, b in params[:-1]:
            h = jnp.tanh(h @ w + b)
        w, b = params[-1]
        return h @ w + b

    def env_reset(key):
        return jax.random.uniform(key, (N, 4), minval=-0.05, maxval=0.05)

    def env_step(state, action):
        x, x_dot, th, th_dot = state.T
        force = jnp.where(action == 1, FORCE_MAG, -FORCE_MAG)
        cos, sin = jnp.cos(th), jnp.sin(th)
        temp = (force + POLEMASS_LENGTH * th_dot**2 * sin) / TOTAL_MASS
        th_acc = (GRAVITY * sin - cos * temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * cos**2 / TOTAL_MASS))
        x_acc = temp - POLEMASS_LENGTH * th_acc * cos / TOTAL_MASS
        x, x_dot = x + TAU * x_dot, x_dot + TAU * x_acc
        th, th_dot = th + TAU * th_dot, th_dot + TAU * th_acc
        done = (jnp.abs(x) > X_LIMIT) | (jnp.abs(th) > THETA_LIMIT)
        return jnp.stack([x, x_dot, th, th_dot], axis=1), done

    def rollout(params, state, ep_t, key):
        def step(carry, _):
            state, ep_t, key = carry
            key, k_act, k_reset = jax.random.split(key, 3)
            action = jax.random.categorical(k_act, policy(params, state))
            nxt, done = env_step(state, action)
            ep_t = ep_t + 1
            done = done | (ep_t >= MAX_EP_STEPS)
            nxt = jnp.where(done[:, None], env_reset(k_reset), nxt)
            ep_t = jnp.where(done, 0, ep_t)
            return (nxt, ep_t, key), (state, action, done)

        (state, ep_t, key), traj = jax.lax.scan(step, (state, ep_t, key), None, length=T)
        return state, ep_t, key, traj

    def discounted_returns(dones):
        # reward is +1 every step; done_t cuts the bootstrap from t+1
        def back(G, d):
            G = 1.0 + GAMMA * G * (1.0 - d)
            return G, G
        _, R = jax.lax.scan(back, jnp.zeros(N), dones.astype(jnp.float32), reverse=True)
        return R

    def loss_fn(params, obs, actions, R):
        logp = jax.nn.log_softmax(policy(params, obs))
        logp_a = jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]
        adv = (R - R.mean()) / (R.std() + 1e-8)
        return -(logp_a * adv).mean()

    def adam(params, grads, m, v, t, b1=0.9, b2=0.999, eps=1e-8):
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, v, grads)
        def upd(p, m, v):
            return p - LR * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)
        return jax.tree_util.tree_map(upd, params, m, v), m, v

    @jax.jit
    def train_iter(params, m, v, t, state, ep_t, key):
        state, ep_t, key, (obs, actions, dones) = rollout(params, state, ep_t, key)
        R = discounted_returns(dones)
        grads = jax.grad(loss_fn)(params, obs, actions, R)
        params, m, v = adam(params, grads, m, v, t)
        avg_ep_len = (N * T) / jnp.maximum(dones.sum(), 1)
        return params, m, v, state, ep_t, key, avg_ep_len

    key = jax.random.PRNGKey(args.seed)
    key, k_p, k_s = jax.random.split(key, 3)
    params = init_params(k_p)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = env_reset(k_s)
    ep_t = jnp.zeros(N, dtype=jnp.int32)

    # first call = trace + XLA compile + one iteration
    t0 = time.perf_counter()
    params, m, v, state, ep_t, key, ep_len = train_iter(params, m, v, 1, state, ep_t, key)
    ep_len.block_until_ready()
    compile_s = time.perf_counter() - t0

    for i in range(args.warmup):
        params, m, v, state, ep_t, key, ep_len = train_iter(params, m, v, i + 2, state, ep_t, key)
    ep_len.block_until_ready()

    t0 = time.perf_counter()
    for i in range(args.iters):
        params, m, v, state, ep_t, key, ep_len = train_iter(params, m, v, args.warmup + i + 2, state, ep_t, key)
    ep_len.block_until_ready()
    wall = time.perf_counter() - t0

    return dict(compile_s=compile_s, wall_s=wall, final_ep_len=float(ep_len),
                device=str(jax.devices()[0]))


# ----------------------------------------------------------------------------- PyTorch
def run_torch(args):
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    N, T = args.num_envs, args.horizon

    policy = nn.Sequential(
        nn.Linear(4, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, 2),
    ).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    def env_reset():
        return torch.empty(N, 4, device=device).uniform_(-0.05, 0.05)

    def env_step(state, action):
        x, x_dot, th, th_dot = state.unbind(1)
        force = torch.where(action == 1, FORCE_MAG, -FORCE_MAG)
        cos, sin = torch.cos(th), torch.sin(th)
        temp = (force + POLEMASS_LENGTH * th_dot**2 * sin) / TOTAL_MASS
        th_acc = (GRAVITY * sin - cos * temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * cos**2 / TOTAL_MASS))
        x_acc = temp - POLEMASS_LENGTH * th_acc * cos / TOTAL_MASS
        x, x_dot = x + TAU * x_dot, x_dot + TAU * x_acc
        th, th_dot = th + TAU * th_dot, th_dot + TAU * th_acc
        done = (x.abs() > X_LIMIT) | (th.abs() > THETA_LIMIT)
        return torch.stack([x, x_dot, th, th_dot], dim=1), done

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    state = env_reset()
    ep_t = torch.zeros(N, dtype=torch.int32, device=device)

    def train_iter():
        nonlocal state, ep_t
        obs_buf, act_buf, done_buf = [], [], []
        with torch.no_grad():
            for _ in range(T):
                action = torch.distributions.Categorical(logits=policy(state)).sample()
                nxt, done = env_step(state, action)
                ep_t = ep_t + 1
                done = done | (ep_t >= MAX_EP_STEPS)
                nxt = torch.where(done[:, None], env_reset(), nxt)
                ep_t = torch.where(done, 0, ep_t)
                obs_buf.append(state); act_buf.append(action); done_buf.append(done)
                state = nxt
        obs, actions = torch.stack(obs_buf), torch.stack(act_buf)
        dones = torch.stack(done_buf).float()

        R = torch.empty(T, N, device=device)
        G = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            G = 1.0 + GAMMA * G * (1.0 - dones[t])
            R[t] = G

        logp = torch.log_softmax(policy(obs), dim=-1)
        logp_a = logp.gather(-1, actions[..., None]).squeeze(-1)
        adv = (R - R.mean()) / (R.std() + 1e-8)
        loss = -(logp_a * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return (N * T) / dones.sum().clamp(min=1)

    t0 = time.perf_counter()
    ep_len = train_iter()
    sync()
    first_s = time.perf_counter() - t0

    for _ in range(args.warmup):
        ep_len = train_iter()
    sync()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        ep_len = train_iter()
    sync()
    wall = time.perf_counter() - t0

    return dict(compile_s=first_s, wall_s=wall, final_ep_len=float(ep_len),
                device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu")


# ----------------------------------------------------------------------------- driver
def single_run(args):
    res = run_jax(args) if args.backend == "jax" else run_torch(args)
    steps = args.num_envs * args.horizon * args.iters
    res.update(backend=args.backend, target=args.device, num_envs=args.num_envs,
               horizon=args.horizon, iters=args.iters, steps_per_s=steps / res["wall_s"])
    return res


def sweep(args):
    import csv

    combos = [(b, d) for b in ("torch", "jax") for d in ("cpu", "gpu")]
    rows = []
    for n in args.env_grid:
        for backend, device in combos:
            cmd = [sys.executable, os.path.abspath(__file__), "--backend", backend, "--device", device,
                   "--num-envs", str(n), "--horizon", str(args.horizon),
                   "--iters", str(max(args.min_iters, args.steps_budget // (n * args.horizon))),
                   "--warmup", str(args.warmup), "--cpu-threads", str(args.cpu_threads), "--json"]
            # each run in its own process: clean JAX platform choice, no cross-framework GPU contention
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"[skip] {backend}-{device} N={n}: {out.stderr.strip().splitlines()[-1]}")
                continue
            r = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(r)
            print(f"{backend:>5}-{device:<3} N={n:>6}  {r['steps_per_s']:>14,.0f} steps/s   "
                  f"first-iter {r['compile_s']:6.2f}s   ep_len {r['final_ep_len']:6.1f}")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {csv_path}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for backend, device in combos:
        pts = [(r["num_envs"], r["steps_per_s"]) for r in rows
               if r["backend"] == backend and r["target"] == device]
        if pts:
            xs, ys = zip(*pts)
            plt.plot(xs, ys, "o-" if backend == "jax" else "s--", label=f"{backend} ({device})")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("parallel envs N")
    plt.ylabel("env steps / second (training, incl. update)")
    plt.title(f"REINFORCE on CartPole, horizon T={args.horizon}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(args.out_dir, "steps_per_sec.png")
    plt.savefig(png_path, dpi=150)
    print(f"saved {png_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["jax", "torch"], default="jax")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="print result as one JSON line")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4, help="torch intra-op threads on CPU")
    p.add_argument("--env-grid", type=int, nargs="+", default=[1, 16, 256, 4096, 65536])
    p.add_argument("--steps-budget", type=int, default=2_000_000, help="sweep: env steps per run")
    p.add_argument("--min-iters", type=int, default=5, help="sweep: lower bound on iters per run")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()

    if args.sweep:
        sweep(args)
    else:
        res = single_run(args)
        print(json.dumps(res) if args.json else json.dumps(res, indent=2))

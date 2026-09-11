# Phase 2 convergence experiments

Question: do EMPO's learning-based Phase 2 solvers (DQN-style trainer, tabular
lookup-table trainer, PPO path) converge to the exact backward-induction solution of
eqs. (4) to (9), and under which settings?

Ground truth comes from `src/empo/backward_induction/phase2.py` on worlds small enough
to enumerate. Learned solutions are scored on **all states** of the DAG by:

- action agreement between argmax pi_r (learned) and argmax pi_r (exact),
- KL(exact pi_r || learned pi_r), mean / median / max,
- V_r error, and where available V_h^e and X_h error.

Every comparison must use the **same world YAML and the same theory parameters**
(beta_h, beta_r, gamma_h, gamma_r, zeta, xi, eta) for the exact solver and the
learner. The demo scripts do not guarantee this by default: the DQN demo anneals to
`beta_r = 10` by default while `phase2_ppo_tabular_reward.py` solves with
`beta_r = 100`, and the demo's "trivial" world is built in-script rather than from
`multigrid_worlds/trivial.yaml`.

## Local setup (macOS, CPU)

    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv/bin/python -r setup/requirements.txt
    export PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds

Needs the macOS fix on branch `fix/macos-statvfs-f-type` (statvfs `f_type` is
Linux-only; the exact solver crashed without it).

## Runs so far (2026-09-11, Apple M-series laptop, CPU)

### Exact solver, `multigrid_worlds/trivial.yaml`

    .venv/bin/python examples/phase2/phase2_backward_induction.py --rollouts 5

243 states, 5 cell goals, 6 timesteps. beta_h = 10, beta_r = 100 (script defaults).
Exact pi_r at the initial state: forward 1.000. Wall clock about 1 s.

### PPO with exact tabular U_r, `trivial.yaml`, 100 iterations (script default)

    .venv/bin/python examples/phase2/phase2_ppo_tabular_reward.py --world trivial.yaml --iters 100

This isolates the PPO policy pipeline from the auxiliary networks: rewards are the
exact U_r(s) from backward induction.

| metric | value |
|---|---|
| states compared | 892 |
| action agreement with exact pi_r | 97 / 892 (10.9%) |
| KL(exact ‖ PPO) mean / median / max | 0.438 / 0.115 / 1.776 |
| V_r(root): PPO vs exact (raw) | -51.9 vs -26.6 |
| PPO V_r range over all states | [-52.8, -50.1] (target range [-4.3, 0.0]) |

Reading: after 100 iterations the PPO critic has collapsed to a near-constant value
and the policy is close to uniform (root pi_r = 0.22 / 0.27 / 0.35 / 0.16 vs exact
0.24 / 0.12 / 0.12 / 0.52). Even with exact rewards the PPO path has not reached the
fixed point at this budget. A 500-iteration run is in progress.

### DQN-style trainer, demo "trivial" world, quick mode (1,000 training steps), seed 42

    .venv/bin/python examples/phase2/phase2_robot_policy_demo.py --quick --tabular --seed 42 --save_networks tabular_s42.pt
    .venv/bin/python examples/phase2/phase2_robot_policy_demo.py --quick --seed 42 --save_networks neural_s42.pt

| variant | final V_r(s0) | final v_h_e loss | final q_r loss | wall clock |
|---|---|---|---|---|
| tabular lookup tables | -7.35 | 0.043 | 32.7 | 11 s |
| neural | -14.38 | 0.034 | 36.8 | 48 s |

Not yet comparable to an exact value: the demo world and beta_r differ from the
exact run above. The two variants disagree with each other by a factor of two on
V_r(s0), which by itself says at least one has not converged in 1,000 steps.

## Next

1. `docs/convergence_study/compare_to_exact.py`: load saved Phase 2 networks (or tabular tables), run the
   exact solver on the same world with the same parameters, and report the metrics
   above over all DAG states. Then sweep seeds x steps x
   {one_step, n_step, episode} x {direct, mcts} x warm-up schedules.
2. Same for the PPO path with learned (not tabular) U_r.
3. Move from `trivial.yaml` to `basic/*.yaml` and the bushworld ensembles.

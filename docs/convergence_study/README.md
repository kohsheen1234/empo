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
fixed point at this budget.

### Same, 500 iterations

    .venv/bin/python examples/phase2/phase2_ppo_tabular_reward.py --world trivial.yaml --iters 500

| metric | 100 iters | 500 iters |
|---|---|---|
| action agreement with exact pi_r | 10.9% | 11.4% |
| KL(exact ‖ PPO) mean / median / max | 0.438 / 0.115 / 1.776 | 0.461 / 0.139 / 2.445 |
| root pi_r (still / left / right / forward) | 0.22 / 0.27 / 0.35 / 0.16 | 0.07 / 0.06 / 0.12 / 0.75 |
| PPO V_r range over all states | [-52.8, -50.1] | [-48.2, -45.4] |

Reading: five times more iterations moves the **root** policy onto the exact argmax
(forward), but agreement across all states does not improve and KL gets slightly
worse, so the policy is not converging to the exact fixed point state by state.
Two caveats for interpreting this diagnostic:

- Agreement of 11% is below the 25% a uniform policy would score over 4 actions,
  which suggests a systematic disagreement (for instance the exact policy preferring
  `still` in many states where PPO prefers movement) rather than noise. Worth
  breaking down by exact argmax action.
- The critic's near-constant value about -47 is roughly U_r / (1 - gamma_r) for
  U_r ~ -0.5 and gamma_r = 0.99, i.e. an infinite-horizon value, while the script's
  "clamped target" is the finite-horizon backward-DP value on a 10-step episode
  (range [-4.3, 0]). The V_r comparison therefore mixes two definitions; the policy
  metrics are the ones to trust here.

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

### Bushworld, exact vs learned lookup tables (`examples/bushworld/bushworld_compare.py`)

    .venv/bin/python examples/bushworld/bushworld_compare.py --method lookup --seed 0 --no-movie

Default 7x3 world, 2 humans + 1 robot, 132 states; beta_r = 5, gamma = 0.95,
zeta = xi = eta = 1 (harness defaults); 600 lookup-table training steps (10 s).

| metric | value |
|---|---|
| states compared | 132 |
| argmax agreement, learned vs exact pi_r | 24.2% |
| RMSE of action probabilities | 0.072 |
| final V_r(s0), learned | -14.76 |
| rollouts, exact policy | bushes cleared 1.67, human travel 4.00 |
| rollouts, learned policy | bushes cleared 2.67, human travel 3.67 |

### Bushworld, exact vs learned neural networks

    .venv/bin/python examples/bushworld/bushworld_compare.py --method neural --seed 0 --no-movie

Same world and parameters; 3,000 neural training steps (63 s learner time).

| metric | lookup, 600 steps | neural, 3,000 steps |
|---|---|---|
| argmax agreement with exact pi_r | 24.2% | 31.1% |
| RMSE of action probabilities | 0.072 | 0.056 |
| final V_r(s0), learned | -14.76 | -125.43 |
| rollouts (exact: 1.67 bushes, 4.00 travel) | 2.67 bushes, 3.67 travel | 1.67 bushes, 4.00 travel |

Reading: this harness *is* set up correctly (same world, same human prior, same
parameters), so the gaps here are genuine non-convergence rather than specification
mismatches. Three observations:

- Both learners are far from the exact policy state by state (agreement 24 to 31%,
  near chance for 4 to 5 actions), even though the neural policy's rollouts already
  match the exact policy's behaviour on the 3 evaluation rollouts. Behavioural
  agreement at the start state is a much weaker test than all-states agreement.
- The two learners disagree with each other on V_r(s0) by a factor of about 8
  (-14.8 vs -125.4) with identical theory parameters. At least one value scale is
  wrong; this is the same value-scale instability the repo's open evaluation flags.
  With gamma_r = 0.95 the infinite-horizon value of a constant U_r = -1 is -20, so
  -125 is not a plausible fixed-point value for this world.
- RMSE on probabilities is small partly because beta_r = 5 keeps pi_r soft; argmax
  agreement is the stricter metric and should be the primary one.

A 6,000-step lookup run is in progress to separate "needs more steps" from "does
not converge".

## Why the multigrid demo cannot converge to backward induction as configured

Reading the code (`examples/phase2/phase2_robot_policy_demo.py`,
`src/empo/learning_based/phase2/config.py`, `src/empo/backward_induction/phase2.py`)
shows three settings under which the learner is solving a *different* problem from
the exact solver, independent of training budget:

1. **`include_step_count=False`** (demo line ~1341). Backward-induction states are
   time-indexed (`state[0]`), and the finite-horizon V_r and V_h^e depend on the
   remaining time. With the flag off, the neural encoder drops the step count and
   the lookup tables strip `state[0]` from their keys, so the learner cannot even
   represent the exact solution. Must be `True` for any exact comparison.
2. **Different human policy.** The exact solver uses the Phase 1
   `TabularHumanPolicyPrior` (Boltzmann in beta_h, computed by backward induction);
   the demo trains against `HeuristicPotentialPolicy(beta=1000)`, a shortest-path
   heuristic. Different pi_h means different eqs. (4) to (9). Pass the Phase 1
   prior as `human_policy_prior=` to the trainer.
3. **X_h clamping asymmetry.** Backward induction feeds unclamped X_h into eq. (8);
   the learner clamps X_h to [1e-3, 1] (and floors it by the goal sampler's
   smallest weight). Either compare against a clamped target (there is a helper,
   `compute_clamped_target_vr` in `examples/phase2/phase2_ppo_tabular_reward.py`) or
   derive the exact U_r with the same clamp.

Smaller traps: beta_r is 0 (uniform robot) throughout warm-up and ramps with a
sigmoid afterwards, so learned pi_r must be evaluated with the exact solver's
beta_r after the ramp saturates; the exact V_h^e is stored as float16 with zero
entries omitted (error floor about 1e-3); goal sets differ between the demo's
`DEFINED_GOALS` (5 goals), the `TrivialGoalGenerator` (4) and `trivial.yaml` (6),
and the three code paths aggregate X_h with different goal weights.

`examples/bushworld/bushworld_compare.py` is the one harness in the repo that
already trains a `Phase2Config` learner and scores it against backward induction
(`compare_policies` -> argmax agreement and RMSE of action probabilities over all
states), so it is the starting point for the sweep rather than a new script.

## Next

1. `docs/convergence_study/compare_to_exact.py`: load saved Phase 2 networks (or tabular tables), run the
   exact solver on the same world with the same parameters, and report the metrics
   above over all DAG states. Then sweep seeds x steps x
   {one_step, n_step, episode} x {direct, mcts} x warm-up schedules.
2. Same for the PPO path with learned (not tabular) U_r.
3. Move from `trivial.yaml` to `basic/*.yaml` and the bushworld ensembles.

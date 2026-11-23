# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

- Python: `3.13.18`
- Gymnasium: `1.4.0`

## Part A — Paper calculations

My handwritten calculations are available below.

- [Page 1](photo_2026-11-24_01-03-29.jpg)
- [Page 2](photo_2026-11-24_01-03-47.jpg)
- [Page 3](photo_2026-11-24_01-04-04.jpg)

## Part B — FrozenLake

### Fixed policy, written before the experiments

The following policy was defined in the code before running the experiments.
State numbers run from 0 to 15, left to right and top to bottom.

| Row | Column 1 | Column 2 | Column 3 | Column 4 |
| --- | --- | --- | --- | --- |
| 1 | 0 Down | 1 Right | 2 Down | 3 Left |
| 2 | 4 Down | 5 Hole | 6 Down | 7 Hole |
| 3 | 8 Right | 9 Down | 10 Down | 11 Hole |
| 4 | 12 Hole | 13 Right | 14 Right | 15 Goal |

Without slipping, the policy follows states 0, 4, 8, 9, 13, 14,
and 15. This route avoids holes and reaches the goal in six steps.
Actions in the other nonterminal states lead toward this route
when the intended move occurs.

### Experiment settings

Both policies were tested with is_slippery=False and is_slippery=True.
Each condition used three batches of 50 episodes, with batch seeds
11, 22, and 33. Episode i used seed 1000 * batch_seed + i, for i=0,...,49.
The same episode seeds were used in all four conditions.
The random action generator was also seeded with the episode seed.

Each episode stopped on termination, truncation, or the 100-step limit.
Goals and holes were identified from the final state.
Terminal outcomes took priority if termination happened at step 100.
The environment used the default FrozenLake rewards and slip probabilities.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | fixed | 11 | 50 | 0 | 0 | 100.0 | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100.0 | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100.0 | 6.00 |
| False | random | 11 | 0 | 50 | 0 | 0.0 | 8.14 |
| False | random | 22 | 1 | 49 | 0 | 2.0 | 7.56 |
| False | random | 33 | 0 | 50 | 0 | 0.0 | 6.88 |
| True | fixed | 11 | 1 | 49 | 0 | 2.0 | 4.24 |
| True | fixed | 22 | 2 | 48 | 0 | 4.0 | 5.48 |
| True | fixed | 33 | 1 | 49 | 0 | 2.0 | 4.44 |
| True | random | 11 | 0 | 50 | 0 | 0.0 | 8.06 |
| True | random | 22 | 0 | 50 | 0 | 0.0 | 7.60 |
| True | random | 33 | 0 | 50 | 0 | 0.0 | 7.98 |

Without slipping, the fixed policy reached the goal in all 150 episodes
and always took six steps. With slipping, it reached the goal in 4 of
150 episodes. Its success rate varied from 2% to 4% across batches.
For example, at state 4 the intended action Down leads to state 8,
but a slip to the right leads into hole 5.

The smaller mean step counts for the slippery fixed policy reflect
early failures. Random actions reached the goal once without slipping
and zero times with slipping. These rare successes provide limited
evidence about how slipping affected the random policy.
No time limits were observed in any condition.

The three batches show variation in outcomes. One successful episode
only demonstrates that reaching the goal is possible. Repeated episodes
are needed to assess how consistently a policy succeeds.

These experiments evaluate unchanged policies. The fixed policy uses
the same action table throughout, and the random policy samples uniformly
from four actions. Neither policy learns from rewards or updates its
decisions during the experiments.

### Saved results and traces

- [Episode results](outputs/part_b_episodes.csv)
- [Batch summary](outputs/part_b_summary.csv)
- [Successful episode](outputs/part_b_success.txt)
- [Failed episode](outputs/part_b_failure.txt)

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.


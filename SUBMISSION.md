# HW6 submission

## AI-use declaration

I used ChatGPT to help with calculations, Python code, and report wording.
I ran the code in Google Colab and checked the outputs.

## Setup and reproduction

The analysis was run in Google Colab on Linux.

- Python 3.13.18
- Gymnasium 1.4.0
- Part C uses the Python standard library.

Installed environment packages are recorded below.

```text
cloudpickle==3.1.3
Farama-Notifications==0.0.6
gymnasium==1.4.0
numpy==2.5.5
typing_extensions==4.16.0
```

Run the following shell commands with Python 3.13.18 available.

```bash
cd /content
git clone https://github.com/aiziya-sagyntay/css4007-hw6.git
cd css4007-hw6
python3 -m pip install virtualenv
python3 -m virtualenv /content/hw6-env
/content/hw6-env/bin/python -m pip install gymnasium==1.4.0
/content/hw6-env/bin/python part_b_frozenlake.py
/content/hw6-env/bin/python part_c_solver.py
```

In Colab, run the shell commands in a code cell beginning with `%%bash`.
The programs save their results in `outputs/`.

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

The solver uses synchronous value iteration, starting from V0 = 0.
Each update uses the previous iteration's values throughout the round.
The code reproduces the full V1 map from Part A.

| Row | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | -1 | -1 | 10 | 0 (G) |
| 2 | -1 | # | -1 | 0 (H) |
| 3 | -1 | -1 | -1 | -1 |

| Cell | Paper V2 | Code V2 | Best action |
| --- | --- | --- | --- |
| (1,2) | 8.00 | 8.00 | Right |
| (1,3) | 10.00 | 10.00 | Right |
| (2,3) | 5.21 | 5.21 | Up |

The values match the handwritten calculations. There are no numerical
discrepancies requiring correction. All four V2 action values for these
cells are saved in [the detailed results](outputs/part_c_results.txt).

### Values, policies and convergence

The discount factor is 0.9. Iteration stops when the largest absolute
change in a full update is below 0.000001. The safeguard is 10000 updates.
Both maps converged before reaching the safeguard.

| Map | Iterations | Final maximum change |
| --- | --- | --- |
| Original | 12 | 2.16529310482e-07 |
| Extended | 11 | 2.40588123201e-07 |

The policy is extracted using the final values. For tied action values,
the first action in Up, Right, Down, Left order is selected, using an
absolute tolerance of 1e-10.

#### Original value map

| Row | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | . 4.200000 | M 8.000000 | . 10.000000 | G 0.000000 |
| 2 | . 2.780000 | # | I 5.824176 | H 0.000000 |
| 3 | S 1.535824 | . 2.817582 | . 4.241758 | . 2.817582 |

#### Original policy map

| Row | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | . → | M → | . → | G |
| 2 | . ↑ | # | I ↑ | H |
| 3 | S → | . → | . ↑ | . ← |

#### Extended value map

| Row | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | . 4.200000 | M 8.000000 | . 10.000000 | G 0.000000 |
| 2 | . 2.780000 | # | I 5.824176 | H 0.000000 |
| 3 | S 1.502000 | D 2.817582 | . 4.241758 | W 2.817582 |

#### Extended policy map

| Row | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | . → | M → | . → | G |
| 2 | . ↑ | # | I ↑ | H |
| 3 | S ↑ | D → | . ↑ | W ← |

Floor rewards and movement rules are read from [data/maps.json](data/maps.json).
The same solver handles both maps. Sand has an entry reward of -2 and water
has an entry reward of -5. Movement from both is deterministic.

### Simulation results

Each map uses three batches of 50 episodes with batch seeds 11, 22, and 33.
Episode i uses seed 1000 * batch_seed + i. Each episode starts at S and
stops at a terminal cell or the 100-step limit. A terminal outcome on the
last step takes priority. Discounted return is r0 + 0.9*r1 + 0.9^2*r2 + ... .

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| original | 11 | 42 | 8 | 0 | 4.98 | 0.891260 |
| original | 22 | 46 | 4 | 0 | 5.04 | 1.914776 |
| original | 33 | 48 | 2 | 0 | 5.04 | 2.492144 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.502000 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.502000 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.502000 |

### Explain your results

On the original map, the policy starts by moving right and then approaches
the goal through ice. On the extended map, the extra cost of entering sand
changes the first action to Up. This route goes through mud, avoids ice,
and reaches the goal deterministically in five moves.

The start value changes from approximately 1.535824 to 1.502000. The other
state values and selected actions agree to six decimal places. From the
start, the value of choosing Right decreases from about 1.535824 to 0.535824
because sand costs one extra reward unit on entry. Choosing Up gives 1.502000.

Water reduces the value of moving right from (3,3) from about 1.535824
to -2.464176. The selected action at (3,3) is Up on both maps. Sand and
water retain approximately the same values as the original floor cells
at those positions because their entry costs are paid on the preceding move.
The chosen actions leave these cells for ordinary floor.

Both policies choose Up on ice. The intended move reaches the cell to the left of
the goal with probability 0.8. A slip left hits the wall and stays on ice,
while a slip right enters the hole, each with probability 0.1.

Simulation means estimate the computed policy's expected return. Sampling
variation and the 100-step cutoff can create differences from the
infinite-horizon values. The original map has variable outcomes because
its route crosses ice. The extended route has a return of 1.502000 in
every episode, matching its start value to the reported precision.
A higher observed success rate alone does not establish a higher expected
discounted return, since costs and terminal rewards also matter.

### Rule checks

The checks confirm the following behavior.

- A wall blocks movement and the reward comes from the cell where the agent stays.
- An attempted move above mud stays in mud and gives -3.
- Entering mud, sand, and water gives -3, -2, and -5 respectively.
- Movement from sand and water is deterministic. Entering ice from ordinary floor is deterministic.
- Every ice action has probabilities 0.8, 0.1, and 0.1, summing to 1. Each outcome is resolved against walls and boundaries.
- Entering the goal gives +10 and entering the hole gives -10.
- Goal and hole values remain zero. Further transitions after termination give zero reward.

All checks passed. State coordinates inside the program start at zero;
coordinates in the report and value CSV start at one.

- [Episode results](outputs/part_c_episodes.csv)
- [Batch summary](outputs/part_c_summary.csv)
- [Values and actions](outputs/part_c_values.csv)
- [Printed maps, convergence, comparisons, and rule checks](outputs/part_c_results.txt)

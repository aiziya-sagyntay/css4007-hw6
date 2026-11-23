import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "data/maps.json").read_text(encoding="utf-8"))

ACTIONS = CONFIG["actions"]
MOVES = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
GAMMA = CONFIG["discount"]
OUTPUTS = ROOT / "outputs"


class GridWorld:
    def __init__(self, rows):
        self.rows = rows
        self.rules = CONFIG["tiles"]
        self.states = [
            (r, c)
            for r in range(len(rows))
            for c in range(len(rows[0]))
            if self.rules[rows[r][c]].get("traversable", True)
        ]
        self.start = next(s for s in self.states if self.tile(s) == "S")

    def tile(self, state):
        r, c = state
        return self.rows[r][c]

    def terminal(self, state):
        return self.rules[self.tile(state)].get("terminal", False)

    def transitions(self, state, action):
        if self.terminal(state):
            return [(1.0, state, 0)]

        rule = self.rules[self.tile(state)]
        directions = [(1.0, action)]

        if rule["movement"] == "slippery":
            i = ACTIONS.index(action)
            directions = [
                (rule["intended"], action),
                (rule["perpendicular_each"], ACTIONS[(i - 1) % 4]),
                (rule["perpendicular_each"], ACTIONS[(i + 1) % 4]),
            ]

        outcomes = []
        r, c = state

        for probability, direction in directions:
            dr, dc = MOVES[direction]
            destination = (r + dr, c + dc)

            if destination not in self.states:
                destination = state

            reward = self.rules[self.tile(destination)]["reward"]
            outcomes.append((probability, destination, reward))

        return outcomes


def action_values(world, state, values):
    return {
        action: sum(
            p * (reward + GAMMA * values[destination])
            for p, destination, reward in world.transitions(state, action)
        )
        for action in ACTIONS
    }


def update_values(world, previous):
    return {
        state: 0.0 if world.terminal(state)
        else max(action_values(world, state, previous).values())
        for state in world.states
    }


def solve(world):
    values = {
        s: float(CONFIG["initial_value"])
        for s in world.states
    }
    early_values = []
    converged = False

    for iteration in range(1, CONFIG["solver"]["max_iterations"] + 1):
        updated = update_values(world, values)
        delta = max(abs(updated[s] - values[s]) for s in world.states)
        values = updated

        if iteration <= 2:
            early_values.append(values.copy())

        if delta < CONFIG["solver"]["tolerance"]:
            converged = True
            break

    policy = {}

    for state in world.states:
        if not world.terminal(state):
            q = action_values(world, state, values)
            best = max(q.values())
            policy[state] = next(
                a for a in ACTIONS
                if math.isclose(q[a], best, rel_tol=0, abs_tol=1e-10)
            )

    return values, policy, iteration, delta, converged, early_values


def map_text(world, entries):
    lines = []

    for r, row in enumerate(world.rows):
        cells = []
        for c, tile in enumerate(row):
            cells.append(f"{tile} {entries.get((r, c), '')}".strip())
        lines.append(" | ".join(f"{cell:>12}" for cell in cells))

    return "\n".join(lines)


def check_rules(world, extended):
    checks = []

    def check(label, actual, expected):
        if actual != expected:
            raise AssertionError(
                f"{label}: got {actual}, expected {expected}"
            )
        checks.append(f"PASS {label}: {actual}")

    check(
        "Wall stays on ordinary floor",
        world.transitions((1, 0), "right"),
        [(1.0, (1, 0), -1)],
    )
    check(
        "Boundary stays on mud",
        world.transitions((0, 1), "up"),
        [(1.0, (0, 1), -3)],
    )
    check(
        "Entering mud",
        world.transitions((0, 0), "right"),
        [(1.0, (0, 1), -3)],
    )
    check(
        "Entering sand",
        extended.transitions((2, 0), "right"),
        [(1.0, (2, 1), -2)],
    )
    check(
        "Entering water",
        extended.transitions((2, 2), "right"),
        [(1.0, (2, 3), -5)],
    )
    check(
        "Leaving sand is deterministic",
        extended.transitions((2, 1), "right"),
        [(1.0, (2, 2), -1)],
    )
    check(
        "Leaving water is deterministic",
        extended.transitions((2, 3), "left"),
        [(1.0, (2, 2), -1)],
    )
    check(
        "Entering ice is deterministic",
        world.transitions((2, 2), "up"),
        [(1.0, (1, 2), -1)],
    )
    check(
        "Entering goal",
        world.transitions((0, 2), "right"),
        [(1.0, (0, 3), 10)],
    )
    check(
        "Entering hole",
        world.transitions((2, 3), "up"),
        [(1.0, (1, 3), -10)],
    )

    for action in ACTIONS:
        outcomes = world.transitions((1, 2), action)
        check(
            f"Ice {action} probability sum",
            sum(p for p, _, _ in outcomes),
            1.0,
        )
        check(
            f"Ice {action} probabilities",
            [p for p, _, _ in outcomes],
            [0.8, 0.1, 0.1],
        )
        checks.append(
            f"Ice {action} outcomes (zero-based coordinates): {outcomes}"
        )

        for state in ((0, 3), (1, 3)):
            check(
                f"After terminal {world.tile(state)}, {action}",
                world.transitions(state, action),
                [(1.0, state, 0)],
            )

    return checks


def simulate(world, policy, name):
    episodes = []
    summaries = []
    settings = CONFIG["simulation"]

    for batch_seed in settings["batch_seeds"]:
        batch = []

        for i in range(settings["episodes_per_batch"]):
            seed = 1000 * batch_seed + i
            rng = random.Random(seed)
            state = world.start
            discounted_return = 0.0
            outcome = "time limit"

            for step in range(settings["max_steps"]):
                transitions = world.transitions(state, policy[state])
                probabilities = [p for p, _, _ in transitions]
                _, state, reward = rng.choices(
                    transitions, weights=probabilities, k=1
                )[0]

                discounted_return += GAMMA ** step * reward

                if world.terminal(state):
                    outcome = {"G": "goal", "H": "hole"}[world.tile(state)]
                    break

            batch.append({
                "map": name,
                "batch_seed": batch_seed,
                "episode_seed": seed,
                "steps": step + 1,
                "outcome": outcome,
                "discounted_return": discounted_return,
            })

        episodes.extend(batch)
        counts = Counter(row["outcome"] for row in batch)

        summaries.append({
            "map": name,
            "batch_seed": batch_seed,
            "goals": counts["goal"],
            "holes": counts["hole"],
            "time_limits": counts["time limit"],
            "mean_steps": mean(row["steps"] for row in batch),
            "mean_discounted_return": mean(
                row["discounted_return"] for row in batch
            ),
        })

    return episodes, summaries


def save_csv(filename, rows):
    with (OUTPUTS / filename).open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    original = GridWorld(CONFIG["paper_map"])
    extended = GridWorld(CONFIG["extended_map"])

    report = ["RULE CHECKS", *check_rules(original, extended)]
    report.append(
        "\nTie rule: first action in up, right, down, left order; "
        "absolute tolerance 1e-10."
    )

    solutions = {}
    all_episodes = []
    all_summaries = []
    value_rows = []

    for name, world in (("original", original), ("extended", extended)):
        values, policy, iterations, delta, converged, early = solve(world)
        solutions[name] = (values, policy)

        if name == "original":
            v1, v2 = early
            expected_v1 = {
                s: 0 if world.terminal(s) else -1
                for s in world.states
            }
            expected_v1[(0, 2)] = 10
            assert v1 == expected_v1, "V1 differs from the paper solution."

            report.append(
                "\nORIGINAL V1\n"
                + map_text(world, {
                    s: f"{v1[s]:.2f}" for s in world.states
                })
            )
            report.append("\nORIGINAL V2 CHECK (one-based coordinates)")

            for position, expected in zip(
                CONFIG["paper_v2_positions"], (8.0, 10.0, 5.21)
            ):
                state = (position[0] - 1, position[1] - 1)
                assert math.isclose(v2[state], expected, abs_tol=1e-10)
                report.append(
                    f"{position}: V2={v2[state]:.2f}; "
                    f"all actions={action_values(world, state, v1)}"
                )

            report.append(
                "PASS: V1 and the three requested V2 values "
                "match the paper solution."
            )

        report.append(
            f"\n{name.upper()} CONVERGENCE\n"
            f"Converged={converged}, iterations={iterations}, "
            f"final change={delta:.12g}"
        )

        if not converged:
            report.append(
                "NONCONVERGENCE: reached the iteration safeguard."
            )

        report.append(
            "VALUES\n"
            + map_text(world, {
                s: f"{values[s]:.6f}" for s in world.states
            })
        )
        report.append(
            "POLICY\n"
            + map_text(world, {
                s: ARROWS[a] for s, a in policy.items()
            })
        )

        for state in world.states:
            r, c = state
            value_rows.append({
                "map": name,
                "row": r + 1,
                "column": c + 1,
                "tile": world.tile(state),
                "value": values[state],
                "action": policy.get(state, "terminal"),
            })

        episodes, summaries = simulate(world, policy, name)
        all_episodes.extend(episodes)
        all_summaries.extend(summaries)

    old_values, old_policy = solutions["original"]
    new_values, new_policy = solutions["extended"]

    report.append("\nMAP COMPARISON (one-based coordinates)")

    for state in original.states:
        same_value = math.isclose(
            old_values[state], new_values[state],
            rel_tol=0, abs_tol=1e-6,
        )
        same_action = old_policy.get(state) == new_policy.get(state)

        if same_value and same_action:
            continue

        r, c = state
        report.append(
            f"({r+1},{c+1}): "
            f"V {old_values[state]:.6f} -> {new_values[state]:.6f}; "
            f"action {old_policy.get(state)} -> {new_policy.get(state)}"
        )

    report.append("\nFLOOR COSTS AND ICE")
    report.append(
        f"Original start actions: "
        f"{action_values(original, original.start, old_values)}"
    )
    report.append(
        f"Extended start actions: "
        f"{action_values(extended, extended.start, new_values)}"
    )
    report.append(
        f"Original actions at (3,3): "
        f"{action_values(original, (2, 2), old_values)}"
    )
    report.append(
        f"Extended actions at (3,3): "
        f"{action_values(extended, (2, 2), new_values)}"
    )
    report.append(
        "The sand entry cost changes the start action from right to up. "
        "The extended start route avoids ice."
    )
    report.append(
        "Water makes moving right from (3,3) less attractive; "
        "both policies choose up there."
    )
    report.append(
        "Both policies choose up on ice. Cell values use future rewards; "
        "a floor's entry cost is paid by its predecessor."
    )
    report.append(
        "Simulation means estimate the computed policy's expected return. "
        "A 100-step cutoff and sampling variation can cause differences "
        "from infinite-horizon values."
    )

    report.append("\nSIMULATIONS")

    for row in all_summaries:
        report.append(
            f"{row['map']}, batch={row['batch_seed']}: "
            f"goals={row['goals']}, holes={row['holes']}, "
            f"time limits={row['time_limits']}, "
            f"mean steps={row['mean_steps']:.2f}, "
            f"mean discounted return={row['mean_discounted_return']:.6f}"
        )

    OUTPUTS.mkdir(exist_ok=True)
    save_csv("part_c_episodes.csv", all_episodes)
    save_csv("part_c_summary.csv", all_summaries)
    save_csv("part_c_values.csv", value_rows)

    text = "\n".join(report)
    (OUTPUTS / "part_c_results.txt").write_text(text, encoding="utf-8")

    print(text)
    print(
        f"\nSaved {len(all_episodes)} episodes "
        f"and {len(all_summaries)} batches."
    )


if __name__ == "__main__":
    main()

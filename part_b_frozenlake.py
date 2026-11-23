import csv
import random
import re
from collections import Counter
from contextlib import closing
from pathlib import Path
from statistics import mean

import gymnasium as gym


LEFT, DOWN, RIGHT, UP = range(4)
ACTION_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")

# State IDs run left to right and top to bottom on the 4x4 map.
FIXED_POLICY = {
    0: DOWN, 1: RIGHT, 2: DOWN, 3: LEFT,
    4: DOWN, 6: DOWN,
    8: RIGHT, 9: DOWN, 10: DOWN,
    13: RIGHT, 14: RIGHT,
}
TERMINAL_OUTCOMES = {
    5: "hole", 7: "hole", 11: "hole", 12: "hole", 15: "goal",
}

BATCH_SEEDS = (11, 22, 33)
EPISODES_PER_BATCH = 50
MAX_STEPS = 100
OUTPUTS = Path(__file__).resolve().parent / "outputs"


def run_episode(env, policy, episode_seed, render=False):
    state, _ = env.reset(seed=episode_seed)
    action_rng = random.Random(episode_seed)
    total_reward = 0
    outcome = "time limit"
    frames = []

    if render:
        frames.append(f"Start, state {state}\n{env.render()}")

    for steps in range(1, MAX_STEPS + 1):
        if policy == "fixed":
            action = FIXED_POLICY[state]
        else:
            action = action_rng.randrange(4)

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if render:
            frames.append(
                f"Step {steps}, {ACTION_NAMES[action]} → state {state}\n"
                f"{env.render()}"
            )

        if terminated:
            outcome = TERMINAL_OUTCOMES[state]
            break
        if truncated:
            break

    return steps, total_reward, outcome, frames


def make_lake(slippery, render_mode=None):
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=slippery,
        max_episode_steps=MAX_STEPS,
        render_mode=render_mode,
    )


def write_csv(filename, rows):
    with (OUTPUTS / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    episodes = []
    summaries = []
    examples = {}

    for slippery in (False, True):
        for policy in ("fixed", "random"):
            with closing(make_lake(slippery)) as env:
                for batch_seed in BATCH_SEEDS:
                    counts = Counter()
                    step_counts = []

                    for i in range(EPISODES_PER_BATCH):
                        episode_seed = 1000 * batch_seed + i
                        steps, reward, outcome, _ = run_episode(env, policy, episode_seed)
                        row = {
                            "condition": f"slippery={slippery}, policy={policy}",
                            "batch_seed": batch_seed,
                            "episode_seed": episode_seed,
                            "policy": policy,
                            "steps": steps,
                            "total_reward": reward,
                            "outcome": outcome,
                        }
                        episodes.append(row)
                        counts[outcome] += 1
                        step_counts.append(steps)
                        example_type = "success" if outcome == "goal" else "failure"
                        examples.setdefault(example_type, (slippery, row))

                    summaries.append({
                        "Slippery?": slippery,
                        "Policy": policy,
                        "Batch seed": batch_seed,
                        "Goals /50": counts["goal"],
                        "Holes /50": counts["hole"],
                        "Time limits /50": counts["time limit"],
                        "Success %": f"{100 * counts['goal'] / EPISODES_PER_BATCH:.1f}",
                        "Mean steps": f"{mean(step_counts):.2f}",
                    })

    OUTPUTS.mkdir(exist_ok=True)
    write_csv("part_b_episodes.csv", episodes)
    write_csv("part_b_summary.csv", summaries)

    for example_type in ("success", "failure"):
        if example_type in examples:
            slippery, row = examples[example_type]
            with closing(make_lake(slippery, render_mode="ansi")) as env:
                steps, reward, outcome, frames = run_episode(
                    env, row["policy"], row["episode_seed"], render=True
                )

            logged_result = (row["steps"], row["total_reward"], row["outcome"])
            if (steps, reward, outcome) != logged_result:
                raise RuntimeError("The rendered replay differs from its logged episode.")

            header = (
                f"{row['condition']}, batch seed={row['batch_seed']}, "
                f"episode seed={row['episode_seed']}\n"
                f"Outcome: {outcome}, steps: {steps}, total reward: {reward}\n\n"
            )
            trace = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(frames))
            (OUTPUTS / f"part_b_{example_type}.txt").write_text(
                header + trace, encoding="utf-8"
            )
        else:
            print(f"No {example_type} episode was observed.")

    print(f"Saved {len(episodes)} episodes and {len(summaries)} batches.\n")
    print("| " + " | ".join(summaries[0]) + " |")
    print("|" + "|".join(["---"] * len(summaries[0])) + "|")
    for row in summaries:
        print("| " + " | ".join(str(value) for value in row.values()) + " |")


if __name__ == "__main__":
    main()

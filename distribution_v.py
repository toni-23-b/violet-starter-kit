import random
from pathlib import Path
import polars as pl

TOTAL_AGENTS = 1000
SEEDS = [1, 2, 3, 4, 5]


def truncated_normal(prng, mean, sd, lower=0.0, upper=1.0):
    value = prng.gauss(mean, sd)

    while value < lower or value > upper:
        value = prng.gauss(mean, sd)

    return value


def create_opinion_batch(seed, total_agents=TOTAL_AGENTS):
    prng = random.Random(seed)

    return [
        truncated_normal(prng, mean=0.5, sd=0.15)
        for _ in range(total_agents)
    ]


five_distributions = {
    seed: create_opinion_batch(seed)
    for seed in SEEDS
}

INITIAL_OPINIONS_PATH = Path("initial_opinions.csv")
initial_opinions_data = pl.DataFrame(
    [
        {
            "seed": seed,
            "agent_id": agent_id,
            "initial_opinion": opinion,
        }
        for seed, opinions in five_distributions.items()
        for agent_id, opinion in enumerate(opinions)
    ]
)

initial_opinions_data.write_csv(INITIAL_OPINIONS_PATH)
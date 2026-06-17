import random


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


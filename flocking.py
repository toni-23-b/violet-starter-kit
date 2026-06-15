from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random

import polars as pl

from vi import Agent, Simulation
from vi.config import Config

DATA_PATH = Path("data.csv")
DATA_PATH.unlink(missing_ok=True)

@dataclass
class OpinionConfig(Config):
    epsilon: float = 0.05
    convergence_tolerance: float = 0.001

def truncated_normal(mean, sd, lower=0.0, upper=1.0):
    value = random.gauss(mean, sd)

    while value < lower or value > upper:
        value = random.gauss(mean, sd)

    return value

def stepHK(agents, epsilon):
    next_opinions = []

    for agent in agents:
        neighbors = []

        for other in agents:
            if abs(other.opinion - agent.opinion) <= epsilon:
                neighbors.append(other.opinion)

        average = sum(neighbors) / len(neighbors)
        next_opinions.append(average)

    return next_opinions

class OpinionAgent(Agent[OpinionConfig]):
    def update(self):
        self.save_data("opinion", self.opinion)
        #self.save_data("next_opinion", self.next_opinion)
        self.save_data("agent_type", type(self).__name__)


class OpinionHolder(OpinionAgent):
    def on_spawn(self):
        self.opinion = truncated_normal(mean=0.5, sd=0.15)


class ExtremistYes(OpinionAgent):
    def on_spawn(self):
        self.opinion = 1.0


class ExtremistNo(OpinionAgent):
    def on_spawn(self):
        self.opinion = 0.0


class HKSimulation(Simulation[OpinionConfig]):
    def after_update(self):
        agents = list(self._agents.sprites())
        next_opinions = stepHK(agents, self.config.epsilon)

        max_change = max(
            abs(agent.opinion - next_opinion)
            for agent, next_opinion in zip(agents, next_opinions)
        )

        for agent, next_opinion in zip(agents, next_opinions):
            agent.opinion = next_opinion

        super().after_update()

        if max_change < self.config.convergence_tolerance:
            self.stop()
    
run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
total_agents = 1000
extremist_proportion = 0.6
extremist_count = int(total_agents * extremist_proportion)
normal_count = total_agents - extremist_count
yes_count = extremist_count // 2
no_count = extremist_count - yes_count

config = OpinionConfig(image_rotation=True, movement_speed=1, radius=50, epsilon=0.05, convergence_tolerance=0.001)

sim = (
    # Step 1: Create a new simulation.
    HKSimulation(config)
    # Step 2: Add agents to the simulation.
    .batch_spawn_agents(normal_count, OpinionHolder, images=["images/triangle.png"])
    .batch_spawn_agents(yes_count, ExtremistYes, images=["images/green.png"])
    .batch_spawn_agents(no_count, ExtremistNo, images=["images/red.png"])
    # Step 3: Profit! 🎉
    .run()
)

run_data = sim.snapshots.with_columns(
    pl.lit(run_id).alias("run_id"),
    pl.lit(config.epsilon).alias("epsilon"),
    pl.lit(config.radius).alias("radius"),
    pl.lit(total_agents).alias("total_agents"),
    pl.lit(extremist_proportion).alias("extremist_proportion"),
)

run_data.write_csv(DATA_PATH)

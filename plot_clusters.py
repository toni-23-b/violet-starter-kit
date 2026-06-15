from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

GRAPHS_PATH = Path("graphs")
GRAPHS_PATH.mkdir(exist_ok=True)

df = pl.read_csv("data.csv")

for run_id in df["run_id"].unique():
    run_data = df.filter(pl.col("run_id") == run_id)

    plt.figure(figsize=(10, 6))

    for agent_type in run_data["agent_type"].unique():
        agent_data = run_data.filter(pl.col("agent_type") == agent_type)

        plt.scatter(
            agent_data["frame"],
            agent_data["opinion"],
            s=8,
            alpha=0.5,
            label=agent_type,
        )

    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Opinion")
    plt.title("HK Opinion Cluster Formation")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_PATH / f"opinion_clusters_{run_id}.png", dpi=300)
    plt.show()


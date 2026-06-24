import polars as pl
from scipy.stats import f_oneway, kruskal


RESULTS_PATH = "results_0.05.csv"

df = pl.read_csv(RESULTS_PATH)

print(df)

groups = []

for proportion in df["extremist_proportion"].unique().sort():
    group = (
        df
        .filter(pl.col("extremist_proportion") == proportion)
        ["number_of_clusters"]
        .to_list()
    )

    groups.append(group)

    print(f"{proportion}: {group}")

statistic, p_value = f_oneway(*groups)

print()
print("ANOVA result")
print(f"F-statistic: {statistic}")
print(f"p-value: {p_value}") 


kruskal_statistic, kruskal_p_value = kruskal(*groups)

print()
print("Kruskal-Wallis result")
print(f"H-statistic: {kruskal_statistic}")
print(f"p-value: {kruskal_p_value}")

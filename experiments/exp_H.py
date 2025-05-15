import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
from matplotlib.lines import Line2D

# Get the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from parameters import DATASETS, SHORTNAMES  # noqa: E402

# Dossier contenant les fichiers JSON
exps_folder = "experiments_finale/results_H"
exps_folder2 = "experiments_finale/results"
plots_folder = "experiments_finale/plots/Heuristic"
MARKERS = {
    "TRA_HEURISTIC": "*",
    "TRA_OCEAN": "p",
}
COLORS = {
    "TRA_HEURISTIC": "blue",
    "TRA_OCEAN": "#E5323B",
}


def read_experiments(
    exps_folder, attacks: list[str] = None, datasets: list[str] = None
):
    """
    Read all experiments in the specified folder
    """
    # Charger les fichiers JSON
    files = glob(os.path.join(exps_folder, "exp_*.json"))

    # Préparer les données
    data = []
    for file in files:
        with open(file, "r") as f:
            experiment = json.load(f)
            dataset = experiment["Dataset"]
            if datasets is not None and dataset not in datasets:
                continue
            attacker = experiment["Attacker"]
            if attacks is not None and attacker not in attacks:
                continue
            seed = experiment["Seed"]
            for depth_info in experiment["Depth"]:
                max_depth = depth_info.get("max_depth")
                # Gérer les attaques standard
                fidelity_over_queries = depth_info.get("fidelity_over_queries", {})
                data.append(
                    {
                        "Dataset": dataset,
                        "Attacker": attacker,
                        "Seed": seed,
                        "Max Depth": max_depth,
                        "N Nodes": depth_info.get("N_nodes"),
                        "N Features": depth_info.get("N_features"),
                        "N Query": depth_info.get("N_query"),
                        "N Splits": depth_info.get("N_splits"),
                        "Fidelity": depth_info.get("fidelity"),
                        "Fidelity Over Queries": fidelity_over_queries,
                    }
                )

    df = pd.DataFrame(data)
    df["Attacker"] = df["Attacker"].replace("TR", "TRA")
    return df


def plot_queries_vs_nodes(df, choosed_attacker, xlogsacle=False, ylogscale=False):
    """
    Plot the number of queries vs the number of nodes for the TR and PathFinding attacker
    """
    if not os.path.exists(plots_folder + "/QvsN"):
        os.makedirs(plots_folder + "/QvsN")
    legend_elements = []
    subset_s = df[df["Attacker"].isin(choosed_attacker)]
    datasets = subset_s["Dataset"].unique()
    for i, dataset in enumerate(DATASETS):
        if dataset not in datasets:
            continue
        plt.figure(figsize=(10, 6))
        subset = subset_s[subset_s["Dataset"] == dataset]
        for attacker in subset["Attacker"].unique():
            attacker_subset = subset[subset["Attacker"] == attacker]
            plt.scatter(
                attacker_subset["N Nodes"],
                attacker_subset["N Query"],
                label=attacker,
                marker=MARKERS[attacker],
                color=COLORS[attacker],
                s=60,
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=MARKERS[attacker],
                    color=COLORS[attacker],
                    label=attacker,
                    markersize=10,
                    linestyle="None",
                )
            )
        plt.xlabel("#Nodes", fontsize="15")
        plt.ylabel("#Queries", fontsize="15")
        if xlogsacle:
            plt.xscale("log")
        if ylogscale:
            plt.yscale("log")
        # plt.legend(fontsize="15")
        plt.grid()
        plt.savefig(
            f"{plots_folder}/QvsN/QvsN_{SHORTNAMES[i]}.pdf", bbox_inches="tight"
        )
    legendFig = plt.figure(figsize=(20, 1))
    legendFig.legend(handles=legend_elements[:2], loc="center", ncol=2, fontsize="15")
    plt.axis("off")
    legendFig.savefig(f"{plots_folder}/QvsN/legend.pdf", bbox_inches="tight")
    plt.close()


def compute_fidelity(row, q):
    if row["Attacker"] in ["TRA", "TRA+"]:
        if q > row["N Query"]:  # attack already finished
            return 1.0
    # Get all available query points
    query_points = sorted(map(int, row["Fidelity Over Queries"].keys()))
    # Find the previous and next query points
    prev_q = next_q = None
    for point in query_points:
        if point <= q:
            prev_q = point
        if point >= q:
            next_q = point
            break
    # Interpolate fidelity
    if prev_q is not None and next_q is not None:
        # Linear interpolation
        prev_fid = row["Fidelity Over Queries"].get(str(prev_q), 0.0)
        next_fid = row["Fidelity Over Queries"].get(str(next_q), 1.0)
        if prev_q == next_q:
            fid = prev_fid
        else:
            fid = prev_fid + (next_fid - prev_fid) * (q - prev_q) / (next_q - prev_q)
    elif prev_q is not None:
        # Only previous exists, use its value
        fid = row["Fidelity Over Queries"].get(str(prev_q), 0.0)
    elif next_q is not None:
        # Only next exists, use its value
        fid = row["Fidelity Over Queries"].get(str(next_q), 1.0)
    else:
        # No data available, default to 1.0
        fid = 1.0
    return fid


def plot_meanFid_vs_query(df, choosed_attacker, xlogscale=False, ylogscale=False):
    """
    Plot the mean of fidelity over queries vs the number of queries for the TR attacker.
    """
    if not os.path.exists(plots_folder + "/MFvsQ"):
        os.makedirs(plots_folder + "/MFvsQ")
    subsets = df[df["Attacker"].isin(choosed_attacker)]
    legend_elements = []
    datasets = subsets["Dataset"].unique()
    for i, dataset in enumerate(DATASETS):
        if dataset not in datasets:
            continue
        plt.figure(figsize=(10, 6))
        subset = subsets[subsets["Dataset"] == dataset]
        max_n_query = int(subset["N Query"].max())
        print(max_n_query, subset["Attacker"].unique())
        for attacker in choosed_attacker:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=None,
                    color=COLORS[attacker],
                    label=attacker,
                    linewidth=3,
                )
            )
            subset_s = subset
            subset_ = subset_s[subset_s["Attacker"] == attacker]
            mean_fidelities = {1: 0.0}
            for q in range(20, max_n_query + 1, 20):
                fids = []
                for _, row in subset_.iterrows():
                    fid = compute_fidelity(row, q)
                    fids.append(fid)
                mean_fidelities[q] = np.mean(fids)
            plt.plot(
                list(mean_fidelities.keys()),
                list(mean_fidelities.values()),
                label=attacker,
                color=COLORS[attacker],
                linewidth=3,
            )

        # plt.title(f"Mean Fidelity vs Queries for the dataset : {SHORTNAMES[i]}", fontsize="18")
        plt.xlabel("#Queries", fontsize="15")
        plt.ylabel("Mean Fidelity", fontsize="15")
        if xlogscale:
            plt.xlim(10, max_n_query)
            plt.xscale("log")
        if ylogscale:
            plt.yscale("log")
        plt.ylim(0.5, 1.0)
        # plt.legend(fontsize="15")
        plt.grid()
        plt.savefig(
            f"{plots_folder}/MFvsQ/MFvsQ_{SHORTNAMES[i]}.pdf", bbox_inches="tight"
        )

    legendFig = plt.figure(figsize=(20, 1))
    legendFig.legend(
        handles=legend_elements[: len(choosed_attacker)],
        loc="center",
        ncol=4,
        fontsize="15",
    )
    plt.axis("off")
    legendFig.savefig(f"{plots_folder}/MFvsQ/legend.pdf", bbox_inches="tight")
    plt.close()


datasets = [
    "datasets/Adult_processedMACE.csv",
    "datasets/COMPAS-ProPublica_processedMACE.csv",
    "datasets/Credit-Card-Default_processedMACE.csv",
    "datasets/German-Credit.csv",
    "datasets/Students-Performance-MAT.csv",
]
choosed_attacker = ["TRA_HEURISTIC", "TRA_OCEAN"]
df_h = read_experiments(exps_folder)
print(len(df_h))
df = read_experiments(exps_folder2, attacks=["TR"], datasets=datasets)
print(len(df))
df_h["Attacker"] = df_h["Attacker"].replace({"TRA": "TRA_HEURISTIC"})
df["Attacker"] = df["Attacker"].replace({"TRA": "TRA_OCEAN"})
df = pd.concat([df_h, df], ignore_index=True)
plot_queries_vs_nodes(df, choosed_attacker, xlogsacle=False, ylogscale=True)
plot_meanFid_vs_query(df, choosed_attacker, xlogscale=True)

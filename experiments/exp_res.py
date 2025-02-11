import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
from matplotlib.lines import Line2D
from parameters import DATASETS, SHORTNAMES, MARKERS, COLORS
# import sys
# # Get the absolute path to the parent directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.insert(0, parent_dir)

import matplotlib as mpl

label_size = 15
mpl.rcParams["xtick.labelsize"] = label_size
mpl.rcParams["ytick.labelsize"] = label_size


# Dossier contenant les fichiers JSON
exps_folder = "experiments/results"
plots_folder = "experiments/plots"


def read_experiments(exps_folder: str) -> pd.DataFrame:
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
            attacker = experiment["Attacker"]
            seed = experiment["Seed"]
            for depth_info in experiment["Depth"]:
                max_depth = depth_info.get("max_depth")
                if "performances_DiCE" in depth_info.keys():
                    # Gérer les attaques avec sous-types (MLP, DT, DT+)
                    for sub_attack, performance in depth_info[
                        "performances_OCEAN"
                    ].items():
                        fidelity_over_queries = performance.get(
                            "fidelity_over_queries", {}
                        )
                        data.append(
                            {
                                "Dataset": dataset,
                                "Attacker": f"{attacker}_{sub_attack}",
                                "BaseAttacker": attacker,
                                "Seed": seed,
                                "Surrogate": sub_attack,
                                "Oracle": "OCEAN",
                                "Max Depth": max_depth,
                                "N Query": performance.get("N_query"),
                                "Fidelity": performance.get("fidelity"),
                                "Fidelity Over Queries": fidelity_over_queries,
                            }
                        )
                    for sub_attack, performance in depth_info[
                        "performances_DiCE"
                    ].items():
                        fidelity_over_queries = performance.get(
                            "fidelity_over_queries", {}
                        )
                        data.append(
                            {
                                "Dataset": dataset,
                                "Attacker": f"{attacker}_{sub_attack}",
                                "BaseAttacker": attacker,
                                "Seed": seed,
                                "Surrogate": sub_attack,
                                "Oracle": "DiCE",
                                "Max Depth": max_depth,
                                "N Query": performance.get("N_query"),
                                "Fidelity": performance.get("fidelity"),
                                "Fidelity Over Queries": fidelity_over_queries,
                            }
                        )
                else:
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
    return df


def plot_queries_vs_nodes(df, xlogsacle=False, ylogscale=False):
    """
    Plot the number of queries vs the number of nodes for the TRA and PathFinding attacker
    """
    if not os.path.exists(plots_folder + "/QvsN"):
        os.makedirs(plots_folder + "/QvsN")
    choosed_attacker = ["TRA", "PathFinding"]
    legend_elements = []
    subset_s = df[df["Attacker"].isin(choosed_attacker)]
    for i, dataset in enumerate(DATASETS):
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
        plt.close()

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
    Plot the mean of fidelity over queries vs the number of queries for the TRA attacker.
    """
    if not os.path.exists(plots_folder + "/MFvsQ"):
        os.makedirs(plots_folder + "/MFvsQ")
    subsets = df[df["Attacker"].isin(choosed_attacker)]
    legend_elements = []
    for i, dataset in enumerate(DATASETS):
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
                    label=attacker.split("_")[0],
                    linewidth=3,
                )
            )
            subset_s = subset
            if "CF" in attacker:
                subset_s = subset[subset["Oracle"] == "DiCE"]
            subset_ = subset_s[subset_s["Attacker"] == attacker]
            mean_fidelities = {1: 0.0}

            if attacker == "PathFinding":
                # for PathFinding attacker
                # plot a piecewise constant function
                total_number_of_trees = len(subset_)
                for q in sorted(subset_["N Query"].unique()):
                    mean_fidelities[q] = (
                        len(subset_[subset_["N Query"] <= q]) / total_number_of_trees
                    )
                kwargs = dict(drawstyle="steps-mid")
                plt.plot(
                    list(mean_fidelities.keys()),
                    list(mean_fidelities.values()),
                    label="PathFinding",
                    color=COLORS["PathFinding"],
                    linewidth=3,
                    **kwargs,
                )
                continue
            # print(max_n_query)
            for q in range(20, max_n_query + 1, 20):
                fids = []
                for _, row in subset_.iterrows():
                    fid = compute_fidelity(row, q)
                    fids.append(fid)
                mean_fidelities[q] = np.mean(fids)
            plt.plot(
                list(mean_fidelities.keys()),
                list(mean_fidelities.values()),
                label=attacker.split("_")[0],
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
        plt.close()

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


def plot_CF_surrogates(df, dualCF=False, xlogscale=True, ylogscale=False):
    """
    Plot the mean of fidelity over queries vs the number of queries for the CF attacker.
    """
    attack = "DualCF" if dualCF else "CF"
    if not os.path.exists(plots_folder + "/" + attack):
        os.makedirs(plots_folder + "/" + attack)
    df_s = df[df["BaseAttacker"] == attack]
    legend_elements = []
    for i, dataset in enumerate(DATASETS):
        plt.figure(figsize=(10, 6))
        subset = df_s[df_s["Dataset"] == dataset]
        max_n_query = int(subset["N Query"].max())
        print(max_n_query, subset["Attacker"].unique())
        for k, attacker in enumerate(
            [f"{attack}_MLP", f"{attack}_DT+", f"{attack}_DT"]
        ):
            subset_ = subset[subset["Attacker"] == attacker]
            for p, method in enumerate(["DiCE", "OCEAN"]):
                # Add legend element
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker=None,
                        color=COLORS[attacker],
                        label=attacker + "_" + method,
                        linewidth=3,
                        linestyle=(p + 1) * "-",
                    )
                )
                subset_s = subset_[subset_["Oracle"] == method]
                mean_fidelities = {1: 0.0}
                # Iterate over query range
                for q in range(20, max_n_query + 1, 20):
                    fids = []
                    for _, row in subset_s.iterrows():
                        fid = compute_fidelity(row, q)
                        fids.append(fid)
                    mean_fidelities[q] = np.mean(fids)

                # "OCEAN" dashed line, "DiCE" solid line
                # if method == "OCEAN":
                plt.plot(
                    list(mean_fidelities.keys()),
                    list(mean_fidelities.values()),
                    label=attacker + "_" + method,
                    color=COLORS[attacker],
                    linewidth=3,
                    linestyle=(p + 1) * "-",
                    alpha=1.0 - 0.1 * k,
                )
                # else:
                #     plt.plot(list(mean_fidelities.keys()), list(mean_fidelities.values()), label=attacker + "_" + method, color=COLORS[attacker], linewidth=3, alpha=0.9 - 0.1 * k)

        plt.ylim(0.1, 1.0)
        if xlogscale:
            plt.xscale("log")
        plt.grid()
        if ylogscale:
            plt.yscale("log")
        plt.xlabel("#Queries", fontsize="15")

        plt.ylabel("Mean Fidelity", fontsize="15")
        # plt.legend(fontsize="15")
        plt.savefig(
            f"{plots_folder}/{attack}/{attack}_{SHORTNAMES[i]}.pdf", bbox_inches="tight"
        )
        plt.close()

    legendFig = plt.figure(figsize=(20, 1))
    legendFig.legend(handles=legend_elements[:6], loc="center", ncol=3, fontsize="15")
    plt.axis("off")
    legendFig.savefig(f"{plots_folder}/{attack}/legend.pdf", bbox_inches="tight")
    plt.close()


choosed_attacker = ["TRA", "DualCF_DT", "CF_DT", "PathFinding"]
df = read_experiments(exps_folder)
plot_CF_surrogates(df, dualCF=False, xlogscale=True)
plot_CF_surrogates(df, dualCF=True, xlogscale=True)
plot_queries_vs_nodes(df, xlogsacle=False, ylogscale=True)
plot_meanFid_vs_query(df, choosed_attacker, xlogscale=True)

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from parameters import MARKERS_RF, COLORS_RF

# import sys
from scipy.optimize import curve_fit
# # Get the absolute path to the parent directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.insert(0, parent_dir)

import matplotlib as mpl

label_size = 15
mpl.rcParams["xtick.labelsize"] = label_size
mpl.rcParams["ytick.labelsize"] = label_size

colors = COLORS_RF
MARKERS = MARKERS_RF

experiments_folder = "experiments/RFs"
plots_folder = "experiments/plots"


def read_experiments(experiments_folder):
    """
    Read all experiments in the specified folder
    """
    # Charger les fichiers JSON
    files = glob(os.path.join(experiments_folder, "exp_*.json"))

    # Préparer les données
    data = []
    for file in files:
        with open(file, "r") as f:
            experiment = json.load(f)
            dataset = experiment["Dataset"]
            attacker = experiment["Attacker"]
            seed = experiment["Seed"]
            for est_info in experiment["Estimators"]:
                estimators = est_info.get("n_estimators")
                if "performances_DiCE" in est_info:
                    # Gérer les attaques avec sous-types (MLP, RF, RF+)
                    for sub_attack, performance in est_info[
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
                                "Surrogate": sub_attack,
                                "Seed": seed,
                                "N Estimators": estimators,
                                "Oracle": "OCEAN",
                                "N Query": performance.get("N_query"),
                                "Fidelity": performance.get("fidelity"),
                                "Fidelity Over Queries": fidelity_over_queries,
                            }
                        )
                    for sub_attack, performance in est_info[
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
                                "Surrogate": sub_attack,
                                "Seed": seed,
                                "N Estimators": estimators,
                                "Oracle": "DiCE",
                                "N Query": performance.get("N_query"),
                                "Fidelity": performance.get("fidelity"),
                                "Fidelity Over Queries": fidelity_over_queries,
                            }
                        )
                else:
                    # Gérer les attaques standard
                    fidelity_over_queries = est_info.get("fidelity_over_queries", {})
                    data.append(
                        {
                            "Dataset": dataset,
                            "Attacker": attacker,
                            "Seed": seed,
                            "N Estimators": estimators,
                            "N Nodes": est_info.get("N_nodes"),
                            "N Features": est_info.get("N_features"),
                            "N Query": est_info.get("N_query"),
                            "N Splits": est_info.get("N_splits"),
                            "Fidelity": est_info.get("fidelity"),
                            "Fidelity Over Queries": fidelity_over_queries,
                        }
                    )

    df = pd.DataFrame(data)
    return df


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


def plot_meanFid_vs_query(df):
    """
    Plot the mean of fidelity over queries vs the number of queries for all the attackers.
    """
    plt.figure(figsize=(10, 6))
    max_n_query = df["N Query"].max()
    choosed_attackers = ["TRA", "CF_RF", "DualCF_RF"]
    for attacker in choosed_attackers:
        subset_ = df[df["Attacker"] == attacker]
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
            label=attacker.split("_")[0],
            color=colors[attacker],
            linewidth=3,
        )
        # print(mean_fidelities)

        # using log scale for the x axis
        # plt.plot(np.log(list(mean_fidelities.keys())), list(mean_fidelities.values()), label=attacker, color=colors[attacker])

    # plt.title(f"Mean Fidelity vs Queries for the dataset : {SHORTNAMES[i]}", fontsize="18")
    plt.xlabel("#Queries", fontsize="15")
    plt.ylabel("Mean Fidelity", fontsize="15")
    plt.ylim(0.9, 1.0)
    plt.xlim(10, max_n_query)
    plt.xscale("log")
    plt.legend(fontsize="15")
    plt.grid()
    plt.savefig(f"{plots_folder}/RF_MFvsQ.pdf", bbox_inches="tight")
    plt.close()


def meanFid_vs_query_Estimators(df):
    """
    Plot the mean of fidelity over queries vs the number of queries for the TRA attacker for all the n_estimators.
    """
    df = df[df["Attacker"] == "TRA"]
    N_estimators = df["N Estimators"].unique()
    plt.figure(figsize=(10, 6))
    max_n_query = df["N Query"].max()
    for n_estimators in N_estimators:
        subset_ = df[df["N Estimators"] == n_estimators]
        mean_fidelities = {1: 0.0}
        for q in range(20, max_n_query + 1, 20):
            fids = []
            for _, row in subset_.iterrows():
                fids.append(row["Fidelity Over Queries"].get(str(q), 1.0))
            mean_fidelities[q] = np.mean(fids)
        plt.plot(
            list(mean_fidelities.keys()),
            list(mean_fidelities.values()),
            label=f"n_estimators={n_estimators}",
            linewidth=3,
        )

    plt.xlabel("#Queries", fontsize="15")
    plt.ylabel("Mean Fidelity", fontsize="15")
    plt.yscale("log")
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize="20")
    plt.grid()
    plt.savefig(f"{plots_folder}/RF_FidvsQuery_Estimators.pdf", bbox_inches="tight")
    plt.close()


def query_vs_nodes_Estimators(df):
    """
    Scatter the number of queries vs number of nodes for the TRA attacker for all the n_estimators.
    """
    df = df[df["Attacker"] == "TRA"]
    N_estimators = [5, 25, 50, 75, 100]
    labels = {5: "x", 25: "o", 50: "v", 75: "s", 100: "d"}
    plt.figure(figsize=(10, 6))
    N_nodes = df["N Nodes"].values
    N_queries = df["N Query"].values
    for n_estimators in N_estimators:
        subset_ = df[df["N Estimators"] == n_estimators]
        plt.scatter(
            subset_["N Nodes"],
            subset_["N Query"],
            label=f"#Trees={n_estimators}",
            marker=labels[n_estimators],
            s=50,
        )
        # N_nodes.extend(subset_["N Nodes"].values)

    def power(x, b, c):
        return b * x**c

    popt, pcov = curve_fit(power, N_nodes, N_queries)
    best_func = rf"${popt[0]:.2f}* x^{{{popt[1]:.2f}}}$"
    print(f"Best fit function : {best_func}")
    print(N_nodes, N_queries)
    x = np.linspace(0, max(N_nodes), 100000)
    plt.plot(
        x, power(x, *popt), label=best_func, color="black", linewidth=3, linestyle="--"
    )
    plt.xlabel("Total #Nodes", fontsize="15")
    plt.ylabel("#Queries", fontsize="15")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize="15")
    plt.grid()
    plt.savefig(f"{plots_folder}/RF_QvsN_Estimators.pdf", bbox_inches="tight")


def plot_CF_surrogates(df, dualCF=False, xlogscale=True, ylogscale=False):
    """
    Plot the mean of fidelity over queries vs the number of queries for the CF attacker.
    """
    attack = "DualCF" if dualCF else "CF"
    # legend_elements = []
    subset = df[df["BaseAttacker"] == attack]
    plt.figure(figsize=(10, 6))
    max_n_query = int(subset["N Query"].max())
    print(max_n_query, subset["Attacker"].unique())
    for k, attacker in enumerate([f"{attack}_MLP", f"{attack}_RF+", f"{attack}_RF"]):
        subset_ = subset[subset["Attacker"] == attacker]
        for method in ["OCEAN", "DiCE"]:
            subset_s = subset_[subset_["Oracle"] == method]
            mean_fidelities = {1: 0.0}
            for q in range(20, max_n_query + 1, 20):
                fids = []
                for _, row in subset_s.iterrows():
                    fid = compute_fidelity(row, q)
                    fids.append(fid)
                mean_fidelities[q] = np.mean(fids)
            # "OCEAN" dashed line, "DiCE" solid line
            if method == "OCEAN":
                plt.plot(
                    list(mean_fidelities.keys()),
                    list(mean_fidelities.values()),
                    label=attacker + "_" + method,
                    color=colors[attacker],
                    linewidth=3,
                    linestyle="--",
                    alpha=0.9 - 0.1 * k,
                )
                # legend_elements.append(plt.Line2D([0], [0], color=colors[attacker], linestyle="--", label=attacker+"_"+method))
            else:
                plt.plot(
                    list(mean_fidelities.keys()),
                    list(mean_fidelities.values()),
                    label=attacker + "_" + method,
                    color=colors[attacker],
                    linewidth=3,
                    alpha=0.9 - 0.1 * k,
                )
                # legend_elements.append(plt.Line2D([0], [0], color=colors[attacker], linestyle="-", label=attacker+"_"+method))
    plt.ylim(0.1, 1.0)
    if xlogscale:
        # ax.set_xlim(10, max_n_query)
        plt.xscale("log")
    plt.grid()
    if ylogscale:
        plt.yscale("log")
    plt.xlabel("#Queries", fontsize="15")
    plt.ylabel("Mean Fidelity", fontsize="15")
    plt.legend(fontsize="15")
    plt.savefig(f"{plots_folder}/RF_{attack}.pdf", bbox_inches="tight")
    plt.close()


df = read_experiments(experiments_folder)
plot_meanFid_vs_query(df)
query_vs_nodes_Estimators(df)
plot_CF_surrogates(df, dualCF = False, xlogscale=True, ylogscale=False)
plot_CF_surrogates(df, dualCF = True, xlogscale=True, ylogscale=False)

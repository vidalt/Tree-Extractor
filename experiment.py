# Load packages
import os as os
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load OCEAN modules
from OCEAN.src.DatasetReader import DatasetReader

from utils.CounterFactualExp import CounterFactualOracle
from utils.DiCEOracle import DiCECounterfactualGenerator

from utils.NodeIdOracle import NodeIdOracle

from experiments.parameters import DATASETS, ATTACKS, SEEDS, DEPTH
from utils.utils import (
    generate_uniform_data,
    train_model_with_best_alpha,
    round_decision_tree_thresholds,
    get_n_splits,
    write_results,
    Path_finding_attack_wrapper,
    TRA_attack_wrapper,
    surrogate_attack_wrapper,
)

# Load MLPClassifier
import json
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
write_folder = "experiments/results"


def get_parameters(experiment):
    """
    This function returns the parameters of the experiment.
    Args :
        experiment : the number of the experiment
    Returns :
        parameters : the parameters of the experiment
    """
    assert experiment >= 0 and experiment < 100, (
        f"Invalid experiment number {experiment + 1} must be between 1 and 100."
    )
    parameters = {}
    parameters["Dataset"] = DATASETS[experiment // (len(ATTACKS) * len(SEEDS))]
    parameters["Attacker"] = ATTACKS[(experiment // len(SEEDS)) % len(ATTACKS)]
    parameters["Seed"] = SEEDS[experiment % len(SEEDS)]
    return parameters


def main(args):
    parameters = get_parameters(args.experiment - 1)
    np.random.seed(parameters["Seed"])
    print("Running experiment", args.experiment, "with parameters", parameters)
    norm = 2  # euclidean norm
    data_dict = {}
    new_datasetPath = parameters["Dataset"]
    data_dict["Dataset"] = new_datasetPath
    data_dict["Attacker"] = parameters["Attacker"]
    data_dict["Seed"] = parameters["Seed"]
    reader = DatasetReader(new_datasetPath, SEED=parameters["Seed"])
    # put the acctionability to FREE for all the features
    reader.featuresActionnability = np.array(
        ["FREE"] * len(reader.featuresActionnability)
    )
    # Train a random forest using sklearn

    data_dict["Depth"] = []
    fidelity_data = generate_uniform_data(reader, n_samples=3000)
    X = reader.X_train
    y = reader.y_train
    X_test = reader.X_test
    y_test = reader.y_test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=parameters["Seed"]
    )  # 0.2 of the full dataset
    # Train a random forest using sklearn (The MILP formulation is better in OCEAN)
    rf = RandomForestClassifier(n_estimators=1, random_state=parameters["Seed"])
    rf.fit(reader.X_train, reader.y_train)
    fidelity_data = pd.DataFrame(fidelity_data, columns=rf.feature_names_in_)

    # Resume process for the experiment
    last_depth = 0
    if os.path.exists(f"{write_folder}/exp_{args.experiment}.json"):
        data_dict = json.load(open(f"{write_folder}/exp_{args.experiment}.json"))
        # getting the last depth
        last_depth = data_dict["Depth"][-1]["max_depth"]

    if last_depth is None:
        print("Experiment already done")
        return

    for i, depth in enumerate(DEPTH):
        if depth is not None and depth <= last_depth:
            continue
        print("Depth : ", depth)
        data_dict["Depth"].append({"max_depth": depth})
        # clf = train_model(reader.X_train.values, reader.y_train.values,
        #                                max_depth=depth, random_state=parameters['Seed'])
        clf, best_ccp_alpha = train_model_with_best_alpha(
            X_train,
            y_train,
            X_val,
            y_val,
            random_state=parameters["Seed"],
            max_depth=depth,
        )
        round_decision_tree_thresholds(clf)
        rf.estimators_ = [clf]

        data_dict["Depth"][i]["Train accuracy"] = clf.score(X_train, y_train)
        data_dict["Depth"][i]["Test accuracy"] = clf.score(X_test, y_test)
        data_dict["Depth"][i]["Val accuracy"] = clf.score(X_val, y_val)
        n = clf.tree_.node_count
        data_dict["Depth"][i]["N_nodes"] = n
        data_dict["Depth"][i]["N_features"] = clf.tree_.n_features
        data_dict["Depth"][i]["Norm"] = norm
        data_dict["Depth"][i]["N_splits"] = get_n_splits(clf)

        if parameters["Attacker"] == "PathFinding":
            NodeOracle = NodeIdOracle(
                clf,
                reader.featuresType,
                reader.featuresPossibleValues,
                FeaturesNames=[
                    str(column) for column in reader.data.columns if column != "Class"
                ],
            )
            fidelity, N_query, attack_time, fidelity_test = Path_finding_attack_wrapper(
                NodeOracle, fidelity_data.values, test_data=X_test.values
            )
            data_dict["Depth"][i]["fidelity_test_data"] = fidelity_test
            data_dict["Depth"][i]["fidelity"] = fidelity
            data_dict["Depth"][i]["N_query"] = N_query
            data_dict["Depth"][i]["attack_time"] = attack_time

        else:
            Oracle = CounterFactualOracle(
                rf,
                reader.featuresType,
                reader.featuresPossibleValues,
                norm=norm,
                n_classes=len(np.unique(reader.y_train.values)),
                SEED=parameters["Seed"],
            )
            continuous_features = reader.data.drop(columns=["Class"]).columns.to_list()
            outcome_name = "Class"
            DiceOracle = DiCECounterfactualGenerator(
                rf,
                reader.data,
                outcome_name=outcome_name,
                continuous_features=continuous_features,
                n_classes=len(np.unique(reader.y_train.values)),
                seed=parameters["Seed"],
            )
        dualCF = True if parameters["Attacker"] == "DualCF" else False
        if parameters["Attacker"].split("_")[0] == "TRA":
            Strategy = "BFS"
            fidelity, N_query, attack_time, fidelity_over_queries, fidelity_test = (
                TRA_attack_wrapper(
                    Oracle,
                    reader,
                    norm,
                    fidelity_data.values,
                    test_data=X_test.values,
                    strategy=Strategy,
                )
            )
            data_dict["Depth"][i]["fidelity_test_data"] = fidelity_test
            data_dict["Depth"][i]["fidelity"] = fidelity
            data_dict["Depth"][i]["N_query"] = N_query
            data_dict["Depth"][i]["attack_time"] = attack_time
            data_dict["Depth"][i]["fidelity_over_queries"] = fidelity_over_queries

        elif "CF" in parameters["Attacker"]:
            dc = surrogate_attack_wrapper(
                Oracle,
                reader,
                fidelity_data,
                max_queries=50 * n,
                dualcf=dualCF,
                seed=parameters["Seed"],
                test_data=X_test,
                depth=depth,
                ccp_alpha=best_ccp_alpha,
            )
            data_dict["Depth"][i]["performances_OCEAN"] = dc
            try:
                dc = surrogate_attack_wrapper(
                    DiceOracle,
                    reader,
                    fidelity_data,
                    max_queries=50 * n,
                    dualcf=dualCF,
                    seed=parameters["Seed"],
                    test_data=X_test,
                    depth=depth,
                    ccp_alpha=best_ccp_alpha,
                )
            except Exception as e:
                print(f"Error with DiCE: {e}")
                dc = {}
            data_dict["Depth"][i]["performances_DiCE"] = dc

        elif parameters["Attacker"] == "PathFinding":
            pass

        else:
            print("Unknown attacker")

        write_results(data_dict, args, write_folder)
    print(data_dict)
    print("End of the experiment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Define the --experiment argument
    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
        required=True,
        help="The number of the experiment to run (1-120).",
    )

    # Parse the arguments
    args = parser.parse_args()
    main(args)

# Load packages
import os as os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Load OCEAN modules
from OCEAN.src.DatasetReader import DatasetReader

from utils.CounterFactualExp import CounterFactualOracle
from utils.DiCEOracle import DiCECounterfactualGenerator

from experiments.parameters import DATASET_RF, ATTACKS_RF, SEEDS, ESTIMATORS
from utils.utils import (
    generate_uniform_data,
    train_model_with_best_alpha,
    round_RF_thresholds,
    get_n_splitsRF,
    write_results,
    TRA_attack_wrapper,
    surrogate_attack_wrapper,
)

# Load MLPClassifier
import json
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

DATASET = DATASET_RF
ATTACKS = ATTACKS_RF

write_folder = "experiments/RFs"


def get_parameters(experiment):
    """
    This function returns the parameters of the experiment.
    Args :
        experiment : the number of the experiment
    Returns :
        parameters : the parameters of the experiment
    """
    assert experiment >= 0 and experiment < 20, (
        f"Invalid experiment number {experiment + 1} must be between 1 and 20."
    )
    parameters = {}
    parameters["Dataset"] = DATASET
    parameters["Attacker"] = ATTACKS[experiment // len(SEEDS)]
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

    data_dict["Estimators"] = []
    fidelity_data = generate_uniform_data(reader, n_samples=3000)

    X = reader.X_train
    y = reader.y_train
    X_test = reader.X_test
    y_test = reader.y_test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=parameters["Seed"]
    )  # 0.2 of the full dataset

    fidelity_data = pd.DataFrame(fidelity_data, columns=reader.X_train.columns)

    # Resume process for the experiment
    last_estimators = 0
    if os.path.exists(f"experiments/RFs/exp_{args.experiment}.json"):
        data_dict = json.load(open(f"experiments/RFs/exp_{args.experiment}.json"))
        # getting the last depth
        last_estimators = data_dict["Estimators"][-1]["n_estimators"]

    if last_estimators is None:
        print("Experiment already done")
        return

    for i, estimators in enumerate(ESTIMATORS):
        print("Estimators : ", estimators)
        # to resume the experiment
        if estimators is not None and estimators <= last_estimators:
            continue

        rf, best_ccp_alpha = train_model_with_best_alpha(
            X_train,
            y_train,
            X_val,
            y_val,
            random_state=parameters["Seed"],
            estimators=estimators,
        )
        round_RF_thresholds(rf)

        data_dict["Estimators"].append({"n_estimators": estimators})
        data_dict["Estimators"][i]["Train accuracy"] = rf.score(X_train, y_train)
        data_dict["Estimators"][i]["Test accuracy"] = rf.score(X_test, y_test)
        data_dict["Estimators"][i]["Val accuracy"] = rf.score(X_val, y_val)
        data_dict["Estimators"][i]["N_nodes"] = sum(
            [clf.tree_.node_count for clf in rf.estimators_]
        )
        data_dict["Estimators"][i]["N_features"] = rf.n_features_in_
        data_dict["Estimators"][i]["N_nodes"] = sum(
            [clf.tree_.node_count for clf in rf.estimators_]
        )
        data_dict["Estimators"][i]["Norm"] = norm
        data_dict["Estimators"][i]["N_splits"] = get_n_splitsRF(rf)

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
            data_dict["Estimators"][i]["fidelity_test_data"] = fidelity_test
            data_dict["Estimators"][i]["fidelity"] = fidelity
            data_dict["Estimators"][i]["N_query"] = N_query
            data_dict["Estimators"][i]["attack_time"] = attack_time
            data_dict["Estimators"][i]["fidelity_over_queries"] = fidelity_over_queries

        elif "CF" in parameters["Attacker"]:
            dc = surrogate_attack_wrapper(
                Oracle,
                reader,
                fidelity_data,
                max_queries=3000,
                dualcf=dualCF,
                seed=parameters["Seed"],
                test_data=X_test,
                estimators=estimators,
                ccp_alpha=best_ccp_alpha,
            )
            data_dict["Estimators"][i]["performances_OCEAN"] = dc
            try:
                dc = surrogate_attack_wrapper(
                    DiceOracle,
                    reader,
                    fidelity_data,
                    max_queries=3000,
                    dualcf=dualCF,
                    seed=parameters["Seed"],
                    test_data=X_test,
                    estimators=estimators,
                    ccp_alpha=best_ccp_alpha,
                )
            except Exception as e:
                print(f"Error with DiCE: {e}")
                dc = {}
            data_dict["Estimators"][i]["performances_DiCE"] = dc
        else:
            print("Unknown attacker")

        write_results(data_dict, args, write_folder)
    print("End of the experiment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Define the --experiment argument
    parser.add_argument(
        "--experiment",
        type=int,
        required=False,
        default=21,
        help="No of the experiment to run",
    )
    # Parse the arguments
    args = parser.parse_args()
    main(args)

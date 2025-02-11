from attacks.TreeReconstructionAttack import TRAttack
from attacks.SurrogateAttacks import SurrogateAttack
from attacks.StealML.tree_stealer import TreeExtractor
import numpy as np
import time
import json
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from OCEAN.src.CounterFactualParameters import FeatureType


def train_model(X_train, y_train, random_state=0, max_depth=None):
    """
    This function trains a decision tree classifier.
    Args :
        X_train : the training data
        y_train : the training labels
        random_state : the random state
        max_depth : the maximum depth of the tree
    Returns :
        clf : the classifier
    """
    clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    clf.fit(X_train, y_train)
    return clf


def train_model_with_best_alpha(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state=0,
    max_depth=None,
    estimators=None,
):
    """
    This function trains a decision tree classifier using the best alpha.
    Args :
        X_train : the training data
        y_train : the training labels
        X_val : the validation data
        y_val : the validation labels
        random_state : the random state
        max_depth : the maximum depth of the tree
    Returns :
        best_clf : the best classifier
    """
    best_clf = None
    best_accuracy = 0.0
    alphas = np.linspace(0.0, 0.2, 50)
    for alpha in alphas:
        if estimators is not None:
            clf = RandomForestClassifier(
                random_state=random_state, ccp_alpha=alpha, n_estimators=estimators
            )
        else:
            clf = DecisionTreeClassifier(
                random_state=random_state, max_depth=max_depth, ccp_alpha=alpha
            )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        # print("Alpha = {:.4f}, Accuracy = {:.4f}, Node count = {}".format(alpha, accuracy, clf.tree_.node_count))
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
            best_clf = clf
    print(
        "Best choosed : Alpha = {:.4f} Accuracy = {:.4f}".format(
            best_alpha, best_accuracy
        )
    )
    return best_clf, best_alpha


def generate_uniform_data(reader, n_samples=1000):
    """
    Generate uniform grid data for to test the fidelity.
    Args :
        reader : the dataset reader (OCEAN dataset reader)
        n_samples : the number of samples to generate
    Returns :
        data : the generated data (np.array)
    """
    n_features = len(reader.featuresType)
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        if reader.featuresType[i].value == FeatureType.Numeric.value:
            data[:, i] = np.random.uniform(0.0, 1.0, n_samples)
        elif reader.featuresType[i].value == FeatureType.Binary.value:
            data[:, i] = np.random.choice([0, 1], n_samples)
        else:
            data[:, i] = np.random.choice(reader.featuresPossibleValues[i], n_samples)
    return data


def round_decision_tree_thresholds(clf, n_digits=6):
    """
    Round the decision tree thresholds.
    Args :
        clf : the decision tree classifier
        n_digits : the number of digits to round
    """
    if clf.tree_.n_features == 0:
        return clf
    for i, t in enumerate(clf.tree_.threshold):
        clf.tree_.threshold[i] = np.float32(round(t, n_digits))


def round_RF_thresholds(rf, n_digits=6):
    """
    Round the random forest trees thresholds.
    Args :
        rf : the random forest classifier
        n_digits : the number of digits to round
    """
    for clf in rf.estimators_:
        for i, t in enumerate(clf.tree_.threshold):
            clf.tree_.threshold[i] = np.float32(round(t, n_digits))


def get_n_splitsRF(rf):
    """
    Get the number of different splits (distinct threshold values) per feature in a forest.

    Args:
        rf: RandomForestClassifier - The random forest classifier.

    Returns:
        dict: A dictionary where keys are feature indices and values are the number of distinct thresholds for that feature.
    """
    splits = {}
    for clf in rf.estimators_:
        clf_splits = get_n_splits(clf)
        for feat, n_splits in clf_splits.items():
            if feat not in splits:
                splits[feat] = 0
            splits[feat] += n_splits
    return splits


def get_n_splits(clf):
    """
    Get the number of different splits (distinct threshold values) per feature in a decision tree.

    Args:
        clf: DecisionTreeClassifier - The decision tree classifier.

    Returns:
        dict: A dictionary where keys are feature indices and values are the number of distinct thresholds for that feature.
    """
    splits = {}
    n_nodes = clf.tree_.node_count
    feature = clf.tree_.feature  # Array of feature indices for each node
    threshold = clf.tree_.threshold  # Array of thresholds for each node

    for node_idx in range(n_nodes):
        # Ignore leaf nodes (feature index == -2)
        if feature[node_idx] != -2:
            feat_idx = feature[node_idx]
            thresh = threshold[node_idx]
            if feat_idx not in splits:
                splits[feat_idx] = set()  # Use a set to store unique thresholds
            splits[feat_idx].add(thresh)

    # Convert sets to counts
    splits = {str(feat): int(len(thresh_set)) for feat, thresh_set in splits.items()}
    return splits


def TRA_attack_wrapper(
    Oracle,
    reader,
    norm,
    fidelity_data,
    strategy="BFS",
    test_data=None,
):
    """
    Perform the TRA attack and return the fidelity and the number of queries.
    Args :
        Oracle : the counterfactual oracle
        reader : the dataset reader
        norm : the norm used for the attack
    Returns :
        fidelity : the fidelity of the attack
        N_query : the number of queries
    """
    attacker = TRAttack(
        Oracle,
        FeaturesType=reader.featuresType,
        FeaturesPossibleValues=reader.featuresPossibleValues,
        ObjNorm=norm,
        strategy=strategy,
        verbose=False,
    )
    start = time.time()
    attacker.attack(
        max_budget=200000,
        Compute_fidelity=True,
        fidelity_each=20,
        fidelity_data=fidelity_data,
    )
    end_time = time.time() - start
    fidelity = attacker.compute_fidelity(
        Oracle.classifier, fidelity_data, verbose=False
    )
    fidelity_test = attacker.compute_fidelity(
        Oracle.classifier, test_data, verbose=False
    )
    N_query = attacker.niter
    fidelity_over_queries = attacker.fidelity_over_queries
    return fidelity, N_query, end_time, fidelity_over_queries, fidelity_test


def surrogate_attack_wrapper(
    Oracle,
    reader,
    fidelity_data,
    max_queries=1000,
    dualcf=False,
    seed=0,
    test_data=None,
    depth=None,
    ccp_alpha=0.0,
    estimators=None,
):
    """
    Perform the surrogate attack and return the fidelity and the number of queries.
    Args :
        Oracle : the counterfactual oracle
        reader : the dataset reader
        max_queries : the maximum number of queries
        dualcf : if True, use the DualCF attack
    Returns :
        fidelity : the fidelity of the attack
        N_query : the number of queries
    """
    Surrogates = (
        {
            "MLP": MLPClassifier(random_state=seed, hidden_layer_sizes=(20, 20)),
            "DT": DecisionTreeClassifier(random_state=seed),
            "DT+": DecisionTreeClassifier(
                random_state=seed, ccp_alpha=ccp_alpha, max_depth=depth
            ),
        }
        if estimators is None
        else {
            "MLP": MLPClassifier(random_state=seed, hidden_layer_sizes=(20, 20)),
            "RF": RandomForestClassifier(random_state=seed),
            "RF+": RandomForestClassifier(
                random_state=seed, ccp_alpha=ccp_alpha, n_estimators=estimators
            ),
        }
    )
    dc = {}
    attacker = SurrogateAttack(
        Oracle,
        FeaturesType=reader.featuresType,
        FeaturesPossibleValues=reader.featuresPossibleValues,
        max_queries=max_queries,
        dualcf=dualcf,
    )
    start = time.time()
    X_train, y_train = attacker.generate_surrogate_training_data(
        surrogate_models=Surrogates,
        fidelity_data=fidelity_data,
        compute_fidelity=True,
        verbose=False,
        fidelity_each=20,
    )
    end_time = time.time() - start
    for name, surrogate_clf in Surrogates.items():
        surrogate_clf.fit(X_train, y_train)
        y_pred = surrogate_clf.predict(fidelity_data)
        fidelity = np.mean(y_pred == Oracle.classifier.predict(fidelity_data))
        y_pred = surrogate_clf.predict(test_data)
        fidelity_test = np.mean(y_pred == Oracle.classifier.predict(test_data))
        N_query = max_queries
        dc[name] = {
            "fidelity_test_data": fidelity_test,
            "fidelity": fidelity,
            "N_query": N_query,
            "attack_time": end_time / len(list(Surrogates.keys())),
            "fidelity_over_queries": attacker.fidelity_over_queries[name],
        }
    return dc


def Path_finding_attack_wrapper(
    NodeOracle, fidelity_data, test_data=None, epsilon=1e-5
):
    """
    Perform the Path finding attack and return the fidelity and the number of queries.
    Args :
        NodeOracle : the NodeId oracle
        fidelity_data : the dataset to compute the fidelity on
        epsilon : the epsilon value used for the attack (granularity)
    Returns :
        fidelity : the fidelity of the attack
        N_query : the number of queries
    """
    attacker = TreeExtractor(NodeOracle, epsilon=epsilon)
    start = time.time()
    all_leaves = attacker.extract()
    end_time = time.time() - start
    formatted_data = [
        {
            feature.name: fidelity_data[i, j]
            for j, feature in enumerate(attacker.features)
        }
        for i in range(len(fidelity_data))
    ]
    formatted_test_data = [
        {feature.name: test_data[i, j] for j, feature in enumerate(attacker.features)}
        for i in range(len(test_data))
    ]
    fidelity = compute_fidelity_from_all_leaves(all_leaves, formatted_data, NodeOracle)
    fidelity_test = compute_fidelity_from_all_leaves(
        all_leaves, formatted_test_data, NodeOracle
    )
    N_query = attacker.queryCount
    return fidelity, N_query, end_time, fidelity_test


def compute_fidelity_from_all_leaves(all_leaves, data, NodeOracle):
    """
    Compute the fidelity of the attack from the leaves.
    Args :
        all_leaves : the leaves of the tree
        data : the data to predict
        NodeOracle : the NodeId oracle
    Returns :
        fidelity : the fidelity of the attack over the data
    """
    y = []
    y_pred = []
    for query in data:
        x = [list(query.values())]
        y.append(NodeOracle.getNodeId(x))
        for y_, predicates in all_leaves.items():
            all_satisfied = True
            for pred in predicates[0]:
                if not pred.is_valid(query[pred.feature.name]):
                    all_satisfied = False
                    break
            if all_satisfied:
                y_pred.append(y_)
                break
        assert all_satisfied

    y = np.array(y)
    y_pred = np.array(y_pred)
    fidelity = np.sum(y == y_pred) / len(y)

    return fidelity


def write_results(data_dict, args, write_folder):
    """Write results to a JSON file."""
    # File to save the results
    EXP_FILE = f"{write_folder}/exp_{args.experiment}.json"
    print("Writing results to", EXP_FILE)
    with open(EXP_FILE, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)
    print("Data has been written to", EXP_FILE)

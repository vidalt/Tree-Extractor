# In this file we implement the model extraction attack using counterfactual data as presented in the papers :
#  - "Model extraction from counterfactual explanations" by Aïvodji et al. (2020)
#  - "DualCF: Efficient Model Extraction Attack from Counterfactual Explanations" by Wang et al. (2022)

import numpy as np
import pandas as pd
from OCEAN.src.CounterFactualParameters import FeatureType


class SurrogateAttack:
    """
    This class implements the model extraction attack using counterfactual data as presented in the papers :
    - "Model extraction from counterfactual explanations" by Aïvodji et al. (2020)
    - "DualCF: Efficient Model Extraction Attack from Counterfactual Explanations" by Wang et al. (2022)
    """

    def __init__(
        self,
        CounterfactualOracle,
        FeaturesType: list[str],
        FeaturesPossibleValues: list[tuple],
        max_queries: int = 10000,
        surrogate_data: pd.DataFrame = None,
        dualcf: bool = False,
    ):
        """
        Parameters
        ----------
        CounterfactualOracle : object
            The counterfactual oracle.
        FeaturesType : list[str]
            The type of each feature.
        FeaturesPossibleValues : list[tuple]
            The possible values of each feature. (This assume that the numerical features are normalized between 0 and 1)
        max_queries : int, optional
            The maximum number of queries allowed. The default is 10000.
        surrogate_data : pd.DataFrame, optional
            The surrogate data used to train the surrogate model. The default is None.
        dualcf : bool, optional
            If True, the DualCF attack is used. The default is False.
        """

        self.oracle = CounterfactualOracle
        self.max_queries = max_queries
        self.dualcf = dualcf
        if self.dualcf:
            self.max_queries = int(max_queries / 2)
        self.FeaturesType = FeaturesType
        self.__format_features_possible_values(FeaturesPossibleValues)
        if surrogate_data is not None:
            self.surrogate_data = surrogate_data
        else:
            self.surrogate_data = self.__generate_uniform_data()
        self.fidelity_over_queries = {}
        self.featureNames = self.oracle.classifier.feature_names_in_

    def __format_features_possible_values(self, FeaturesPossibleValues: list[tuple]):
        """
        Formats the features possible values.
        """
        self.FeaturesPossibleValues = FeaturesPossibleValues
        for i in range(len(FeaturesPossibleValues)):
            if (
                self.FeaturesType[i].value == FeatureType.Numeric.value
                or self.FeaturesType[i].value == FeatureType.Binary.value
            ):
                self.FeaturesPossibleValues[i] = [0.0, 1.0]

    def __generate_a_random_query(self):
        """
        Generates a random query.
        """
        query = []
        for j in range(len(self.FeaturesType)):
            if self.FeaturesType[j].value == FeatureType.Numeric.value:
                query.append(np.random.uniform(0.0, 1.0))
            else:
                query.append(np.random.choice(self.FeaturesPossibleValues[j]))
        return query

    def __generate_uniform_data(self):
        """
        Generates a uniform random dataset of size max_queries.
        """
        data = []
        for i in range(self.max_queries):
            data.append(self.__generate_a_random_query())
        return np.array(data)

    def generate_surrogate_training_data(
        self,
        surrogate_models: dict = None,
        fidelity_data: np.ndarray = None,
        compute_fidelity: bool = False,
        verbose: bool = False,
        fidelity_each: int = 10,
    ):
        """
        Generates the training data for the surrogate model.

        Returns
        -------
        final_data : np.array
            The final training data.
        labels : np.array
            The labels of the training data.
        """
        # check if the oracle is a DiCEOracle,
        if self.oracle.__class__.__name__ == "DiCECounterfactualGenerator":
            print("Using DiCE to generate counterfactuals")
            return self.generate_surrogate_training_data_DiCE(
                surrogate_models,
                fidelity_data,
                compute_fidelity,
                verbose,
                fidelity_each,
            )
        Queue = self.surrogate_data
        for model_name in surrogate_models.keys():
            self.fidelity_over_queries[model_name] = {}
        final_data = np.array([])
        labels = []
        while len(Queue) > 0:
            query = Queue[0]
            Queue = Queue[1:]
            x_cf, y_cf, y = self.oracle.getCounterFactual([query], return_label=True)
            queries = np.array([query])
            if self.dualcf and (len(final_data) <= self.max_queries):
                Queue = np.vstack((Queue, x_cf))
                final_data = (
                    np.vstack((final_data, queries)) if final_data.size else queries
                )
                labels += list(y)
            else:
                final_data = (
                    np.vstack((final_data, queries, x_cf))
                    if final_data.size
                    else np.vstack((queries, x_cf))
                )
                labels += list(y) + list(y_cf)
            if (
                compute_fidelity
                and len(final_data) != 0
                and len(final_data) % fidelity_each == 0
            ):
                for model_name, surrogate_model in surrogate_models.items():
                    data = pd.DataFrame(final_data, columns=self.featureNames)
                    fidelity = self.compute_fidelity_surrogate_model(
                        surrogate_model, data, labels, fidelity_data
                    )
                    self.fidelity_over_queries[model_name][len(final_data)] = fidelity
                    if verbose:
                        print("Fidelity of the surrogate model: ", fidelity)
        X = pd.DataFrame(final_data, columns=self.featureNames)
        y = np.array(labels)
        return X, y

    def generate_surrogate_training_data_DiCE(
        self,
        surrogate_models: dict = None,
        fidelity_data: np.ndarray = None,
        compute_fidelity: bool = False,
        verbose: bool = False,
        fidelity_each: int = 10,
    ):
        """
        Generates the training data for the surrogate model.
        Passes all the data once by batches to the DiCE model to get the counterfactuals.
        """
        Queue = self.surrogate_data
        for model_name in surrogate_models.keys():
            self.fidelity_over_queries[model_name] = {}
        final_data = np.array([])
        labels = []
        while len(Queue) > 0:
            queries = Queue[: min(fidelity_each, len(Queue))]
            Queue = Queue[fidelity_each:]
            x_cf, y_cf, y = self.oracle.getCounterFactual(queries, return_label=True)
            if self.dualcf and (len(final_data) <= self.max_queries):
                Queue = np.vstack((Queue, x_cf))
                final_data = (
                    np.vstack((final_data, queries)) if final_data.size else queries
                )
                labels += list(y)
            else:
                final_data = (
                    np.vstack((final_data, queries, x_cf))
                    if final_data.size
                    else np.vstack((queries, x_cf))
                )
                labels += list(y) + list(y_cf)
            if (
                compute_fidelity
                and len(final_data) != 0
                and len(final_data) % fidelity_each == 0
            ):
                for model_name, surrogate_model in surrogate_models.items():
                    data = pd.DataFrame(final_data, columns=self.featureNames)
                    fidelity = self.compute_fidelity_surrogate_model(
                        surrogate_model, data, labels, fidelity_data
                    )
                    self.fidelity_over_queries[model_name][len(final_data)] = fidelity
                    if verbose:
                        print("Fidelity of the surrogate model: ", fidelity)
        X = pd.DataFrame(final_data, columns=self.featureNames)
        y = np.array(labels)
        return X, y

    def compute_fidelity_surrogate_model(
        self,
        surrogate_model: object,
        X: np.ndarray,
        y: np.ndarray,
        fidelity_data: np.ndarray,
    ):
        """
        Trains the surrogate model.

        Parameters
        ----------
        surrogate_model : object
            The surrogate model. must have the methods fit and predict.
        X : np.array
            The training data.
        y : np.array
            The labels of the training data.
        fidelity_data : np.array
            The data used to compute the fidelity.

        Returns
        -------
        float
            The fidelity of the surrogate model.
        """
        surrogate_model.fit(X, y)
        y_pred = surrogate_model.predict(fidelity_data)
        y_true = self.oracle.classifier.predict(fidelity_data)
        return np.mean(y_pred == y_true)

from OCEAN.src.RfClassifierCounterFactual import RfClassifierCounterFactualMilp
import numpy as np
import copy
import gurobipy as gp


class CounterFactualOracle:
    def __init__(
        self,
        classifier: object,
        FeaturesType: list,
        FeaturesPossibleValues: list,
        n_classes: int = 2,
        norm: int = 2,
        SEED: int = 0,
    ):
        """
        Parameters
        ----------
        classifier : object, optional
            The classifier. The default is None.
        FeaturesType : list
            The type of each feature.
        FeaturesPossibleValues : list
            The possible values for each feature.
        n_classes : int, optional
            The number of classes. The default is 2.
        norm : int, optional
            The norm to use. The default is 2."""
        self.classifier = classifier
        self.FeaturesType = FeaturesType
        self.FeaturesPossibleValues = FeaturesPossibleValues
        self.norm = norm
        self.n_classes = n_classes
        self.epsilon = 1e-6
        self.env = gp.Env()
        self.env.setParam("OutputFlag", 0)
        self.env.setParam("Seed", SEED)
        self.env.setParam("FeasibilityTol", 1e-6)
        self.env.start()

    def getCounterFactual(
        self,
        sample: np.ndarray,
        currentLabel: int = None,
        BoundingBox: list = None,
        return_label: bool = False,
    ):
        """
        Get the counterfactual of the sample.
        Args:
            sample: the sample
            currentLabel: the current label of the sample
            BoundingBox: the bounding box of the features
        Returns:
            the counterfactual of the sample
        """
        if currentLabel is None:
            currentLabel = self.classifier.predict(sample)[0]

        return self.getOCEANCounterFactual(
            sample, currentLabel, BoundingBox, return_label
        )

    def getOCEANCounterFactual(
        self, sample, currentLabel=None, BoundingBox=None, return_label=False
    ):
        """
        Get the counterfactual of the sample using the OCEAN algorithm.
        Args:
            sample: the sample
            currentLabel: the current label of the sample
        Returns:
            the counterfactual of the sample
        """

        x_sol = copy.deepcopy(sample)
        previous_norm = np.linalg.norm(sample[0] - x_sol[0], ord=self.norm)
        label_cf = copy.deepcopy(currentLabel)
        strict_c = False
        for i in range(self.n_classes):
            if i == currentLabel:
                continue
            if i > currentLabel:  # Break ties in favor of the lowest class
                strict_c = True
            counterFactual = RfClassifierCounterFactualMilp(
                self.classifier,
                sample,
                outputDesired=i,
                objectiveNorm=self.norm,
                boundingBox=BoundingBox,
                featuresType=self.FeaturesType,
                featuresPossibleValues=self.FeaturesPossibleValues,
                gurobi_env=self.env,
                verbose=False,
                strictCounterFactual=strict_c,
            )
            counterFactual.buildModel()
            counterFactual.solveModel()
            new_norm = np.linalg.norm(
                sample[0] - counterFactual.x_sol[0], ord=self.norm
            )
            if new_norm > self.epsilon:
                if previous_norm < self.epsilon:
                    x_sol = counterFactual.x_sol
                    previous_norm = new_norm
                    label_cf = i
                elif new_norm < previous_norm:
                    x_sol = counterFactual.x_sol
                    previous_norm = new_norm
                    label_cf = i
        if return_label:
            return x_sol[0], [label_cf], [currentLabel]
        return x_sol[0], [label_cf]

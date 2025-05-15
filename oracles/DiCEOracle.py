import dice_ml  # type: ignore
from dice_ml import Dice  # type: ignore
import numpy as np
import pandas as pd


class DiCECounterfactualGenerator:
    def __init__(
        self,
        model: object,
        data: pd.DataFrame,
        outcome_name: str = "Class",
        method: str = "random",
        continuous_features: list = None,
        n_classes: int = 2,
        diversity_weight: float = 0.5,
        proximity_weight: float = 1.0,
        seed: int = 0,
    ):
        """
        Initialize the DiCE counterfactual generator.

        Parameters
        ----------
        model : object
            The trained model for which counterfactuals need to be generated.
        data : pd.DataFrame
            The dataset used for the model.
        method : str, optional
            Method to generate counterfactuals, e.g., 'random' or 'genetic'. Default is 'random'.
        diversity_weight : float, optional
            Weight for diversity in counterfactual generation. Default is 1.0.
        seed : int, optional
            Random seed for reproducibility. Default is 0.
        """
        self.model = dice_ml.Model(model=model, backend="sklearn")
        self.featureNames = model.feature_names_in_
        self.n_classes = n_classes
        self.continuous_features = continuous_features
        self.classifier = model
        self.data = dice_ml.Data(
            dataframe=data,
            continuous_features=self.continuous_features,
            outcome_name=outcome_name,
            model_type="classifier",
        )
        self.method = method
        self.diversity_weight = diversity_weight
        self.proximity_weight = proximity_weight
        self.seed = seed
        self.dice = Dice(self.data, self.model, method=self.method)
        np.random.seed(self.seed)
        self.outcome_name = outcome_name

    def check_instance(self, instance: dict):
        """
        Check if the instance is a dictionary and has the correct number of features.
        """
        if not isinstance(instance, dict):
            # raise ValueError("The instance must be a dictionary.")
            if len(instance[0]) != len(self.featureNames):
                raise ValueError(
                    f"The instance must have the same number of features ({len(instance[0])}) as the data ({len(self.featureNames)})."
                )
            if len(instance[0]) == 1:
                dc = dict(zip(self.featureNames, instance[0]))
                return pd.DataFrame([dc])
            else:
                return pd.DataFrame(instance, columns=self.featureNames)
        else:
            dc = instance
            return pd.DataFrame([instance])

    def getCounterFactual(
        self,
        instance: dict,
        total_CFs: int = 1,
        desired_class: str = "opposite",
        return_label: bool = False,
    ):
        """
        Generate counterfactuals for a given instance.

        Parameters
        ----------
        instance : dict
            The instance for which counterfactuals are to be generated.
        total_CFs : int, optional
            Number of counterfactuals to generate. Default is 1.
        desired_class : int or 'opposite',
            Desired class for the counterfactual. Default is 'opposite'.
        return_label : bool, optional
            Whether to return labels of counterfactuals along with the counterfactual instances. Default is False.

        Returns
        -------
        list or tuple
            If return_label is False, returns a list of counterfactual instances.
            If return_label is True, returns a tuple of counterfactual instances and their labels.
        """
        instance = self.check_instance(instance)
        explanation = self.dice.generate_counterfactuals(
            query_instances=instance,
            total_CFs=total_CFs,
            desired_class=desired_class,
            diversity_weight=self.diversity_weight,
            proximity_weight=self.proximity_weight,
            random_seed=self.seed,
        )
        cf_instances = explanation.cf_examples_list
        outputs = [cf.final_cfs_df for cf in cf_instances]
        outputs = pd.concat(outputs)
        cf_labels = outputs[self.outcome_name].values
        outputs = outputs.drop(columns=[self.outcome_name])
        x_cf = np.array(outputs.values)
        y_pred = self.model.model.predict(instance)
        if return_label:
            return x_cf, cf_labels, y_pred
        return x_cf, y_pred

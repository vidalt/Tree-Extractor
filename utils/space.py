from OCEAN.src.CounterFactualParameters import FeatureType
import numpy as np


class space:
    """
    This class represents a space.
    """

    def __init__(
        self,
        features_type: list,
        features_possible_values: list,
    ):
        """
        Parameters
        ----------
        features_type : list
            The types of the features.
        features_possible_values : list
            The possible values of the features.
        """
        self.features_type = features_type
        self.fpv = features_possible_values
        for i in range(len(self.features_type)):
            if (
                self.features_type[i].value == FeatureType.Discrete.value
                or self.features_type[i].value == FeatureType.Categorical.value
            ):
                self.fpv[i] = np.sort(self.fpv[i])

    def contains(
        self,
        x: np.ndarray,
        threshold: float = 1e-5,
    ):
        """
        Check if the point x is in the space.
        Args:
            x: the point to check.
            threshold: the threshold to use.
        Returns:
            True if x is in the space, False otherwise.
        """
        for i in range(len(x)):
            if self.features_type[i].value == FeatureType.Numeric.value:
                if (
                    x[i] < self.fpv[i][0] + threshold
                    or x[i] > self.fpv[i][1] - threshold
                ):
                    return False
            elif self.features_type[i].value == FeatureType.Binary.value:
                if int(x[i]) not in self.fpv[i]:
                    return False
            elif (
                self.features_type[i].value == FeatureType.Discrete.value
                or self.features_type[i].value == FeatureType.Categorical.value
            ):
                if x[i] not in self.fpv[i]:
                    return False
        return True

    def is_empty(self):
        """
        Check if the space is empty.
        Returns:
            True if the space is empty, False otherwise.
        """
        for i in range(len(self.features_type)):
            if self.features_type[i].value == FeatureType.Numeric.value:
                if self.fpv[i][0] > self.fpv[i][1]:
                    return True
            elif (
                self.features_type[i].value == FeatureType.Binary.value
                or self.features_type[i].value == FeatureType.Discrete.value
                or self.features_type[i].value == FeatureType.Categorical.value
            ):
                if len(self.fpv[i]) == 0:
                    return True
        return False

    def get_center_point(self):
        """
        Get the center point of the space.
        Returns:
            center_point: the center point.

        For binary features, we take the first value of the feature.
        """
        center_point = []
        for i in range(len(self.features_type)):
            if self.features_type[i].value == FeatureType.Numeric.value:
                center_point.append((self.fpv[i][0] + self.fpv[i][1]) / 2)
            elif self.features_type[i].value == FeatureType.Binary.value:
                center_point.append(self.fpv[i][0])
            else:
                center_point.append(self.fpv[i][int(len(self.fpv[i]) / 2)])
        return np.array(center_point)

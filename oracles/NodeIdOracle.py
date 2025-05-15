from attacks.StealML.feature import ContFeature, CatFeature
from OCEAN.src.CounterFactualParameters import FeatureType


class NodeIdOracle:
    def __init__(
        self,
        classifier: object,
        FeaturesType: list,
        FeaturesPossibleValues: list,
        FeaturesNames: list,
        n_classes: int = 2,
    ):
        """
        Parameters
        ----------
        classifier : Scikit-learn decision tree classifier
            The classifier.
        FeaturesType : list
            The type of each feature.
        FeaturesPossibleValues : list
            The possible values for each feature.
        FeaturesNames : list
            The names of the features.
        n_classes : int, optional
            The number of classes. The default is 2.
        """
        self.classifier = classifier
        self.FeaturesType = FeaturesType
        self.FeaturesPossibleValues = FeaturesPossibleValues
        self.FeaturesNames = FeaturesNames
        self.n_classes = n_classes
        self.epsilon = 1e-5
        self.encodings = {}

    def getFeatureNames(self):
        """
        Get the feature names.
        Returns:
            the feature names
        """
        featuresNames = []
        for i, feature in enumerate(self.FeaturesNames):
            if self.FeaturesType[i].value == FeatureType.Numeric.value:
                featuresNames.append(ContFeature(feature, feature, 0.0, 1.0))
            else:
                if self.FeaturesType[i].value == FeatureType.Binary.value:
                    featuresNames.append(CatFeature(feature, feature, [0, 1]))
                else:
                    featuresNames.append(
                        CatFeature(feature, feature, self.FeaturesPossibleValues[i])
                    )
        return featuresNames

    def getNodeId(self, sample: list):
        """
        Get the node id of the sample.
        Args:
            classifier: the classifier
            sample: the input sample
        Returns:
            the node id of the sample
        """
        path = self.classifier.decision_path(sample)
        path = path.indices
        node_id = path[-1]
        return node_id

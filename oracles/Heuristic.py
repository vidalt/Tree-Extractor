import numpy as np
from itertools import product, islice
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _check_candidate(args):
    """
    Fonction utilitaire qui sera picklée dans chaque worker.
    args = (model, candidate_tuple, target_label)
    """
    model, candidate, y = args
    # on renvoie candidate si y_pred != y, sinon None
    return candidate if model.predict([candidate])[0] != y else None


class HeuristicCounterfactualGenerator:
    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_types: list[str] = None,
        feature_values: list[list] = None,
        epsilon=1e-6,
        verbose=False,
        reader=None,
        SEED=0,
    ):
        """
        Initialize the counterfactual generator heuristic.

        Parameters
        ----------
        model : object
            A trained classifier with a predict method.
        training_data : np.ndarray
            Array containing the training data.
        epsilon : float, optional
            Threshold for convergence in the line search. Default is 1e-6.
        feature_types : list[str], optional
            List of feature types. Default is None, meaning all features are numeric.
            Allowed types: 'N' (numeric), 'B' (binary), 'D' (discrete), 'C' (categorical).
        feature_values : list[list], optional
            For features that are discrete ('D') or categorical ('C'), a list of the allowed values.
            Must be provided for these types.
        """
        self.classifier = model
        self.training_data = training_data
        self.epsilon = epsilon
        self.y_pred = model.predict(training_data)
        self._n_columns = training_data.shape[1]
        self.features = feature_types
        self.feature_values = self.__format_feature_values(
            feature_values,
            feature_types,
        )
        print("Feature types: ", self.features)
        self.verbose = verbose
        self.n_classes = len(np.unique(self.y_pred))
        np.random.seed(SEED)
        self.feature_levels = self.__get_feature_levels()
        print("Feature levels: ", self.feature_levels)

    def __format_feature_values(self, feature_values, feature_types):
        """
        Format the feature values based on the feature types.
        """
        formatted_values = []
        for i, feature_type in enumerate(feature_types):
            if feature_type in ["D", "C"]:
                formatted_values.append(sorted(list(feature_values[i])))
            else:
                formatted_values.append([0.0, 1.0])
        return formatted_values

    def __get_feature_levels(self):
        """
        Get the levels of each feature.
        """
        feature_levels = {f: [] for f in range(self._n_columns)}
        for estimator in self.classifier.estimators_:
            thresholds = estimator.tree_.threshold
            feature = estimator.tree_.feature
            for i in range(len(thresholds)):
                if thresholds[i] != -2.0:
                    feature_levels[feature[i]].append(thresholds[i])
        for i in range(self._n_columns):
            feature_levels[i] = np.unique(feature_levels[i])
            feature_levels[i].sort()
        n_combinations = 1
        for i in range(self._n_columns):
            n_combinations *= len(feature_levels[i]) + 1
        print("Maximum number of combinations: ", n_combinations)
        return feature_levels

    def getCounterFactual(
        self,
        instance,
        return_label: bool = False,
        BoundingBox=None,
    ):
        instance = np.array(instance)
        if len(instance.shape) != 1:
            instance = instance[0]
        if BoundingBox is None:
            BoundingBox = [[0.0, 1.0] for _ in range(len(self.features))]
        # print("Searching for a counterfactual in the bounding box: ", BoundingBox)
        x_cf = self.get_nearest_counterfactual(instance, bounding_box=BoundingBox)
        if x_cf is None:
            x_cf = instance.copy()
        y_pred = self.classifier.predict([instance])
        if return_label:
            y_cf = self.classifier.predict([x_cf])[0]
            return x_cf, y_cf, y_pred
        return x_cf, y_pred

    def get_nearest_counterfactual(self, instance, bounding_box=None):
        """
        Generate a counterfactual for a given instance with an optional bounding box.

        Parameters
        ----------
        instance : np.ndarray
            Array of feature values for a single instance.
        bounding_box : list or None, optional
            A list of [lower_bound, upper_bound] for each feature. If provided, the candidate
            counterfactual will be adjusted to lie within these bounds.

        Returns
        -------
        np.ndarray
            The counterfactual instance as an array of feature values.

        """
        x = instance
        original_prediction = self.classifier.predict([x])[0]

        # Find candidate instances that yield a different prediction.
        mask = self.y_pred != original_prediction
        candidates = self.training_data[mask]
        candidates = self.__postprocess_candidates(
            candidates,
            bounding_box,
            original_prediction,
        )
        if candidates is None:
            if self.verbose:
                print("No counterfactual candidate found in training data.")
            return None

        # Compute Euclidean distances and select the nearest candidate.
        distances = np.linalg.norm(candidates - x, axis=1)
        nearest_candidate = candidates[np.argmin(distances)].copy()
        cf = self.__process_candidate(
            nearest_candidate,
            x,
            bounding_box,
            original_prediction,
        )
        if cf is not None:
            return cf
        # Search for a candidate in the bounding box.
        candidate = self.__search_candidate_in_bbox(
            x,
            bounding_box,
            candidates,
        )
        if candidate is not None:
            return candidate
        return None

    def __postprocess_candidates(self, candidates, bounding_box, y):
        """
        Remove the candidates that are outside the bounding box.
        """
        if bounding_box is None:
            return candidates
        new_candidates = []
        for candidate in candidates:
            if not self.__out_bbox(candidate, bounding_box):
                new_candidates.append(candidate)
        if len(new_candidates) == 0:
            if self.verbose:
                print(
                    "No counterfactual candidate found in bounding box. Generating new candidates..."
                )
            new_candidates = self.__generate_candidates(bounding_box, y=y)
            if len(new_candidates) == 0:
                if self.verbose:
                    print("No counterfactual candidate found in bounding box.")
                return None
            else:
                return np.array(new_candidates)

        print("Found without sampling new candidates")
        return np.array(new_candidates)

    def __generate_candidates(self, bounding_box, y, n_candidates=1000):
        """
        Generate candidates within the bounding box.
        """
        eps = 1e-6
        per_feature_values = []
        for i, box in enumerate(bounding_box):
            lb, ub = box[0], box[-1]
            levels = self.feature_levels[i]
            vals = []
            if len(levels) == 0:
                vals.append(self.feature_values[i][0])
            elif self.features[i] == "N":
                for lvl in levels:
                    v = lvl - eps
                    if lb <= v <= ub:
                        vals.append(v)
                last_plus = levels[-1] + eps
                if lb <= last_plus <= ub:
                    vals.append(last_plus)
                if len(vals) == 0:
                    vals.append((lb + ub) / 2)
            else:
                below = self.feature_values[i][0]
                if lb <= below <= ub:
                    vals.append(below)
                for l1, l2 in zip(levels[:-1], levels[1:]):
                    for mid in self.feature_values[i]:
                        if l1 < mid <= l2 and lb <= mid <= ub:
                            vals.append(mid)
                            break
                above = self.feature_values[i][-1]
                if lb <= above <= ub:
                    vals.append(above)
            if len(vals) == 0:
                raise ValueError(
                    f"Aucun candidat généré pour la feature {i} dans {box}"
                )
            assert len(vals) <= len(levels) + 1, (
                f"Il y a {len(vals)} candidats générés {vals} pour la feature {i} (type={self.features[i]}) dans {box}"
            )
            # shuffle vals
            vals = np.array(list(set(vals)))
            np.random.shuffle(vals)
            per_feature_values.append(list(vals))

        # 2) On crée un générateur pour toutes les combinaisons
        combo_gen = product(*per_feature_values)

        # 3) Pool de workers
        n_workers = cpu_count()
        pool = Pool(processes=n_workers)
        try:
            CHUNK_SIZE = 3000
            N_combs = n_candidates // n_workers
            for _ in tqdm(range(N_combs), desc="Generating candidates..."):
                chunk = list(islice(combo_gen, CHUNK_SIZE))
                if not chunk:
                    break  # plus rien à traiter

                args = [(self.classifier, candidate, y) for candidate in chunk]

                results = pool.map(_check_candidate, args)

                for res in results:
                    if res is not None:
                        print("######## FOUND A COUNTERFACTUAL #########")
                        return [res]
            return []
        finally:
            pool.close()
            pool.join()

    def __search_candidate_in_bbox(self, bounding_box, candidates):
        """
        Search for a counterfactual candidate within the bounding box.

        Parameters
        ----------
        x : np.ndarray
            The original instance.
        bounding_box : list
            A list of [lower_bound, upper_bound] for each feature.

        Returns
        -------
        np.ndarray
            The counterfactual candidate instance.

        """
        for candidate in candidates:
            if not self.__out_bbox(candidate, bounding_box):
                return candidate
        return None

    def __process_candidate(
        self,
        nearest_candidate,
        x,
        bounding_box,
        original_prediction,
    ):
        """
        Process the nearest candidate to generate a valid counterfactual.

        Parameters
        ----------
        nearest_candidate : np.ndarray
            The nearest candidate instance.
        x : np.ndarray
            The original instance.
        bounding_box : list or None
            A list of [lower_bound, upper_bound] for each feature.

        Returns
        -------
        np.ndarray

        """

        # Refine candidate using feature-specific line search with the optional bounding box.
        counterfactual = nearest_candidate.copy()
        f = 0
        while f < self._n_columns:
            counterfactual, optimized = self.__line_search_feature(x, counterfactual, f)
            f = 0 if optimized else f + 1
        if self.__out_bbox(counterfactual, bounding_box):
            return None
        # Verify that the counterfactual flips the prediction.
        new_prediction = self.classifier.predict([counterfactual])[0]
        if new_prediction == original_prediction:
            raise ValueError("Failed to generate a valid counterfactual.")
        return counterfactual

    def __out_bbox(self, x_cf, bbox, return_verbose=False):
        """
        Check if the counterfactual is within the bounding box constraints.

        Parameters
        ----------
        x_cf : array-like
            Counterfactual instance.
        bbox : list or None
            Bounding box constraints.

        Returns
        -------
        bool
            True if the counterfactual is outside the bounding box, False otherwise.
        """
        num_eps = 1e-10
        if bbox is None:
            return False
        for i in range(len(x_cf)):
            if self.features[i] == "N":
                if (
                    x_cf[i] < bbox[i][0] - num_eps  # equals to the TRA eps value
                    or x_cf[i] > bbox[i][-1] + num_eps
                ):
                    if return_verbose:
                        print(
                            f"Feature {i} {self.features[i]} is outside the bounding box: {x_cf[i]} not in {bbox[i]}"
                        )
                    return True
            else:
                if x_cf[i] < bbox[i][0] or x_cf[i] > bbox[i][-1]:
                    if return_verbose:
                        print(
                            f"Feature {i} {self.features[i]} is outside the bounding box: {x_cf[i]} not in {bbox[i]}"
                        )
                    return True
        return False

    def __line_search_feature(self, x, cf, f):
        """
        Perform a line search on feature f to adjust the counterfactual until it yields a different prediction.

        Parameters
        ----------
        cf : array-like
            Current counterfactual.
        x : array-like
            Original instance.
        f : int
            Feature index to optimize.

        Returns
        -------
        tuple
            (Optimized counterfactual, flag indicating whether optimization occurred)
        """
        if self.verbose:
            print(f"Line search for feature {f}: {cf[f]} -> {x[f]}")
        optimized = False
        y_x = self.classifier.predict([x])[0]
        x_tmp = x.copy()
        cf_tmp = cf.copy()
        if cf_tmp[f] == x[f]:
            return cf_tmp, optimized
        cf_tmp[f] = x[f]
        if self.classifier.predict([cf_tmp])[0] != y_x:
            if self.verbose:
                print(
                    f"\tFeature {f} optimized to {cf_tmp[f]}, y_cf = {self.classifier.predict([cf_tmp])[0]}, y_x = {y_x}"
                )
            return cf_tmp, True
        else:
            cf_tmp = cf.copy()
        distance = self._get_distance(f, cf_tmp[f], x_tmp[f])
        if distance < self.epsilon:
            return cf_tmp, optimized
        while distance > self.epsilon:
            mid_val = self._get_midpoint(f, cf_tmp[f], x_tmp[f])
            x_mid = cf_tmp.copy()
            x_mid[f] = mid_val
            y_mid = self.classifier.predict([x_mid])[0]
            if y_mid != y_x:
                cf_tmp[f] = mid_val
                optimized = True
            else:
                x_tmp[f] = mid_val
            distance = self._get_distance(f, cf_tmp[f], x_tmp[f])
        if optimized:
            if self.verbose:
                print(f"\tFeature {f} optimized to {cf_tmp[f]}")
        return cf_tmp, optimized

    def _get_distance(self, feature, a, b):
        """
        Compute the distance between two feature values.
        For continuous features, this is the absolute difference.
        For discrete features, it is the difference in indices within the permitted range.
        """
        if self.features[feature] == "N":
            return abs(a - b)
        else:
            tmp = self.feature_values[feature]
            return abs(tmp.index(a) - tmp.index(b)) * self.epsilon / 1.5

    def _get_midpoint(self, feature, a, b):
        """
        Compute the midpoint between two feature values.
        For continuous features, returns the arithmetic mean.
        For discrete features, returns the value at the midpoint index in the permitted range.
        """
        if self.features[feature] == "N":
            return (a + b) / 2
        else:
            tmp = self.feature_values[feature]
            return tmp[(tmp.index(a) + tmp.index(b)) // 2]

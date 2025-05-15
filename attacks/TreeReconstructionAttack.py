import copy
import numpy as np
from OCEAN.src.CounterFactualParameters import FeatureType
from oracles.CounterFactualExp import CounterFactualOracle
from utils.ExtractedTree import DecisionTree, Node
from utils.space import space


class TRAttack:
    """
    This class implements the Tree Reconstruction Attack.
    """

    def __init__(
        self,
        Oracle: CounterFactualOracle,
        FeaturesType: list[str],
        FeaturesPossibleValues: list[tuple],
        ObjNorm: int = 2,
        verbose=False,
        strategy="DFS",
        data=None,
    ):
        """
        Parameters
        ----------
        clf : the classifier to attack.
        Oracle : the counterfactual oracle.
        FeaturesType : the types of the features.
        FeaturesPossibleValues : the possible values of the features.
        ObjNorm : the norm to use. The default is 2.
        verbose : if True, print the logs. The default is False.
        strategy : the strategy to use (DFS or BFS). The default is DFS.
        """
        self.name = "TreeReconstructionAttack"
        self.oracle = Oracle
        self.ObjNorm = ObjNorm
        self.FeaturesType = FeaturesType
        self.strategy = strategy
        self.__formatFeaturePossibleValues(FeaturesPossibleValues)
        self.verbose = verbose
        self.Tree = DecisionTree(None)
        self.DIV = []
        self.QueryList = []
        self.niter = 0
        self.first_leaf = False
        self.epsilon = 1e-5
        self.fidelity_over_queries = {}

    def __formatFeaturePossibleValues(
        self,
        FeaturesPossibleValues: list[tuple],
    ):
        """
        This function format the FeaturesPossibleValues to the format used by the counterfactual oracle.
        Args:
            FeaturesPossibleValues: the possible values for each feature (for continuous,
                                discrete and categorical featues :(min, max)).
                                For binary features: (0, 1)
        """
        formattedFPV = []
        for i in range(len(self.FeaturesType)):
            if self.FeaturesType[i].value == FeatureType.Numeric.value:
                formattedFPV.append([0.0, 1.0])
            elif self.FeaturesType[i].value == FeatureType.Binary.value:
                formattedFPV.append(np.array([0, 1]))
            elif self.FeaturesType[i].value == FeatureType.Discrete.value:
                formattedFPV.append(np.array(FeaturesPossibleValues[i]))
            elif self.FeaturesType[i].value == FeatureType.Categorical.value:
                formattedFPV.append(np.array(FeaturesPossibleValues[i]))
        self.FeaturesPossibleValues = formattedFPV

    def __split_space(
        self,
        initial_space: space,
        d_var: int,
        decision_value: int | float,
        d_operator: str,
    ):
        """
        Split the space into two subspaces.
        Args:
            initial_space: Space
                the initial space to split.
            d_var: int
                the decision variable (or feature).
            decision_value: float
                the threshold value.
            d_operator: str
                the operator ('l' or 'g') to use.
        Returns:
            right_space: Space
                the right space. The space that contains the query point.
            left_space: Space
                the left space. The space that contains the counterfactual.
        """
        # Copy the initial space
        right_space = space(self.FeaturesType, copy.deepcopy(initial_space.fpv))
        left_space = space(self.FeaturesType, copy.deepcopy(initial_space.fpv))
        # Split the space
        # The right space contains the query point
        # The left space contains the counterfactual
        if self.FeaturesType[d_var].value == FeatureType.Numeric.value:
            if d_operator == "l":
                right_space.fpv[d_var][1] = decision_value - self.epsilon
                left_space.fpv[d_var][0] = decision_value + self.epsilon
            else:
                right_space.fpv[d_var][0] = decision_value + self.epsilon
                left_space.fpv[d_var][1] = decision_value - self.epsilon
        else:
            if d_operator == "l":
                right_space.fpv[d_var] = right_space.fpv[d_var][
                    right_space.fpv[d_var] < decision_value - self.epsilon
                ]
                left_space.fpv[d_var] = left_space.fpv[d_var][
                    left_space.fpv[d_var] >= decision_value - self.epsilon
                ]
            else:
                right_space.fpv[d_var] = right_space.fpv[d_var][
                    right_space.fpv[d_var] > decision_value + self.epsilon
                ]
                left_space.fpv[d_var] = left_space.fpv[d_var][
                    left_space.fpv[d_var] <= decision_value + self.epsilon
                ]
        return right_space, left_space

    def __get_changed_features(
        self,
        x: np.ndarray,
        x_cf: np.ndarray,
    ):
        """
        Get the features that changed between x and x_cf.
        We return a list of tuples (index, decision)
        decision = 'l' means if the feature value of x is less than x_cf
                    and so forall x' if x'[index] < x_cf[index] then f(x') = f(x)
                    (if we are at a leaf node)
        decision = 'g' means if the feature value of x is greater than x_cf
                    and so forall x' if x'[index] > x_cf[index] then f(x') = f(x)
                    (if we are at a leaf node)
        Args:
            x: the point x
            x_cf: the counterfactual of x
        Returns:
            changed_features: list
                the list of tuples (index, decision)
        """
        changed_features = []
        for i in range(len(x)):
            changed, direction = self.__get_decision_direction(x, x_cf, i)
            if changed and self.Tree.root is not None:
                if (
                    self.Tree.root.space.features_type[i].value
                    == FeatureType.Numeric.value
                ):
                    changed_features.append((i, x_cf[i], direction))
                else:
                    step = (
                        self.Tree.root.space.fpv[i][1] - self.Tree.root.space.fpv[i][0]
                    ) / 2
                    if direction == "l":
                        changed_features.append((i, x_cf[i] - step, direction))
                    else:
                        changed_features.append((i, x_cf[i] + step, direction))
            elif changed:
                changed_features.append((i, x_cf[i], direction))
        return changed_features

    def __get_decision_direction(
        self,
        x: np.ndarray,
        x_cf: np.ndarray,
        i: int,
    ):
        """
        Get the decision direction of the feature i.
        Args:
            x: the point x.
            x_cf: the counterfactual of x.
            i: the feature index.
        Returns:
            decision: the decision direction (less 'l' or  greater 'g')
        """
        if x[i] - x_cf[i] > self.epsilon:
            return True, "g"
        elif x[i] - x_cf[i] < -self.epsilon:
            return True, "l"
        return False, None

    def __getBoundingBox(self, space: space):
        """
        Get the bounding box of the space.
        Args:
            space: the space to get the bounding box.
        Returns:
            BoundingBox: the bounding box (i.e Hypercube).
        """
        BoundingBox = copy.deepcopy(space.fpv)
        for i in range(len(space.fpv)):
            BoundingBox[i] = [space.fpv[i][0], space.fpv[i][-1]]
        return BoundingBox

    def predict(self, X: np.ndarray):
        """
        Predict the label of the point X.
        Args:
            X: the point to predict.
        Returns:
            y: the predicted label.
        """
        return self.Tree.predict(X)

    def compute_fidelity(
        self,
        clf: object,
        data: np.ndarray,
        verbose=False,
    ):
        """
        This function return the computed fidelity of this reconstructed model with the classifier clf on the data.
        Args:
            clf: the classifier to compute the fidelity with
            data: the data to compute the fidelity on. The data format should be OCEAN.src.DatasetReader
            verbose: if True, print the fidelity
            use_random: if True, use random to predict the data in balls attack
        """
        reconst_model_pred = self.predict(data)
        true_model_pred = clf.predict(data)
        fidelity = np.sum(
            np.array(reconst_model_pred, dtype=object)
            == np.array(true_model_pred, dtype=object)
        ) / len(true_model_pred)
        if verbose:
            print(f"Fidelity = {100 * fidelity:.4f}%")
        return fidelity

    def getLabelAndCounterfactual(
        self,
        x: np.ndarray,
        BoundingBox: np.ndarray = None,
    ):
        """
        Get the label and the counterfactual of the point x.
        Args:
            x: the point to get the label and the counterfactual.
            BoundingBox: the bounding box to use.
        Returns:
            y: the label of x.
            x_cf: the counterfactual of x.
        """
        x_cf, y_cf, y = self.oracle.getCounterFactual(
            [x], BoundingBox=BoundingBox, return_label=True
        )
        self.niter += 1
        return y[0], x_cf

    def attack(
        self,
        max_budget: int = None,
        Compute_fidelity: bool = False,
        fidelity_data: np.ndarray = None,
        fidelity_each: int = 10,
    ):
        """
        The main function to attack the model.
        Args:
            dynamic_plot: if True, plot the dynamic plot.
            max_budget: the maximum budget to use.
        Returns:
            Tree: the reconstructed tree.
        """
        if max_budget is None:
            max_budget = np.inf
        # Create the initial space
        self.space = space(self.FeaturesType, self.FeaturesPossibleValues)
        self.id = "0"  # The id is used to identify the nodes 0 : left, 1 : right
        # Get the center point x of the space and its counterfactual x_cf
        x = self.space.get_center_point()
        y, x_cf = self.getLabelAndCounterfactual(x)
        y_cf = (
            self.oracle.n_classes - y - 1
        )  # Here y_cf does not refer to the true label of x_cf (Just an initial value)
        # DIV is the list of the decision variables and their values and operators
        # that changed between x and x_cf
        self.DIV = [
            (c, d_value, d_operator, x, y)
            for c, d_value, d_operator in self.__get_changed_features(x, x_cf)
        ]
        if len(self.DIV) == 0:
            # No change between x and x_cf : x_cf is the same as x
            self.Tree.root = Node(y, id=self.id, space=self.space)
            return self.Tree
        c, d_value, d_operator, x, y = self.DIV.pop(0)
        # Update the root node of the tree
        # The root node is the decision that changed between x and x_cf
        # If the decision is "l" then we shift the value of the decision by -epsilon
        if d_operator == "l":
            self.Tree.root = Node(
                y,
                d_operator,
                c,
                d_value - self.epsilon,
                name="root",
                id=self.id,
                space=self.space,
            )
        else:  # Otherwise we shift the value of the decision by +epsilon
            self.Tree.root = Node(
                y,
                d_operator,
                c,
                d_value + self.epsilon,
                name="root",
                id=self.id,
                space=self.space,
            )
        # Split the space into two subspaces
        r_space, l_space = self.__split_space(
            self.Tree.root.space, c, d_value, d_operator
        )

        # Create the left and right nodes of the root node
        # Left node must have the same label as x_cf
        self.Tree.root.left = Node(y_cf, id=self.id + "0", space=l_space, DIV=self.DIV)
        # Right node must have the same label as x
        self.Tree.root.right = Node(y, id=self.id + "1", space=r_space)
        if self.verbose:
            self.xx_cf = (x, x_cf)
            print(f"""Initial space: , {self.space.fpv}
                    \tInitial x: {x} 
                    \tLabel of x: , {y}
                    \tx_cf: , {x_cf}
                    \tdecision: {c}, {d_value}, {d_operator}
                    \tleft space: {l_space.fpv}, {y_cf}
                    \tright space: {r_space.fpv}, {y}""")
        # Add the left and right nodes to the QueryList to explore them
        self.QueryList.append((self.Tree.root.left, l_space, y_cf))
        self.QueryList.append((self.Tree.root.right, r_space, y))

        # Main loop to attack the model
        while len(self.QueryList) != 0 and self.niter < max_budget:
            self.__attack_with_querylist(max_budget)
            if self.niter % 1000 == 0:
                print("Nb queries: ", self.niter)
            if Compute_fidelity and self.niter % fidelity_each == 0:
                self.fidelity_over_queries[self.niter] = self.compute_fidelity(
                    self.oracle.classifier, fidelity_data
                )

        print("Nb queries: ", self.niter)
        return self.Tree

    def __attack_with_querylist(self, max_budget: int):
        """
        A function to attack the model by exploring and updating the QueryList.
        Args:
            max_budget: the maximum budget to use.
        """
        # Get the next node to explore
        node, space, y = self.QueryList.pop(0)
        if len(node.DIV) == 0:
            # If there is no split to do.
            if self.niter >= max_budget:
                return
            # Get the center point of the space
            x = space.get_center_point()
            # Get the Bounding box (i.e Hypercube), the label of x, and its counterfactual x_cf
            boundingBox = self.__getBoundingBox(space)
            y, x_cf = self.getLabelAndCounterfactual(x, BoundingBox=boundingBox)
            y_cf = (
                self.oracle.n_classes - y - 1
            )  # Here y_cf does not refer to the true label of x_cf (Just an arbitrary value)
            if np.linalg.norm(x - x_cf, ord=self.ObjNorm) < self.epsilon:
                # Check if the counterfactual is the same as x
                node.update_label(y)

                if self.verbose:
                    print(f"""###Counterfactual is the same as x : not found, leaf node reached.
                        Final space: , {node.space.fpv}
                        x = {x}, with label = {y}""")
                return
            # Add the changed features to the DIV list
            node.DIV = [
                (c, d_value, d_operator, x, y)
                for c, d_value, d_operator in self.__get_changed_features(x, x_cf)
            ]
            self.xx_cf = (x, x_cf)
        c, d_value, d_operator, x, y = node.DIV.pop(0)
        y_cf = self.oracle.n_classes - y - 1  # Arbitrary value
        # Update the node
        # The node is the decision that changed between x and x_cf
        # If the decision is "l" then we shift the decision value by -epsilon
        # Otherwise we shift the decision value by +epsilon
        if d_operator == "l":
            node.update_decision(c, d_value - self.epsilon, d_operator)
        else:
            node.update_decision(c, d_value + self.epsilon, d_operator)
        # Split the space into two subspaces
        r_space, l_space = self.__split_space(space, c, d_value, d_operator)

        # Create the left and right nodes of the current node
        node.left = Node(y_cf, id=node.id + "0", space=l_space, DIV=node.DIV)
        node.right = Node(y, id=node.id + "1", space=r_space)
        if self.verbose:
            x, x_cf = self.xx_cf
            print(f"""Initial space: , {node.space.fpv}
                    \tInitial x: {x} 
                    \tLabel of x: , {y}
                    \tx_cf: , {x_cf}
                    \tdecision: {c}, {d_value}, {d_operator}
                    \tleft space: {l_space.fpv}, {y_cf}
                    \tright space: {r_space.fpv}, {y}""")

        # Update the QueryList to explore the left and right nodes (DFS strategy)
        if self.strategy == "DFS":
            self.QueryList = [
                (node.left, l_space, y_cf),
                (node.right, r_space, y),
            ] + self.QueryList
        elif self.strategy == "BFS":
            self.QueryList = self.QueryList + [
                (node.left, l_space, y_cf),
                (node.right, r_space, y),
            ]
        else:
            print("Warning: Unknown strategy, using DFS")
            self.QueryList = [
                (node.left, l_space, y_cf),
                (node.right, r_space, y),
            ] + self.QueryList
        # self.niter += 1
        return

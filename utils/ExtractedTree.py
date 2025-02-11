
import numpy as np

class DecisionTree :
    """
    The decision tree class.
    """
    def __init__(self, root) :
        """
        Parameters
        ----------
        root : Node
            The root of the tree.
        """
        self.root = root

    def get_node_list(self) :
        """
        Get the list of nodes of the tree.
        """
        node_list = []
        self.get_node_list_rec(self.root, node_list)
        return node_list
    
    def get_node_list_rec(self, node, node_list) :
        """
        Get the list of nodes of the tree recursively.
        """
        if node.name == "leaf" :
            node_list.append(node)
        else :
            node_list.append(node)
            self.get_node_list_rec(node.left, node_list)
            self.get_node_list_rec(node.right, node_list)
    
    def not_in_thresholds(self, thresholds, node) :
        """
        Check if the node is not in the thresholds list.
        Args:
            thresholds: the list of thresholds
            node: the node
        Returns:
            True if the node is not in the list, False otherwise."""
        for t in thresholds :
            if t[0] == node.decision_variable and abs(t[1] - node.decision_value)<=1e-6 :
                return False
        return True
    
    def get_thresholds(self) :
        """
        Get the thresholds of the tree.
        returns:
            thresholds : List
                the thresholds of the tree.
        """
        thresholds = []
        for node in self.get_node_list() :
            if node.name != "leaf" and self.not_in_thresholds(thresholds, node):
                thresholds.append((node.decision_variable,node.decision_value))
        return thresholds

    def predict(self,X):
        """
        Predict the labels of the samples.
        Args:
            X: the samples
        Returns:
            the labels of the samples.
        """
        predictions = np.zeros(len(X))
        for i,x in enumerate(X) :
            predictions[i] = self.root.get_decision(x)
        return predictions
    
    def print_tree(self):
        """
        Print the tree.
        """
        node_list = self.get_node_list()
        for node in node_list :
            node.print_node()


class Node :
    """
    The node class.
    """
    def __init__(self, 
                label = None, 
                decison_operator = None,
                decision_variable = None,
                decision_value = None,
                id = None,
                space = None,
                name="leaf",
                DIV= []):
        """
        Parameters
        ----------
        label : int, optional
            The label of the node. The default is None.
        decison_operator : str, optional
            The decision operator. The default is None.
            'l' for less than or equal, 'g' for greater than or equal.
        decision_variable : int, optional
            The decision variable (or feature). The default is None.
        decision_value : float, optional
            The decision value. The default is None.
        id : int, optional
            The id of the node. The default is None.
        space : Space, optional
            The space of the node. The default is None.
        name : str, optional
            The name of the node. The default is "leaf".
        """
        self.name = name
        self.label = label
        self.left = None
        self.right = None
        self.decision_operator = decison_operator
        self.decision_variable = decision_variable
        self.decision_value = decision_value
        self.id = id
        self.space = space
        self.DIV = DIV

    def update_decision(self, decision_variable, decision_value, decision_operator) :
        """
        Update the decision of the node.
        Args:
            decision_variable: the decision variable
            decision_value: the decision value
            decision_operator: the decision operator
        """
        self.decision_variable = decision_variable
        self.decision_value = decision_value
        self.decision_operator = decision_operator
        self.name = "decision_node"

    def update_label(self, label) :
        """
        Update the label of the node.
        Args:
            label: the label
        """
        self.label = label

    def satisfy_decision(self, x):
        """ 
        Check if the sample satisfies the decision of the node.
        Args:
            x: the sample
        Returns:
            True if the sample satisfies the decision, False otherwise.
        """
        if self.decision_operator == 'l':
            return x[self.decision_variable] <= self.decision_value
        else :
            return x[self.decision_variable] >= self.decision_value
        
    # def get_decision(self, x):
    #     """
    #     Get the decision of the sample. Used for prediction of the sample when using the root node. 
    #     Args:
    #         x: the sample
    #     Returns:
    #         the decision of the sample."""
    #     if self.name=='leaf':
    #         return self.label
    #     if self.satisfy_decision(x) :
    #         return self.right.get_decision(x)
    #     else :
    #         return self.left.get_decision(x)  
          
    def get_decision(self, x):
        """
        Get the decision of the sample. Used for prediction of the sample when using the root node. 
        Args:
            x: the sample
        Returns:
            the decision of the sample."""
        while self.name != 'leaf' :
            if self.satisfy_decision(x) :
                self = self.right
            else :
                self = self.left  
        return self.label
        
    def print_node(self):
        """
        Print the node.
        """
        if self.name == "leaf" :
            print( f'{self.id} ,{self.name}, {self.label}')
        else :
            dc = {'g':'>=', 'l':'<='}
            print(f'{self.id} ,{self.name},  x[{self.decision_variable}] {dc[self.decision_operator]} {self.decision_value} ? {self.label}, space = {self.space.fpv}')


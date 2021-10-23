import numpy as np
import math
from collections import deque


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.depth = None
        self.leaf = True

class DecisionTreeClassifier:
    def __init__(self, S, attributes, labels, max_depth, sample_weight=None, split_scheme='Information Gain', feature_sample = None):
        self.S = S  # features or predictors
        self.attributes = np.array(attributes)  # name of the features
        self.labels = np.array(labels)  # categories
        self.sample_weight = sample_weight
        self.labels_categories = set(self.labels)  # unique categories
        # number of instances of each category
        self.node = None  # nodes
        # calculate the initial entropy of the system
        self.entropy = self._get_entropy(np.arange(len(self.labels)))
        self.max_depth = max_depth
        self.split_scheme = split_scheme

    def _get_entropy(self, s_ids):
        # sorted labels by instance id
        if len(s_ids)==0:
            return 0
        labels = [self.labels[i] for i in s_ids]
        if not self.sample_weight:
            labels = [self.labels[i] for i in s_ids]
            labels = np.array(labels)
            # count number of instances of each category
            label_count = [(labels == category).sum() for category in self.labels_categories]
            # calculate the entropy for each category and sum them
            entropy = sum([-count / len(s_ids) * math.log(count / len(s_ids), 2)
                           if count else 0
                           for count in label_count
                           ])
        else:
            weight = [self.sample_weight[i] for i in s_ids]
            labels = np.array(labels)*np.array(weight)
            # count number of instances of each category
            pos_label = [label for label in labels if label>0]
            neg_label = [label for label in labels if label<0]
            pos_label = np.sum(pos_label)
            neg_label = -np.sum(neg_label)

            entropy = -pos_label*math.log(pos_label,2)-neg_label*math.log(neg_label,2)
        return entropy

    def _get_ME(self, s_ids):
        # sorted labels by instance id
        if len(s_ids)==0:
            return 0
        labels = [self.labels[i] for i in s_ids]
        labels = np.array(labels)
        # count number of instances of each category
        label_count = [(labels == category).sum() for category in self.labels_categories]
        max_label = max(label_count)
        label_count.remove(max_label)
        max_error = max(label_count)
        # calculate the ME
        ME = max_error/len(s_ids)
        return ME

    def _get_GI(self, s_ids):
        # sorted labels by instance id
        if len(s_ids)==0:
            return 0
        labels = [self.labels[i] for i in s_ids]
        labels = np.array(labels)
        # count number of instances of each category
        label_count = [(labels == category).sum() for category in self.labels_categories]
        # calculate the entropy for each category and sum them
        GI = 1- sum([(count / len(s_ids))**2
                       if count else 0
                       for count in label_count
                       ])
        return GI

    def _get_information_gain(self, s_ids, feature_id):
        # store in a list all the values of the chosen feature
        S_features = [self.S[x][feature_id] for x in s_ids]
        if is_integer(S_features[0]):
            S_features = [int(s) for s in S_features]
            thresh_hold = np.median(S_features)
            small_v_id = []
            large_v_id = []
            smaller_than_threshold = []
            larger_than_threshold = []
            for s, s_id in zip(S_features, s_ids):
                if int(s)<=thresh_hold:
                    smaller_than_threshold.append(s)
                    small_v_id.append(s_id)
                else:
                    larger_than_threshold.append(s)
                    large_v_id.append(s_id)
            feature_v_count = [len(smaller_than_threshold), len(larger_than_threshold)]
            feature_v_id = [small_v_id, large_v_id]
        else:
            feature_vals = list(set(S_features))
            # get frequency of each value
            feature_v_count = [S_features.count(val) for val in feature_vals]
            # get the feature values ids
            feature_v_id = [
                [s_ids[i]
                 for i, x in enumerate(S_features)
                 if x == y]
                for y in feature_vals
            ]
        if self.split_scheme == 'Majority Error':
            total_entropy = self._get_ME(s_ids)
            feature_entropy = sum([v_counts / len(s_ids) * self._get_ME(v_ids)
                                   for v_counts, v_ids in zip(feature_v_count, feature_v_id)])
        elif self.split_scheme == 'Gini Index':
            total_entropy = self._get_GI(s_ids)
            feature_entropy = sum([v_counts / len(s_ids) * self._get_GI(v_ids)
                                   for v_counts, v_ids in zip(feature_v_count, feature_v_id)])
        else:
            total_entropy = self._get_entropy(s_ids)
            feature_entropy = sum([v_counts / len(s_ids) * self._get_entropy(v_ids)
                                   for v_counts, v_ids in zip(feature_v_count, feature_v_id)])

        gain = total_entropy - feature_entropy

        return gain

    def _get_feature_max_information_gain(self, s_ids, feature_ids):
        # get the entropy for each feature
        features_entropy = [self._get_information_gain(s_ids, feature_id) for feature_id in feature_ids]
        # find the feature that maximises the information gain
        max_id = feature_ids[features_entropy.index(max(features_entropy))]
        return self.attributes[max_id], max_id

    def id3(self):
        # assign an unique number to each instance
        depth = 0
        #x_ids = [x for x in range(len(self.S))]
        s_ids = np.arange(len(self.S))
        # assign an unique number to each feature
        feature_ids = [x for x in range(len(self.attributes))]
        # define node variable - instance of the class Node
        if self.max_depth==1:
            node = Node()
            best_feature_name, best_feature_id = self._get_feature_max_information_gain(s_ids, feature_ids)
            node.value = best_feature_name
            node.childs = []
            feature_values = list(set([self.S[s][best_feature_id] for s in s_ids]))
            labels_in_features = [self.labels[s] for s in s_ids]
            if is_integer(feature_values[0]):
                feature_values = [int(value) for value in feature_values]
                threshold = np.median(feature_values)

                small_value_ids = [s for s in s_ids if int(self.S[s][best_feature_id]) <= threshold]
                small_value_labels =  [self.labels[s] for s in small_value_ids]
                small_value_leaf_label = max(set(small_value_labels), key=small_value_labels.count)

                large_value_ids = [s for s in s_ids if int(self.S[s][best_feature_id]) > threshold]
                large_value_labels = [self.labels[s] for s in large_value_ids]
                large_value_leaf_label = max(set(large_value_labels), key=large_value_labels.count)
                if small_value_leaf_label == large_value_leaf_label:
                    if small_value_labels.count(-small_value_leaf_label) > large_value_labels.count(-large_value_leaf_label):
                        large_value_leaf_label = -large_value_leaf_label
                    else:
                        small_value_leaf_label = -small_value_labels

                small_child = Node()
                small_child.value = threshold
                small_child.next = small_value_leaf_label

                large_child = Node()
                large_child.value = threshold
                large_child.next = large_value_leaf_label
                node.childs.append(small_child)
                node.childs.append(large_child)
            else:
                for value in feature_values:
                    child = Node()
                    child.value = value  # add a branch from the node to each feature value in our feature
                    node.childs.append(child)  # append new child node to current node
                    child.next = max(set(labels_in_features), key=labels_in_features.count)
            self.node = node
        else:
            self.node = self._id3_recv(s_ids, feature_ids, self.node, depth)

    def _id3_recv(self, s_ids, feature_ids, node, current_depth):
        if not node:
            node = Node()  # initialize nodes
        # sorted labels by instance id
        labels_in_features = [self.labels[s] for s in s_ids]
        # if all the example have the same class (pure node), return node
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[s_ids[0]]
            node.leaf = True
            return node
        # if there are not more feature to compute or max depth reach, return node with the most probable class
        if len(feature_ids) == 0 or current_depth==self.max_depth:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            node.leaf = True
            return node
        # choose the feature that maximizes the information gain
        best_feature_name, best_feature_id = self._get_feature_max_information_gain(s_ids, feature_ids)
        node.value = best_feature_name
        node.leaf = False
        node.childs = []
        # value of the chosen feature for each instance
        feature_values = list(set([self.S[s][best_feature_id] for s in s_ids]))
        if is_integer(feature_values[0]):
            feature_values = [int(value) for value in feature_values]
            threshold = np.median(feature_values)
            small_value_ids = [s for s in s_ids if int(self.S[s][best_feature_id]) <= threshold]
            large_value_ids = [s for s in s_ids if int(self.S[s][best_feature_id]) > threshold]
            child_s_ids = [small_value_ids,large_value_ids]
            for child_ids in child_s_ids:
                child = Node()
                child.value = threshold  # add a branch from the node to each feature value in our feature
                node.childs.append(child)  # append new child node to current node
                if not child_ids:
                    child.next = max(set(labels_in_features), key=labels_in_features.count)
                else:
                    if best_feature_id in feature_ids:
                        to_remove = feature_ids.index(best_feature_id)
                        feature_ids.pop(to_remove)
                    # recursively call the algorithm
                    next_depth = current_depth + 1
                    child.leaf = False
                    child.next = self._id3_recv(child_ids, feature_ids, child.next, next_depth)
        # loop through all the values
        else:
            for value in feature_values:
                child = Node()
                child.value = value  # add a branch from the node to each feature value in our feature
                node.childs.append(child)  # append new child node to current node
                child_s_ids = [s for s in s_ids if self.S[s][best_feature_id] == value]
                if not child_s_ids:
                    child.next = max(set(labels_in_features), key=labels_in_features.count)
                    print('')
                else:
                    if best_feature_id in feature_ids:
                        to_remove = feature_ids.index(best_feature_id)
                        feature_ids.pop(to_remove)
                    # recursively call the algorithm
                    next_depth = current_depth + 1
                    child.leaf = False
                    child.next = self._id3_recv(child_s_ids, feature_ids, child.next, next_depth)
        return node

    def printTree(self):
        if not self.node:
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = nodes.popleft()
            print("value:",node.value)
            if node.childs:
                for child in node.childs:
                    print('(child:{})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print("next:", node.next)

    def single_predict(self, S, node=None):
        result = None
        if node == None:
            node = self.node
        if not isinstance(node, Node):
            return node
        if node.childs:
            feature_id = np.where(self.attributes == node.value)[0][0]
            if isinstance(node.childs[0].value, float):
                threshold = node.childs[0].value
                if float(S[feature_id]) <= threshold:
                    child = node.childs[0]
                    result = self.single_predict(S, child.next)
                else:
                    child = node.childs[1]
                    result = self.single_predict(S, child.next)
            else:
                for child in node.childs:
                    if child.value == S[feature_id]:
                        result = self.single_predict(S, child.next)
        elif node.next:
            if node.next.value in self.attributes:
                result = self.single_predict(S, node.next)
            else:
                result = node.next.value
        else:
            result = node.value
        return result

    def predict(self, S):
        y_pred = []
        for s in S:
            pred = self.single_predict(s)
            y_pred.append(pred)
        return np.array(y_pred)

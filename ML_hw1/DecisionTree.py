import numpy as np
import pandas as pd
import math
from collections import deque

description = open("D:\ML_hw1\data-desc.txt", 'r')
lines = description.readlines()
labels_categories = lines[2].strip('\n').split(',')
labels_categories = [x.strip() for x in labels_categories]
attributes = lines[14].strip('\n').strip(' ').split(',')[:-1]


data_dic = {}
for attribute in attributes:
    data_dic[attribute] = []
data_dic['label'] = []
train = open('train.csv', 'r')
for line in train:
    values = line.strip('\n').split(',')
    for index,key in enumerate(data_dic.keys()):
        data_dic[key].append(values[index])

data_df = pd.DataFrame(columns=data_dic.keys())
for key, value in data_dic.items():
    data_df[key] = value
S= np.array(data_df.drop('label', axis=1).copy())
labels = np.array(data_df['label'].copy())



class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.depth = None
        self.leaf = True

class DecisionTreeClassifier:
    def __init__(self, S, attributes, labels, labels_categories, max_depth, split_scheme='Information Gain'):
        self.S = S  # features or predictors
        self.attributes = np.array(attributes)  # name of the features
        self.labels = np.array(labels)  # categories
        self.labels_categories = np.array(labels_categories)  # unique categories
        # number of instances of each category
        self.labelCategoriesCount = [(self.labels==category).sum() for category in self.labels_categories]
        self.node = None  # nodes
        # calculate the initial entropy of the system
        self.entropy = self._get_entropy(np.arange(len(self.labels)))
        self.max_depth = max_depth
        self.split_scheme = split_scheme

    def _get_entropy(self, s_ids):
        # sorted labels by instance id
        labels = [self.labels[i] for i in s_ids]
        labels = np.array(labels)
        # count number of instances of each category
        label_count = [(labels==category).sum() for category in self.labels_categories]
        # calculate the entropy for each category and sum them
        entropy = sum([-count / len(s_ids) * math.log(count / len(s_ids), 2)
                       if count else 0
                       for count in label_count
                       ])
        return entropy

    def _get_ME(self, s_ids):
        # sorted labels by instance id
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
        # get unique values
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
        if len(feature_ids) == 0 or current_depth == self.max_depth:
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
        # loop through all the values
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

    def predict(self, S, node=None):
        result = None
        if node == None:
            node = self.node
        if node.childs:
            feature_id = np.where(self.attributes == node.value)[0][0]
            for child in node.childs:
                if child.value == S[feature_id]:
                    result = self.predict(S, child.next)
        elif node.next:
            if node.next.value in self.attributes:
                result = self.predict(S, node.next)
            else:
                result = node.next.value
        else:
            result = node.value
        return result


test_data_dic = {}
for attribute in attributes:
    test_data_dic[attribute] = []
test_data_dic['label'] = []
test = open('test.csv', 'r')
for line in test:
    values = line.strip('\n').split(',')
    for index,key in enumerate(data_dic.keys()):
        test_data_dic[key].append(values[index])

test_df = pd.DataFrame(columns=test_data_dic.keys())
for key, value in test_data_dic.items():
    test_df[key] = value
S_test= np.array(test_df.drop('label', axis=1).copy())
labels_test = np.array(test_df['label'].copy())


#tree.printTree()
for i in range(1,7):
    accurate = 0
    tree = DecisionTreeClassifier(S=S, attributes=attributes, labels=labels, labels_categories=labels_categories,
                                  max_depth=i,split_scheme='Gini Index')
    tree.id3()
    for s,y in zip(S_test,labels_test):
        predict = tree.predict(s)
        if predict == y:
            accurate+=1
    error_rate = (len(labels_test)-accurate)/len(labels_test)
    print(i,error_rate)
#result = tree.predict(['vhigh','high','5more','2','small','low'], tree.node)
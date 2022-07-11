import pandas as pd
import numpy as np


class Expandable:
    pass

class Result:
    def __init__(self):
        self.count = 0
        self.accuracy = 0

    def append_accuracy(self, accuracy):
        self.accuracy += accuracy
        self.count += 1

    def get_average_accuracy(self):
        return 0 if  self.count == 0 else round(self.accuracy / self.count, 2)



def calculate_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entropy = 0

    for class_ in class_list:
        class_count = train_data[train_data[label] == class_].shape[0]
        class_entropy = - (class_count / total_row) * np.log2(class_count / total_row)
        total_entropy += class_entropy

    return total_entropy


def calculate_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for class_ in class_list:

        label_class_count = feature_value_data[feature_value_data[label] == class_].shape[0]

        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy_class = - probability_class * np.log2(probability_class)
            entropy += entropy_class

    return entropy


def calculate_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:

        feature_value_data = train_data[
            train_data[feature_name] == feature_value
        ]

        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calculate_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count / total_row
        feature_info += feature_value_probability * feature_value_entropy

    return calculate_total_entropy(train_data, label, class_list) - feature_info


def find_best_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)

    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:
        feature_info_gain = calculate_info_gain(feature, train_data, label, class_list)

        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}

    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[
            train_data[feature_name] == feature_value]

        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]

            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = Expandable()

    return tree, train_data


def generate_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        best_feature = find_best_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(
            best_feature,
            train_data,
            label,
            class_list
        )

        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][best_feature] = tree
            next_root = root[prev_feature_value][best_feature]
        else:
            root[best_feature] = tree
            next_root = root[best_feature]

        for node, branch in list(next_root.items()):
            if isinstance(branch, Expandable): # if expandable
                feature_value_data = train_data[train_data[best_feature] == node]
                generate_tree(next_root, node, feature_value_data, label, class_list)


def construct_id3(train_data, label):
    train_data_copy = train_data.copy()
    class_list = train_data_copy[label].unique()
    root = {}
    generate_tree(root, None, train_data, label, class_list)
    return root


def predict(tree, instance):
    if not isinstance(tree, dict):  # if leaf node
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None


def evaluate(tree, test_data_m, label):
    correct_prediction = 0
    wrong_prediction = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_prediction += 1
        else:
            wrong_prediction += 1
    accuracy = correct_prediction / (correct_prediction + wrong_prediction)
    return accuracy


def construct_and_evaluate(data_source, label, iteration=100, test_percentage=0.75):

    data_frame = pd.read_csv(data_source)

    results = {}

    for _, c in enumerate(data_frame.columns):
        results[c] = Result()

    for x in range(iteration):
        first_split = data_frame.sample(frac=test_percentage)
        second_split = data_frame.drop(first_split.index)

        first_split.reset_index(drop=True, inplace=True)
        second_split.reset_index(drop=True, inplace=True)

        id3_tree = construct_id3(first_split, label)
        base = list(id3_tree.keys())[0]

        accuracy = evaluate(id3_tree, second_split, label)

        results[base].append_accuracy(accuracy)

    results_data = []

    for column, result in results.items():
        results_data.append([
            column,
            result.count,
            result.get_average_accuracy()
        ])

    df = pd.DataFrame(results_data, columns=['Column', 'Count', 'Accuracy'])

    return df, iteration

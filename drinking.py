import numpy
import json
import random
import numpy as np
from sklearn import svm
from collections import namedtuple
from functools import partial
from sklearn.naive_bayes import GaussianNB

def load_json_file(file_path):
    """ JsonFilePath -> JSON """
    f = open(file_path)
    return json.load(f)

def svm_train(X_train, y_train):
    """
    (X, y) -> Classification
    X = [[Value]]
    y = [Value]
    """
    classification = svm.SVC()
    classification.fit(X_train, y_train)
    return classification

def naive_bayes_train(X_train, y_train):
    """
    (X, y) -> Classification
    X = [[Value]]
    y = [Value]
    """
    classification = GaussianNB()
    classification.fit(X_train, y_train)
    return classification

def split_train_test(data, train_percent):
    """ ([[Value]], Percent)-> ([[Value]], [[Value]]) """
    m = len(data)
    num_to_select = int(train_percent * m)
    training = data[0:num_to_select]
    test = data[num_to_select:m]
    return training, test

def cat_to_num(mapper, row):
    """ ([Category -> Value], [Category]) -> [Value] """
    mapper_rows = zip(mapper, row)
    return map(lambda (xform, cat): xform(cat), mapper_rows)

def main():
    """ IO """
    raw = load_json_file('./students.json')
    value_map = [
        (1, lambda sex: 1 if sex == u'M' else 0), # [M, F]
        (3, lambda address: 1 if address == u'U' else 0), # [R, U]
        (4, lambda family_size: 1 if family_size == u'GT3' else 0), # [LT3, GT3]
        (5, lambda parent_status: 1 if parent_status == u'T' else 0), # [A, T]
        (27, lambda weekend_alcohol: 1 if weekend_alcohol == u'4' or weekend_alcohol == u'5' else 0), # [1-3, 4-5]
    ]
    value_indices = map(lambda mapping: mapping[0], value_map)
    value_mapper = map(lambda mapping: mapping[1], value_map)
    cat_values = value_matrix(raw, value_indices)
    num_values = map(partial(cat_to_num, value_mapper), cat_values)
    train, test = split_train_test(num_values, 0.7)
    X_train, y_train = split_x_y(train)
    X_test, y_test = split_x_y(test)

    print 'Classifier: (Accuracy, F1 Score)'
    print 'svm: ', predict(svm_train, X_train, y_train, X_test, y_test)
    print 'naive_bayes: ', predict(naive_bayes_train, X_train, y_train, X_test, y_test) 


def predict(clf, X_train, y_train, X_test, y_test):
    """
    ((X, y) -> Classification, X, y, X, y) -> Percent
    X = [[Value]]
    y = [Value]
    """
    classification = clf(X_train, y_train)
    predicted_y = classification.predict(X_test)
    accuracy = get_accuracy(predicted_y, y_test)
    ys = zip(predicted_y, y_test)
    true_pos = get_true_pos(ys)
    false_pos = get_false_pos(ys)
    false_neg = get_false_neg(ys)
    try:
        f1 = f1_score(true_pos, false_pos, false_neg)
    except ZeroDivisionError:
        f1 = None
    return accuracy, f1

def get_accuracy(predicted, actual):
    """(Count, Count) -> Percent"""
    # accuracy = correct predictions/total data points
    num_data_points = len(predicted)
    compare = zip(predicted, actual)
    num_correct = get_count(compare, lambda predict, actual: predict == actual)
    return num_correct / float(num_data_points)

def get_count(ys, predicate):
    """([(y_predict, y_test)], (y_predict, y_test) -> Bool) -> Count"""
    return reduce(lambda count, y: count + (1 if predicate(y[0], y[1]) else 0), ys, 0)

def get_true_pos(ys):
    """[(y_predict, y_test)] -> Count"""
    return get_count(ys, lambda predict, test: predict == 1 and test == 1)

def get_false_pos(ys):
    """[(y_predict, y_test)] -> Count"""
    return get_count(ys, lambda predict, test: predict == 1 and test == 0)

def get_false_neg(ys):
    """[(y_predict, y_test)] -> Count"""
    return get_count(ys, lambda predict, test: predict == 0 and test == 1)

def split_x_y(data):
    """
    [[Value]] -> (X, y)
    X = [[Value]]
    y = [Value]
    """
    X = []
    y = []
    y_index = len(data[0]) - 1
    for row in data:
        X.append(row[0:(y_index)])
        y.append(row[y_index])
    return X, y

def value_matrix(raw, value_indices):
    """ (DictOfJson, [Index]) -> [[Value]] """
    return [
        [ item['values'][idx] for idx in value_indices ]
        for item in raw['data']
    ]

def precision(true_pos, false_pos):
    """ (Count, Count) -> Percent """
    return true_pos / float(true_pos + false_pos)

def recall(true_pos, false_neg):
    """ (Count, Count) -> Percent """
    all_actual_pos = float(true_pos + false_neg)
    return true_pos / all_actual_pos

def f1_score_formula(precision, recall):
    """ (Percent, Percent) -> F1Score """
    return (2 * precision * recall) / float(precision + recall)

def f1_score(true_pos, false_pos, false_neg):
    """ (Count, Count, Count) -> F1Score """
    p = precision(true_pos, false_pos)
    r = recall(true_pos, false_neg)
    return f1_score_formula(p, r)

if __name__ == '__main__':
    main()

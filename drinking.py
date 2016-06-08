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

def svm_train(X_test, y_test):
  """
  (X, y) -> Classification
  X = [[Value]]
  y = [Value]
  """
  classification = svm.SVC()
  classification.fit(X_test,y_test)
  return classification

def naive_bayes_train(X_test, y_test):
  """
  (X, y) -> Classification
  X = [[Value]]
  y = [Value]
  """
  classification = GaussianNB()
  classification.fit(X_test, y_test)
  return classification

def split_train_test(data, train_percent):
  """ ([[Value]], Percent)-> ([[Value]], [[Value]]) """
  m = len(data)
  num_to_select = int(train_percent * m)
  training = data[0:num_to_select]
  test = data[num_to_select:m]
  return training, test

def get_accuracy(predicted, actual):
  """ (Count, Count) -> Percent """
  # accuracy = correct predictions/total data points
  num_data_points = len(predicted)
  compare = zip(predicted, actual)
  num_correct = 0
  for predicted, actual in compare:
      num_correct = num_correct + (1 if predicted == actual else 0)
  return (float(num_correct)/float(num_data_points))

def cat_to_num(mapper, row):
  """ ([Category -> Value], [Category]) -> [Value] """
  mapper_rows = zip(mapper, row)
  return map(lambda (xform, cat): xform(cat), mapper_rows)

def main():
  """ IO """
  raw = load_json_file('./students.json')
  value_map = [
    (1, lambda sex: 1 if sex == 'M' else 0), # [M, F]
    (3, lambda address: 1 if address == 'U' else 0), # [R, U]
    (4, lambda family_size: 1 if family_size == 'GT3' else 0), # [LT3, GT3]
    (5, lambda parent_status: 1 if parent_status == 'T' else 0), # [A, T]
    (27, lambda weekend_alcohol: 1 if weekend_alcohol == '4' or weekend_alcohol == '5' else 0), # [1-3, 4-5]
  ]
  value_indices = map(lambda mapping: mapping[0], value_map)
  value_mapper = map(lambda mapping: mapping[1], value_map)
  cat_values = value_matrix(raw, value_indices)
  num_values = map(partial(cat_to_num, value_mapper), cat_values)
  train, test = split_train_test(num_values, .7)
  X_train, y_train = split_x_y(train)
  X_test, y_test = split_x_y(test)
  print predict_accuracy(svm_train, X_train, y_train, X_test, y_test)
  print predict_accuracy(naive_bayes_train, X_train, y_train, X_test, y_test) 

def predict_accuracy(clf, X_train, y_train, X_test, y_test):
  """
  ((X, y) -> Classification, X, y, X, y) -> Percent
  X = [[Value]]
  y = [Value]
  """
  classification = clf(X_train, y_train)
  predicted_y = classification.predict(X_test)
  accuracy = get_accuracy(predicted_y, y_test)
  return accuracy

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
  return true_pos / (true_pos + false_pos)

def recall(true_pos, false_neg):
  """ (Count, Count) -> Percent """
  all_actual_pos = true_pos + false_neg
  return true_pos / all_actual_pos

def f1_score(precision, recall):
  """ (Count, Count) -> F1Score """
  return (2 * precision * recall) / (precision + recall)

def f1_score_prim(true_pos, false_pos, false_neg):
  """ (Count, Count, Count) -> F1Score """
  p = precision(true_pos, false_pos)
  r = recall(true_pos, false_neg)
  return f1_score(p, r)

if __name__ == '__main__':
  main()

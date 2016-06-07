import numpy
import json
import random
from collections import namedtuple
from functools import partial

def load_json_file(file_path):
  f = open(file_path)
  return json.load(f)

def split_train_test(data, train_percent):
    m = len(data)
    num_to_select = int(train_percent * m)
    training = data[0:num_to_select]
    test = data[num_to_select:m]
    return training, test

def cat_to_num(mapper, row):
  mapper_rows = zip(mapper, row)
  return map(lambda (xform, cat): xform(cat), mapper_rows)

def main():
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
  print train[0]

def value_matrix(raw, value_indices):
  return [
    [ item['values'][idx] for idx in value_indices ]
    for item in raw['data']
  ]

if __name__ == '__main__':
  main()

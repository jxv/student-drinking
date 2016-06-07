import numpy
import json
import random
from collections import namedtuple

def load_json_file(file_path):
  f = open(file_path)
  return json.load(f)

def split_train_test(data, train_percent):
    m = len(data)
    num_to_select = int(train_percent * m)
    training = data[0:num_to_select]
    test = data[num_to_select:m]
    return training, test

def cat_to_num(row):
    return [
      1 if row[0] == 'M' else 0,
      1 if row[1] == 'U' else 0,
      1 if row[2] == 'GT3' else 0,
      1 if row[3] == 'T' else 0,
      1 if row[4] == '4' or row[4] == '5' else 0
    ]


def main():
  raw = load_json_file('./students.json')
  # 2 sex
  # 4 address
  # 5 family size
  # 6 parent status
  # 28 weekend alcohol (Y value)
  value_indices = [1, 3, 4, 5, 27]
  cat_values = value_matrix(raw, value_indices)
  num_values = map(cat_to_num, cat_values)
  train, test = split_train_test(num_values, .7)
  print train[0]

def value_matrix(raw, value_indices):
  return [
    [ item['values'][idx] for idx in value_indices ]
    for item in raw['data']
  ]

if __name__ == '__main__':
  main()

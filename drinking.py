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


def main():
  raw = load_json_file('./students.json')
  # 2 sex
  # 4 address
  # 5 family size
  # 6 parent status
  # 28 weekend alcohol (Y value)
  value_indices = [1, 3, 4, 5, 27]
  values = value_matrix(raw, value_indices)
  train, test = split_train_test(values, .7)
  print len(train), len(test)

def value_matrix(raw, value_indices):
  matrix = []
  for item in raw['data']:
    row = []
    for idx in value_indices:
      row.append(item['values'][idx])
    matrix.append(row)
  return matrix

if __name__ == '__main__':
  main()

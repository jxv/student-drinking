import numpy
import json
from collections import namedtuple

def load_json_file(file_path):
  f = open(file_path)
  return json.load(f)

def main():
  raw = load_json_file('./students.json')
  # 2 sex
  # 4 address
  # 5 family size
  # 6 parent status
  # 28 weekend alcohol (Y value)
  value_indices = [1, 3, 4, 5, 27]
 
  matrix = []
  for item in raw['data']:
    row = []
    for idx in value_indices:
      row.append(item['values'][idx])
    matrix.append(row)
  print matrix

if __name__ == '__main__':
  main()

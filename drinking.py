import numpy
import json
from collections import namedtuple

def load_json_file(file_path):
  f = open(file_path)
  return json.load(f)

def main():
  print 'hello, world!'
  some_json = load_json_file('./students.json')
  print some_json

if __name__ == '__main__':
  main()

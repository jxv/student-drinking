# SVM Classification
import pandas
import numpy as np
import csv
from sklearn import cross_validation
from sklearn.svm import SVC

def unstringer(s):
    return np.int64(s.split('"')[1])

def klass(xs):
    def go(x):
        for i in range(0, len(xs)):
            if eval(x) == xs[i]: # instead of 's.split('"')[1]' just eval the string of string.
                return np.int64(i)
        raise ValueError('cannot find value \'' + x + '\' in list')
    return go

def get_df():
    c = {
        'school': klass(['GP', 'MS']),
        'sex': klass(['F','M']),
        'address': klass(['U','R']),
        'famsize': klass(['LE3','GT3']),
        'Pstatus': klass(['A','T']),
        'Mjob': klass(['teacher','health','services','at_home','other']),
        'Fjob': klass(['teacher','health','services','at_home','other']),
        'reason': klass(['home','reputation','course','other']),
        'guardian': klass(['mother','father','other']),
        'schoolsup': klass(['no','yes']),
        'famsup': klass(['no','yes']),
        'paid': klass(['no','yes']),
        'activities': klass(['no','yes']),
        'nursery': klass(['no','yes']),
        'higher': klass(['no','yes']),
        'internet': klass(['no','yes']),
        'romantic': klass(['no','yes']),
        'G1': unstringer,
        'G2': unstringer,
    }
    return pandas.read_csv('student.csv', delimiter=';', quoting=csv.QUOTE_NONE, converters=c)

def main():
    dataframe = get_df()
    array = dataframe.values
    X = array[:,0:31]
    Y = array[:,32]
    num_folds = 10
    num_instances = len(X)
    seed = 7
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = SVC()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

if __name__ == '__main__':
    main()

# SVM Classification
import pandas
import numpy as np
import csv
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def unstringer(s):
    return np.int64(s.split('"')[1])

def klass(xs):
    def go(x):
        for i in range(0, len(xs)):
            if eval(x) == xs[i]: # instead of 's.split('"')[1]' just eval the string of string.
                return np.int64(i)
        raise ValueError('cannot find value \'' + x + '\' in list')
    return go

# from sklearn.neighbors import KNeighborsClassifier

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
        'Dalc': lambda x: np.int64(1 if x == '4' or x == '5' else 0),
        'Walc': lambda x: np.int64(1 if x == '4' or x == '5' else 0),
    }
    return pandas.read_csv('student.csv', delimiter=';', quoting=csv.QUOTE_NONE, converters=c)

def split_data(arr):
    n = len(arr[0])
    nx = n - 2
    ny = n - 1
    X = arr[:,0:nx]
    Y = arr[:,ny]
    return X, Y

def main():
    cols = ['sex','address','famsize','Pstatus','Dalc','Walc']
    dataframe = get_df()[cols]
    X, Y = split_data(dataframe.values)
    num_folds = 10
    num_instances = len(X)
    seed = 7
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

    models = [
        ('SVC', SVC()),
        ('KNeighborsClassifier', KNeighborsClassifier()),
        ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
        ('LogisticRegression', LogisticRegression()),
        ('GaussianNB', GaussianNB()),
        ('DecisionTreeClassifier', DecisionTreeClassifier())
    ]

    for (name, model) in models:
        results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
        print(name + ': %.3f%%') % (results.mean() * 100.0)

if __name__ == '__main__':
    main()

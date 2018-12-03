
import pandas as pd
from lxml import html
import quandl
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, preprocessing, cross_validation


from matplotlib import style
def get_label(cur_hpi,fut_hpi):
    if cur_hpi < fut_hpi:
        return 1
    else:
        return 0



style.use('fivethirtyeight')

housing_data = pd.read_pickle('HPI.pickle')
housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data.dropna(inplace=True)


housing_data['US_HPI_future'] = housing_data['United States'].shift(-1)
housing_data['label'] = list(map(get_label,housing_data['United States'],housing_data['US_HPI_future']))
print(housing_data.head())

X = np.array(housing_data.drop(['label','US_HPI_future'],1))
X = preprocessing.scale(X)
y = np.array(housing_data['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

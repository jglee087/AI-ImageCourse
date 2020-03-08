import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings(action='ignore')

iris_data =pd.read_csv("./data/iris2.csv",encoding='utf-8')

y=iris_data.loc[:,'Name']#.values
x=iris_data.iloc[:,:-1]#.values

warnings.filterwarnings(action='ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)


#RandomForestClassifier

parameters = [ {"n_estimators":[10,50,100], "criterion":['gini','entropy']} ]

kfold_cv = KFold(n_splits=5, shuffle=True)

clf=GridSearchCV( RandomForestClassifier(), parameters, cv=kfold_cv)

cl=clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
print("acc", accuracy_score(y_test, y_pred))

#print(cl.best_estimator_)
print(cl.best_params_)
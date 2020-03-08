import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings(action='ignore')

iris_data =pd.read_csv("./data/iris2.csv",encoding='utf-8')

y=iris_data.loc[:,'Name']#.values
x=iris_data.iloc[:,:-1]#.values

warnings.filterwarnings(action='ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

values = np.array([0.001, 0.01, 0.1, 1, 10, 100])
params = {'svc__C':values, 'svc__gamma':values}

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe=Pipeline([("scaler",MinMaxScaler()) , ('svc', SVC())])

kfold_cv = KFold(n_splits=5, shuffle=True)

clf=GridSearchCV(pipe, params, cv=kfold_cv)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
print("acc", accuracy_score(y_test, y_pred))

#print(clf.best_estimator_)
print(clf.best_params_)

########### Hyperparameter Tuning ###############
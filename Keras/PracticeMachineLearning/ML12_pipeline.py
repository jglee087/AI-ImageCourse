import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings(action='ignore')

iris_data =pd.read_csv("./data/iris2.csv",encoding='utf-8')

y=iris_data.loc[:,'Name']#.values
x=iris_data.iloc[:,:-1]#.values

print(x.shape)
warnings.filterwarnings(action='ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.6, test_size=0.3)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe=Pipeline([("scaler",MinMaxScaler()) , ('svm', SVC())])
pipe.fit(x_train, y_train)

print("테스트 점수:", pipe.score(x_test, y_test))

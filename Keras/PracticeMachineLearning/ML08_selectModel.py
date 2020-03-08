import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings(action='ignore')

iris_data =pd.read_csv("./data/iris2.csv",encoding='utf-8')

y=iris_data.loc[:,'Name'].values
x=iris_data.iloc[:,:-1].values

from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()
y=enc.fit_transform(y)

warnings.filterwarnings(action='ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

warnings.filterwarnings(action='ignore')
allAlgorithms= all_estimators(type_filter='classifier')

#print(allAlgorithms)
print(len(allAlgorithms))
#print(type(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    
    try:
        clf=algorithm()
    
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(name,"의 정답률= ", accuracy_score(y_test,y_pred))
    
    except:
        print("\n오류:",name,"\n")
        pass
    
    
# pip uninstall scikit-learn (0.21 version)
# pip install scikit-learn == 0.20.3
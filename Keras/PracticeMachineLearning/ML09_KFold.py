import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings(action='ignore')

iris_data =pd.read_csv("./data/iris2.csv",encoding='utf-8')

y=iris_data.loc[:,'Name']
x=iris_data.iloc[:,:-1]

warnings.filterwarnings(action='ignore')
allAlgorithms= all_estimators(type_filter='classifier')

#kfold_cv = KFold(n_splits=10, shuffle=True)

#print(allAlgorithms)
#print(len(allAlgorithms))
#print(type(allAlgorithms))

print("start")
for i in range(4,12,2):

    print(i," \n")
    kfold_cv = KFold(n_splits=i, shuffle=True)

    li=[]
    na=[]
    for (name, algorithm) in allAlgorithms:
    
        try:
            clf=algorithm()
        
            if hasattr(clf,"score"):
                scores=cross_val_score(clf,x,y, cv=kfold_cv)
                print(name,"의 정답률:", scores.mean())
 
               
            
            
        except:
            print("\n오류:",name,"\n")

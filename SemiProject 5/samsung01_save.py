
import numpy as np
import pandas as pd

# index_col=0 첫 번째 열을 column으로 하겠다.
# header=0 첫 번째 행을 row으로 하겠다.

df1=pd.read_csv('./samsung.csv',encoding='cp949', index_col=0, header=0, sep=',')
df2=pd.read_csv('./kospi200.csv',encoding='cp949', index_col=0, header=0, sep=',')

# 삼성전자의 모든 데이터
for i in range(len(df1.index)):
    for j in range(len(df1.iloc[i])):
        df1.iloc[i,j] = int(df1.iloc[i,j].replace(',',''))

# kospi200의 모든 데이터
for i in range(len(df2.index)):
    df2.iloc[i,4] = int(df2.iloc[i,4].replace(',',''))    

df1=df1.sort_values(['일자'], ascending=True)
df2=df2.sort_values(['일자'], ascending=True)

# padnas를 numpy로 변환

df1 = df1.values
df2 = df2.values

np.save('./data/samsung.npy',arr=df1)
np.save('./data/kospi.npy',arr=df2)
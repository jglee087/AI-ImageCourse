import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings(action='ignore')

wine = pd.read_csv("./data/winequality-white.csv",sep=";", encoding='utf-8')

count_data = wine.groupby('quality')["quality"].count()
print(count_data)

count_data.plot()
plt.savefig("wine-count-plot.png")
plt.show()


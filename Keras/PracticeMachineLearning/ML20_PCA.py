from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer= load_breast_cancer()

scaler=StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA, KernelPCA

pca=PCA(n_components=0.8)
#pca=PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)

print("원본 데이터 형태: ", X_scaled.shape)
print("축소된 데이터 형태: ", X_pca.shape)

#rbf_pca = KernelPCA(n_components =0.1, kernel="rbf", gamma=0.04)
#X_reduced = rbf_pca.fit_transform(X_scaled)

#print(X_reduced.shape)
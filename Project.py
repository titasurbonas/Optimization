#!/usr/bin/env python

import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('./ORL_txt/orl_data.txt' ,sep="	", header=None)
data.drop(data.columns[400], axis=1, inplace=True)
print(data)
plt.plot(data)
plt.show()
############################################################
# Fit a PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(data)
# Project the data in 2D
X_pca = pca.transform(data)

# Visualize the data
#target_ids = range(len(data.target_names))


plt.figure(figsize=(10, 10))
plt.plot(X_pca)
plt.show()
"""
for i, c, label in zip(target_ids, 'rgbcmykw', data.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],c=c, label=label)
plt.legend()
plt.show()
"""
"""
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

############################################################
# Fit a PCA
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
pca = PCA(n_components=4, whiten=True)
pca.fit(X)

############################################################
# Project the data in 2D
X_pca = pca.transform(X)

############################################################
# Visualize the data
target_ids = range(len(iris.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
for i, c, label in zip(target_ids, 'rgbcmykw', iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],c=c, label=label)
plt.legend()
plt.show()
"""
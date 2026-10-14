import pandas as pd
import numpy as np
import sys
from ucimlrepo import fetch_ucirepo 
  
# # fetch dataset 
# parkinsons = fetch_ucirepo(id=174) 
  
# # data (as pandas dataframes) 
# X = parkinsons.data.features 
# y = parkinsons.data.targets

# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 
  

X_std = (X - X.mean()) / X.std()

y['Class'] = np.where(y == 4, 1, 0)

def power_set(set):
    num_permutations = 2 ** len(set)
    return [(i, [a[1] for a in filter(lambda a: i & (2 ** a[0]) != 0, enumerate(set))]) for i in range(num_permutations)]

feature_power_set = power_set(X.columns)

from sklearn.model_selection import GridSearchCV
from knn import KnnClassifier
import random

from sklearn.model_selection import GridSearchCV
from knn import KnnClassifier

param_grid = {
    "k": range(1, 31)
}

grid_search = GridSearchCV(
    estimator=KnnClassifier(),
    param_grid=param_grid,
    cv=10,
    scoring="f1",
    n_jobs=-1
)

def train_feature_set(X, y):
    X.dropna(axis=0, inplace=True)
    y = y.iloc[list(X.index)]

    grid_search.fit(X, y)
    return pd.DataFrame(grid_search.cv_results_)

all_results = []
current_best_score = 0
random.shuffle(feature_power_set)
for subset_id, feature_subset in feature_power_set:
    if len(feature_subset) == 0:
        continue

    results = train_feature_set(X[feature_subset], y)

    for result in results.itertuples():
        if result.mean_test_score > current_best_score:
            current_best_score = result.mean_test_score
            print(f"{subset_id}\n{result.param_k}\t{result.mean_test_score} NEW BEST SCORE", file=sys.stderr)

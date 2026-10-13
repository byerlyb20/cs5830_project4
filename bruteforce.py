import pandas as pd
import numpy as np
import sys

df = pd.read_csv("cleveland.csv")

df["age_s"] = (df["age"] - df["age"].mean()) / df["age"].std()
# sex

# one-hot encoding for cp: chest pain type
df["typical_anigma"] = (df["cp"] == 1).astype(int)
df["atypical_anigma"] = (df["cp"] == 2).astype(int)
df["nonanigmal"] = (df["cp"] == 3).astype(int)
df["asymptopmatic"] = (df["cp"] == 4).astype(int)

df["trestbps_s"] = (df["trestbps"] - df["trestbps"].mean()) / df["trestbps"].std()
df["chol_s"] = (df["chol"] - df["chol"].mean()) / df["chol"].std()
# fbs

# one-hot encoding for restecg: resting electrocardiographic results 
df["ecg_normal"] = (df["restecg"] == 0).astype(int)
df["ecg_stt_abnormal"] = (df["restecg"] == 1).astype(int)
df["ecg_lvh_estes"] = (df["restecg"] == 2).astype(int)

df["thalach_s"] = (df["thalach"] - df["thalach"].mean()) / df["thalach"].std()
# exang
df["oldpeak_s"] = (df["oldpeak"] - df["oldpeak"].mean()) / df["oldpeak"].std()

# one-hot encoding for slope: the slope of the peak exercise ST segment
df["upsloping"] = (df["slope"] == 1).astype(int)
df["flat"] = (df["slope"] == 2).astype(int)
df["downsloping"] = (df["slope"] == 3).astype(int)

df["ca"] = df["ca"].replace("?", np.nan).astype(np.float64)
df["ca_s"] = (df["ca"] - df["ca"].mean()) / df["ca"].std()

# one-hot encoding for thal
df["thal_normal"] = (df["thal"] == 3).astype(int)
df["thal_fixed"] = (df["thal"] == 6).astype(int)
df["thal_reversable"] = (df["thal"] == 7).astype(int)

df["num"] = df["num"] != 0

quantitative_cols = ["age_s", "trestbps_s", "chol_s", "thalach_s", "oldpeak_s", "ca_s"]
onehot_cols = ["sex", "typical_anigma", "atypical_anigma", "nonanigmal", "asymptopmatic", "fbs", "ecg_normal", "ecg_stt_abnormal", "ecg_lvh_estes", "exang", "upsloping", "flat", "downsloping", "thal_normal", "thal_fixed", "thal_reversable"]

def power_set(set):
    num_permutations = 2 ** len(set)
    return [(i, [a[1] for a in filter(lambda a: i & (2 ** a[0]) != 0, enumerate(set))]) for i in range(num_permutations)]

feature_power_set = power_set(quantitative_cols + onehot_cols)

from sklearn.model_selection import GridSearchCV
from knn import KnnClassifier
import random

param_grid = {
    "k": range(1, 21)
}

grid_search = GridSearchCV(
    estimator=KnnClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

def train_feature_set(feature_set):
    df_filtered = df[feature_set + ["num"]].copy()
    df_filtered.dropna(inplace=True)

    X = df_filtered.drop(columns=["num"])
    y = df_filtered["num"].to_numpy()

    grid_search.fit(X, y)
    return pd.DataFrame(grid_search.cv_results_)

all_results = []
current_best_score = 0
random.shuffle(feature_power_set)
for subset_id, feature_subset in feature_power_set:
    if len(feature_subset) == 0:
        continue

    results = train_feature_set(feature_subset)

    for result in results.itertuples():
        print(subset_id, result.param_k, result.mean_test_score, sep=",")
        if result.mean_test_score > current_best_score:
            current_best_score = result.mean_test_score
            print(f"{subset_id}\n{result.param_k}\t{result.mean_test_score} NEW BEST SCORE", file=sys.stderr)

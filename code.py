
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)



SEED = 42
POSITIVE, NEGATIVE = "Left", "Stayed"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")

columns = list(train_raw.columns)
#print(columns)

ID_COL = "Employee ID"
train = train_raw.drop(columns=[ID_COL])
test = test_raw.drop(columns=[ID_COL])
train["Attrition"] = (train["Attrition"] == "Left").astype(int)
prop = train["Attrition"].value_counts(normalize=True)
print("Distribuzione target (train):\n", prop, "\n")

fig, ax = plt.subplots()
prop.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
ax.set_title("Distribuzione Attrition (train)")
ax.set_ylabel("Proporzione")
fig.savefig(OUT / "01_distribuzione_y.png")
plt.close(fig)

METRIC = "roc_auc"

n_na = int(train.isna().sum().sum())
print(f"Valori mancanti nel train: {n_na}")
if n_na > 0:
    raise ValueError("Ci sono NA: serve una strategia di imputazione prima di proseguire.")

x = train.drop(columns=["Attrition"])
y = train["Attrition"]
 
X_fit, X_val, y_fit, y_val = train_test_split(
    x, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Dimensioni train_fit: {X_fit.shape} | valid: {X_val.shape}")


num_vars = X_fit.select_dtypes(include="number")
cols_num = list(num_vars.columns)
M = num_vars.corr()

fig, ax = plt.subplots()
im = ax.imshow(M.values, vmin=-1, vmax=1)
ax.set_xticks(range(len(cols_num)))
ax.set_yticks(range(len(cols_num)))
ax.set_xticklabels(cols_num, rotation=45, ha="right")
ax.set_yticklabels(cols_num)
plt.colorbar(im, ax=ax)
fig.savefig(OUT / "02_corr_numeriche.png", dpi=150, bbox_inches="tight")
plt.close(fig)
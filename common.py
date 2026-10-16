from pathlib import Path
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


CAT_VARS = list(X_fit.select_dtypes(exclude="number").columns)

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

cramer_df = (pd.DataFrame({"Variable": CAT_VARS,
                           "CramerV": [cramers_v(X_fit[c], y_fit) for c in CAT_VARS]})
             .round(3)
             .sort_values("CramerV", ascending=False))
print("\nCramer's V:\n", cramer_df.to_string(index=False), "\n")

fig, ax = plt.subplots()
ax.barh(cramer_df["Variable"][::-1], cramer_df["CramerV"][::-1], color="steelblue")
ax.set_title("Associazione categoriche con Attrition (Cramer's V)")
ax.set_xlabel("Cramer's V")
fig.savefig(OUT / "03_cramer_v.png", dpi=150, bbox_inches="tight")
plt.close(fig)

ALL_COL = list(X_fit.columns)
CAT_WEAK = cramer_df.loc[cramer_df["CramerV"] < 0.05, "Variable"].tolist()
COL_A = ALL_COL
COL_B = [c for c in ALL_COL if c not in CAT_WEAK]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

def make_preprocessor(X, scaler=StandardScaler):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    return ColumnTransformer([
        ("num", scaler(), num),
        ("cat", make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                              VarianceThreshold(0.0475)), cat),
    ], sparse_threshold=0)

def run(name, model, cols, grid=None, scaler=StandardScaler):
    Path("models").mkdir(exist_ok=True)
    Xf, Xv = X_fit[cols], X_val[cols]
    pipe = Pipeline([("prep", make_preprocessor(Xf, scaler)), ("model", model)])

    search = GridSearchCV(pipe, grid or {}, cv=cv, scoring=METRIC, n_jobs=-1)
    search.fit(Xf, y_fit)

    i = search.best_index_
    folds = [search.cv_results_[f"split{k}_test_score"][i] for k in range(cv.get_n_splits())]
    proba = search.predict_proba(Xv)[:, 1]

    print(f"[{name}] parametri scelti: {search.best_params_}")
    print(f"[{name}] CV AUC: {np.mean(folds):.3f} ± {np.std(folds, ddof=1):.3f}")
    print(f"[{name}] AUC validation: {roc_auc_score(y_val, proba):.3f}")

    joblib.dump(search.best_estimator_, f"models/{name}.joblib")
    joblib.dump(proba, f"models/{name}_proba.joblib")
    joblib.dump(folds, f"models/{name}_cv.joblib")
    return search
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from common import y_fit, COL_A, OUT

model = joblib.load("models/gbm.joblib")
soglia = joblib.load("models/gbm_soglia.joblib")

test = pd.read_csv("test.csv")
p = model.predict_proba(test[COL_A])[:, 1]

p_left = 0.15 / y_fit.mean()
p_stayed = 0.85 / (1 - y_fit.mean())
p_corr = p * p_left / (p * p_left + (1 - p) * p_stayed)

pred = np.where(p_corr >= soglia, "Left", "Stayed")
out = pd.DataFrame({"Employee ID": test["Employee ID"], "prob_Left": p_corr, "pred": pred})
out.to_csv(OUT / "predizioni_test.csv", index=False)
print("Quota prevista Left:", round((pred == "Left").mean(), 3))


if "Attrition" in test.columns:
    y_test = (test["Attrition"] == "Left").astype(int)
    print("AUC test:", round(roc_auc_score(y_test, p), 3))
    cm = confusion_matrix(y_test, (pred == "Left").astype(int))
    tn, fp, fn, tp = cm.ravel()
    print(cm)
    print(f"sens = {tp/(tp+fn):.3f} | spec = {tn/(tn+fp):.3f} | acc = {(tp+tn)/cm.sum():.3f}")
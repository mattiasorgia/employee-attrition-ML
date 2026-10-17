import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from common import y_val, OUT

p = joblib.load("models/gbm_proba_corr.joblib")


fpr, tpr, thr = roc_curve(y_val, p)
t = pd.DataFrame({"soglia": thr, "sens": tpr, "spec": 1 - fpr})
t["geo"] = np.sqrt(t["sens"] * t["spec"])

def soglia_con_spec_min(s):
    """Soglia con sensibilità massima tra quelle con specificità >= s."""
    return t.loc[t.loc[t["spec"] >= s, "sens"].idxmax(), "soglia"]

soglie = {
    "media geometrica": t.loc[t["geo"].idxmax(), "soglia"],
    "spec >= 0.60": soglia_con_spec_min(0.60),
    "spec >= 0.70": soglia_con_spec_min(0.70),
}


for nome, s in soglie.items():
    cm = confusion_matrix(y_val, (p >= s).astype(int))
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{nome}: soglia = {s:.3f}")
    print(cm)
    print(f"sens = {tp/(tp+fn):.3f} | spec = {tn/(tn+fp):.3f} | acc = {(tp+tn)/cm.sum():.3f}")


m = t[(t["soglia"] >= 0) & (t["soglia"] <= 1)].sort_values("soglia").copy()
m["acc"] = m["sens"] * y_val.mean() + m["spec"] * (1 - y_val.mean())
m["youden"] = m["sens"] + m["spec"] - 1
ax = m.set_index("soglia")[["sens", "spec", "acc", "youden"]].plot(title="Metriche vs soglia - GBM (validation)")
ax.axvline(soglie["spec >= 0.70"], ls="--", color="black")
ax.figure.savefig(OUT / "07_soglia_gbm.png", dpi=150, bbox_inches="tight")
plt.close(ax.figure)


joblib.dump(float(soglie["spec >= 0.70"]), "models/gbm_soglia.joblib")
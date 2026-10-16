from pathlib import Path
import glob
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from code import y_val

Path("output").mkdir(exist_ok=True)

fig, ax = plt.subplots()
for f in sorted(glob.glob("models/*_prob.joblib")):
    name = Path(f).stem.replace("_prob", "")
    prob = joblib.load(f)
    fpr, tpr, _ = roc_curve(y_val, prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_val, prob):.3f})")

ax.plot([0, 1], [0, 1], "--", color="grey")
ax.set_xlabel("Falsi positivi (1 - specificità)")
ax.set_ylabel("Veri positivi (sensibilità)")
ax.set_title("Curve ROC sul validation")
ax.legend()
fig.savefig("output/04_roc_validation.png")
plt.close(fig)
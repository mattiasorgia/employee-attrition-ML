import glob
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from common import OUT

cv_auc = pd.DataFrame({Path(f).stem.replace("_cv", ""): joblib.load(f)
                       for f in sorted(glob.glob("models/*_cv.joblib"))})

print(cv_auc.agg(["mean", "std", "min", "max"]).T.round(3).sort_values("mean", ascending=False))

ax = cv_auc.plot.box(title="AUC in CV (10 fold)", ylabel="AUC")
ax.figure.savefig(OUT / "05_confronto_cv.png", dpi=150, bbox_inches="tight")
plt.close(ax.figure)
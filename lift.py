import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import y_fit, y_val, OUT

p = joblib.load(f"models/gbm_proba.joblib")

p_left = 0.15 / y_fit.mean()
p_stayed = 0.85 / (1 - y_fit.mean())
p_corr = p * p_left / (p * p_left + (1 - p) * p_stayed)
joblib.dump(p_corr, f"models/gbm_proba_corr.joblib")

df = pd.DataFrame({"y": y_val.values, "p": p_corr}).sort_values("p", ascending=False)
df["decile"] = pd.qcut(np.arange(len(df)), 10, labels=False) + 1
gain = df.groupby("decile")["y"].sum().cumsum() / df["y"].sum()
lift = gain / (gain.index.values / 10)
print(pd.DataFrame({"gain": gain, "lift": lift}).round(3))

# 3. Grafici
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
gain.plot(ax=ax1, marker="o", title="Gain cumulato")
lift.plot(ax=ax2, marker="o", title="Lift cumulato")
fig.savefig(OUT / "06_gain_lift_gbm.png")
plt.close(fig)
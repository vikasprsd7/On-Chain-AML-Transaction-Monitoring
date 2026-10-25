"""Wallet-level AML risk model: feature engineering -> SMOTE-in-CV -> model comparison -> held-out test."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix, roc_curve
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

SEED = 42
tx = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"]); w = pd.read_csv("data/wallets.csv")

# ---------- features (no labels used) ----------
out = tx.groupby("from_wallet").agg(n_out=("tx_id","count"), v_out=("amount_usd","sum"), max_out=("amount_usd","max"),
                                    uniq_dst=("to_wallet","nunique"), avg_out=("amount_usd","mean"))
inc = tx.groupby("to_wallet").agg(n_in=("tx_id","count"), v_in=("amount_usd","sum"), max_in=("amount_usd","max"), uniq_src=("from_wallet","nunique"))
f = w.set_index("wallet_id")[["kyc_status","country_risk","created_day"]].join(out).join(inc).fillna(0)
f["retention"] = np.where(f.v_in > 0, (f.v_in - f.v_out) / f.v_in, 0).clip(-5, 1)
f["flow_ratio"] = (f.v_out + 1) / (f.v_in + 1)
near = tx[(tx.amount_usd >= 9000) & (tx.amount_usd < 10000)].groupby("from_wallet").size()
f["near_threshold_n"] = near.reindex(f.index).fillna(0)
f["share_near_threshold"] = f.near_threshold_n / f.n_out.replace(0, np.nan); f["share_near_threshold"] = f.share_near_threshold.fillna(0)

# burstiness: max tx count (in or out) inside any 48h window
def max_burst(ts):
    t = np.sort(ts.values.astype("datetime64[s]").astype(np.int64)); 
    if len(t) == 0: return 0
    j = 0; best = 0
    for i in range(len(t)):
        while t[i] - t[j] > 48*3600: j += 1
        best = max(best, i - j + 1)
    return best
allw = pd.concat([tx[["from_wallet","timestamp"]].rename(columns={"from_wallet":"wid"}), tx[["to_wallet","timestamp"]].rename(columns={"to_wallet":"wid"})])
f["burst_48h"] = allw.groupby("wid").timestamp.apply(max_burst).reindex(f.index).fillna(0)

# fastest forward: min minutes between an incoming and a following outgoing tx (pass-through speed)
tx_sorted = tx.sort_values("timestamp")
last_in = {}
fast = {}
for r in tx_sorted.itertuples():
    if r.from_wallet in last_in:
        dt, amt = last_in[r.from_wallet]
        mins = (r.timestamp - dt).total_seconds()/60
        if amt >= 5000 and r.amount_usd >= 0.8*amt:
            fast[r.from_wallet] = min(fast.get(r.from_wallet, 1e9), mins)
    last_in[r.to_wallet] = (r.timestamp, r.amount_usd)
f["fastest_forward_min"] = pd.Series(fast).reindex(f.index).fillna(1e5).clip(upper=1e5)
f["fast_forward_flag"] = (f.fastest_forward_min < 180).astype(int)

# 1-hop graph context (label-free): degree of the busiest counterparty, avg counterparty degree
deg = (f.uniq_src + f.uniq_dst)
nb = pd.concat([tx[["from_wallet","to_wallet"]].rename(columns={"from_wallet":"a","to_wallet":"b"}),
                tx[["to_wallet","from_wallet"]].rename(columns={"to_wallet":"a","from_wallet":"b"})]).drop_duplicates()
nb["deg_b"] = nb.b.map(deg)
g = nb.groupby("a").deg_b.agg(max_nb_deg="max", mean_nb_deg="mean")
f = f.join(g).fillna(0)
# neighbour-behaviour features (still label-free): is this wallet transacting with a burst-collector or a pass-through wallet?
is_collector = (f.uniq_src >= 8) & (f.uniq_dst >= 4) & (f.burst_48h >= 12)
is_passthru = (f.v_in >= 8000) & (f.retention.abs() < 0.2) & (f.n_in <= 4)
nb["nb_collector"] = nb.b.map(is_collector).astype(int); nb["nb_passthru"] = nb.b.map(is_passthru).astype(int)
nb["nb_burst"] = nb.b.map(f.burst_48h)
g2 = nb.groupby("a").agg(nb_collector=("nb_collector","sum"), nb_passthru=("nb_passthru","sum"), max_nb_burst=("nb_burst","max"))
f = f.join(g2).fillna(0)
f["kyc_unverified"] = (f.kyc_status == "unverified").astype(int); f["kyc_pending"] = (f.kyc_status == "pending").astype(int)
f["country_high"] = (f.country_risk == "high").astype(int); f["country_med"] = (f.country_risk == "medium").astype(int)
f["log_vol"] = np.log1p(f.v_in + f.v_out)
feat_cols = [c for c in f.columns if c not in ["kyc_status","country_risk"]]
X = f[feat_cols]; y = w.set_index("wallet_id").loc[f.index, "is_illicit"]
role = w.set_index("wallet_id").loc[f.index, "typology_role"]
f.assign(is_illicit=y).to_csv("outputs/wallet_features.csv")

# ---------- split, SMOTE-inside-CV model comparison ----------
Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, y, role, test_size=0.25, stratify=y, random_state=SEED)
print(f"train {len(Xtr)} ({ytr.sum()} illicit)  test {len(Xte)} ({yte.sum()} illicit)")
models = {
  "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, random_state=SEED),
  "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED),
  "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
res, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("scale", RobustScaler()), ("smote", SMOTE(random_state=SEED, k_neighbors=5)), ("clf", clone(m))])
    cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["roc_auc","average_precision","recall","precision"])
    pipe.fit(Xtr, ytr); fitted[name] = pipe
    p = pipe.predict_proba(Xte)[:,1]; pred = (p >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    res.append(dict(model=name, cv_roc_auc=cvr["test_roc_auc"].mean(), cv_pr_auc=cvr["test_average_precision"].mean(),
        test_roc_auc=roc_auc_score(yte, p), test_pr_auc=average_precision_score(yte, p),
        test_precision=precision_score(yte, pred), test_recall=recall_score(yte, pred), tp=tp, fp=fp, fn=fn, tn=tn))
R = pd.DataFrame(res).round(3); print(R.to_string(index=False)); R.to_csv("outputs/model_results.csv", index=False)

# baseline without SMOTE for the best model, to show whether SMOTE helps (honest check)
best = R.sort_values("cv_pr_auc", ascending=False).iloc[0].model  # chosen on CV only, not on the test set
nos = Pipeline([("scale", RobustScaler()), ("clf", clone(models[best]))]).fit(Xtr, ytr)
pn = nos.predict_proba(Xte)[:,1]; prn = (pn >= 0.5).astype(int)
nosmote = dict(model=best, test_roc_auc=round(roc_auc_score(yte, pn),3), test_pr_auc=round(average_precision_score(yte, pn),3),
               test_precision=round(precision_score(yte, prn),3), test_recall=round(recall_score(yte, prn),3))
print("\nBest model without SMOTE:", nosmote)

# recall by wallet role on test set (do mules get caught?)
p = fitted[best].predict_proba(Xte)[:,1]; pred = (p >= 0.5).astype(int)
byrole = pd.DataFrame({"role": rte.values, "y": yte.values, "pred": pred}); byrole = byrole[byrole.y == 1]
rr = byrole.groupby("role").agg(n=("y","size"), caught=("pred","sum")); rr["recall"] = (rr.caught/rr.n).round(3)
print("\nTest-set recall by illicit wallet role (", best, "):\n", rr.to_string()); rr.to_csv("outputs/recall_by_role.csv")

# threshold sensitivity for best model
rows = []
for th in [0.3, 0.5, 0.7, 0.9]:
    pr = (p >= th).astype(int); rows.append(dict(threshold=th, precision=round(precision_score(yte, pr, zero_division=0),3), recall=round(recall_score(yte, pr),3), false_alarms=int(((pr==1)&(yte==0)).sum())))
T = pd.DataFrame(rows); print("\nThreshold sensitivity:\n", T.to_string(index=False)); T.to_csv("outputs/threshold_sensitivity.csv", index=False)

# importances / coefficients
clf = fitted[best].named_steps["clf"]
imp = pd.Series(getattr(clf, "feature_importances_", None), index=feat_cols).sort_values(ascending=False).head(8) if hasattr(clf, "feature_importances_") else None
if imp is not None: print("\nTop features:\n", imp.round(3).to_string()); imp.to_csv("outputs/top_features.csv")

# ROC plot
plt.figure(figsize=(5.5,4.5))
for name, pipe in fitted.items():
    fpr, tpr, _ = roc_curve(yte, pipe.predict_proba(Xte)[:,1]); plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, pipe.predict_proba(Xte)[:,1]):.3f})")
plt.plot([0,1],[0,1],"k--",lw=0.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Wallet-level AML model – held-out test"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("outputs/roc_curves.png", dpi=150)
json.dump(dict(best_model=best, no_smote=nosmote, n_test=int(len(yte)), n_test_illicit=int(yte.sum())), open("outputs/model_summary.json","w"), indent=2)

import sqlite3, pandas as pd, re, json
con = sqlite3.connect("data/chain.db")
wallets = pd.read_sql("select * from wallets", con)
sql = open("sql/aml_red_flags.sql").read()
queries = [q.strip() for q in re.split(r"-- Q\d+ [^\n]*\n", sql)[1:]]
names = ["Q1 structuring", "Q2 rapid pass-through", "Q3 fan-in/fan-out", "Q4 KYC gap"]
typ = {"Q1 structuring": "structuring", "Q2 rapid pass-through": "layering", "Q3 fan-in/fan-out": "fan_in_out", "Q4 KYC gap": "kyc_gap"}
rows, all_flagged = [], set()
for n, q in zip(names, queries):
    flagged = set(pd.read_sql(q, con).wallet_id)
    all_flagged |= flagged
    seeded = set(wallets[(wallets.typology == typ[n]) & (wallets.primary_target == 1)].wallet_id)
    tp = len(flagged & seeded)
    rows.append(dict(rule=n, flagged=len(flagged), seeded_targets=len(seeded), caught=tp,
                     false_positives=len(flagged - set(wallets[wallets.is_illicit == 1].wallet_id)),
                     recall=round(tp / len(seeded), 3), precision=round(len(flagged & set(wallets[wallets.is_illicit==1].wallet_id)) / max(len(flagged),1), 3)))
res = pd.DataFrame(rows); print(res.to_string(index=False))
ill = set(wallets[wallets.is_illicit == 1].wallet_id)
combined = dict(flagged=len(all_flagged), caught=len(all_flagged & ill), total_illicit=len(ill),
                false_positives=len(all_flagged - ill),
                recall=round(len(all_flagged & ill)/len(ill), 3), precision=round(len(all_flagged & ill)/len(all_flagged), 3))
print("\nCombined rules:", combined)
res.to_csv("outputs/sql_rule_results.csv", index=False)
json.dump(combined, open("outputs/sql_combined.json", "w"), indent=2)

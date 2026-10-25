# On-Chain AML & Transaction Monitoring (Synthetic Ledger)

A small, end-to-end blockchain-AML project: SQL red-flag rules, a wallet-level ML risk model, and a
tamper-evident hash-chain ledger check. **All data is synthetic and seeded (seed = 42)**; nothing here is a
production system or real on-chain data.

## Why this project
Public-chain analytics and AML monitoring are a core RegTech use case. This project shows 
- How known laundering patterns look in transaction data,
- How far rules vs. ML get you,
- How a hash chain makes a ledger tamper-evident.

## Data (generate_data.py)
- 43,480 transactions, 3,000 wallets, 870 blocks, 90 days; 125 illicit wallets (4.17%).
- Seeded typologies: **structuring** (repeated transfers just under $10,000), **layering** (rapid pass-through chains),
  **fan-in / fan-out** (collector wallet with mule wallets on both sides), **KYC gap** (unverified, high-risk-country wallets with large flows).
- Deliberate legitimate look-alikes so false positives are realistic: merchants, payroll wallets, large-balance wallets,
  honest users paying just under $10,000, and group-pooling wallets.

## Method
1. **SQL rules (sql/aml_red_flags.sql, run_sql_rules.py)** - four red-flag queries on SQLite.
2. **Wallet model (features_and_model.py)** - 27 label-free features (volume, retention, burstiness in 48h, pass-through speed,
   near-threshold count, KYC/country risk, neighbour behaviour). 75/25 stratified split; scaling + SMOTE applied **inside**
   5-fold CV (imblearn Pipeline) to avoid leakage; Logistic Regression vs Decision Tree vs Random Forest; model picked on CV PR-AUC, scored once on the held-out test set.
3. **Ledger integrity (ledger_integrity.py)** - SHA-256 hash chain with a Merkle root per block and a light proof-of-work.

## Results
**SQL rules** (recall measured against the wallet type each rule is designed to catch):

| Rule | Flagged | Designed targets | Caught | False positives |
|---|---|---|---|---|
| Q1 Structuring | 13 | 14 | 13 | 0 |
| Q2 Rapid pass-through | 11 | 11 | 11 | 0 |
| Q3 Fan-in / fan-out collector | 6 | 4 | 4 | 2 |
| Q4 KYC gap | 13 | 12 | 12 | 1 |

Combined: 40 of 41 designed targets, 3 false positives (precision 0.93). But the rules reach only **32% (40/125) of all illicit wallets**,
because mule, origin and destination wallets look ordinary on their own.

**ML model** (held-out test: 750 wallets, 31 illicit):

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.896 | 0.433 | 0.905 | 0.166 | 0.806 |
| Decision Tree | 0.824 | 0.207 | 0.749 | 0.118 | 0.548 |
| Random Forest (chosen) | 0.905 | 0.468 | 0.910 | 0.241 | 0.452 |

- Random Forest at 0.5: 14 of 31 illicit wallets caught, 44 false alarms. Threshold trade-off: 0.3 gives 90% recall with 128 false alarms; 0.7 gives 53% precision at 26% recall with 7 false alarms.
- SMOTE did **not** improve ranking quality (Random Forest without SMOTE: test ROC-AUC 0.93), it only moved the operating point: recall 19% -> 45% while precision fell from 100% to 24%.
- By role (Random Forest, test): Intermediaries 3/3, structurers 4/4, collectors 1/2, mule-in 5/14, mule-out 1/6, destinations 0/2.

**Ledger integrity:** Altering one transaction amount in block 100 was detected at block 100 (Merkle root mismatch); re-mining block 100 to hide it was detected at block 101 (broken previous-hash link).

## Limitations 
- Synthetic data with typologies I designed; features were built with those typologies in mind, so scores will be optimistic vs. real chains.
- Small test set (31 illicit wallets) - precision/recall swing a lot with the split; no repeated-split confidence intervals.
- Fixed 0.5 default threshold; thresholds in the table are shown on the test set for illustration only, not tuned.
- Mule wallets are the weak spot; label-propagation or graph-neural methods would be the next step.
- Not a blockchain implementation: the ledger demo is a teaching-scale hash chain, no consensus, no network, no smart contracts.
- The rule thresholds (e.g. $9,000-$9,999, $15,000 materiality floor) are illustrative, not regulatory values.

## Run it
```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
python generate_data.py && python run_sql_rules.py && python features_and_model.py && python ledger_integrity.py
```

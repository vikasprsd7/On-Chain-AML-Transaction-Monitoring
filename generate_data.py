"""
Generate a fully synthetic, seeded on-chain transaction ledger with embedded AML typologies.

NOTE: All data is synthetic. Typologies are seeded so detection can be measured.
Legitimate "decoy" wallets (payroll, merchants, whales, occasional near-threshold payers)
are included on purpose so that rules and models produce realistic false positives.
"""
import numpy as np, pandas as pd, sqlite3, hashlib

SEED = 42
rng = np.random.default_rng(SEED)
N_WALLETS = 3000
START = pd.Timestamp("2026-01-01")
DAYS = 90
THRESHOLD = 10_000          # reporting-style threshold (USD)

def wid(i): return "0x" + hashlib.sha1(f"w{i}".encode()).hexdigest()[:12]
wallets = pd.DataFrame({"wallet_id": [wid(i) for i in range(N_WALLETS)]})
wallets["kyc_status"] = rng.choice(["verified", "pending", "unverified"], N_WALLETS, p=[0.72, 0.13, 0.15])
wallets["country_risk"] = rng.choice(["low", "medium", "high"], N_WALLETS, p=[0.78, 0.15, 0.07])
wallets["created_day"] = rng.integers(0, 400, N_WALLETS)
wallets["role"] = "retail"
wallets["typology"] = "none"
wallets["typology_role"] = "none"
wallets["primary_target"] = 0

idx = np.arange(N_WALLETS)
rng.shuffle(idx)
pool = list(idx)
def take(n): 
    out = [pool.pop() for _ in range(n)]; return out

txs = []
def add(src, dst, amt, ts):
    txs.append((src, dst, round(float(amt), 2), ts))

def rand_ts(day_lo=0, day_hi=DAYS):
    return START + pd.Timedelta(seconds=int(rng.integers(day_lo*86400, day_hi*86400)))

# ---- Legitimate decoys -------------------------------------------------
merchants = take(30); payroll = take(10); whales = take(15); near_thr_legit = take(40)
for m in merchants: wallets.loc[m, "role"] = "merchant"
for p in payroll:   wallets.loc[p, "role"] = "payroll"
for w in whales:    wallets.loc[w, "role"] = "whale"
for n in near_thr_legit: wallets.loc[n, "role"] = "legit_near_threshold"

# retail background
retail_pool = [i for i in range(N_WALLETS)]
for _ in range(38000):
    s, d = rng.choice(retail_pool, 2, replace=False)
    amt = float(np.clip(rng.lognormal(5.0, 1.3), 5, 6000))
    add(wid(s), wid(d), amt, rand_ts())
# merchants receive from many wallets (legit fan-in)
for m in merchants:
    for _ in range(rng.integers(60, 160)):
        s = rng.choice(retail_pool); 
        if s == m: continue
        add(wid(s), wid(m), float(np.clip(rng.lognormal(4.3, 0.9), 3, 1500)), rand_ts())
    # merchants pay out to a supplier in bulk, but only occasionally
    for _ in range(rng.integers(2, 6)):
        add(wid(m), wid(rng.choice(retail_pool)), float(rng.uniform(2000, 9000)), rand_ts())
# payroll wallets fan-out monthly (legit fan-out)
for p in payroll:
    emps = rng.choice(retail_pool, rng.integers(25, 60), replace=False)
    for month in range(3):
        for e in emps:
            if e == p: continue
            add(wid(p), wid(e), float(rng.uniform(900, 4200)), START + pd.Timedelta(days=month*30 + int(rng.integers(0, 2)), minutes=int(rng.integers(0, 600))))
# whales: large legit transfers incl. over threshold
for w in whales:
    for _ in range(rng.integers(8, 25)):
        add(wid(w), wid(rng.choice(retail_pool)), float(rng.uniform(15_000, 90_000)), rand_ts())
# legit users who sometimes pay just under the threshold (e.g. rent, car deposits)
for n in near_thr_legit:
    for _ in range(rng.integers(1, 4)):
        add(wid(n), wid(rng.choice(retail_pool)), float(rng.uniform(9000, 9900)), rand_ts())


# legit "pooling" wallets (group purchases / crowdfunding) that also collect from many and pay out to several
poolers = take(4)
for p in poolers:
    wallets.loc[p, "role"] = "legit_pool"
    for burst in range(2):
        t0 = rand_ts(0, DAYS-4)
        srcs = rng.choice(retail_pool, int(rng.integers(9, 18)), replace=False)
        tot = 0
        for s_ in srcs:
            if s_ == p: continue
            a = float(rng.uniform(300, 3000)); tot += a
            add(wid(s_), wid(p), a, t0 + pd.Timedelta(hours=int(rng.integers(0, 40))))
        for d_ in rng.choice(retail_pool, int(rng.integers(4, 7)), replace=False):
            if d_ == p: continue
            add(wid(p), wid(d_), tot / 5 * rng.uniform(0.9, 0.99), t0 + pd.Timedelta(hours=int(rng.integers(41, 70))))

# ---- Seeded illicit typologies ----------------------------------------
# 1. Structuring: repeated transfers just below threshold within a short window
struct = take(14)
for s in struct:
    wallets.loc[s, ["typology","role","typology_role","primary_target"]] = ["structuring","illicit","structurer",1]
    dst = rng.choice(retail_pool, 3, replace=False)
    t0 = rand_ts(0, DAYS-5)
    k = rng.integers(4, 9)
    for j in range(k):
        # ~25% of the time the amount varies more widely to resemble normal behaviour
        amt = rng.uniform(9000, 9950) if rng.random() > 0.25 else rng.uniform(6500, 9000)
        add(wid(s), wid(rng.choice(dst)), amt, t0 + pd.Timedelta(hours=int(rng.integers(0, 60))))

# 2. Layering / rapid pass-through chains: A -> B -> C -> D within minutes, ~85-99% forwarded
n_chains = 6; chain_members = []
for c in range(n_chains):
    L = int(rng.integers(3, 6)); chain = take(L)
    chain_members += chain
    amt = float(rng.uniform(12_000, 60_000)); t = rand_ts(0, DAYS-2)
    for k, w in enumerate(chain):
        r = "origin" if k == 0 else ("destination" if k == L-1 else "intermediary")
        wallets.loc[w, ["typology","role","typology_role","primary_target"]] = ["layering","illicit",r, int(r=="intermediary")]
    for a, b in zip(chain[:-1], chain[1:]):
        amt *= rng.uniform(0.85, 0.99)
        t += pd.Timedelta(minutes=int(rng.integers(2, 90)))
        add(wid(a), wid(b), amt, t)
# 3. Fan-in then fan-out through a collector
n_clusters = 4
for c in range(n_clusters):
    collector = take(1)[0]; sources = take(int(rng.integers(8, 16))); sinks = take(int(rng.integers(4, 8)))
    wallets.loc[collector, ["typology","role","typology_role","primary_target"]] = ["fan_in_out","illicit","collector",1]
    for s in sources: wallets.loc[s, ["typology","role","typology_role"]] = ["fan_in_out","illicit","mule_in"]
    for s in sinks:   wallets.loc[s, ["typology","role","typology_role"]] = ["fan_in_out","illicit","mule_out"]
    t0 = rand_ts(0, DAYS-4); total = 0
    for s in sources:
        a = float(rng.uniform(800, 3500)); total += a
        add(wid(s), wid(collector), a, t0 + pd.Timedelta(hours=int(rng.integers(0, 36))))
    t1 = t0 + pd.Timedelta(hours=40)
    for s in sinks:
        add(wid(collector), wid(s), total / len(sinks) * rng.uniform(0.9, 0.99), t1 + pd.Timedelta(hours=int(rng.integers(0, 24))))
# 4. High-risk-country, unverified wallets with unusually large flows (KYC gap typology)
kyc_gap = take(12)
for w in kyc_gap:
    wallets.loc[w, ["typology","role","typology_role","primary_target","kyc_status","country_risk"]] = ["kyc_gap","illicit","kyc_gap",1,"unverified","high"]
    for _ in range(rng.integers(6, 14)):
        add(wid(w), wid(rng.choice(retail_pool)), float(rng.uniform(8_000, 40_000)), rand_ts())
        add(wid(rng.choice(retail_pool)), wid(w), float(rng.uniform(8_000, 40_000)), rand_ts())

tx = pd.DataFrame(txs, columns=["from_wallet", "to_wallet", "amount_usd", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
tx["tx_id"] = ["tx" + str(i).zfill(6) for i in range(len(tx))]
tx["block_number"] = (tx.index // 50).astype(int)
tx["gas_fee_usd"] = np.round(rng.gamma(2.0, 0.6, len(tx)), 3)
tx = tx[["tx_id", "block_number", "timestamp", "from_wallet", "to_wallet", "amount_usd", "gas_fee_usd"]]
wallets["is_illicit"] = (wallets["role"] == "illicit").astype(int)

tx["timestamp"] = tx["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
tx.to_csv("data/transactions.csv", index=False)
wallets.to_csv("data/wallets.csv", index=False)
con = sqlite3.connect("data/chain.db")
tx.to_sql("transactions", con, if_exists="replace", index=False)
wallets.to_sql("wallets", con, if_exists="replace", index=False)
con.close()
print(f"transactions: {len(tx):,} | wallets: {len(wallets):,} | illicit wallets: {wallets.is_illicit.sum()} ({wallets.is_illicit.mean():.2%})")
print(wallets[wallets.is_illicit == 1].typology.value_counts().to_string())

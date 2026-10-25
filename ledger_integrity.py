"""Tamper-evident ledger demo: SHA-256 hash chain + Merkle root per block over the synthetic transactions."""
import pandas as pd, hashlib, json, copy

def h(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()

def merkle_root(leaves):
    level = [h(x) for x in leaves] or [h("")]
    while len(level) > 1:
        if len(level) % 2: level.append(level[-1])
        level = [h(level[i] + level[i+1]) for i in range(0, len(level), 2)]
    return level[0]

def build_chain(tx: pd.DataFrame):
    chain, prev = [], "0" * 64
    for bn, g in tx.groupby("block_number"):
        leaves = [f"{r.tx_id}|{r.from_wallet}|{r.to_wallet}|{r.amount_usd:.2f}|{r.timestamp}" for r in g.itertuples()]
        mr = merkle_root(leaves)
        # light proof-of-work: hash must start with '000'
        nonce = 0
        while True:
            bh = h(f"{bn}|{prev}|{mr}|{nonce}")
            if bh.startswith("000"): break
            nonce += 1
        chain.append(dict(block=int(bn), prev_hash=prev, merkle_root=mr, nonce=nonce, hash=bh, leaves=leaves))
        prev = bh
    return chain

def verify(chain):
    prev = "0" * 64
    for b in chain:
        if b["prev_hash"] != prev: return False, b["block"], "broken prev_hash link"
        if merkle_root(b["leaves"]) != b["merkle_root"]: return False, b["block"], "merkle root mismatch (transaction altered)"
        if h(f"{b['block']}|{b['prev_hash']}|{b['merkle_root']}|{b['nonce']}") != b["hash"]: return False, b["block"], "block hash mismatch"
        prev = b["hash"]
    return True, None, "chain valid"

if __name__ == "__main__":
    tx = pd.read_csv("data/transactions.csv")
    chain = build_chain(tx)
    ok, blk, msg = verify(chain); print(f"blocks: {len(chain)} | original chain: {msg}")
    tampered = copy.deepcopy(chain); victim = 100
    parts = tampered[victim]["leaves"][3].split("|"); parts[5 - 2] = "9999.99"   # alter amount of one tx
    tampered[victim]["leaves"][3] = "|".join(parts)
    ok2, blk2, msg2 = verify(tampered); print(f"tampered block {victim}: valid={ok2} | detected at block {blk2} | {msg2}")
    # rewriting the block's merkle root + hash still breaks the next block's prev_hash link
    t2 = copy.deepcopy(tampered); b = t2[victim]; b["merkle_root"] = merkle_root(b["leaves"]); 
    nonce = 0
    while not h(f"{b['block']}|{b['prev_hash']}|{b['merkle_root']}|{nonce}").startswith("000"): nonce += 1
    b["nonce"] = nonce; b["hash"] = h(f"{b['block']}|{b['prev_hash']}|{b['merkle_root']}|{nonce}")
    ok3, blk3, msg3 = verify(t2); print(f"attacker re-mines block {victim}: valid={ok3} | detected at block {blk3} | {msg3}")
    json.dump(dict(blocks=len(chain), tamper_block=victim, detect_naive=[blk2, msg2], detect_remined=[blk3, msg3]), open("outputs/ledger_integrity.json", "w"), indent=2)

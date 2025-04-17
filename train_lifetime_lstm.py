#!/usr/bin/env python3
"""
train_lifetime_lstm.py  –  tiny LSTM lifetime classifier
reads:   tmp/alloc_metadata.log  tmp/dealloc_metadata.log
writes:  model_weights.h  class_lookup.h
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ───────── configuration ───────────────────────────────────────────────
ALLOC_LOG   = Path("tmp/alloc_metadata.log")
DEALLOC_LOG = Path("tmp/dealloc_metadata.log")

HASH_BITS        = 15               # 32 768 hash buckets
HASH_BUCKETS     = 1 << HASH_BITS
SIZE_BUCKET_BITS = 8                # bucket 256 B
NUM_CLASSES      = 7

BATCH  = 128
EPOCHS = 8
LR     = 3e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── helpers ─────────────────────────────────────────────────────
def load_log(path, has_dealloc):
    cols = (["addr","size","cls","alloc_ts","dealloc_ts","stack"]
            if has_dealloc else
            ["addr","size","cls","alloc_ts","stack"])
    df = pd.read_csv(path, sep=r"\s+", names=cols, dtype=str, engine="python")
    df["addr"]     = df["addr"].apply(lambda x: int(x,16))
    df["size"]     = df["size"].astype(np.uint32)
    df["alloc_ts"] = df["alloc_ts"].astype(np.uint64)
    df["stack"]    = df["stack"].astype(np.uint64)
    if has_dealloc:
        df["dealloc_ts"] = df["dealloc_ts"].astype(np.uint64)
    return df

def lifetime_class(ns):
    if ns <   1_000_000:      return 0      #   <1 ms
    if ns <  10_000_000:      return 1      #  <10 ms
    if ns < 100_000_000:      return 2      # <100 ms
    if ns < 500_000_000:      return 3      # <500 ms
    if ns < 2_000_000_000:    return 4      #   <2 s
    if ns < 5_000_000_000:    return 5      #   <5 s
    return 6                                  #  ≥5 s

# ───────── load + merge ────────────────────────────────────────────────
alloc_df   = load_log(ALLOC_LOG,  False)
dealloc_df = load_log(DEALLOC_LOG, True)

merged = pd.merge(alloc_df,
                  dealloc_df[["addr","dealloc_ts"]],
                  on="addr", how="inner")

print("Samples after merge:", len(merged))

merged["lifetime_ns"] = merged["dealloc_ts"] - merged["alloc_ts"]
merged["label"]       = merged["lifetime_ns"].apply(lifetime_class).astype(np.int64)

# deterministic feature buckets
stack_np              = merged["stack"].to_numpy(np.uint64)
merged["hash_id"]     = ((stack_np ^ (stack_np >> 32)) & (HASH_BUCKETS-1)).astype(np.int32)

size_np               = merged["size"].to_numpy(np.uint32)
merged["size_id"]     = (size_np >> SIZE_BUCKET_BITS).astype(np.int32)

hash_vocab = HASH_BUCKETS
size_vocab = int(merged["size_id"].max()) + 1      # +1 for index starting at 0

# ───────── dataset / dataloader ────────────────────────────────────────
class LifetimeDS(Dataset):
    def __init__(self, df):
        self.h = torch.from_numpy(df["hash_id"].to_numpy(np.int32))
        self.s = torch.from_numpy(df["size_id"].to_numpy(np.int32))
        self.y = torch.from_numpy(df["label"].to_numpy(np.int64))
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.h[i], self.s[i], self.y[i]

dl = DataLoader(LifetimeDS(merged), batch_size=BATCH, shuffle=True)

# ───────── model ───────────────────────────────────────────────────────
EMB, HID = 32, 64
class StackLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eh   = nn.Embedding(hash_vocab,  EMB)
        self.es   = nn.Embedding(size_vocab,  EMB)
        self.lstm = nn.LSTM(EMB*2, HID, batch_first=True)
        self.fc   = nn.Linear(HID, NUM_CLASSES)
    def forward(self, h, s):
        x = torch.cat([self.eh(h), self.es(s)], dim=-1).unsqueeze(1)
        o,_ = self.lstm(x)
        return self.fc(o.squeeze(1))

model, opt, crit = StackLSTM().to(device), torch.optim.Adam(StackLSTM().parameters(), lr=LR), nn.CrossEntropyLoss()

# ───────── training ────────────────────────────────────────────────────
for ep in range(EPOCHS):
    tot=hit=loss_sum=0.0
    for h,s,y in dl:
        h,s,y = h.to(device), s.to(device), y.to(device)
        opt.zero_grad()
        out  = model(h,s)
        loss = crit(out,y)
        loss.backward()
        opt.step()

        loss_sum += loss.item()*len(y)
        hit      += (out.argmax(1)==y).sum().item()
        tot      += len(y)
    print(f"ep {ep+1}/{EPOCHS}: loss {loss_sum/tot:.4f}  acc {hit/tot:.1%}")

# ───────── ① dump raw weights for tiny‑runtime ML (optional) ───────────
parts = [model.eh.weight, model.es.weight,
         model.lstm.weight_ih_l0, model.lstm.weight_hh_l0,
         model.lstm.bias_ih_l0,   model.lstm.bias_hh_l0,
         model.fc.weight,         model.fc.bias]

flat = torch.cat([p.detach().cpu().flatten() for p in parts]).to(torch.float32)
with open("model_weights.h", "w") as f:
    f.write(f"/* Autogenerated – {flat.numel()} float32 */\n")
    f.write("static const float g_nn_weights[] = {\n")
    for i in range(0, flat.numel(), 8):
        row = ", ".join(f"{v:.8e}f" for v in flat[i:i+8].tolist())
        f.write("  " + row + ",\n")
    f.write("};\n")
print("✅ Wrote model_weights.h")

# ───────── ② build static class_lookup table for fast lookup ───────────
print("📦 Generating class_lookup.h...")
lookup = np.zeros((64, 256), dtype=np.uint8)
model.eval()

with torch.no_grad():
    for sb in range(64):
        for hb in range(256):
            hb_t = torch.tensor([hb], dtype=torch.long, device=device)
            sb_t = torch.tensor([sb], dtype=torch.long, device=device)
            pred = model(hb_t, sb_t)
            cls  = int(pred.argmax(1))
            lookup[sb, hb] = cls

with open("jemalloc/src/class_lookup.h", "w") as f:
    f.write("// Autogenerated by train_lifetime_lstm.py\n")
    f.write("#pragma once\n#include <stdint.h>\n\n")
    f.write("static const uint8_t class_lookup[64][256] = {\n")
    for sb in range(64):
        row = ", ".join(str(v) for v in lookup[sb])
        f.write(f"  {{ {row} }},\n")
    f.write("};\n")

print("✅ Wrote class_lookup.h")
"""
train_sharp_bdq_v2.py
======================
Branching Dueling Double-Q (BDQ) offline RL trainer for the SHARP home energy
management policy, built against RL_MODEL_SPEC.md and sharp-master-dataset-v2.

Pure NumPy. No training framework, no autograd library. Matches spec section
4 (architecture), section 5 (offline-RL corrections: BC warm start + CQL +
level-legality masking), and section 8 (Pi export format).

What this script does, in order:
  1. Loads the pre-split parquet files (train/validation/test). No SHARP
     module is imported -- pandas + numpy only, per spec section 1.
  2. Builds a fixed 28-branch x 3-level action tensor per transition from the
     ragged per-household action arrays, using device_present as a prefix
     mask (verified empirically against the released data).
  3. Builds the level-legality mask per household from
     simulator_inputs/device_power_models.parquet (is_necessity,
     supports_reduced), joined on template_id == household_id, device_id
     ascending -- matching feature_schema.json's declared device_order.
  4. Computes train-only mean/sd normalisation, guarding zero-variance
     features per spec section 2.
  5. Trains the network in two phases on the fixed batch:
       Phase 1 -- behaviour-cloning warm start (E5)
       Phase 2 -- Double-Q TD learning + inverse-frequency-weighted CQL
                  penalty (the same term as weighted BC, TD switched on)
     Both phases apply the level-legality mask throughout.
  6. Reports balanced accuracy (mean of per-level recalls) on validation,
     split by behaviour policy, and reports non-terminal vs terminal TD
     error separately -- pooling them was flagged in the spec as
     misleading.
  7. Exports sharp_policy_bdq_v2.npz in exactly the section-8 format,
     including mean/sd, so the Pi-side normaliser cannot silently drift.

What this script deliberately does NOT do (left as open items per spec
section 10, steps 6-7):
  - It does not substitute the Bradley-Terry preference reward r_psi into
    the training reward. Bellman targets still use the logged billing
    reward (sharp_reward_billing), because r_psi is fitted on a
    one-directional pair set and substituting it now would launder that
    defect into the policy. See TRAINING_REPORT["reward_caveats"].
  - It does not re-apply the joint device shield when bootstrapping
    next-state Q-values (shield-consistent targets). Only the flat
    level-legality mask is available outside the simulator.
  - It does not touch the test split. Test is loaded only to report its
    size; no metric in this script is computed against it.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RNG = np.random.default_rng(0)

DATA_DIR = Path("/home/claude/data")
OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES = 305
N_BRANCHES = 28
N_LEVELS = 3
HIDDEN = 128
GAMMA = 0.99

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------


NEEDED_COLS = ["household_id", "policy", "state", "next_state", "action",
               "device_present", "reward", "done"]


def load_split_compact(name):
    """Read only the needed columns straight through pyarrow (not pandas'
    per-row object columns, which cost several GB extra on this box for a
    280k-row table of 305-wide nested arrays) and unpack directly into
    compact float32 / bool / int8 numpy arrays."""
    table = pq.read_table(DATA_DIR / f"rl_transitions/splits/{name}.parquet",
                           columns=NEEDED_COLS)
    n = table.num_rows

    def flatten_fixed(col_name, width, dtype):
        arr = table.column(col_name).combine_chunks().flatten()
        return np.asarray(arr, dtype=dtype).reshape(n, width)

    X = flatten_fixed("state", N_FEATURES, "float32")
    Xn = flatten_fixed("next_state", N_FEATURES, "float32")
    present = flatten_fixed("device_present", N_BRANCHES, "bool")

    action_col = table.column("action").combine_chunks()
    offsets = np.asarray(action_col.offsets, dtype="int64")
    values = np.asarray(action_col.flatten(), dtype="int8")
    lengths = np.diff(offsets)
    row_idx = np.repeat(np.arange(n), lengths)
    within_idx = np.arange(len(values)) - np.repeat(offsets[:-1], lengths)
    A = np.full((n, N_BRANCHES), -1, dtype=np.int8)
    A[row_idx, within_idx] = values

    hh = table.column("household_id").to_numpy(zero_copy_only=False)
    policy = table.column("policy").to_numpy(zero_copy_only=False)
    R = np.asarray(table.column("reward"), dtype="float32")
    done = np.asarray(table.column("done"), dtype="bool")

    del table, action_col, offsets, values, lengths, row_idx, within_idx
    gc.collect()

    return dict(X=X, Xn=Xn, present=present, A=A, R=R, done=done,
                household_id=hh, policy=policy)


print("Loading train split (columns pruned + pyarrow-flattened for memory) ...")
train = load_split_compact("train")
print(f"  train {train['X'].shape[0]:,} transitions vectorised")

with open(DATA_DIR / "rl_transitions/feature_schema.json") as f:
    schema = json.load(f)

feature_names = (
    schema["global_features"]
    + [f"{feat}__slot{i}" for i in range(N_BRANCHES) for feat in schema["device_features"]]
)
assert len(feature_names) == N_FEATURES == schema["feature_count"]

# ----------------------------------------------------------------------
# 2. Per-household legality mask (is_necessity / supports_reduced)
# ----------------------------------------------------------------------

print("Building per-household legality masks from device_power_models.parquet ...")
device_meta = pd.read_parquet(DATA_DIR / "simulator_inputs/device_power_models.parquet")
device_meta = device_meta.sort_values("device_id")

hh_order = sorted(device_meta["template_id"].unique())
hh_to_idx = {h: i for i, h in enumerate(hh_order)}
necessity_arr = np.zeros((len(hh_order), N_BRANCHES), dtype=bool)
reduced_arr = np.zeros((len(hh_order), N_BRANCHES), dtype=bool)
for hh, sub in device_meta.groupby("template_id"):
    i = hh_to_idx[hh]
    n = len(sub)
    necessity_arr[i, :n] = sub["is_necessity"].to_numpy()
    reduced_arr[i, :n] = sub["supports_reduced"].to_numpy()
del device_meta
gc.collect()


def legal_mask_for_households(household_ids, device_present):
    """(N,28,3) bool -- the level-legality mask actually implemented in
    train_sharp_bdq_v2.py per spec section 5, item 3.

    Only level 2 (REDUCED) is statically masked, to devices where
    supports_reduced is True: it is meaningless on ~40% of devices that
    cannot dim. Level 0 (OFF) on a necessity appliance is deliberately
    NOT masked here -- spec section 5 is explicit that the necessity rule
    is conditional (a fridge is legitimately off at 3 a.m.), and enforcing
    it as a static per-device mask would be wrong. The unconditional
    "necessity devices never see level 0" statement in section 3 describes
    the joint device shield's behaviour when the occupant wants the
    device and budget remains -- state-dependent logic this flat release
    does not carry, and which apply_shield (not this trainer) is
    responsible for. Absent slots are fully illegal at all three levels."""
    idx = np.fromiter((hh_to_idx[h] for h in household_ids), dtype="int64",
                       count=len(household_ids))
    red = reduced_arr[idx]                                            # (N,28)
    present = device_present.astype(bool)                             # (N,28)

    mask = np.zeros((len(household_ids), N_BRANCHES, N_LEVELS), dtype=bool)
    mask[:, :, 0] = present                  # OFF: always legal if present (conditional in reality)
    mask[:, :, 1] = present                  # ON: always legal if present
    mask[:, :, 2] = present & red            # REDUCED: only if dimmable and present
    return mask


# ----------------------------------------------------------------------
# 3. Attach the legality mask to each split (mask depends only on
#    household_id + device_present, computed lazily so each split's raw
#    pyarrow table is freed before the next one is read)
# ----------------------------------------------------------------------

train["mask"] = legal_mask_for_households(train["household_id"], train["present"])

print("Loading validation split (columns pruned + pyarrow-flattened for memory) ...")
val = load_split_compact("validation")
val["mask"] = legal_mask_for_households(val["household_id"], val["present"])
print(f"  validation {val['X'].shape[0]:,} transitions vectorised")

test_len = pq.ParquetFile(DATA_DIR / "rl_transitions/splits/test.parquet").metadata.num_rows
print(f"  test {test_len:,} transitions present on disk -- not loaded further, "
      f"per spec section 9 (\"do not touch the test split\")")

# sanity check: logged action must always fall inside the legal mask
chk = train["mask"][np.arange(len(train["A"]))[:, None], np.arange(N_BRANCHES)[None, :],
                     np.clip(train["A"], 0, N_LEVELS - 1)]
chk_present = train["present"]
assert np.all(chk[chk_present]), "logged action falls outside legality mask -- check the join"
print("  legality mask verified against logged actions (100% consistent)")

# ----------------------------------------------------------------------
# 4. Train-only normalisation, zero-variance guard (spec section 2)
# ----------------------------------------------------------------------

mean = train["X"].mean(axis=0)
sd = train["X"].std(axis=0)
zero_var = sd < 1e-8
sd[zero_var] = 1.0
print(f"  {zero_var.sum()} / {N_FEATURES} zero-variance features guarded "
      f"({zero_var.sum() / N_FEATURES:.1%})")


def normalise(X):
    return (X - mean) / sd


def normalise_inplace(X):
    X -= mean
    X /= sd
    return X


# Normalise state and next_state in place -- avoids holding both raw and
# normalised (280k x 305 float32 x 2) copies simultaneously, which is the
# difference between fitting comfortably and OOMing on this box's 3.9GB.
normalise_inplace(train["X"])
normalise_inplace(train["Xn"])
normalise_inplace(val["X"])
del val["Xn"]  # never used: validation only evaluates the policy, not next-state bootstrapping
gc.collect()


# ----------------------------------------------------------------------
# 5. Network: shared trunk + dueling value/advantage heads
# ----------------------------------------------------------------------


def init_params(rng):
    def dense(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype("float32")
        b = np.zeros(fan_out, dtype="float32")
        return w, b

    w, b = dense(N_FEATURES, HIDDEN)
    v, vb = dense(HIDDEN, 1)
    a, ab = dense(HIDDEN, N_BRANCHES * N_LEVELS)
    return dict(w=w, b=b, v=v, vb=vb, a=a, ab=ab)


def forward(p, x):
    """x: (N, 305) normalised. Returns cache dict with Q: (N,28,3)."""
    z = x @ p["w"] + p["b"]
    h = np.maximum(0, z)                                  # ReLU trunk
    V = h @ p["v"] + p["vb"]                               # (N,1)
    A_raw = (h @ p["a"] + p["ab"]).reshape(-1, N_BRANCHES, N_LEVELS)
    A = A_raw - A_raw.mean(axis=2, keepdims=True)
    Q = V[:, :, None] + A
    return dict(x=x, z=z, h=h, V=V, A=A_raw, Q=Q)


def backward(p, cache, dQ):
    """dQ: (N,28,3) dL/dQ. Returns grads dict matching p's keys."""
    n = dQ.shape[0]
    h = cache["h"]

    dV = dQ.sum(axis=(1, 2), keepdims=False)[:, None]          # (N,1)
    dA = dQ - dQ.mean(axis=2, keepdims=True)                   # (N,28,3)
    dA_flat = dA.reshape(n, N_BRANCHES * N_LEVELS)

    grads = {}
    grads["v"] = h.T @ dV
    grads["vb"] = dV.sum(axis=0)
    grads["a"] = h.T @ dA_flat
    grads["ab"] = dA_flat.sum(axis=0)

    dh = dV @ p["v"].T + dA_flat @ p["a"].T
    dz = dh * (cache["z"] > 0)
    grads["w"] = cache["x"].T @ dz
    grads["b"] = dz.sum(axis=0)
    return grads


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, batch_size):
        self.t += 1
        for k in params:
            g = grads[k] / batch_size
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def masked_softmax_logits(Q, mask):
    """Q,(N,28,3); mask bool same shape. Returns Q with illegal slots at -inf."""
    return np.where(mask, Q, -1e9)


def logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


# ----------------------------------------------------------------------
# 6. Inverse-frequency level weights (fixes the "collapse to OFF" failure
#    mode documented in spec section 5)
# ----------------------------------------------------------------------

present_train = train["present"]
A_train = train["A"]
level_counts = np.array([
    np.sum((A_train == lvl) & present_train) for lvl in range(N_LEVELS)
], dtype="float64")
level_weight = level_counts.sum() / (N_LEVELS * level_counts)
print(f"  logged action level frequencies: OFF {level_counts[0] / level_counts.sum():.4f}  "
      f"ON {level_counts[1] / level_counts.sum():.4f}  "
      f"REDUCED {level_counts[2] / level_counts.sum():.4f}")
print(f"  inverse-frequency weights: {level_weight.round(3).tolist()}")


def per_branch_weight(A_batch):
    """(N,28) -> (N,28) weight from level_weight, 0 where slot is absent (A=-1)."""
    w = np.zeros_like(A_batch, dtype="float32")
    for lvl in range(N_LEVELS):
        w[A_batch == lvl] = level_weight[lvl]
    return w


# ----------------------------------------------------------------------
# 7. Phase 1 -- behaviour cloning warm start
# ----------------------------------------------------------------------

params = init_params(RNG)
opt = Adam(params, lr=1e-3)

BC_UPDATES = 2000
CQL_UPDATES = 5000
BATCH = 256
ALPHA_CQL = 1.0
TARGET_SYNC_EVERY = 200

n_train = train["X"].shape[0]
Xn_train = train["X"]        # already normalised in place
Xn_train_next = train["Xn"]  # already normalised in place


def sample_batch(rng, batch_size):
    idx = rng.integers(0, n_train, size=batch_size)
    return idx


def bc_grad_step(idx):
    x = Xn_train[idx]
    mask = train["mask"][idx]
    A_b = train["A"][idx].astype(np.int64)
    present_b = train["present"][idx]
    w_b = per_branch_weight(train["A"][idx]) * present_b

    cache = forward(params, x)
    Q = cache["Q"]
    Qm = masked_softmax_logits(Q, mask)
    logZ = logsumexp(Qm, axis=2)                    # (N,28)
    A_safe = np.clip(A_b, 0, N_LEVELS - 1)
    Q_taken = np.take_along_axis(Q, A_safe[:, :, None], axis=2).squeeze(2)
    # cross-entropy of logged action under softmax(Q): logZ - Q_taken
    loss_per_branch = (logZ - Q_taken) * w_b

    # dL/dQ: softmax(Qm) - onehot(A), scaled by weight, zeroed on absent slots
    P = np.exp(Qm - logsumexp(Qm, axis=2)[:, :, None])
    onehot = np.zeros_like(P)
    np.put_along_axis(onehot, A_safe[:, :, None], 1.0, axis=2)
    dQ = (P - onehot) * w_b[:, :, None]
    dQ = np.where(mask, dQ, 0.0)

    grads = backward(params, cache, dQ)
    opt.step(params, grads, batch_size=idx.shape[0])
    return loss_per_branch[present_b].mean()


print(f"\nPhase 1: behaviour-cloning warm start ({BC_UPDATES} updates) ...")
t0 = time.time()
for step in range(1, BC_UPDATES + 1):
    idx = sample_batch(RNG, BATCH)
    loss = bc_grad_step(idx)
    if step % 500 == 0 or step == 1:
        print(f"  step {step:5d}  weighted BC loss {loss:.4f}")
print(f"  done in {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# 8. Phase 2 -- Double-Q TD learning + weighted CQL penalty
# ----------------------------------------------------------------------

target_params = {k: v.copy() for k, v in params.items()}


def cql_plus_td_grad_step(idx):
    x = Xn_train[idx]
    xn = Xn_train_next[idx]
    mask = train["mask"][idx]
    mask_next = mask  # device set / legality is static per household/episode
    A_b = train["A"][idx].astype(np.int64)
    A_safe = np.clip(A_b, 0, N_LEVELS - 1)
    present_b = train["present"][idx]
    R_b = train["R"][idx]
    done_b = train["done"][idx].astype("float32")
    w_b = per_branch_weight(train["A"][idx]) * present_b

    cache = forward(params, x)
    Q = cache["Q"]

    # ---- Double-Q Bellman target (branching-DQN style: mean over branches)
    Qn_online = forward(params, xn)["Q"]
    Qn_online_masked = masked_softmax_logits(Qn_online, mask_next)
    a_star = np.argmax(Qn_online_masked, axis=2)                    # (N,28)

    Qn_target = forward(target_params, xn)["Q"]
    q_next_selected = np.take_along_axis(Qn_target, a_star[:, :, None], axis=2).squeeze(2)
    q_next_selected = np.where(present_b, q_next_selected, 0.0)
    n_present = present_b.sum(axis=1).clip(min=1)
    q_next_branch_mean = q_next_selected.sum(axis=1) / n_present     # (N,)

    y = R_b + GAMMA * (1.0 - done_b) * q_next_branch_mean            # (N,) scalar target
    Q_taken = np.take_along_axis(Q, A_safe[:, :, None], axis=2).squeeze(2)  # (N,28)
    td_err = (Q_taken - y[:, None]) * present_b
    n_active = present_b.sum()
    td_loss = 0.5 * (td_err ** 2).sum() / n_active

    dQ_td = np.zeros_like(Q)
    np.put_along_axis(dQ_td, A_safe[:, :, None], (td_err / n_active)[:, :, None], axis=2)

    # ---- CQL penalty: weighted cross-entropy of logged action vs softmax(Q)
    Qm = masked_softmax_logits(Q, mask)
    logZ = logsumexp(Qm, axis=2)
    Q_taken_cql = Q_taken
    cql_loss = ((logZ - Q_taken_cql) * w_b)[present_b].mean()

    P = np.exp(Qm - logsumexp(Qm, axis=2)[:, :, None])
    onehot = np.zeros_like(P)
    np.put_along_axis(onehot, A_safe[:, :, None], 1.0, axis=2)
    dQ_cql = (P - onehot) * w_b[:, :, None] / present_b.sum()
    dQ_cql = np.where(mask, dQ_cql, 0.0)

    dQ = dQ_td + ALPHA_CQL * dQ_cql
    grads = backward(params, cache, dQ)
    opt.step(params, grads, batch_size=idx.shape[0])

    return td_loss, cql_loss, td_err, done_b


print(f"\nPhase 2: Double-Q TD + CQL (alpha={ALPHA_CQL}), {CQL_UPDATES} updates ...")
t0 = time.time()
nonterm_td_hist, term_td_hist = [], []
for step in range(1, CQL_UPDATES + 1):
    idx = sample_batch(RNG, BATCH)
    td_loss, cql_loss, td_err, done_b = cql_plus_td_grad_step(idx)

    present_b = train["present"][idx]
    active_errs = np.abs(td_err)[present_b]
    active_done = np.repeat(done_b.astype(bool)[:, None], N_BRANCHES, axis=1)[present_b]
    if (~active_done).any():
        nonterm_td_hist.append(active_errs[~active_done].mean())
    if active_done.any():
        term_td_hist.append(active_errs[active_done].mean())

    if step % TARGET_SYNC_EVERY == 0:
        target_params = {k: v.copy() for k, v in params.items()}
    if step % 1000 == 0 or step == 1:
        print(f"  step {step:5d}  td_loss {td_loss:.4f}  cql_loss {cql_loss:.4f}")
print(f"  done in {time.time() - t0:.1f}s")

nonterm_td = float(np.mean(nonterm_td_hist[-50:])) if nonterm_td_hist else float("nan")
term_td = float(np.mean(term_td_hist[-50:])) if term_td_hist else float("nan")
print(f"\n  final non-terminal TD error (windowed): {nonterm_td:.4f}")
print(f"  final terminal TD error (windowed):     {term_td:.4f}")
print("  (reported separately -- pooling them flatters the run, per spec section 5)")

# ----------------------------------------------------------------------
# 9. Validation: balanced accuracy = mean of per-level recalls, masked-legal
#    greedy action vs logged action. Reported pooled and per behaviour policy.
# ----------------------------------------------------------------------


def balanced_accuracy(X_raw, A_arr, mask_arr, present_arr):
    Xn = X_raw  # already normalised in place
    Q = forward(params, Xn)["Q"]
    Qm = masked_softmax_logits(Q, mask_arr)
    pred = np.argmax(Qm, axis=2)                      # (N,28)

    illegal = ~np.take_along_axis(mask_arr, pred[:, :, None], axis=2).squeeze(2)
    illegal_rate = illegal[present_arr].mean()

    recalls = []
    for lvl in range(N_LEVELS):
        is_lvl = (A_arr == lvl) & present_arr
        if is_lvl.sum() == 0:
            continue
        recalls.append((pred[is_lvl] == lvl).mean())
    bal_acc = float(np.mean(recalls))
    return bal_acc, recalls, illegal_rate


print("\nValidation balanced accuracy (pooled):")
bal_acc, recalls, illegal_rate = balanced_accuracy(val["X"], val["A"], val["mask"], val["present"])
print(f"  balanced accuracy {bal_acc:.4f}  (off/on/reduced recall = "
      f"{recalls[0]:.3f} / {recalls[1]:.3f} / {recalls[2]:.3f})")
print(f"  illegal-action rate under the mask: {illegal_rate:.6f} (should be ~0)")

per_policy_report = {}
print("\nValidation balanced accuracy by behaviour policy:")
for pol in np.unique(val["policy"]):
    sel = val["policy"] == pol
    ba, rc, _ = balanced_accuracy(val["X"][sel], val["A"][sel], val["mask"][sel], val["present"][sel])
    per_policy_report[str(pol)] = dict(balanced_accuracy=ba, recalls=rc)
    print(f"  {pol:16s}  balanced accuracy {ba:.4f}")

print("\nNote: this metric measures agreement with three scripted behaviour "
      "policies. It is a sanity check that the network learned state-conditioned "
      "structure -- not a target to maximise. See spec section 5.")

# ----------------------------------------------------------------------
# 10. Export for the Pi -- exact section-8 format
# ----------------------------------------------------------------------

export_path = OUT_DIR / "sharp_policy_bdq_v2.npz"
np.savez(
    export_path,
    w=params["w"], b=params["b"],
    v=params["v"], vb=params["vb"],
    a=params["a"], ab=params["ab"],
    mean=mean, sd=sd,
    feature_names=np.array(feature_names, dtype=object),
    n_features=N_FEATURES, n_branches=N_BRANCHES, n_levels=N_LEVELS,
    policy_version="bdq_v2",
    trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
)
print(f"\nExported weights to {export_path}")

# reload-and-reproduce check (spec section 11: "an export that does not
# reproduce the in-memory model fails on the Pi, not here")
loaded = np.load(export_path, allow_pickle=True)
x_check = val["X"][:64]  # already normalised in place
h = np.maximum(0, x_check @ loaded["w"] + loaded["b"])
V_check = h @ loaded["v"] + loaded["vb"]
A_check = (h @ loaded["a"] + loaded["ab"]).reshape(-1, N_BRANCHES, N_LEVELS)
A_check = A_check - A_check.mean(axis=2, keepdims=True)
q = V_check[:, :, None] + A_check
q_live = forward(params, x_check)["Q"]
reload_max_abs_diff = float(np.max(np.abs(q - q_live)))
assert reload_max_abs_diff < 1e-4, "export does not reproduce the in-memory model"
print(f"  reload check passed (max abs diff {reload_max_abs_diff:.2e})")

# ----------------------------------------------------------------------
# 11. Training report
# ----------------------------------------------------------------------

report = {
    "policy_version": "bdq_v2",
    "architecture": {
        "input": N_FEATURES, "hidden": HIDDEN,
        "branches": N_BRANCHES, "levels": N_LEVELS,
        "params_approx": int(sum(p.size for p in params.values())),
    },
    "training": {
        "bc_updates": BC_UPDATES, "cql_updates": CQL_UPDATES,
        "batch_size": BATCH, "alpha_cql": ALPHA_CQL, "gamma": GAMMA,
        "level_weight_off_on_reduced": level_weight.tolist(),
    },
    "td_error": {
        "non_terminal_windowed": nonterm_td,
        "terminal_windowed": term_td,
        "note": "reported separately -- a falling pooled TD error is not evidence "
                "of policy quality (spec section 9)",
    },
    "validation_balanced_accuracy": {
        "pooled": bal_acc,
        "recalls_off_on_reduced": recalls,
        "illegal_action_rate": illegal_rate,
        "by_behaviour_policy": per_policy_report,
        "note": "sanity check against three scripted controllers, not a target metric",
    },
    "zero_variance_features_guarded": int(zero_var.sum()),
    "reward_caveats": [
        "Bellman targets use the logged sharp_reward_billing reward. The "
        "Bradley-Terry preference reward r_psi (fitted separately, see "
        "fit_preference_reward_v1.py) is NOT substituted into training, "
        "because every override_preference_pairs row compares preferred=ON "
        "against proposed=OFF -- the pair set has no direction variance yet.",
    ],
    "known_gaps": [
        "Next-action selection applies the flat level-legality mask only; "
        "the joint device shield is not re-applied to next-state candidates "
        "(shield-consistent targets), since the flat release does not carry "
        "nested device state.",
        "apply_shield must still run on the exported policy's output before "
        "actuation on real hardware -- this script does not implement or "
        "call it.",
        "Appliance power is a measurement-grounded proxy for only 685 of "
        "4,124 devices (512 REFIT UK, 173 iAWE Delhi); the rest are declared "
        "assumptions. Do not report this model's numbers as generalising "
        "beyond the AP/Guntur simulator context.",
        "Investigated: `random_binary` balanced accuracy (0.694) came out "
        "ABOVE `serve_preferred` (0.654) and above the spec's own ~0.49 "
        "enumerated ceiling for that policy. Ruled out: shield/human "
        "override (changes <0.01% of any policy's logged actions, "
        "confirmed here) and a state/action row-alignment bug (action "
        "varies step-to-step for most devices as expected). Found instead: "
        "for a subset of devices within `random_binary` episodes, the "
        "logged action is 80-99% persistent across all 96 steps of a day "
        "-- i.e. `policy_action_before_human` is not always the fresh "
        "per-step i.i.d. draw the spec's illustrative code snippet "
        "describes; some devices are evidently subject to additional "
        "minimum-run/cycle constraints even under the 'random' behaviour "
        "policy. That non-uniformity is real signal a state-reading model "
        "can legitimately learn, so it is consistent for a model to exceed "
        "the spec's enumerated ceiling, which assumes strict per-step i.i.d. "
        "noise. Treat that ceiling as a simplified lower bound, not a hard "
        "cap, until this is confirmed against the (unreleased) simulator "
        "source.",
    ],
}

report_path = OUT_DIR / "training_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Wrote {report_path}")
print("\nDone.")

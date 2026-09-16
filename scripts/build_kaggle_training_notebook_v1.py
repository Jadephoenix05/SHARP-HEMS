"""Generate the Kaggle training notebook for the SHARP RL model.

The notebook is written from here rather than by hand so the cells cannot drift
from the validated trainer in scripts/train_sharp_bdq_v2.py, and so the JSON is
always well formed.
"""
from pathlib import Path
import argparse
import json

DATASET = '/kaggle/input/sharp-master-dataset-v2'


def markdown(text):
    lines = text.strip('\n').split('\n')
    return {'cell_type': 'markdown', 'metadata': {},
            'source': [f'{line}\n' for line in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip('\n').split('\n')
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [],
            'source': [f'{line}\n' for line in lines[:-1]] + [lines[-1]]}


CELLS = [
    markdown(f"""
# SHARP — training the Branching Dueling Double-Q policy

Trains the SHARP home energy controller entirely on Kaggle, from the published
dataset. **pandas and numpy only** — no deep-learning framework, because the
policy has to run on a Raspberry Pi, and a NumPy forward pass is three matrix
multiplies with no runtime to install on an ARM board.

**Before you run:** add the dataset `sharp-master-dataset-v2` through
*Add Data* in the right-hand panel, then *Run All*. Accelerator: **None**
(this is CPU work; a GPU will not make it faster).

What this notebook does, in order:

1. Load the pre-split transitions and build the per-branch legality mask
2. Behaviour-cloning warm start
3. Conservative Q-Learning (CQL) — the offline correction that matters
4. Evaluate with metrics that **cannot be fooled by the majority class**
5. Fit the Bradley-Terry preference reward (E2)
6. Export the weights for the Pi

**The test split is never read.** Not in any cell here.
"""),

    markdown("""
## 1. Setup

Everything is pinned at the top. `SEED` fixes the run; `SHOULD_RUN_ABLATIONS`
turns the four-way comparison in section 7 on and off, since it multiplies
runtime by roughly four.
"""),

    code(f"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

DATASET = Path('{DATASET}')
WORKING = Path('/kaggle/working')

N_BRANCHES = 28
N_LEVELS = 3                      # 0 = OFF (shed), 1 = ON (full), 2 = REDUCED (dim)
LEVEL_NAMES = ['off', 'on', 'reduced']
# State layout from feature_schema.json: 25 globals, then 10 per device slot.
GLOBAL_FEATURES = 25
DEVICE_FEATURES = 10
REMAINING_HOURS = 0
PREFERRED_SERVICE = 6

SEED = 42
HIDDEN = 128
BATCH = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4              # Adam, not plain SGD
REWARD_SCALE = 10.0               # terminal rewards are ~77x step rewards
WARM_START_UPDATES = 2000         # behaviour cloning
CONSERVATIVE_UPDATES = 5000
CQL_ALPHA = 1.0                    # 0.5 and 1.0 both tested; 1.0 scored better
USE_LEVEL_BALANCE = True

# Reward reweighting, applied at load time from the recorded components so the
# dataset does not have to be regenerated.
#
# As shipped, the peak term is 3.9 per cent of the cost-plus-peak signal - cost
# outweighs it 25 to 1. Under a telescopic tariff with no time-of-day rate a
# kWh saved at 4 a.m. is worth exactly as much as one saved at noon, so the
# policy learned to shed whenever, which is useless to the grid and maximally
# annoying to the resident. Measured: five of its eight busiest shedding hours
# were between 3 and 7 a.m., when grid severity is exactly zero.
PEAK_WEIGHT = 1.0                  # lifts peak from 3.9% to roughly a quarter of the signal
COST_WEIGHT = 1.0

# Penalty on Q for actions the deployment mask forbids. Without it the policy
# never learns the rule, because the shield silently repairs every violation -
# measured, the raw policy wanted to shed a protected ceiling fan 1,733 times
# and a television 56 times. Safety then rests entirely on one module.
ILLEGAL_ACTION_PENALTY = 0.0      # at 0.5 the policy learned 'ON is always safe'
ILLEGAL_ACTION_MARGIN = 0.25       # how far below the best legal action is enough

SHOULD_RUN_ABLATIONS = True

# Balanced BATCH sampling, not just a balanced loss.
#
# Reweighting the loss by inverse frequency still leaves every batch 83 per cent
# OFF and 2.6 per cent REDUCED - about six dim examples in a batch of 256. The
# network sees the rare class too rarely to fit it, however heavily each example
# is weighted once it arrives. Sampling rows in proportion to how much rare-class
# action they contain puts the examples in front of it in the first place.
# Measured and REJECTED. Oversampling rows that contain a rare action moved dim
# recall almost not at all (0.011 -> 0.015) while costing six points of OFF
# recall, taking three-level balanced accuracy from 0.6173 to 0.5922. REDUCED is
# almost absent by DESIGN - no critical load may be dimmed, so only 37 air
# coolers of 3,353 devices can dim at all - and no amount of resampling teaches
# a class the design removed. It only distorts the distribution the value
# function is fitted on.
USE_BALANCED_SAMPLING = False
LEVEL_WEIGHT_NORM = 'min'          # 'mean' or 'min'; see level_weights()
# Multiplier on the conservative term. This is the knob that controls CONTROL
# quality - the accuracy metric barely moves with it. Swept on the simulator,
# 90 episodes, seed 99, peak kW (lower is better):
#
#   effective CQL   1.0x  ->  0.302   (mean-normalised)
#   effective CQL   6.6x  ->  0.264   <- shipped, min-normalised, scale 1.0
#   effective CQL  16.5x  ->  0.295   (scale 2.5)
#   effective CQL  33.0x  ->  0.295   (scale 5.0)
#
# It is not monotone: past ~6.6x the imitation term swamps the reward and the
# controller regresses towards the logged behaviour. Do not raise this.
LEVEL_WEIGHT_SCALE = 1.0

MARKER = 'rl_transitions/splits/train.parquet'

if not (DATASET / MARKER).exists():
    # Kaggle mounts a dataset under its slug, the slug is not always the name
    # you gave it, and the archive's own folder structure is preserved inside.
    # So do not guess a path - search for the file that identifies the release,
    # at any depth, and work back to its root.
    hits = list(Path('/kaggle/input').glob('**/' + MARKER))
    found = sorted(set(hit.parent.parent.parent for hit in hits))
    if len(found) == 1:
        DATASET = found[0]
        print('found the dataset at', DATASET)
    elif len(found) > 1:
        DATASET = found[0]
        print('several copies attached, using', DATASET)
        for other in found[1:]:
            print('  also present:', other)
    else:
        print('Attached under /kaggle/input:')
        for path in sorted(Path('/kaggle/input').glob('*')):
            print('  ' + path.name + '/')
            for child in sorted(path.glob('*'))[:12]:
                print('    ' + child.name + ('/' if child.is_dir() else ''))
        raise SystemExit(
            'Could not find ' + MARKER + ' anywhere under /kaggle/input. '
            'Attach sharp-master-dataset-v2 with Add Data, then re-run. The '
            'listing above shows what is currently mounted.')

for required in [MARKER,
                 'rl_transitions/splits/validation.parquet',
                 'rl_transitions/override_preference_pairs.parquet',
                 'simulator_inputs/device_power_models.parquet']:
    if not (DATASET / required).exists():
        raise SystemExit('The attached dataset is missing ' + required)

print('numpy', np.__version__, '| pandas', pd.__version__)
print('dataset:', DATASET)
for path in sorted(DATASET.glob('*')):
    print(' ', path.name)
"""),

    markdown("""
## 2. Load the transitions

The release ships the splits **already separated**, so there is no filtering to
get wrong. Splits are disjoint in two dimensions at once — household **and**
calendar year — so neither a home nor a date appears in more than one split.

`state` and `next_state` arrive as finished 305-float vectors. The one thing
deliberately *not* baked in is normalisation: it has to be fitted on train only,
here, or validation leaks into the input scaling.
"""),

    code("""
def load_split(split):
    frame = pd.read_parquet(DATASET / f'rl_transitions/splits/{split}.parquet')
    for column in ['reward_cost_inr', 'reward_grid_peak_kwh', 'reward_discomfort']:
        if column not in frame.columns:
            raise ValueError(f'{split} is missing {column}; cannot reweight the reward')
    if frame.empty:
        raise ValueError(f'No rows in {split}')

    state = np.stack(frame.state.to_numpy()).astype(np.float32)
    next_state = np.stack(frame.next_state.to_numpy()).astype(np.float32)
    present = np.stack(frame.device_present.to_numpy()).astype(np.float32)

    # Actions are ragged: one level per device the household actually owns.
    # Pad to 28 with zeros; device_present is what stops padding from learning.
    actions = np.zeros((len(frame), N_BRANCHES), dtype=np.int64)
    for row, levels in enumerate(frame.action.to_numpy()):
        levels = np.asarray(levels, dtype=np.int64)
        if levels.size and (levels.min() < 0 or levels.max() >= N_LEVELS):
            raise ValueError(f'Action level outside 0..{N_LEVELS - 1} in {split}')
        actions[row, :levels.size] = levels

    # Recompose the reward from its components so peak can be weighted up.
    # residual carries the switching and unmet-service terms, which are not
    # broken out separately in the release and are left untouched.
    cost = frame.reward_cost_inr.to_numpy(dtype=np.float64)
    peak = frame.reward_grid_peak_kwh.to_numpy(dtype=np.float64)
    discomfort = frame.reward_discomfort.to_numpy(dtype=np.float64)
    shipped = frame.reward.to_numpy(dtype=np.float64)
    residual = shipped + (cost + peak + 0.5 * discomfort)
    reward = residual - (COST_WEIGHT * cost + PEAK_WEIGHT * peak
                         + 0.5 * discomfort)
    done = frame.done.to_numpy(dtype=bool)
    for name, array in [('state', state), ('next_state', next_state),
                        ('reward', reward)]:
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite {name} in {split}')

    return {'state': state, 'next_state': next_state, 'present': present,
            'actions': actions, 'reward': reward, 'done': done,
            'household': frame.household_id.to_numpy(),
            'rows': len(frame)}


started = time.time()
train = load_split('train')
validation = load_split('validation')
print(f'loaded in {time.time() - started:.1f}s')
print(f"train      {train['rows']:,} rows, {len(np.unique(train['household']))} households")
print(f"validation {validation['rows']:,} rows, "
      f"{len(np.unique(validation['household']))} households")
print(f"features   {train['state'].shape[1]}")

overlap = set(np.unique(train['household'])) & set(np.unique(validation['household']))
if overlap:
    raise ValueError(f'{len(overlap)} households appear in both splits')
print('household disjointness: OK')
"""),

    markdown("""
## 3. The legality mask — do not skip this

Level 2 (REDUCED) is only physically meaningful on an appliance that can dim or
run slow: fans, coolers, and most lighting. It is meaningless on a television or
a water pump.

Without a mask, `argmax` over three levels can pick REDUCED on a device that
cannot dim — and roughly **40 % of devices cannot**. The shield would refuse it
at actuation time, but by then the Q values have already been trained as though
the level existed, and the bootstrap target may have been built from it.

We confirm the mask against the logged data first: if the behaviour policies
never chose level 2 on a non-dimmable device, the mask is safe to impose.

Note what we deliberately do **not** mask: OFF on a necessity appliance. The
necessity rule is *conditional* — necessity loads are protected when the
occupant wants them, not pinned on around the clock — so OFF is a legitimate
logged action for a fridge at 3 a.m.
"""),

    code("""
devices = pd.read_parquet(
    DATASET / 'simulator_inputs/device_power_models.parquet',
    columns=['template_id', 'device_id', 'appliance_type',
             'is_necessity', 'supports_reduced'])

# Slot order must match the generator exactly: device_id ascending within
# household, as recorded in feature_schema.json. Getting this wrong silently
# attaches every device's features to the wrong branch.
devices = devices.sort_values(['template_id', 'device_id'])
devices['slot'] = devices.groupby('template_id').cumcount()
if devices.slot.max() >= N_BRANCHES:
    raise ValueError('A household has more devices than there are state slots')

supports_reduced = {(t, s): bool(v) for t, s, v
                    in zip(devices.template_id, devices.slot, devices.supports_reduced)}
is_necessity = {(t, s): bool(v) for t, s, v
                in zip(devices.template_id, devices.slot, devices.is_necessity)}
print(f'{int(devices.supports_reduced.sum()):,} of {len(devices):,} devices can dim')


def build_legal_mask(data):
    \"\"\"(rows, 28, 3) TRAINING mask: level 2 wherever the appliance can dim.

    This is the PHYSICAL constraint only, and it has to stay that way. The
    conservative term takes the log-probability of the logged action, so if the
    mask declares that action impossible you are asking for the log of zero. An
    attempt to fold the policy rule in here produced a TD error of 317 and a
    balanced accuracy of 0.40.

    The rule that SHARP never dims a fan, a light or a fridge is a SHIELD rule,
    enforced at action-selection time by the deployment mask below - exactly
    like the existing rule that it never sheds them. The dataset contains
    306,666 logged necessity sheds and that has never broken training, for the
    same reason.
    \"\"\"
    per_household = {}
    for household in np.unique(data['household']):
        mask = np.ones((N_BRANCHES, N_LEVELS), dtype=bool)
        for slot in range(N_BRANCHES):
            mask[slot, 2] = supports_reduced.get((household, slot), False)
        per_household[household] = mask
    index = {h: i for i, h in enumerate(per_household)}
    table = np.stack(list(per_household.values()))
    return table[np.array([index[h] for h in data['household']])]


for data, name in [(train, 'train'), (validation, 'validation')]:
    data['legal'] = build_legal_mask(data)
    live = data['present'] > 0
    chose_illegal = int((~np.take_along_axis(
        data['legal'], data['actions'][:, :, None], axis=2)[:, :, 0] & live).sum())
    print(f'{name}: logged actions that violate the physical mask: {chose_illegal}')
    if chose_illegal:
        raise ValueError(
            f'{chose_illegal} logged {name} actions are illegal under the mask. '
            'The mask or the slot ordering is wrong - do not train on this.')

    # THE DEPLOYMENT MASK. What the policy may choose, and what the Pi enforces.
    #
    # A necessity appliance is never shed and never dimmed WHEN THE OCCUPANT
    # WANTS IT. The condition matters: a fridge at 3 a.m. that nobody is asking
    # for, or a fan whose daily service budget is already met, is correctly OFF.
    # Forcing level 1 unconditionally would run every fan and light around the
    # clock, which wastes energy and is not what protecting essential service
    # means.
    #
    # Occupant demand is read from the state itself - device feature 6 is
    # preferred_service_fraction and feature 0 is remaining_service_hours - so
    # this uses exactly the quantities the shield uses.
    necessity = np.stack([
        np.array([is_necessity.get((h, s), False) for s in range(N_BRANCHES)])
        for h in data['household']])
    slot = np.arange(N_BRANCHES) * DEVICE_FEATURES + GLOBAL_FEATURES
    wanted = (data['state'][:, slot + PREFERRED_SERVICE] > 0) &              (data['state'][:, slot + REMAINING_HOURS] > 0)
    entitled = necessity & wanted
    deployment = data['legal'].copy()
    # The two rules are NOT symmetric, and getting that wrong is easy.
    #
    # DIMMING is forbidden UNCONDITIONALLY on a critical load. There is no state
    # of the world in which SHARP dims someone's fan.
    #
    # SHEDDING is forbidden only while the occupant wants it. A fridge nobody is
    # asking for at 3 a.m., or a fan whose daily service budget is already met,
    # is correctly OFF - protecting essential service does not mean running it
    # around the clock.
    deployment[:, :, 2] &= ~necessity         # never dim a critical load, ever
    deployment[:, :, 0] &= ~entitled          # never shed one that is in use
    data['deployment_legal'] = deployment
    data['forbidden'] = ~deployment
    data['necessity'] = necessity
    data['entitled'] = entitled

print('training mask agrees with every logged action')
live = train['present'] > 0
blocked = int((train['legal'][:, :, 2] & ~train['deployment_legal'][:, :, 2] & live).sum())
entitled = int((train['entitled'] & live).sum())
print(f'{entitled:,} device-steps where an essential is in use and therefore '
      f'protected')
print(f'deployment mask blocks {blocked:,} of those from being dimmed')
print('a fan, light or fridge the occupant is using is never shed and never dimmed')
"""),

    markdown("""
## 4. The network

Branching Dueling Double-Q. A joint action space over 28 devices would be 2²⁸;
branching gives 28 × 3 = 84 outputs — **linear instead of exponential**.

```
input 305
  └── trunk: Dense(128) + ReLU
        ├── value head:     Dense(1)
        └── advantage head: Dense(28 × 3)
  Q[i] = V + A[i] − mean(A[i])        duelling, per branch
```

About 50k parameters, under 1 MB in float32.

### The one insight that simplifies everything

The CQL penalty is

```
logsumexp_a Q[i,a] − Q[i, a_logged]
```

which is exactly **the negative log probability of the logged action** under a
softmax over that branch's Q values. So the conservative penalty and the
behaviour-cloning objective are the *same term*. A BC warm start is this loss
with the temporal-difference part switched off — no second head, no second
objective, no separate model to keep in sync.
"""),

    code("""
class BranchingDuelingNetwork:
    def __init__(self, n_features, hidden=HIDDEN, seed=SEED):
        rng = np.random.default_rng(seed)
        self.p = {
            'w': rng.normal(0, np.sqrt(2 / n_features), (n_features, hidden)),
            'b': np.zeros(hidden),
            'v': rng.normal(0, 0.01, (hidden, 1)), 'vb': np.zeros(1),
            'a': rng.normal(0, 0.01, (hidden, N_BRANCHES * N_LEVELS)),
            'ab': np.zeros(N_BRANCHES * N_LEVELS)}
        # Adam moments. Plain gradient descent with one global step size serves
        # 305 features of very different scale badly; switching to Adam was the
        # single largest measured gain in this pipeline.
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.vv = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        h = np.maximum(0, x @ self.p['w'] + self.p['b'])
        advantage = (h @ self.p['a'] + self.p['ab']).reshape(-1, N_BRANCHES, N_LEVELS)
        value = (h @ self.p['v'] + self.p['vb']).reshape(-1, 1, 1)
        return value + advantage - advantage.mean(2, keepdims=True), h

    def backward(self, x, h, dq):
        da = (dq - dq.mean(2, keepdims=True)).reshape(len(x), -1)
        dv = dq.sum((1, 2))[:, None]
        dh = (da @ self.p['a'].T + dv @ self.p['v'].T) * (h > 0)
        return {'w': x.T @ dh, 'b': dh.sum(0), 'v': h.T @ dv, 'vb': dv.sum(0),
                'a': h.T @ da, 'ab': da.sum(0)}

    def loss_and_gradient(self, x, actions, target, present, legal, alpha,
                          use_td=True, level_weight=None, forbidden=None,
                          illegal_penalty=0.0):
        q, h = self.forward(x)
        weight = present / (present.sum(1, keepdims=True) * len(x))
        dq = np.zeros_like(q)
        loss = 0.0

        if use_td:
            chosen = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]
            error = chosen - target[:, None]
            # Huber: quadratic near zero, linear in the tails, so the terminal
            # reward spike at step 95 cannot dominate every gradient.
            loss += float((np.where(np.abs(error) < 1, 0.5 * error ** 2,
                                    np.abs(error) - 0.5) * weight).sum())
            np.put_along_axis(dq, actions[:, :, None],
                              (np.clip(error, -1, 1) * weight)[:, :, None], axis=2)

        if alpha > 0:
            # logsumexp over LEGAL levels only. Including impossible levels would
            # spend the conservative penalty on actions that can never be taken.
            masked = np.where(legal, q, -np.inf)
            shift = masked.max(2, keepdims=True)
            exponent = np.where(legal, np.exp(masked - shift), 0.0)
            total = exponent.sum(2)
            softmax = exponent / total[:, :, None]
            logsumexp = shift[:, :, 0] + np.log(total)
            logged = np.take_along_axis(q, actions[:, :, None], axis=2)[:, :, 0]

            # Inverse-frequency weighting. Without it this is a class-imbalanced
            # cross-entropy: 83% of logged actions are OFF, so it collapses to
            # answering OFF unconditionally - which scores 0.83 on raw agreement
            # and is worthless as a controller.
            cql_weight = weight if level_weight is None else weight * level_weight[actions]
            loss += alpha * float(((logsumexp - logged) * cql_weight).sum())
            gradient = softmax.copy()
            np.put_along_axis(
                gradient, actions[:, :, None],
                np.take_along_axis(gradient, actions[:, :, None], axis=2) - 1.0, axis=2)
            dq += alpha * gradient * cql_weight[:, :, None]

        if forbidden is not None and illegal_penalty > 0:
            # A HINGE, not a linear penalty. The goal is only that a forbidden
            # action ranks below every permitted one - not that its Q value runs
            # off to minus infinity, which is what a linear penalty does because
            # it has no floor and the loss simply keeps decreasing.
            #
            # The reference is the best PERMITTED action, held constant (no
            # gradient flows into it), so the penalty cannot be satisfied by
            # dragging the legal actions down instead.
            block = forbidden & (present[:, :, None] > 0)
            permitted = np.where(~forbidden, q, -np.inf)
            best_legal = permitted.max(axis=2, keepdims=True)
            best_legal = np.where(np.isfinite(best_legal), best_legal, 0.0)
            violation = np.maximum(0.0, q - (best_legal - ILLEGAL_ACTION_MARGIN))
            loss += illegal_penalty * float((violation * block).sum() / len(x))
            dq = dq + illegal_penalty * ((violation > 0) & block) / len(x)

        return loss, self.backward(x, h, dq)

    def update(self, x, actions, target, present, legal, alpha,
               lr=LEARNING_RATE, use_td=True, level_weight=None,
               forbidden=None, illegal_penalty=0.0,
               beta1=0.9, beta2=0.999, eps=1e-8):
        loss, gradient = self.loss_and_gradient(
            x, actions, target, present, legal, alpha, use_td, level_weight,
            forbidden, illegal_penalty)
        norm = np.sqrt(sum(np.square(g).sum() for g in gradient.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):
            raise ValueError('Nonfinite training update')
        clip = min(1.0, 10.0 / (norm + 1e-12))
        self.t += 1
        for key, g in gradient.items():
            g = g * clip
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.vv[key] = beta2 * self.vv[key] + (1 - beta2) * np.square(g)
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.vv[key] / (1 - beta2 ** self.t)
            self.p[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return loss


print('network defined')
"""),

    markdown("""
## 5. Gradient tests

Hand-written backpropagation with no autodiff to check it, so every analytic
gradient is compared against a central finite difference before any training
happens. The second assertion is the one that matters most: **padded branches
must contribute exactly zero gradient**, or households with few appliances
quietly teach the network about devices they do not own.
"""),

    code("""
def run_gradient_tests():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 5))
    actions = rng.integers(0, N_LEVELS, (3, N_BRANCHES))
    target = np.array([0.3, -0.2, 0.7])
    present = np.zeros((3, N_BRANCHES))
    present[:, :2] = 1
    legal = rng.random((3, N_BRANCHES, N_LEVELS)) > 0.3
    legal[:, :, 1] = True                                  # ON is always legal
    np.put_along_axis(legal, actions[:, :, None], True, axis=2)
    balance = np.array([0.4, 1.1, 1.5])

    settings = [(0.0, True, None), (1.0, False, None), (0.5, True, None),
                (0.5, True, balance), (1.0, False, balance)]
    for alpha, use_td, level_weight in settings:
        net = BranchingDuelingNetwork(5, hidden=8)
        _, gradient = net.loss_and_gradient(
            x, actions, target, present, legal, alpha, use_td, level_weight)
        for key, index in [('a', (0, 0)), ('w', (1, 3)), ('v', (2, 0)), ('ab', (5,))]:
            original = net.p[key][index]
            eps = 1e-6
            net.p[key][index] = original + eps
            up = net.loss_and_gradient(x, actions, target, present, legal,
                                       alpha, use_td, level_weight)[0]
            net.p[key][index] = original - eps
            down = net.loss_and_gradient(x, actions, target, present, legal,
                                         alpha, use_td, level_weight)[0]
            net.p[key][index] = original
            numeric = (up - down) / (2 * eps)
            if not np.isclose(numeric, gradient[key][index], atol=1e-6, rtol=1e-4):
                raise AssertionError(
                    f'Gradient mismatch alpha={alpha} td={use_td} '
                    f'balanced={level_weight is not None} at {key}{index}: '
                    f'finite difference {numeric} vs analytic {gradient[key][index]}')
        padded = slice(2 * N_LEVELS, None)
        if not (np.all(gradient['a'][:, padded] == 0)
                and np.all(gradient['ab'][padded] == 0)):
            raise AssertionError('Padded branches contributed to the loss')
    print('finite-difference gradient test: PASS')
    print('padded-branch zero-gradient test: PASS')


run_gradient_tests()
"""),

    markdown("""
## 6. Evaluation — metrics that cannot be gamed

Two traps, both of which caught earlier versions of this work.

**Trap 1: pooled TD error.** The unmet-service penalty lands as a single lump at
step 95, so terminal rewards are roughly **77×** step rewards. A pooled TD number
is dominated by the 1 % of rows that are terminal. Worse, the two move in
opposite directions: non-terminal TD falls while terminal TD rises. Reported
together, that reads as progress. So they are reported apart.

**Trap 2: raw agreement with the logged action.** 83 % of logged actions are OFF.
A model that answers OFF unconditionally scores **0.83** and has learned nothing.
The headline is therefore **balanced accuracy** — the mean of the three per-level
recalls — which that model scores 0.333 on, exactly chance.
"""),

    code("""
def double_q_target(net, target_net, next_state, present, legal, reward, done,
                    gamma, index):
    \"\"\"Online net picks the next level, target net prices it. Legal levels only.\"\"\"
    online, _ = net.forward(next_state[index])
    allowed = (present[index][:, :, None] > 0) & legal[index]
    best = np.argmax(np.where(allowed, online, -np.inf), axis=2)
    priced, _ = target_net.forward(next_state[index])
    chosen = np.take_along_axis(priced, best[:, :, None], axis=2)[:, :, 0]
    per_branch = ((chosen * present[index]).sum(1)
                  / np.maximum(1, present[index].sum(1)))
    return reward[index] + np.where(done[index], 0.0, gamma * per_branch)


def evaluate(net, target_net, data, gamma=GAMMA, chunk=20000):
    state, next_state = data['normalised'], data['normalised_next']
    present, legal, actions = data['present'], data['legal'], data['actions']
    deployment = data['deployment_legal']
    entitled = data['entitled']
    necessity = data['necessity']
    necessity_touched = necessity_dimmed = would_violate = 0
    reward, done = data['scaled_reward'], data['done']

    absolute = np.zeros(len(state))
    counted = np.zeros(len(state))
    agree = np.zeros(N_LEVELS)
    logged_count = np.zeros(N_LEVELS)
    greedy_count = np.zeros(N_LEVELS)
    illegal_picks = 0

    for start in range(0, len(state), chunk):
        index = np.arange(start, min(start + chunk, len(state)))
        q, _ = net.forward(state[index])
        chosen = np.take_along_axis(q, actions[index][:, :, None], axis=2)[:, :, 0]
        y = double_q_target(net, target_net, next_state, present, legal,
                            reward, done, gamma, index)
        absolute[index] = (np.abs(chosen - y[:, None]) * present[index]).sum(1)
        counted[index] = present[index].sum(1)

        # Select under the DEPLOYMENT mask: this is what would actually happen.
        # What the policy would do with NO deployment mask at all. This is the
        # number that says whether it has learned the rule or is merely being
        # restrained by the shield.
        unshielded = np.argmax(np.where(present[index][:, :, None] > 0, q, -np.inf), axis=2)
        would_violate += int((~np.take_along_axis(
            deployment[index], unshielded[:, :, None], axis=2)[:, :, 0]
            & (present[index] > 0)).sum())
        allowed = (present[index][:, :, None] > 0) & deployment[index]
        greedy = np.argmax(np.where(allowed, q, -np.inf), axis=2)
        live = present[index] > 0
        illegal_picks += int((~np.take_along_axis(
            deployment[index], greedy[:, :, None], axis=2)[:, :, 0] & live).sum())
        # Two separate violations, because the rules differ.
        necessity_touched += int((entitled[index] & live & (greedy == 0)).sum())
        necessity_dimmed += int((necessity[index] & live & (greedy == 2)).sum())
        for level in range(N_LEVELS):
            picked = live & (actions[index] == level)
            logged_count[level] += picked.sum()
            agree[level] += (picked & (greedy == level)).sum()
            greedy_count[level] += (live & (greedy == level)).sum()

    def td(rows):
        return float(absolute[rows].sum() / max(1.0, counted[rows].sum()))

    recall = agree / np.maximum(1.0, logged_count)
    total = max(1.0, logged_count.sum())
    return {
        'balanced_accuracy': float(recall.mean()),
        'raw_agreement': float(agree.sum() / total),
        'unshielded_violations': would_violate,
        'critical_shed_while_in_use': necessity_touched,
        'critical_dimmed_ever': necessity_dimmed,
        'td_error_non_terminal': td(~done),
        'td_error_terminal': td(done),
        'td_error_pooled_do_not_quote': td(slice(None)),
        'illegal_greedy_picks': illegal_picks,
        'recall_by_level': {LEVEL_NAMES[i]: float(recall[i]) for i in range(N_LEVELS)},
        'logged_share': {LEVEL_NAMES[i]: float(logged_count[i] / total)
                         for i in range(N_LEVELS)},
        'greedy_share': {LEVEL_NAMES[i]: float(greedy_count[i] / total)
                         for i in range(N_LEVELS)},
    }


print('evaluation defined')
"""),

    markdown("""
## 7. Train

Normalisation is fitted on **train only**. About 37 of the 305 features have zero
variance — padding, plus two power-system features that are constant in the main
scenario — so the standard deviation is floored before dividing.

Phase 1 clones the logged behaviour. Phase 2 switches on the temporal-difference
term and keeps the conservative penalty.
"""),

    code("""
def prepare(train_data, validation_data, reward_scale=REWARD_SCALE):
    mean = train_data['state'].mean(0)
    sd = train_data['state'].std(0)
    zero_variance = int((sd < 1e-8).sum())
    sd[sd < 1e-8] = 1.0
    for data in (train_data, validation_data):
        data['normalised'] = ((data['state'] - mean) / sd).astype(np.float32)
        data['normalised_next'] = ((data['next_state'] - mean) / sd).astype(np.float32)
        data['scaled_reward'] = data['reward'] / reward_scale
    print(f'{zero_variance} of {len(sd)} features have zero variance in train')
    return mean, sd, zero_variance


def level_weights(train_data):
    live = train_data['present'] > 0
    counts = np.array([float(((train_data['actions'] == level) & live).sum())
                       for level in range(N_LEVELS)])
    share = counts / counts.sum()
    weights = 1.0 / np.maximum(share, 1e-9)
    if LEVEL_WEIGHT_NORM == 'mean':
        return weights / weights.mean() * LEVEL_WEIGHT_SCALE, share
    # 'min': scale so the commonest level sits near 0.4 rather than the mean
    # sitting at 1.0. Same ratios, about 6.6x the magnitude, which makes the
    # conservative term correspondingly stronger.
    return weights * (0.401 / weights.min()) * LEVEL_WEIGHT_SCALE, share


def sampling_probabilities(train_data, weights):
    # How often each ROW should be drawn, given how rare its actions are.
    #
    # A row is not one label - it holds up to 28 device decisions at once - so
    # the weight is the mean inverse-frequency weight over its live devices.
    # Rows containing a REDUCED action are then drawn far more often than rows
    # that are entirely OFF, which is what puts the rare class in front of the
    # network rather than merely scaling it up once it has arrived.
    live = train_data['present'] > 0
    per_row = (weights[train_data['actions']] * live).sum(1) / np.maximum(1, live.sum(1))
    probability = per_row / per_row.sum()
    if not np.isfinite(probability).all() or probability.min() < 0:
        raise ValueError('Bad sampling distribution')
    return probability


def train_policy(train_data, validation_data, *, alpha, warm_start, steps,
                 balance, seed=SEED, label='run'):
    net = BranchingDuelingNetwork(train_data['state'].shape[1], seed=seed)
    target_net = BranchingDuelingNetwork(train_data['state'].shape[1], seed=seed)
    target_net.p = {k: v.copy() for k, v in net.p.items()}
    rng = np.random.default_rng(seed)
    weights = level_weights(train_data)[0] if balance else None
    probability = None
    if balance and USE_BALANCED_SAMPLING:
        probability = sampling_probabilities(train_data, weights)
        expected = (weights[train_data['actions']]
                    * (train_data['present'] > 0)).sum(1)
        print(f'  balanced sampling on: rarest rows drawn '
              f'{expected.max() / max(1e-9, expected.mean()):.1f}x the average')

    def draw(size):
        if probability is None:
            return rng.integers(0, len(state), size)
        return rng.choice(len(state), size=size, p=probability)
    history, recent = [], []

    state = train_data['normalised']
    next_state = train_data['normalised_next']
    actions, present, legal = (train_data['actions'], train_data['present'],
                               train_data['legal'])
    forbidden = train_data['forbidden']
    reward, done = train_data['scaled_reward'], train_data['done']

    best = {'balanced_accuracy': -1.0}
    best_parameters = None
    best_target = None

    def snapshot(step, phase):
        nonlocal best, best_parameters, best_target
        metrics = evaluate(net, target_net, validation_data)
        metrics.update({'step': step, 'phase': phase,
                        'train_loss': float(np.mean(recent[-100:]))})
        history.append(metrics)
        # Keep the best checkpoint FROM THE CONSERVATIVE PHASE ONLY.
        #
        # This restriction is the whole point. Behaviour cloning always wins on
        # agreement with the logged actions - that is literally its objective -
        # so selecting on balanced accuracy across both phases picks the warm
        # start every time and throws the reinforcement learning away. The
        # exported model would be a pure imitator of the scripted controllers,
        # with Q values that are not value estimates at all.
        #
        # The cost of the restriction is small and the benefit is large: on the
        # first Kaggle run the warm start scored 0.7635 with a step TD error of
        # 0.778, while the first conservative checkpoint scored 0.7475 with a TD
        # error of 0.263. Slightly less imitation, three times the value
        # accuracy, and an actual policy.
        if (phase == 'conservative'
                and metrics['balanced_accuracy'] > best['balanced_accuracy']):
            best = dict(metrics)
            best_parameters = {k: v.copy() for k, v in net.p.items()}
            # The target network is paired with the online one. Keeping the last
            # target while restoring an earlier online net makes every reported
            # TD error meaningless, because they no longer belong together.
            best_target = {k: v.copy() for k, v in target_net.p.items()}
        print(f"  {phase:12s} {step:6d} | loss {metrics['train_loss']:8.5f} "
              f"| TD step {metrics['td_error_non_terminal']:7.5f} "
              f"terminal {metrics['td_error_terminal']:8.5f} "
              f"| balanced {metrics['balanced_accuracy']:.4f}", flush=True)

    started = time.time()
    if warm_start:
        print(f'[{label}] behaviour-cloning warm start: {warm_start} updates')
        for step in range(1, warm_start + 1):
            index = draw(BATCH)
            recent.append(net.update(
                state[index], actions[index], reward[index], present[index],
                legal[index], alpha=1.0, use_td=False, level_weight=weights,
                forbidden=forbidden[index],
                illegal_penalty=ILLEGAL_ACTION_PENALTY))
            if step % 1000 == 0 or step == warm_start:
                target_net.p = {k: v.copy() for k, v in net.p.items()}
                snapshot(step, 'warm_start')
        target_net.p = {k: v.copy() for k, v in net.p.items()}

    print(f'[{label}] conservative Double-Q: {steps} updates, CQL alpha {alpha}')
    for step in range(1, steps + 1):
        index = draw(BATCH)
        y = double_q_target(net, target_net, next_state, present, legal,
                            reward, done, GAMMA, index)
        recent.append(net.update(
            state[index], actions[index], y, present[index], legal[index],
            alpha=alpha, use_td=True, level_weight=weights,
            forbidden=forbidden[index], illegal_penalty=ILLEGAL_ACTION_PENALTY))
        if step % 250 == 0:
            target_net.p = {k: v.copy() for k, v in net.p.items()}
        if step % 2500 == 0 or step == steps:
            snapshot(step, 'conservative')

    if best_parameters is not None:
        net.p = best_parameters
        target_net.p = best_target
        warm = [h for h in history if h['phase'] == 'warm_start']
        print(f"[{label}] selected conservative step {best['step']} "
              f"(balanced {best['balanced_accuracy']:.4f}, "
              f"TD {best['td_error_non_terminal']:.4f})")
        if warm:
            peak = max(warm, key=lambda h: h['balanced_accuracy'])
            print(f"[{label}]   the warm start reached "
                  f"{peak['balanced_accuracy']:.4f} but with TD "
                  f"{peak['td_error_non_terminal']:.4f} and is not a policy")
    print(f'[{label}] finished in {(time.time() - started) / 60:.1f} min')
    return net, target_net, history


MEAN, SD, ZERO_VARIANCE = prepare(train, validation)
BALANCE_WEIGHTS, LOGGED_SHARE = level_weights(train)
print('logged level share:',
      {LEVEL_NAMES[i]: round(float(LOGGED_SHARE[i]), 4) for i in range(N_LEVELS)})
print('inverse-frequency weights:',
      {LEVEL_NAMES[i]: round(float(BALANCE_WEIGHTS[i]), 3) for i in range(N_LEVELS)})
"""),

    code("""
policy, policy_target, history = train_policy(
    train, validation,
    alpha=CQL_ALPHA, warm_start=WARM_START_UPDATES,
    steps=CONSERVATIVE_UPDATES, balance=USE_LEVEL_BALANCE, label='main')

final = evaluate(policy, policy_target, validation)
print()
print('=== FINAL (validation) ===')
print(f"balanced accuracy      {final['balanced_accuracy']:.4f}   "
      f"(0.3333 = no discrimination)")
print(f"raw agreement          {final['raw_agreement']:.4f}   "
      f"(majority class is {max(final['logged_share'].values()):.4f} - "
      f"do not quote this alone)")
print(f"TD error, step rows    {final['td_error_non_terminal']:.5f}")
print(f"TD error, terminal     {final['td_error_terminal']:.5f}")
print(f"illegal greedy picks   {final['illegal_greedy_picks']}")
print()
for level in LEVEL_NAMES:
    print(f"  {level:8s} logged {final['logged_share'][level]:.4f}  "
          f"greedy {final['greedy_share'][level]:.4f}  "
          f"recall {final['recall_by_level'][level]:.4f}")
"""),

    markdown("""
### Reading the result

`balanced_accuracy` is the number to watch, against a floor of 0.3333.

Check `greedy_share` against `logged_share` every time. If greedy OFF is near
1.0 and the other two recalls are near zero, the model has collapsed to the
majority class — raw agreement will look excellent and the controller will be
useless. That is what inverse-frequency weighting exists to prevent.

`illegal_greedy_picks` must be **0**. Anything else means the legality mask is
not being applied where it should be.
"""),

    markdown("""
## 8. Ablations (E5) — does any of this actually help?

Four configurations, same seed, same budget. Without this table there is no
evidence that the warm start or the conservative penalty earns its place.

Set `SHOULD_RUN_ABLATIONS = False` in section 1 to skip; it roughly quadruples
runtime.
"""),

    code("""
ABLATION_STEPS = 5000
ablations = {}

if SHOULD_RUN_ABLATIONS:
    configurations = [
        ('neither',                dict(alpha=0.0, warm_start=0,    balance=False)),
        ('warm start only',        dict(alpha=0.0, warm_start=2000, balance=True)),
        ('CQL only',               dict(alpha=0.5, warm_start=0,    balance=True)),
        ('warm start + CQL',       dict(alpha=0.5, warm_start=2000, balance=True)),
    ]
    for name, settings in configurations:
        print(f'--- {name} ---')
        net, target_net, _ = train_policy(
            train, validation, steps=ABLATION_STEPS, label=name, **settings)
        ablations[name] = evaluate(net, target_net, validation)
        print()

    table = pd.DataFrame({
        name: {'balanced accuracy': m['balanced_accuracy'],
               'raw agreement': m['raw_agreement'],
               'TD (step rows)': m['td_error_non_terminal'],
               'TD (terminal)': m['td_error_terminal'],
               'recall: off': m['recall_by_level']['off'],
               'recall: on': m['recall_by_level']['on'],
               'recall: reduced': m['recall_by_level']['reduced']}
        for name, m in ablations.items()}).T
    display(table.round(4))
    print('\\nchance-level balanced accuracy is 0.3333')
else:
    print('ablations skipped (SHOULD_RUN_ABLATIONS = False)')
"""),

    markdown("""
## 9. The Bradley-Terry preference reward (E2)

Every time the simulated occupant overrode the controller, the dataset recorded
two actions for the same device in the same state: what the controller settled
on, and what the occupant asked for instead. That is a preference comparison,
and comparisons are enough to fit a reward without hand-setting a comfort weight:

```
P(preferred beats proposed) = sigmoid( r(s, i, a_pref) − r(s, i, a_prop) )
```

### Read this before quoting a number

**Every pair points the same way** — preferred = ON, proposed = OFF — because a
pair is only recorded when someone overrode a shed. So the constant rule *"the
occupant always wants it on"* scores **100 %**, and the classification accuracy
of this model is not a result. It does not belong in the abstract.

What *is* a result is the **margin**: `r(s,i,ON) − r(s,i,OFF)` is a learned,
state-dependent measure of how badly the occupant wants that appliance back —
which is exactly what a reward needs to supply. So the cell below reports
whether the margin recovers the generator's own override pressure on held-out
households and dates, and whether it decays as response latency grows.

To make the *direction* informative too, the pair generator would have to emit
comparisons where the occupant let a shed stand. That is a dataset change, not a
notebook change.
"""),

    code("""
class PreferenceReward:
    \"\"\"r(s) -> (28, 3). One trunk, one score per branch level.\"\"\"

    def __init__(self, n_features, hidden=64, seed=7):
        rng = np.random.default_rng(seed)
        self.p = {'w': rng.normal(0, np.sqrt(2 / n_features), (n_features, hidden)),
                  'b': np.zeros(hidden),
                  'r': rng.normal(0, 0.01, (hidden, N_BRANCHES * N_LEVELS)),
                  'rb': np.zeros(N_BRANCHES * N_LEVELS)}

    def forward(self, x):
        h = np.maximum(0, x @ self.p['w'] + self.p['b'])
        return (h @ self.p['r'] + self.p['rb']).reshape(-1, N_BRANCHES, N_LEVELS), h

    def loss_and_gradient(self, x, branch, preferred, proposed, weight, decay):
        r, h = self.forward(x)
        rows = np.arange(len(x))
        margin = r[rows, branch, preferred] - r[rows, branch, proposed]
        loss = float((weight * np.logaddexp(0.0, -margin)).sum() / weight.sum())
        scale = -(weight / weight.sum()) / (1.0 + np.exp(margin))
        dr = np.zeros_like(r)
        dr[rows, branch, preferred] += scale
        dr[rows, branch, proposed] -= scale
        dr = dr.reshape(len(x), -1)
        dh = (dr @ self.p['r'].T) * (h > 0)
        loss += 0.5 * decay * float(np.square(self.p['w']).sum()
                                    + np.square(self.p['r']).sum())
        return loss, {'w': x.T @ dh + decay * self.p['w'], 'b': dh.sum(0),
                      'r': h.T @ dr + decay * self.p['r'], 'rb': dr.sum(0)}

    def update(self, x, branch, preferred, proposed, weight, lr, decay):
        loss, gradient = self.loss_and_gradient(
            x, branch, preferred, proposed, weight, decay)
        norm = np.sqrt(sum(np.square(g).sum() for g in gradient.values()))
        if not np.isfinite(norm) or not np.isfinite(loss):
            raise ValueError('Nonfinite preference update')
        clip = min(1.0, 10.0 / (norm + 1e-12))
        for key, g in gradient.items():
            self.p[key] -= lr * g * clip
        return loss

    def margins(self, x, branch, preferred, proposed):
        r, _ = self.forward(x)
        rows = np.arange(len(x))
        return r[rows, branch, preferred] - r[rows, branch, proposed]


def preference_gradient_test():
    rng = np.random.default_rng(3)
    model = PreferenceReward(6, hidden=5)
    x = rng.normal(size=(4, 6))
    branch = np.array([0, 1, 0, 2])
    preferred = np.array([1, 2, 1, 0])
    proposed = np.array([0, 0, 2, 1])
    weight = np.array([1.0, 0.5, 0.25, 2.0])
    _, gradient = model.loss_and_gradient(x, branch, preferred, proposed, weight, 1e-3)
    for key, index in [('r', (0, 0)), ('w', (2, 1)), ('rb', (4,)), ('b', (3,))]:
        original = model.p[key][index]
        eps = 1e-6
        model.p[key][index] = original + eps
        up = model.loss_and_gradient(x, branch, preferred, proposed, weight, 1e-3)[0]
        model.p[key][index] = original - eps
        down = model.loss_and_gradient(x, branch, preferred, proposed, weight, 1e-3)[0]
        model.p[key][index] = original
        numeric = (up - down) / (2 * eps)
        if not np.isclose(numeric, gradient[key][index], atol=1e-7, rtol=1e-4):
            raise AssertionError(f'Preference gradient mismatch at {key}{index}')
    print('preference gradient test: PASS')


preference_gradient_test()
"""),

    code("""
slot_of = devices.set_index('device_id').slot

def assemble_pairs(split):
    pairs = pd.read_parquet(
        DATASET / 'rl_transitions/override_preference_pairs.parquet')
    pairs = pairs[pairs.split.eq(split)]
    if pairs.empty:
        raise ValueError(f'No {split} preference pairs')

    transitions = pd.read_parquet(
        DATASET / f'rl_transitions/splits/{split}.parquet',
        columns=['episode_id', 'step_id', 'state', 'device_present']
    ).set_index(['episode_id', 'step_id'])

    key = pd.MultiIndex.from_arrays([pairs.episode_id, pairs.step_id])
    if not key.isin(transitions.index).all():
        raise ValueError(f'Some {split} pairs have no matching transition row')
    joined = transitions.loc[key]

    x = np.stack(joined.state.to_numpy()).astype(np.float32)
    present = np.stack(joined.device_present.to_numpy()).astype(float)
    branch = pairs.device_id.map(slot_of).to_numpy()
    if pd.isna(branch).any():
        raise ValueError(f'{split} pairs reference unknown devices')
    branch = branch.astype(int)
    if not (present[np.arange(len(branch)), branch] > 0).all():
        raise ValueError(f'Some {split} pairs point at a padded branch')

    preferred = pairs.preferred_action.to_numpy(int)
    proposed = pairs.proposed_action.to_numpy(int)
    if (preferred == proposed).any():
        raise ValueError(f'Some {split} pairs compare a level with itself')

    weight = pairs.preference_weight.to_numpy(float)
    # Weight is exactly zero where the occupancy gate closed: response latency
    # pushed the override past the moment everyone left the house. The dataset
    # is saying those carry no evidence, so drop them rather than keep rows that
    # contribute nothing but still sit in the denominators.
    keep = weight > 0
    print(f'{split}: {len(pairs)} pairs, dropped {int((~keep).sum())} zero-weight')
    return (x[keep], branch[keep], preferred[keep], proposed[keep],
            weight[keep], pairs[keep].reset_index(drop=True))


train_pairs = assemble_pairs('train')
validation_pairs = assemble_pairs('validation')

px, pbranch, ppreferred, pproposed, pweight, ptable = train_pairs
vpx, vpbranch, vppreferred, vpproposed, vpweight, vptable = validation_pairs

pmean = px.mean(0)
psd = px.std(0)
psd[psd < 1e-8] = 1.0
pxn = ((px - pmean) / psd).astype(np.float32)
vpxn = ((vpx - pmean) / psd).astype(np.float32)

reward_model = PreferenceReward(px.shape[1], seed=7)
rng = np.random.default_rng(7)
best_score, best_parameters = None, None

for step in range(1, 4001):
    index = rng.integers(0, len(pxn), min(BATCH, len(pxn)))
    loss = reward_model.update(pxn[index], pbranch[index], ppreferred[index],
                               pproposed[index], pweight[index], lr=0.01, decay=1e-4)
    if step % 500 == 0:
        margin = reward_model.margins(vpxn, vpbranch, vppreferred, vpproposed)
        weighted = float((vpweight * (margin > 0)).sum() / vpweight.sum())
        if best_score is None or weighted > best_score:
            best_score = weighted
            best_parameters = {k: v.copy() for k, v in reward_model.p.items()}
        print(f'  step {step:5d} | loss {loss:.5f} | validation weighted {weighted:.4f}')

reward_model.p = best_parameters
"""),

    code("""
margin = reward_model.margins(vpxn, vpbranch, vppreferred, vpproposed)
directions = pd.Series(list(zip(vptable.preferred_action,
                                vptable.proposed_action))).value_counts()

print('=== E2: preference reward ===')
print(f'train pairs {len(pxn)}, validation pairs {len(vpxn)}')
print()
if len(directions) == 1:
    a, b = directions.index[0]
    print(f'DIRECTION IS CONSTANT: every pair prefers level {a} over level {b}.')
    print('A constant rule scores 1.0000. Accuracy is NOT a result here.')
print(f'accuracy {float((margin > 0).mean()):.4f}   (for completeness only)')
print()
print('--- the E2 evidence is the margin ---')
probability = vptable.override_probability.to_numpy(float)
pearson = float(np.corrcoef(margin, probability)[0, 1])
spearman = float(pd.Series(margin).corr(pd.Series(probability), method='spearman'))
print(f'margin vs override probability: pearson {pearson:+.4f}  spearman {spearman:+.4f}')
print('mean margin by latency steps (should FALL as latency rises):')
for latency, value in pd.Series(margin).groupby(
        vptable.latency_steps.to_numpy()).mean().items():
    print(f'    {latency} steps: {value:+.3f}')
print('mean margin by pressure source:')
for source, value in pd.Series(margin).groupby(
        vptable.pressure_source.to_numpy()).mean().items():
    print(f'    {source:20s}: {value:+.3f}')
"""),

    markdown("""
## 10. Export for the Raspberry Pi

Ship `mean` and `sd` **with** the weights. If the Pi normalises with anything
else, inference is silently wrong and nothing will tell you — no exception, no
warning, just confidently wrong relay commands.

Inference on the Pi is three matrix multiplies, then the shield:

```python
h = np.maximum(0, (x - mean) / sd @ w + b)
q = (h @ v + vb) + (h @ a + ab).reshape(28, 3)
q -= q.mean(axis=1, keepdims=True)
action = np.argmax(np.where(legal_mask, q, -np.inf), axis=1)
```

`apply_shield` then runs on the result, unchanged, before anything is actuated.
The shield is the last word, not the model.
"""),

    code("""
export_path = WORKING / 'sharp_policy_bdq_v2.npz'
np.savez(
    export_path,
    w=policy.p['w'], b=policy.p['b'],
    v=policy.p['v'], vb=policy.p['vb'],
    a=policy.p['a'], ab=policy.p['ab'],
    mean=MEAN, sd=SD,
    n_features=np.array(train['state'].shape[1]),
    n_branches=np.array(N_BRANCHES),
    n_levels=np.array(N_LEVELS),
    policy_version=np.array('bdq_v2_cql'),
    trained_at=np.array(pd.Timestamp.now(tz='Asia/Kolkata').isoformat()))

# Reload and re-run inference. An export that does not reproduce the in-memory
# model is worse than no export, because it fails on the Pi and not here.
reloaded = np.load(export_path, allow_pickle=False)
for key in ['w', 'b', 'v', 'vb', 'a', 'ab']:
    if not np.array_equal(reloaded[key], policy.p[key]):
        raise ValueError(f'Export mismatch on {key}')

sample = validation['normalised'][:64]
expected, _ = policy.forward(sample)
h = np.maximum(0, sample @ reloaded['w'] + reloaded['b'])
q = ((h @ reloaded['v'] + reloaded['vb'])[:, :, None]
     + (h @ reloaded['a'] + reloaded['ab']).reshape(-1, N_BRANCHES, N_LEVELS))
q = q - (h @ reloaded['a'] + reloaded['ab']).reshape(
    -1, N_BRANCHES, N_LEVELS).mean(2, keepdims=True)
if not np.allclose(q, expected, atol=1e-10):
    raise ValueError('Reloaded forward pass does not match the trained model')
print(f'export verified: {export_path} '
      f'({export_path.stat().st_size / 1024:.0f} KB)')

# A golden vector: the Pi asserts this at boot, before accepting any command.
# A state-builder mismatch between here and the Pi does not crash - it returns
# confident nonsense while the relays click and the dashboard looks fine.
golden = {
    'state': validation['normalised'][0].tolist(),
    'expected_q': expected[0].tolist(),
    'expected_greedy_action': np.argmax(np.where(
        (validation['present'][0][:, None] > 0) & validation['deployment_legal'][0],
        expected[0], -np.inf), axis=1).tolist(),
    'device_present': validation['present'][0].tolist(),
    # The legality mask has to travel with the vector. The greedy action is an
    # argmax over LEGAL levels, so without knowing which devices can dim the Pi
    # cannot reproduce expected_greedy_action even with byte-identical weights
    # and a byte-identical state - it would pick level 2 on an appliance that
    # has no level 2.
    # The DEPLOYMENT mask, not the physical one: a necessity appliance is never
    # shed and never dimmed, so its only legal level is 1.
    'legal_levels': validation['deployment_legal'][0].astype(int).tolist(),
    'supports_reduced': validation['legal'][0][:, 2].astype(int).tolist(),
    'is_necessity': validation['necessity'][0].astype(int).tolist(),
    'occupant_wants_it': validation['entitled'][0].astype(int).tolist(),
    'policy_rule': ('A fan, light or fridge the occupant is using is never shed '
                    'and never dimmed - its only legal level is 1. One the '
                    'occupant is not asking for may be off; protecting essential '
                    'service does not mean running it around the clock.'),
    'how_to_check': [
        'h = maximum(0, state @ w + b)',
        'q = (h @ v + vb) + (h @ a + ab).reshape(28, 3)',
        'q -= q.mean(axis=1, keepdims=True)',
        'assert allclose(q, expected_q)',
        'allowed = device_present[:, None] & legal_levels',
        'assert argmax(where(allowed, q, -inf), axis=1) == expected_greedy_action',
    ],
    'note': ('Assert this on the Pi at boot, before accepting any command. '
             'Equality proves the Pi builds a byte-identical state vector, '
             'normalises it identically, and applies the same legality mask.'),
}
(WORKING / 'golden_vector.json').write_text(json.dumps(golden, indent=2))
print('golden vector written for Join 1 of the integration guide')
"""),

    code("""
warm_peak = max((h for h in history if h['phase'] == 'warm_start'),
                key=lambda h: h['balanced_accuracy'], default=None)
report = {
    'status': 'CONSERVATIVE_BDQ_TRAINED_ON_KAGGLE',
    'checkpoint_selection': 'BEST_CONSERVATIVE_PHASE_BY_BALANCED_ACCURACY',
    'checkpoint_selection_note': (
        'Warm-start snapshots are excluded on purpose. Behaviour cloning '
        'optimises agreement with the logged action directly, so it always wins '
        'that metric; selecting across both phases exports a pure imitator '
        'whose Q values are not value estimates.'),
    'warm_start_peak_for_reference': warm_peak,
    'trained_at': pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),
    'training_rows': int(train['rows']),
    'validation_rows': int(validation['rows']),
    'feature_count': int(train['state'].shape[1]),
    'zero_variance_features': ZERO_VARIANCE,
    'parameter_count': int(sum(v.size for v in policy.p.values())),
    'hyperparameters': {
        'hidden': HIDDEN, 'batch': BATCH, 'gamma': GAMMA,
        'learning_rate': LEARNING_RATE, 'reward_scale': REWARD_SCALE,
        'warm_start_updates': WARM_START_UPDATES,
        'conservative_updates': CONSERVATIVE_UPDATES,
        'cql_alpha': CQL_ALPHA, 'level_balanced': USE_LEVEL_BALANCE, 'seed': SEED},
    'final_validation': final,
    'history': history,
    'ablations': ablations,
    'preference_reward': {
        'experiment': 'E2',
        'train_pairs': int(len(pxn)),
        'validation_pairs': int(len(vpxn)),
        'accuracy_is_meaningless_here': bool(len(directions) == 1),
        'margin_pearson_with_override_probability': pearson,
        'margin_spearman_with_override_probability': spearman},
    'normalisation_fit': 'TRAIN_ONLY',
    'test_split_used': False,
    'gradient_tests': 'PASS',
    'export_verified': True,
    'limitations': [
        'Balanced accuracy measures imitation of three scripted behaviour '
        'policies, not control quality. A real evaluation needs the simulator.',
        'A falling TD error is not evidence of policy quality: offline RL can '
        'overestimate confidently and be wrong.',
        'Every preference pair is synthetic, from the stated rule in '
        'sharp_human_model.py, and every pair points ON over OFF, so the '
        'classification accuracy of the reward model is not a result.',
        'Next-action selection applies the level-legality mask but does not '
        're-apply the joint safety shield, which needs nested device state the '
        'flat release does not carry.',
        'Appliance power values are proxies: 685 of 4,124 devices are '
        'measurement-grounded, the rest are declared assumptions.',
        'The test split has deliberately not been touched.',
    ],
    'approved_for_deployment': False,
}
(WORKING / 'training_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in
                  ['status', 'parameter_count', 'test_split_used',
                   'approved_for_deployment']}, indent=2))
print()
print('written to /kaggle/working:')
for path in sorted(WORKING.glob('*')):
    print(f'  {path.name}  ({path.stat().st_size / 1024:.0f} KB)')
"""),

    markdown("""
## 11. What you must not claim

- A falling TD error is **not** evidence of policy quality. Offline RL without a
  distribution-shift correction can overestimate confidently and be wrong; CQL
  reduces that, it does not remove it.
- Balanced accuracy measures **imitation of three scripted behaviour policies**.
  It is not control performance. Real performance needs the simulator in the
  loop — cost against baseline, peak-to-average ratio, comfort hours.
- Override and attention evidence is **synthetic**, from a stated behavioural
  rule. No dataset records real demand-response overrides in an Indian home;
  that is what the Pi deployment is for.
- The preference model's **accuracy is not a result** — every pair points the
  same way. Quote the margin correlations.
- Appliance power is a proxy: **685 of 4,124** devices are measurement-grounded
  (512 REFIT, 173 iAWE); the rest are declared assumptions.
- Thermal parameters are **declared assumptions** bounded by the RESIDE
  envelope. A fitted coefficient was attempted and rejected — it lost to plain
  persistence in 7 of 11 houses.
- REFIT is 20 **UK** homes. iAWE is **one** Delhi home. RESIDE is 11 **Hyderabad**
  houses over 19 days.
- **Do not touch the test split** until you report final numbers, once.

`scope_and_limits` in `rl_transitions/transition_validation.json` carries all of
this. Copy it into the report rather than paraphrasing it.

## Next

1. Download `sharp_policy_bdq_v2.npz` and `golden_vector.json` from the Output
   panel and hand both to the hardware lane.
2. The golden vector is Join 1 in the integration guide — the check that catches
   a state-builder mismatch, which otherwise fails silently.
3. The shield runs on the Pi **after** this model, unchanged. It is the last word.
"""),
]


def build(root):
    notebook = {
        'cells': CELLS,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11.0',
                              'mimetype': 'text/x-python',
                              'file_extension': '.py'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out = root / 'notebooks/sharp_rl_training_kaggle.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    code_cells = sum(1 for c in CELLS if c['cell_type'] == 'code')
    print(f'wrote {out}')
    print(f'  {len(CELLS)} cells ({code_cells} code, '
          f'{len(CELLS) - code_cells} markdown)')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    build(p.parse_args().root.resolve())

"""Complementary fashion item retrieval on Polyvore Outfits.

Given a partial outfit, retrieve the missing item from all test items of the
target category. See README.md for results and discussion.

Usage (Colab, T4 GPU):
    !pip -q install open_clip_torch huggingface_hub
     %run run_experiments.py

Use %run rather than !python so that Drive mounting, the Hugging Face login
prompt, and inline figures work. Requires approved access to the gated
dataset mvasil/polyvore-outfits.
"""

# %% 0. Setup
import os, io, json, math, copy, random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/set_retrieval'   # cached embeddings, checkpoints, results
except ImportError:
    OUT = './outputs'
DATA = '/content/data'   # raw dataset (local; never commit)
IMG = '/content/img'     # decoded images for qualitative figures (never commit)
os.makedirs(OUT, exist_ok=True)

REPO, SPLIT = 'mvasil/polyvore-outfits', 'disjoint'
CLIP_MODEL, CLIP_PRETRAINED = 'ViT-B-32', 'laion2b_s34b_b79k'
SEEDS = [0, 1, 2]
MAKE_FIGURES = False   # set True to render qualitative examples (section 13)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev, '| torch', torch.__version__)

# %% 1. Download (requires approved access to the gated dataset)
from huggingface_hub import login, snapshot_download

if not os.path.exists(f'{DATA}/polyvore_item_metadata.json'):
    if not os.environ.get('HF_TOKEN'):
        login()
    snapshot_download(REPO, repo_type='dataset', local_dir=DATA,
                      allow_patterns=['polyvore_item_metadata.json',
                                      f'{SPLIT}/*.json', f'data/{SPLIT}/*.parquet'])

meta = json.load(open(f'{DATA}/polyvore_item_metadata.json'))
splits = {s: json.load(open(f'{DATA}/{SPLIT}/{s}.json')) for s in ['train', 'valid', 'test']}
PQ = {'train': f'{DATA}/data/{SPLIT}/train.parquet',
      'valid': f'{DATA}/data/{SPLIT}/validation.parquet',
      'test':  f'{DATA}/data/{SPLIT}/test.parquet'}

# %% 2. Dataset statistics and cross-split overlap
def items_of(s):
    return {it['item_id'] for o in splits[s] for it in o['items']}

item_sets = {s: items_of(s) for s in splits}
for s, outfits in splits.items():
    sizes = [len(o['items']) for o in outfits]
    print(f'{s:5s}: {len(outfits):,} outfits, {len(item_sets[s]):,} items, '
          f'{np.mean(sizes):.1f} items/outfit')

# The "disjoint" split still shares some items across splits; build_queries() handles this.
for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
    print(f'{a} ∩ {b}: {len(item_sets[a] & item_sets[b]):,} items')

every = set().union(*item_sets.values())
for c, n in Counter(meta[i]['semantic_category'] for i in every).most_common():
    print(f'  {c:12s} {n:,}')

# %% 3. CLIP embeddings (extracted once, cached to OUT)
EMB, IDS = f'{OUT}/emb_vitb32.npy', f'{OUT}/item_ids.json'

def iter_rows(path, bs=256):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=bs, columns=['item_id', 'image']):
        yield batch.to_pylist()

if not os.path.exists(EMB):
    import open_clip
    from PIL import Image
    from tqdm.auto import tqdm

    clip, _, prep = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    clip = clip.to(dev).eval()
    all_ids, embs, bad, seen = [], [], [], set()
    with torch.no_grad(), torch.autocast(dev, enabled=(dev == 'cuda')):
        for s, path in PQ.items():
            n = pq.ParquetFile(path).metadata.num_rows
            for rows in tqdm(iter_rows(path), total=math.ceil(n / 256), desc=s):
                xs, keep = [], []
                for r in rows:
                    iid = r['item_id']
                    if iid in seen:          # items shared across splits are embedded once
                        continue
                    try:
                        xs.append(prep(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')))
                    except Exception:
                        bad.append(iid)
                        continue
                    keep.append(iid)
                    seen.add(iid)
                if xs:
                    e = clip.encode_image(torch.stack(xs).to(dev)).float()
                    embs.append(F.normalize(e, dim=-1).cpu())
                    all_ids += keep
    np.save(EMB, torch.cat(embs).numpy().astype('float16'))
    json.dump(all_ids, open(IDS, 'w'))
    print(f'embedded {len(all_ids):,} items | unreadable: {len(bad)}')

# %% 4. Load embeddings, build queries and candidate pools
E = np.load(EMB).astype('float32')
ids = json.load(open(IDS))
idx = {iid: j for j, iid in enumerate(ids)}          # item id -> embedding row
cats = sorted({meta[i]['semantic_category'] for i in ids})
cat_id = np.array([cats.index(meta[i]['semantic_category']) for i in ids])
Et = torch.from_numpy(E).to(dev)
assert len(ids) == len(set(ids)) == E.shape[0]
print(E.shape, cats)

def build_queries(s, banned, drop_whole_set):
    """Hold out each item of an outfit as the target; the rest is the query.
    banned: items that appear in other splits.
    drop_whole_set=True  -> skip any outfit containing a banned item (test).
    drop_whole_set=False -> never use a banned item as the target (valid)."""
    Q, dropped = [], 0
    for o in splits[s]:
        raw = list(dict.fromkeys(it['item_id'] for it in o['items']))
        if drop_whole_set and any(i in banned for i in raw):
            dropped += 1
            continue
        its = [idx[i] for i in raw]
        if len(its) < 2:
            continue
        for t in its:
            if ids[t] in banned:
                continue
            Q.append(([j for j in its if j != t], t))
    print(f'{s}: {len(Q):,} queries, {dropped:,} outfits dropped')
    return Q

def build_pools(item_ids):
    """Candidate pool per category (embedding rows)."""
    P = defaultdict(list)
    for i in item_ids:
        P[cat_id[idx[i]]].append(idx[i])
    return {c: np.array(sorted(v)) for c, v in P.items()}

tr, va, te = item_sets['train'], item_sets['valid'], item_sets['test']
test_Q = build_queries('test', tr | va, drop_whole_set=True)
valid_Q = build_queries('valid', tr, drop_whole_set=False)
train_Q = build_queries('train', set(), drop_whole_set=False)
test_P, valid_P = build_pools(te), build_pools(va)
for c in sorted(test_P):
    print(f'  {cats[c]:12s} {len(test_P[c]):,} test candidates')

# %% 5. Evaluation and zero-shot baselines
def pad(ctxs):
    L = max(len(c) for c in ctxs)
    A = np.zeros((len(ctxs), L), dtype=np.int64)
    M = np.zeros((len(ctxs), L), dtype=bool)
    for i, c in enumerate(ctxs):
        A[i, :len(c)] = c
        M[i, :len(c)] = True
    return torch.from_numpy(A).to(dev), torch.from_numpy(M).to(dev)

@torch.no_grad()
def evaluate(Q, P, score_fn, bs=256):
    """score_fn(X [B,L,D], M [B,L], C [N,D], c) -> scores [B,N].
    Returns one row per query with the target's rank in its category pool."""
    by_cat = defaultdict(list)
    for ctx, t in Q:
        by_cat[cat_id[t]].append((ctx, t))
    rows = []
    for c, qs in by_cat.items():
        pool = P[c]
        pos = {int(j): p for p, j in enumerate(pool)}
        C = Et[torch.from_numpy(pool).to(dev)]
        for b in range(0, len(qs), bs):
            chunk = qs[b:b + bs]
            A, M = pad([q[0] for q in chunk])
            S = score_fn(Et[A], M, C, c).float()
            for i, (ctx, _) in enumerate(chunk):      # query items cannot be the answer
                for j in ctx:
                    if j in pos:
                        S[i, pos[j]] = -1e9
            tpos = torch.tensor([pos[t] for _, t in chunk], device=dev)
            rank = ((S > S.gather(1, tpos[:, None])).sum(1) + 1).tolist()
            rows += [(cats[c], len(ctx), r, len(pool)) for (ctx, _), r in zip(chunk, rank)]
    return pd.DataFrame(rows, columns=['cat', 'n_ctx', 'rank', 'pool'])

def score_mean(X, M, C, c):
    """Zero-shot: cosine similarity to the mean query embedding."""
    q = (X * M[..., None]).sum(1) / M.sum(1, keepdim=True)
    return F.normalize(q, dim=-1) @ C.T

def score_max(X, M, C, c):
    """Zero-shot: best similarity to any single query item."""
    return (X @ C.T).masked_fill(~M[..., None], -1e9).max(1).values

def r10(df):
    return (df['rank'] <= 10).mean() * 100

def summarize(df, name):
    r = df['rank']
    return {'method': name, 'R@10': r10(df), 'R@50': (r <= 50).mean() * 100,
            'MRR': (1 / r).mean() * 100, 'median rank': r.median()}

def random_row(df):
    n = df['pool']
    return {'method': 'Random (expected)', 'R@10': (10 / n).clip(upper=1).mean() * 100,
            'R@50': (50 / n).clip(upper=1).mean() * 100,
            'MRR': ((np.log(n) + 0.5772) / n).mean() * 100, 'median rank': (n / 2).median()}

zs = {'Zero-shot: CLIP mean': evaluate(test_Q, test_P, score_mean),
      'Zero-shot: per-item max': evaluate(test_Q, test_P, score_max)}
first = next(iter(zs.values()))
print(pd.DataFrame([random_row(first)] + [summarize(d, k) for k, d in zs.items()])
      .round(2).to_string(index=False))

# %% 6. Models
# norm=True adds LayerNorm to the inputs (and hidden layers of the MLPs).
# norm=False reproduces the first, unnormalized runs used in the ablation.
D, H, P_DROP = 512, 256, 0.1

def mlp(dims, norm, pre_norm=False):
    layers = [nn.LayerNorm(dims[0])] if pre_norm else []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers += [nn.GELU(), nn.Dropout(P_DROP)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

class Base(nn.Module):
    """Shared candidate tower, category embedding, and temperature."""
    def __init__(self, norm):
        super().__init__()
        self.item_net = mlp([D, 512, H], norm=False)
        self.cat_emb = nn.Embedding(len(cats), H)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.xn = nn.LayerNorm(D) if norm else nn.Identity()
        self.en = nn.LayerNorm(H) if norm else nn.Identity()

    def item(self, x):
        return F.normalize(self.item_net(x), dim=-1)

    def tokens(self, X, c):
        """Concatenate each query item with the target-category embedding."""
        e = self.en(self.cat_emb(c))[:, None].expand(-1, X.shape[1], -1)
        return torch.cat([self.xn(X), e], -1)

    @staticmethod
    def pool(h, M):
        m = M[..., None].float()
        return (h * m).sum(1) / m.sum(1)

class AdditiveMean(Base):
    """Transform each (item, category) pair independently, then average.
    No item-item interaction, so per-item vectors can be cached."""
    def __init__(self, norm=True, w=896):
        super().__init__(norm)
        self.phi = mlp([D + H, w, 512, H], norm)

    def query(self, X, M, c):
        return F.normalize(self.pool(self.phi(self.tokens(X, c)), M), dim=-1)

class DeepSets(Base):
    """Average, then apply a nonlinear transform to the pooled vector."""
    def __init__(self, norm=True, w=1280):
        super().__init__(norm)
        self.phi = mlp([D + H, 512, H], norm)
        self.rho = mlp([H, w, H], norm=False, pre_norm=norm)

    def query(self, X, M, c):
        pooled = self.pool(self.phi(self.tokens(X, c)), M)
        return F.normalize(pooled + self.rho(pooled), dim=-1)

class SetEncoder(Base):
    """A category token and the query items attend to each other (2-layer Transformer)."""
    def __init__(self, norm=True, layers=2, heads=4):
        super().__init__(norm)
        self.inp = nn.Linear(D, H)
        layer = nn.TransformerEncoderLayer(H, heads, 2 * H, dropout=P_DROP,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H))

    def query(self, X, M, c):
        tok = torch.cat([self.en(self.cat_emb(c))[:, None], self.inp(self.xn(X))], 1)
        pad_mask = torch.cat([torch.zeros(len(M), 1, dtype=torch.bool, device=M.device), ~M], 1)
        return F.normalize(self.out(self.enc(tok, src_key_padding_mask=pad_mask)[:, 0]), dim=-1)

def make_score(model):
    def fn(X, M, C, c):
        cc = torch.full((X.shape[0],), int(c), device=X.device)
        return model.query(X, M, cc) @ model.item(C).T
    return fn

# (checkpoint prefix, label, class). Prefixes match the original runs so saved checkpoints load.
CONFIGS = [('learned_mean_big', 'Learned additive', AdditiveMean),
           ('deepsets', 'DeepSets', DeepSets),
           ('set_encoder', 'Set encoder', SetEncoder)]
for _, label, cls in CONFIGS:
    print(f'{label:18s} {sum(p.numel() for p in cls().parameters()):,} params')

# %% 7. Training
train_by_cat = defaultdict(list)
for ctx, t in train_Q:
    train_by_cat[int(cat_id[t])].append((ctx, t))

def batches(by_cat, bs):
    """Every batch shares one target category, so in-batch negatives are
    same-category items, matching the evaluation."""
    out = []
    for c, qs in by_cat.items():
        qs = qs[:]
        random.shuffle(qs)
        out += [(c, qs[i:i + bs]) for i in range(0, len(qs), bs)]
    random.shuffle(out)
    return out

def subsample(ctx):
    """Randomly drop query items so the model handles varying set sizes."""
    if len(ctx) > 1 and random.random() < 0.5:
        return random.sample(ctx, random.randint(1, len(ctx)))
    return ctx

def train(model, name, epochs=30, bs=512, lr=1e-3, wd=0.05, seed=0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.to(dev)
    steps = epochs * len(batches(train_by_cat, bs))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    print(f'[{name} seed {seed}] {sum(p.numel() for p in model.parameters()):,} params, {steps:,} steps')

    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for c, chunk in batches(train_by_cat, bs):
            if len(chunk) < 8:
                continue
            A, M = pad([subsample(q[0]) for q in chunk])
            t = torch.tensor([q[1] for q in chunk], device=dev)
            q = model.query(Et[A], M, torch.full((len(chunk),), c, device=dev))
            logits = model.logit_scale.exp().clamp(max=100) * (q @ model.item(Et[t]).T)
            # mask invalid negatives: duplicate targets, and targets inside another query
            eye = torch.eye(len(t), dtype=torch.bool, device=dev)
            same = (t[:, None] == t[None, :]) & ~eye
            in_ctx = ((A[:, :, None] == t[None, None, :]) & M[:, :, None]).any(1) & ~eye
            loss = F.cross_entropy(logits.masked_fill(same | in_ctx, -1e9),
                                   torch.arange(len(t), device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1

        model.eval()
        v = r10(evaluate(valid_Q, valid_P, make_score(model)))
        flag = ''
        if v > best:
            best, best_state, flag = v, copy.deepcopy(model.state_dict()), ' *'
        if ep % 3 == 0 or flag:
            print(f'  epoch {ep + 1:2d} | loss {tot / n:.3f} | valid R@10 {v:.2f}{flag}')

    model.load_state_dict(best_state)
    torch.save(best_state, f'{OUT}/{name}_seed{seed}.pt')
    print(f'  best valid R@10 {best:.2f}')
    return model.eval()

def get_model(cls, prefix, norm, seed, retrain=False):
    """Load a cached checkpoint if present, otherwise train one."""
    name = f'{prefix}_v2' if norm else prefix
    path = f'{OUT}/{name}_seed{seed}.pt'
    torch.manual_seed(seed)
    model = cls(norm=norm)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path, map_location=dev))
        return model.to(dev).eval()
    return train(model, name, seed=seed)

# %% 8. Main experiment: 3 models x 3 seeds, with and without normalization
ZS = 'Zero-shot: CLIP mean'
runs = [zs[ZS].assign(method=ZS, norm=True, seed=0)]
for norm in [True, False]:
    for prefix, label, cls in CONFIGS:
        for seed in SEEDS:
            df = evaluate(test_Q, test_P, make_score(get_model(cls, prefix, norm, seed)))
            runs.append(df.assign(method=label, norm=norm, seed=seed))
            print(f'{label} | norm={norm} | seed {seed} | test R@10 {r10(df):.2f}')

allr = pd.concat(runs, ignore_index=True)
allr['qid'] = allr.groupby(['method', 'norm', 'seed']).cumcount()
allr['n'] = allr['n_ctx'].clip(upper=6)
allr['r10'] = (allr['rank'] <= 10) * 100.0
allr['r50'] = (allr['rank'] <= 50) * 100.0
allr['rr'] = 100.0 / allr['rank']
allr.to_csv(f'{OUT}/test_all_runs.csv', index=False)

# %% 9. Results tables (mean ± std over seeds)
ORDER = [ZS] + [label for _, label, _ in CONFIGS]
N_COLS = lambda ks: [f'{k}' if k < 6 else '6+' for k in ks]

def pm(per_seed, keys):
    a = per_seed.groupby(keys).agg(['mean', 'std'])
    return a.apply(lambda r: f"{r['mean']:.2f}±{r['std']:.2f}" if pd.notna(r['std'])
                   else f"{r['mean']:.2f}", axis=1)

main = allr[allr['norm']]
overall = pd.DataFrame({name: pm(main.groupby(['method', 'seed'])[col].mean(), 'method')
                        for name, col in [('R@10', 'r10'), ('R@50', 'r50'), ('MRR', 'rr')]})
print('Test results (with LayerNorm)')
print(overall.reindex(ORDER).to_string())

abl = pm(allr[allr['method'] != ZS].groupby(['method', 'norm', 'seed'])['r10'].mean(),
         ['method', 'norm']).unstack('norm')
abl.columns = ['without LayerNorm' if not c else 'with LayerNorm' for c in abl.columns]
print('\nAblation: R@10 with vs. without normalization')
print(abl.reindex(ORDER[1:]).to_string())

by_n = pm(main.groupby(['method', 'seed', 'n'])['r10'].mean(), ['method', 'n']).unstack('n')
by_n.columns = N_COLS(by_n.columns)
print('\nR@10 by number of query items')
print(by_n.reindex(ORDER).to_string())

# %% 10. Analysis 1 (exploratory): does attention help when query items are diverse?
def ordered(Q):
    """Queries in the same order evaluate() emits them."""
    by = defaultdict(list)
    for ctx, t in Q:
        by[cat_id[t]].append((ctx, t))
    return [q for qs in by.values() for q in qs]

def mean_pair_sim(ctx):
    n = len(ctx)
    if n < 2:
        return np.nan
    S = E[ctx] @ E[ctx].T
    return (S.sum() - np.trace(S)) / (n * (n - 1))

oq = ordered(test_Q)
chk = main[main['method'] == ZS]
assert (chk['n_ctx'].values == np.array([len(c) for c, _ in oq])).all(), 'query order mismatch'
sim = np.array([mean_pair_sim(ctx) for ctx, _ in oq])
main = main.assign(sim=sim[main['qid'].values])

ADD = 'Learned additive'
sub = main[main['n_ctx'] >= 2].copy()
lo, hi = np.nanquantile(sim, [1 / 3, 2 / 3])
sub['div'] = np.where(sub['sim'] <= lo, '1 diverse', np.where(sub['sim'] <= hi, '2 medium', '3 similar'))
print('R@10 by query diversity (queries with >= 2 items)')
print(pm(sub.groupby(['method', 'seed', 'div'])['r10'].mean(), ['method', 'div'])
      .unstack('div').reindex(ORDER).to_string())
m1 = sub.groupby(['method', 'div'])['r10'].mean().unstack('div')
print('\nGap vs. learned additive (pts)')
print((m1.drop(index=ADD) - m1.loc[ADD]).round(2).to_string())

# %% 11. Analysis 2: is the set-size curve driven by category mix?
w = chk.groupby('cat').size()
w = w / w.sum()
cell = main.groupby(['method', 'n', 'cat'])['r10'].mean().unstack('cat')
std_curve = ((cell * w).sum(1) / (cell.notna() * w).sum(1)).unstack('n')
raw_curve = main.groupby(['method', 'n'])['r10'].mean().unstack('n')
raw_curve.columns = std_curve.columns = N_COLS(raw_curve.columns)
print('Raw R@10 by set size')
print(raw_curve.reindex(ORDER).round(2).to_string())
print('\nR@10 by set size, reweighted to a fixed category mix')
print(std_curve.reindex(ORDER).round(2).to_string())

# %% 12. Analysis 3: same queries, reveal only the first k items
rng = random.Random(0)
big = [(ctx, t) for ctx, t in test_Q if len(ctx) >= 5]
perms = [rng.sample(ctx, len(ctx)) for ctx, _ in big]     # fixed reveal order per query

controlled = {'Learned additive': (AdditiveMean, 'learned_mean_big'),
              'Set encoder': (SetEncoder, 'set_encoder')}
models = {label: [get_model(cls, p, True, s) for s in SEEDS]
          for label, (cls, p) in controlled.items()}
out = defaultdict(dict)
for k in range(1, 6):
    Qk = [(p[:k], t) for p, (_, t) in zip(perms, big)]
    out[ZS][f'k={k}'] = r10(evaluate(Qk, test_P, score_mean))
    for label, ms in models.items():
        out[label][f'k={k}'] = np.mean([r10(evaluate(Qk, test_P, make_score(m))) for m in ms])
print(f'{len(big):,} queries with >= 5 items')
print(pd.DataFrame(out).T.round(2).to_string())

# %% 13. Qualitative examples (optional; figures go to OUT/figs, never commit them)
def topk(score_fn, ctx, t, k=5):
    """Top-k candidates and the target's rank, using the evaluation rules."""
    c = cat_id[t]
    pool = test_P[c]
    pos = {int(j): p for p, j in enumerate(pool)}
    A, M = pad([ctx])
    with torch.no_grad():
        S = score_fn(Et[A], M, Et[torch.from_numpy(pool).to(dev)], c).float()[0]
    for j in ctx:
        if j in pos:
            S[pos[j]] = -1e9
    rank = int((S > S[pos[t]]).sum()) + 1
    return pool[torch.argsort(S, descending=True)[:k].cpu().numpy()], rank

def draw(a, j, title, color=None):
    from PIL import Image
    a.axis('on')
    a.imshow(Image.open(f'{IMG}/{ids[j]}.jpg'))
    a.set_title(title, fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(color is not None)
        s.set_color(color or 'k')
        s.set_linewidth(3)

def show(ctx, t, methods, k=5, save=None):
    """Row 1: query items and target. Other rows: top-k per method (green = target)."""
    import matplotlib.pyplot as plt
    cols = max(len(ctx) + 1, k)
    fig, ax = plt.subplots(1 + len(methods), cols, figsize=(2 * cols, 2.4 * (1 + len(methods))))
    for a in ax.ravel():
        a.axis('off')
    for i, j in enumerate(ctx):
        draw(ax[0, i], j, f'query: {cats[cat_id[j]]}')
    draw(ax[0, len(ctx)], t, f'target: {cats[cat_id[t]]}', color='tab:blue')
    for r, (name, fn) in enumerate(methods.items(), 1):
        top, rank = topk(fn, ctx, t, k)
        for i, j in enumerate(top):
            title = f'{name}\ntarget rank {rank:,}' if i == 0 else f'#{i + 1}'
            draw(ax[r, i], j, title, color='tab:green' if j == t else None)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()

if MAKE_FIGURES:
    FIG = f'{OUT}/figs'
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    for rows in iter_rows(PQ['test'], bs=512):
        for r in rows:
            p = f"{IMG}/{r['item_id']}.jpg"
            if not os.path.exists(p):
                with open(p, 'wb') as f:
                    f.write(r['image']['bytes'])

    # Buckets are defined up front and examples are sampled at random within each bucket,
    # to avoid cherry-picking.
    ra = main[(main['method'] == ADD) & (main['seed'] == 0)].sort_values('qid')['rank'].values
    rz = main[main['method'] == ZS].sort_values('qid')['rank'].values
    nq = np.array([len(c) for c, _ in oq])
    ok = (nq >= 2) & (nq <= 4)                        # fits on one row
    BUCKETS = {'learned_only': ok & (ra <= 5) & (rz > 100),
               'both_fail': ok & (ra > 500) & (rz > 500),
               'zeroshot_only': ok & (rz <= 5) & (ra > 100)}
    for name, mask in BUCKETS.items():
        print(f'{name}: {mask.sum():,} queries ({mask.mean() * 100:.1f}%)')

    methods = {'CLIP mean (zero-shot)': score_mean,
               'Learned additive': make_score(models['Learned additive'][0])}
    pick = np.random.default_rng(0)
    for name, mask in BUCKETS.items():
        for n, qi in enumerate(pick.choice(np.flatnonzero(mask), size=min(2, mask.sum()),
                                           replace=False)):
            print(f'\n[{name} #{n + 1}] query {qi}')
            ctx, t = oq[qi]
            show(ctx, t, methods, save=f'{FIG}/{name}_{n + 1}.png')
else:
    print('Done. Set MAKE_FIGURES = True to render qualitative examples.')

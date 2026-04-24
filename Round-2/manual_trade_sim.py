"""
=============================================================================
TRADING OUTPOST ALLOCATION OPTIMIZER  — v3 (realistic)
=============================================================================
Changes from v2:
  1. DISCRETE INTEGER speed distributions with hard clustering at round
     numbers (30, 40, 50, 60) — mirrors real human behaviour
  2. BUNCHING EFFECT in rank: all tied speeds share the same rank, so a
     cluster of 2000 at S=40 creates a massive dead-zone — explicitly
     modelled and punished
  3. R/C/S JOINT DEPENDENCE enforced — total must equal 100, so R and C
     are derived from leftover = 100 - S; their tradeoff is baked in
  4. CROWDING PENALTY: multiplier is computed exactly on the discrete
     integer grid; regions with high cluster density cause rank jumps
     of hundreds, not one-at-a-time
  5. INTEGER-ONLY everywhere: all speeds are ints, BO rounds candidates
     before evaluation, no float speed comparisons
  6. SNIPE STRATEGY for copycat: copycat models players who pick S+1 to
     just beat the cluster, not players who blindly mirror you
  7. HIGH-SPEED CONVERGENCE CLUSTER: quant_meta peaks at 55-65 (not 45)
     because sophisticated players converge on the multiplier breakeven;
     special "round_hunter" meta for players who cluster exactly at
     round numbers with high density spikes

Usage:
    pip install numpy scipy matplotlib pandas scikit-learn tqdm
    python trading_sim.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings, os, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "N_OPPONENTS":       4999,
    "BUDGET":            50_000,
    "MC_ITERATIONS":     2000,
    "SWEEP_ITERS":       500,
    "SPEED_STEP":        1,
    "TRIAL_ITERS_LIST":  [200, 500, 1000, 2000, 5000],
    "BAYES_CALLS":       60,
    "BAYES_ITERS":       300,
    "N_WORKERS":         max(1, multiprocessing.cpu_count() - 1),
    "OUTPUT_DIR":        "trading_sim_results",
    "VERBOSE":           True,
    # Crowding penalty: multiplier bonus subtracted when landing in a dense cluster
    # Set to 0 to disable. Realistic range 0.02-0.08.
    "CROWDING_PENALTY":  0.05,
    # Fraction of cluster players who snipe +1 above their cluster anchor
    "SNIPE_FRACTION":    0.15,
}

BUDGET  = CONFIG["BUDGET"]
N_OPP   = CONFIG["N_OPPONENTS"]
N_TOTAL = N_OPP + 1
_LOG101 = np.log(101.0)

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)


# =============================================================================
# CORE MATH — all integer inputs enforced at call sites
# =============================================================================

def research_val(r):
    """r must be non-negative int or array of ints."""
    return 200_000.0 * np.log1p(r) / _LOG101

def scale_val(c):
    return 7.0 * (c / 100.0)

def base_score(r, c):
    return research_val(r) * scale_val(c)

def optimal_rc_integer(L: int) -> tuple:
    """
    Integer-only optimal R/C split for leftover budget L.
    R + C = L, both non-negative integers.
    Vectorised argmax — no Python loop.
    Constraint: R, C >= 0, R + C = L exactly.
    """
    L = int(round(L))
    if L <= 0:
        return 0, 0, 0.0
    r_arr  = np.arange(0, L + 1, dtype=np.int32)
    scores = base_score(r_arr, L - r_arr)
    idx    = int(np.argmax(scores))
    return idx, L - idx, float(scores[idx])

# Pre-build integer-optimal (R, C) for every integer speed 0..100
_RC_TABLE = {s: optimal_rc_integer(100 - s) for s in range(101)}


# =============================================================================
# DISCRETE INTEGER OPPONENT GENERATION
# =============================================================================
#
# Key design decisions (all changes from v2):
#
# Change 1 & 5: All speeds are INTEGERS. Real humans pick round numbers.
#   We use a mixture distribution that spikes at multiples of 5 and 10.
#
# Change 2: Bunching is explicit. Large blocks of opponents are set to
#   exactly the same integer, creating rank plateaus.
#
# Change 7: quant_meta now clusters at 55-65 (breakeven zone for the
#   rank curve) rather than 45. round_hunter meta is a new profile.
#
# Change 6: copycat is split into two behaviours:
#   - Pure mirrors: same integer as you (competes directly)
#   - Snipers: your S+1 (just enough to beat you)

def _round_spike_sample(rng, center: int, spread: int, n: int) -> np.ndarray:
    """
    Sample integers that cluster around `center` but with spikes at
    the nearest multiples of 5 and 10. Matches human round-number bias.
    """
    raw = rng.normal(center, spread, n * 3).astype(np.int32)
    raw = np.clip(raw, 0, 100)
    # With 30% probability, snap to nearest multiple of 5
    snap_mask = rng.random(len(raw)) < 0.30
    raw[snap_mask] = np.round(raw[snap_mask] / 5).astype(np.int32) * 5
    raw = np.clip(raw, 0, 100)
    return raw[:n]

def _discrete_normal(rng, mu: float, sigma: float, n: int) -> np.ndarray:
    """
    Discrete version of normal — round to int, clip to [0,100].
    With round-number snapping for realism.
    """
    raw = rng.normal(mu, sigma, n)
    rounded = np.clip(np.round(raw).astype(np.int32), 0, 100)
    # 25% snap to nearest 5
    snap = rng.random(n) < 0.25
    rounded[snap] = np.clip(
        (np.round(rounded[snap] / 5) * 5).astype(np.int32), 0, 100
    )
    return rounded

def gen_opponent_matrix(meta: str, iters: int, my_s: int = 50,
                        rng: np.random.Generator = None) -> np.ndarray:
    """
    Returns integer (iters, N_OPP) speed matrix.
    All values are integers in [0, 100].

    Metas:
      naive        — discrete normal ~33, heavy snapping to round numbers
      speed_heavy  — cluster around 55-65 with spikes at 60
      blotto       — uniform integers 0-100
      nash_approx  — concentrated around 48-52 (smart convergence zone),
                     with snapping
      mixed        — realistic field: 30% naive / 25% nash / 20% speed_heavy
                     / 15% round_hunter / 10% anti_speed
      copycat      — Change 6: SNIPE model: (1-snipe_frac) players at my_s,
                     snipe_frac players at my_s+1 to beat the cluster
      quant_meta   — Change 7: convergence cluster at 55-65 (breakeven
                     zone where mult gain > base_score loss)
      anti_speed   — cluster at 15-25, pure base-score maximisers
      bimodal      — 50% naive / 50% nash
      round_hunter — Change 7: players who deliberately pick exact round
                     numbers (20, 30, 40, 50, 60, 70) with equal weight;
                     huge discrete spikes
      high_conv    — Change 7: sophisticated players who model the rank
                     curve and converge at 55-58 (just above the naives)
      contrarian   — bimodal 15 / 75 (anti-coordinate)
    """
    if rng is None:
        rng = np.random.default_rng()
    my_s = int(round(my_s))

    if meta == "naive":
        return _discrete_normal(rng, 33, 9, iters * N_OPP).reshape(iters, N_OPP)

    elif meta == "speed_heavy":
        # Spike at 60 + spread
        base = _discrete_normal(rng, 60, 10, iters * N_OPP)
        # Extra spike: 20% exactly at 60
        spike_mask = rng.random(iters * N_OPP) < 0.20
        base[spike_mask] = 60
        return np.clip(base, 0, 100).reshape(iters, N_OPP)

    elif meta == "blotto":
        return rng.integers(0, 101, size=(iters, N_OPP))

    elif meta == "nash_approx":
        # Converged field: tight cluster 46-52, snap to 50 frequently
        base = _discrete_normal(rng, 49, 4, iters * N_OPP)
        snap = rng.random(iters * N_OPP) < 0.35
        base[snap] = 50
        return np.clip(base, 0, 100).reshape(iters, N_OPP)

    elif meta == "mixed":
        n_naive  = int(N_OPP * 0.30)
        n_nash   = int(N_OPP * 0.25)
        n_speed  = int(N_OPP * 0.20)
        n_round  = int(N_OPP * 0.15)
        n_anti   = N_OPP - n_naive - n_nash - n_speed - n_round

        rows = []
        for _ in range(iters):
            p_naive = _discrete_normal(rng, 33, 9, n_naive)
            p_nash  = _discrete_normal(rng, 49, 4, n_nash)
            # extra 50-spike for nash
            snap = rng.random(n_nash) < 0.30; p_nash[snap] = 50

            p_speed = _discrete_normal(rng, 60, 10, n_speed)
            spike   = rng.random(n_speed) < 0.20; p_speed[spike] = 60

            # round_hunter: pick from {20,30,40,50,60,70} uniformly
            p_round = rng.choice([20, 30, 40, 50, 60, 70], size=n_round)

            p_anti  = _discrete_normal(rng, 20, 7, n_anti)

            row = np.concatenate([p_naive, p_nash, p_speed, p_round, p_anti])
            row = np.clip(row, 0, 100).astype(np.int32)
            rng.shuffle(row)
            rows.append(row)
        return np.array(rows, dtype=np.int32)

    elif meta == "copycat":
        # Change 6: snipe model, not pure mirror
        snipe_frac = CONFIG["SNIPE_FRACTION"]
        n_snipe  = max(1, int(N_OPP * snipe_frac))
        n_mirror = N_OPP - n_snipe
        snipe_val  = int(np.clip(my_s + 1, 0, 100))
        mirror_val = my_s

        mat = np.empty((iters, N_OPP), dtype=np.int32)
        mat[:, :n_mirror] = mirror_val
        mat[:, n_mirror:] = snipe_val
        # Add small noise to mirror group (±2) so not perfectly identical
        noise = rng.integers(-2, 3, size=(iters, n_mirror))
        mat[:, :n_mirror] = np.clip(mat[:, :n_mirror] + noise, 0, 100)
        return mat

    elif meta == "quant_meta":
        # Change 7: breakeven zone 55-65 (where mult gain > base_score loss)
        base = _discrete_normal(rng, 60, 5, iters * N_OPP)
        # Spike at 60 (most common quant pick)
        spike = rng.random(iters * N_OPP) < 0.25; base[spike] = 60
        # Secondary spike at 55
        spike2 = rng.random(iters * N_OPP) < 0.15; base[spike2] = 55
        return np.clip(base, 0, 100).reshape(iters, N_OPP)

    elif meta == "anti_speed":
        base = _discrete_normal(rng, 20, 7, iters * N_OPP)
        snap = rng.random(iters * N_OPP) < 0.30; base[snap] = 20
        return np.clip(base, 0, 100).reshape(iters, N_OPP)

    elif meta == "bimodal":
        half = N_OPP // 2
        rows = []
        for _ in range(iters):
            p1 = _discrete_normal(rng, 33, 9, half)
            p2 = _discrete_normal(rng, 49, 4, N_OPP - half)
            row = np.clip(np.concatenate([p1, p2]), 0, 100).astype(np.int32)
            rng.shuffle(row)
            rows.append(row)
        return np.array(rows, dtype=np.int32)

    elif meta == "round_hunter":
        # Change 7: pure round-number pickers — massive discrete spikes
        anchors = np.array([20, 30, 40, 50, 60, 70], dtype=np.int32)
        # 60% go exactly to anchor, 40% ±1 from anchor
        raw = rng.choice(anchors, size=(iters, N_OPP))
        jitter_mask = rng.random((iters, N_OPP)) < 0.40
        jitter = rng.integers(-1, 2, size=(iters, N_OPP))
        raw = np.where(jitter_mask, raw + jitter, raw)
        return np.clip(raw, 0, 100).astype(np.int32)

    elif meta == "high_conv":
        # Change 7: sophisticated players converge at 55-58 (just above naive cluster)
        base = _discrete_normal(rng, 56, 3, iters * N_OPP)
        spike = rng.random(iters * N_OPP) < 0.35; base[spike] = 55
        return np.clip(base, 0, 100).reshape(iters, N_OPP)

    elif meta == "contrarian":
        half = N_OPP // 2
        rows = []
        for _ in range(iters):
            p1 = _discrete_normal(rng, 15, 5, half)
            p2 = _discrete_normal(rng, 75, 5, N_OPP - half)
            row = np.clip(np.concatenate([p1, p2]), 0, 100).astype(np.int32)
            rng.shuffle(row)
            rows.append(row)
        return np.array(rows, dtype=np.int32)

    else:
        raise ValueError(f"Unknown meta: {meta}")


ALL_METAS = [
    "naive", "speed_heavy", "blotto", "nash_approx", "mixed",
    "copycat", "quant_meta", "anti_speed", "bimodal",
    "round_hunter", "high_conv", "contrarian",
]
PRIMARY_METAS = ["nash_approx", "mixed"]


# =============================================================================
# VECTORISED INTEGER RANK + CROWDING PENALTY
# =============================================================================
#
# Change 2 & 4: Because speeds are integers, ties are exact and common.
# The multiplier formula uses method='min' ties — all tied players share
# the BEST rank in their group.
#
# Crowding penalty: if the density at my_s is above a threshold, apply
# a penalty to the effective multiplier. This captures the real effect
# that landing in a 2000-player cluster means you share that rank level
# with everyone and small errors put you at the bottom of the cluster.

def speed_multipliers_batch(my_s: int, opp_matrix: np.ndarray) -> np.ndarray:
    """
    opp_matrix: (iters, N_OPP) integer array.
    my_s: integer.

    Rank = 1 + count(opp > my_s)  [method='min', ties share best rank]

    Crowding penalty: if count(opp == my_s) / N_TOTAL > cluster_threshold,
    subtract CROWDING_PENALTY from the multiplier. This reflects that in a
    real-world discrete system a large cluster at your value means rank
    uncertainty — you might be near the bottom of that cluster.

    Returns (iters,) float32.
    """
    my_s = int(round(my_s))
    opp  = opp_matrix.astype(np.int32)

    faster  = np.sum(opp > my_s, axis=1, dtype=np.int32)   # strictly faster
    tied    = np.sum(opp == my_s, axis=1, dtype=np.int32)  # tied with you

    ranks = faster + 1  # best rank in the tied group (method='min')
    mults = 0.9 - ((ranks - 1) / (N_TOTAL - 1)) * 0.8

    # Change 4: crowding penalty proportional to cluster density
    cluster_density = (tied + 1) / N_TOTAL   # +1 for yourself
    penalty_threshold = 0.10                  # penalise if >10% share your speed
    penalty = CONFIG["CROWDING_PENALTY"] * np.maximum(
        0.0,
        (cluster_density - penalty_threshold) / (1.0 - penalty_threshold)
    )
    mults = mults - penalty

    return np.clip(mults, 0.1, 0.9).astype(np.float32)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def simulate(r: int, c: int, s: int,
             meta: str, iters: int, seed: int = None) -> dict:
    """
    Change 5: r, c, s are INTEGERS. Enforced at entry.
    R + C + S must equal 100. If not, raises.
    """
    r, c, s = int(round(r)), int(round(c)), int(round(s))
    if r + c + s != 100:
        raise ValueError(f"R+C+S={r+c+s} != 100")
    if any(x < 0 for x in [r, c, s]):
        raise ValueError("Negative allocation")

    rng = np.random.default_rng(seed)
    bs  = float(base_score(r, c))

    opp   = gen_opponent_matrix(meta, iters, my_s=s, rng=rng)  # int matrix
    mults = speed_multipliers_batch(s, opp)                     # float32 vec
    pnls  = bs * mults - BUDGET

    neg  = pnls[pnls < 0]
    dstd = float(np.std(neg)) if len(neg) > 1 else 1e-9
    mu   = float(np.mean(pnls))
    sig  = float(np.std(pnls))
    p5   = float(np.percentile(pnls, 5))

    return {
        "r": r, "c": c, "s": s, "meta": meta,
        "base_score":    bs,
        "avg_pnl":       mu,
        "median_pnl":    float(np.median(pnls)),
        "std_pnl":       sig,
        "min_pnl":       float(np.min(pnls)),
        "max_pnl":       float(np.max(pnls)),
        "pnl_5th":       p5,
        "pnl_25th":      float(np.percentile(pnls, 25)),
        "pnl_75th":      float(np.percentile(pnls, 75)),
        "pnl_95th":      float(np.percentile(pnls, 95)),
        "sharpe":        mu / (sig + 1e-9),
        "sortino":       mu / (dstd + 1e-9),
        "cvar_5":        float(np.mean(pnls[pnls <= p5])) if len(pnls[pnls <= p5]) else p5,
        "win_rate":      float(np.mean(pnls > 0)),
        "top_half_rate": float(np.mean(mults >= 0.5)),
        "avg_mult":      float(np.mean(mults)),
        "mult_std":      float(np.std(mults)),
        "pnls":  pnls,
        "mults": mults,
        "iters": iters,
    }


# =============================================================================
# CLUSTER DENSITY ANALYSER  (diagnostic — shows rank cliffs)
# =============================================================================

def analyse_cluster_density(meta: str, iters: int = 200,
                             seed: int = 0) -> pd.DataFrame:
    """
    For a given meta, compute the expected density of opponents at each
    integer speed 0..100. Shows where the dangerous clusters are.
    """
    rng = np.random.default_rng(seed)
    mat = gen_opponent_matrix(meta, iters, my_s=50, rng=rng)  # (iters, N_OPP)
    counts = np.zeros(101, dtype=np.float64)
    for val in range(101):
        counts[val] = np.mean(np.sum(mat == val, axis=1))
    density = counts / N_OPP
    return pd.DataFrame({"speed": np.arange(101), "density": density,
                          "avg_count": counts})


# =============================================================================
# PARALLEL SWEEP
# =============================================================================

def _sweep_one(args):
    meta, s, iters, seed = args
    s = int(s)
    r_opt, c_opt, _ = _RC_TABLE[s]
    # Change 3: R/C/S are jointly determined — C = 100 - S - R, no independence
    # _RC_TABLE already enforces R + C = 100 - S, so R+C+S = 100 exactly
    res = simulate(r_opt, c_opt, s, meta, iters, seed=seed)
    return {
        "meta": meta, "speed": s, "research": r_opt, "scale": c_opt,
        **{k: res[k] for k in ["avg_pnl","median_pnl","std_pnl","sharpe",
                                "sortino","win_rate","top_half_rate","avg_mult",
                                "base_score","pnl_5th","pnl_95th","cvar_5"]}
    }

def run_sweep_all(metas: list, step: int = 1, iters: int = None,
                  n_workers: int = None) -> dict:
    if iters is None:     iters = CONFIG["SWEEP_ITERS"]
    if n_workers is None: n_workers = CONFIG["N_WORKERS"]
    speeds = list(range(0, 101, step))
    tasks  = [(m, s, iters, hash((m, s)) % (2**31))
               for m in metas for s in speeds]
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_sweep_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks),
                        desc="Parallel sweep", leave=False):
            rows.append(fut.result())
    all_df = pd.DataFrame(rows)
    return {m: all_df[all_df["meta"]==m].sort_values("speed").reset_index(drop=True)
            for m in metas}


# =============================================================================
# REGRET ANALYSIS
# =============================================================================

def regret_analysis(sweep_dfs: dict) -> pd.DataFrame:
    records = [{"meta": m, "speed": r["speed"], "avg_pnl": r["avg_pnl"]}
               for m, df in sweep_dfs.items() for _, r in df.iterrows()]
    pivot  = pd.DataFrame(records).pivot(index="speed", columns="meta", values="avg_pnl")
    regret = pivot.max(axis=0) - pivot
    out = pd.DataFrame({"speed": pivot.index,
                         "max_regret": regret.max(axis=1).values,
                         "avg_regret": regret.mean(axis=1).values})
    for m in regret.columns:
        out[f"regret_{m}"] = regret[m].values
    return out.reset_index(drop=True)


# =============================================================================
# WEIGHTED ENSEMBLE SCORE
# =============================================================================

META_WEIGHTS = {
    "nash_approx":  0.28,
    "mixed":        0.22,
    "round_hunter": 0.12,   # high weight — very realistic
    "high_conv":    0.10,   # sophisticated convergence players
    "bimodal":      0.08,
    "blotto":       0.06,
    "quant_meta":   0.05,
    "naive":        0.04,
    "speed_heavy":  0.02,
    "copycat":      0.01,
    "anti_speed":   0.01,
    "contrarian":   0.01,
}

def ensemble_score(sweep_dfs: dict, weights: dict = None) -> pd.DataFrame:
    if weights is None: weights = META_WEIGHTS
    avail = {m: w for m, w in weights.items() if m in sweep_dfs}
    tw    = sum(avail.values())
    avail = {m: w / tw for m, w in avail.items()}
    speeds = sorted(next(iter(sweep_dfs.values()))["speed"].unique())
    rows = []
    for s in speeds:
        wp = ws = 0.0
        for m, w in avail.items():
            row = sweep_dfs[m][sweep_dfs[m]["speed"] == s]
            if not row.empty:
                wp += w * float(row["avg_pnl"].values[0])
                ws += w * float(row["sharpe"].values[0])
        rows.append({"speed": s, "weighted_pnl": wp, "weighted_sharpe": ws})
    return pd.DataFrame(rows)


# =============================================================================
# BAYESIAN OPTIMISATION — integer-only candidates
# =============================================================================

def bayesian_optimise(meta: str, n_calls: int = None,
                      iters_per_eval: int = None) -> dict:
    """
    Change 5: All (R, S) candidates are rounded to integers before
    evaluating the objective. C = 100 - R - S (integer arithmetic).
    The GP operates on the continuous space but evaluates at integer points,
    giving it a realistic discrete landscape.
    """
    if n_calls is None:        n_calls = CONFIG["BAYES_CALLS"]
    if iters_per_eval is None: iters_per_eval = CONFIG["BAYES_ITERS"]

    rng = np.random.default_rng(42)

    def objective_int(r_float, s_float):
        # Change 5: ROUND to integers before evaluating
        r = int(round(r_float))
        s = int(round(s_float))
        c = 100 - r - s
        if c < 0 or r < 0 or s < 0 or r + c + s != 100:
            return -1e9
        try:
            return simulate(r, c, s, meta, iters_per_eval)["avg_pnl"]
        except ValueError:
            return -1e9

    # Initial samples: uniform on integer simplex
    n_init = max(12, n_calls // 5)
    X_obs, y_obs = [], []

    # Generate valid integer pairs
    init_candidates = []
    attempts = 0
    while len(init_candidates) < n_init * 3 and attempts < 10000:
        s_i = int(rng.integers(0, 101))
        r_i = int(rng.integers(0, 101 - s_i))
        init_candidates.append((r_i, s_i))
        attempts += 1
    init_candidates = init_candidates[:n_init]

    for r_i, s_i in tqdm(init_candidates, desc=f"  BO init [{meta}]", leave=False):
        X_obs.append([float(r_i), float(s_i)])
        y_obs.append(objective_int(r_i, s_i))

    kernel = Matern(nu=2.5, length_scale_bounds=(0.5, 50.0))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                  normalize_y=True, alpha=1e-4)

    for _ in tqdm(range(n_calls - n_init), desc=f"  BO steps [{meta}]", leave=False):
        gp.fit(np.array(X_obs), np.array(y_obs))
        best_y = float(np.max(y_obs))

        # Generate integer candidates on simplex
        cands_raw  = rng.uniform(0, 1, (12000, 2)) * 100
        valid_mask = cands_raw[:, 0] + cands_raw[:, 1] <= 100
        cands_cont = cands_raw[valid_mask]
        # Round to integers — Change 5
        cands_int  = np.round(cands_cont).astype(int)
        # Re-validate after rounding
        valid2 = (cands_int[:, 0] + cands_int[:, 1] <= 100) & \
                 (cands_int[:, 0] >= 0) & (cands_int[:, 1] >= 0)
        cands_int = cands_int[valid2]
        if len(cands_int) == 0:
            cands_int = np.array([[20, 40]])

        # Score on continuous GP (GP doesn't care that we'll round)
        mu, sigma = gp.predict(cands_int.astype(float), return_std=True)
        z  = (mu - best_y) / (sigma + 1e-9)
        ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
        best_cand = cands_int[int(np.argmax(ei))]

        X_obs.append(best_cand.astype(float).tolist())
        y_obs.append(objective_int(best_cand[0], best_cand[1]))

    bi   = int(np.argmax(y_obs))
    br_f, bs_f = X_obs[bi]
    br   = int(round(br_f))
    bs_i = int(round(bs_f))
    bc   = 100 - br - bs_i

    return {"meta": meta, "r": br, "c": bc, "s": bs_i,
            "expected_pnl": y_obs[bi], "X_obs": X_obs, "y_obs": y_obs}


# =============================================================================
# MULTI-TRIAL AUTO-RUNNER
# =============================================================================

def run_multi_trials(strategy: dict, metas: list = None) -> pd.DataFrame:
    if metas is None: metas = PRIMARY_METAS
    r = int(strategy["r"]); c = int(strategy["c"]); s = int(strategy["s"])
    rows = []
    for ti, n_iters in enumerate(CONFIG["TRIAL_ITERS_LIST"]):
        for meta in metas:
            res = simulate(r, c, s, meta, n_iters, seed=ti)
            rows.append({"trial": ti+1, "iters": n_iters, "meta": meta,
                         **{k: res[k] for k in
                            ["avg_pnl","std_pnl","sharpe","sortino",
                             "win_rate","avg_mult","cvar_5"]}})
        if CONFIG["VERBOSE"]:
            print(f"  Trial {ti+1}/{len(CONFIG['TRIAL_ITERS_LIST'])} ({n_iters:,} iters)")
    return pd.DataFrame(rows)


# =============================================================================
# SENSITIVITY
# =============================================================================

def sensitivity_analysis(best_s: int, meta: str = "nash_approx",
                          radius: int = 8, iters: int = 800) -> pd.DataFrame:
    rows = []
    for delta in range(-radius, radius + 1):
        s = int(np.clip(best_s + delta, 0, 100))
        r_opt, c_opt, _ = _RC_TABLE[s]
        res = simulate(r_opt, c_opt, s, meta, iters, seed=delta + 99)
        rows.append({"delta": delta, "speed": s, "research": r_opt, "scale": c_opt,
                     **{k: res[k] for k in
                        ["avg_pnl","std_pnl","sharpe","sortino","cvar_5"]}})
    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

FMT_K = plt.FuncFormatter(lambda x, _: f"{x:,.0f}")

def _vline(ax, x, color="red", lw=1.5, ls="--", label=None):
    ax.axvline(x=x, color=color, linewidth=lw, linestyle=ls, label=label)

def plot_cluster_density(sweep_dfs, outdir):
    """New plot: shows discrete speed bunching per meta."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    for ax, meta in zip(axes, ALL_METAS):
        df = analyse_cluster_density(meta, iters=100)
        ax.bar(df["speed"], df["density"] * 100, width=0.9,
               color="#378ADD", alpha=0.75)
        ax.set_title(f"{meta}", fontsize=10)
        ax.set_xlabel("Speed"); ax.set_ylabel("% of field")
        ax.set_xlim(0, 100)
    for ax in axes[len(ALL_METAS):]:
        ax.set_visible(False)
    fig.suptitle("Opponent speed distribution per meta (discrete integer bunching)",
                 fontsize=13)
    path = os.path.join(outdir, "cluster_density.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")

def plot_rank_cliff(outdir):
    """
    Shows rank cliff effect: at a clustered speed value S, adding 1
    player to the cluster vs jumping above it has very different rank impact.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: effective multiplier when joining vs beating a cluster of size k
    cluster_sizes = np.arange(0, 2001, 50)
    for frac_label, frac in [("10% cluster", 0.10), ("30% cluster", 0.30),
                              ("50% cluster", 0.50)]:
        k = (frac * N_OPP).astype(int) if hasattr(frac, "astype") else int(frac * N_OPP)
        # If you JOIN the cluster: rank = 1 + (opp faster) ~ N_OPP * (1 - frac) + 1
        faster_join = int(N_OPP * (1 - frac))
        mult_join   = 0.9 - ((faster_join) / (N_TOTAL - 1)) * 0.8
        # After crowding penalty
        density = (k + 1) / N_TOTAL
        penalty_thresh = 0.10
        penalty = CONFIG["CROWDING_PENALTY"] * max(
            0, (density - penalty_thresh) / (1 - penalty_thresh))
        mult_join_penalised = max(0.1, mult_join - penalty)

        # If you BEAT the cluster by 1: rank = 1
        mult_beat = 0.9

        axes[0].bar([frac_label],
                    [mult_join_penalised],
                    label=f"Join (penalised)", color="#E24B4A", alpha=0.7)
    axes[0].axhline(0.9, color="green", lw=2, ls="--", label="Beat cluster (rank 1)")
    axes[0].axhline(0.5, color="orange", lw=1.5, ls=":", label="Neutral mult (0.5)")
    axes[0].set_title("Multiplier: join cluster vs beat by 1")
    axes[0].set_ylabel("Speed multiplier"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Right: rank as function of speed, with realistic cluster at 40 and 50
    speeds_x = np.arange(0, 101)
    # Simulate field: 30% at 33, 30% at 50, 20% at 60, 20% random
    field = np.concatenate([
        np.full(int(N_OPP * 0.30), 33),
        np.full(int(N_OPP * 0.30), 50),
        np.full(int(N_OPP * 0.20), 60),
        np.random.randint(0, 101, N_OPP - int(N_OPP * 0.80))
    ])
    ranks_x = np.array([1 + int(np.sum(field > s)) for s in speeds_x])
    mults_x = 0.9 - ((ranks_x - 1) / (N_TOTAL - 1)) * 0.8
    axes[1].plot(speeds_x, mults_x, color="#378ADD", lw=2, label="Effective multiplier")
    axes[1].axvline(33, color="red", ls="--", lw=1, alpha=0.6, label="Cluster @33")
    axes[1].axvline(50, color="orange", ls="--", lw=1, alpha=0.6, label="Cluster @50")
    axes[1].axvline(60, color="purple", ls="--", lw=1, alpha=0.6, label="Cluster @60")
    axes[1].set_title("Rank cliff: multiplier vs your speed (realistic field)")
    axes[1].set_xlabel("Your Speed %"); axes[1].set_ylabel("Multiplier")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle("Discrete bunching & rank cliff effects", fontsize=13)
    path = os.path.join(outdir, "rank_cliff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")

def plot_sweeps(sweep_dfs, regret_df, ensemble_df, best_s, outdir):
    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)
    colors = plt.cm.tab20(np.linspace(0, 0.9, len(sweep_dfs)))

    ax1 = fig.add_subplot(gs[0, :2])
    for (meta, df), col in zip(sweep_dfs.items(), colors):
        ax1.plot(df["speed"], df["avg_pnl"], label=meta, color=col, lw=1.6)
    ax1.plot(ensemble_df["speed"], ensemble_df["weighted_pnl"],
             color="black", lw=2.5, label="Weighted ensemble")
    _vline(ax1, best_s, color="red", lw=2, ls="-", label=f"Rec. S={best_s}")
    ax1.set_title("Expected PnL by Speed (integer, R/C auto-optimised, crowding penalty on)")
    ax1.set_xlabel("Speed"); ax1.set_ylabel("Avg PnL")
    ax1.yaxis.set_major_formatter(FMT_K); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    for (meta, df), col in zip(sweep_dfs.items(), colors):
        ax2.plot(df["speed"], df["sharpe"], color=col, lw=1.4, label=meta)
    ax2.plot(ensemble_df["speed"], ensemble_df["weighted_sharpe"],
             color="black", lw=2.0, label="Ensemble")
    _vline(ax2, best_s)
    ax2.set_title("Sharpe"); ax2.set_xlabel("Speed"); ax2.legend(fontsize=6); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :2])
    for (meta, df), col in zip(sweep_dfs.items(), colors):
        if "cvar_5" in df.columns:
            ax3.plot(df["speed"], df["cvar_5"], color=col, lw=1.4, label=meta)
    _vline(ax3, best_s)
    ax3.set_title("CVaR-5% (tail risk)")
    ax3.set_xlabel("Speed"); ax3.set_ylabel("CVaR-5")
    ax3.yaxis.set_major_formatter(FMT_K); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(regret_df["speed"], regret_df["max_regret"], color="crimson", lw=2, label="Max")
    ax4.plot(regret_df["speed"], regret_df["avg_regret"], color="orange", lw=1.5, ls="--", label="Avg")
    mmr = int(regret_df.loc[regret_df["max_regret"].idxmin(), "speed"])
    _vline(ax4, mmr, color="green", lw=2, ls="-", label=f"MiniMax={mmr}")
    _vline(ax4, best_s)
    ax4.set_title("MiniMax Regret"); ax4.set_xlabel("Speed")
    ax4.yaxis.set_major_formatter(FMT_K); ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    for (meta, df), col in zip(sweep_dfs.items(), colors):
        ax5.plot(df["speed"], df["top_half_rate"]*100, color=col, lw=1.4, label=meta)
    _vline(ax5, best_s)
    ax5.set_title("Top-half rate"); ax5.set_xlabel("Speed"); ax5.legend(fontsize=6); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    sv = np.arange(101)
    rv = np.array([_RC_TABLE[s][0] for s in sv])
    cv = np.array([_RC_TABLE[s][1] for s in sv])
    ax6.step(sv, rv, label="Research", color="#378ADD", lw=2)  # step — integers
    ax6.step(sv, cv, label="Scale",    color="#1D9E75", lw=2)
    _vline(ax6, best_s)
    ax6.set_title("Optimal integer R/C vs Speed"); ax6.set_xlabel("Speed")
    ax6.legend(); ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.step(sv, base_score(rv, cv), color="#D85A30", lw=2)
    _vline(ax7, best_s)
    ax7.set_title("Base score vs Speed"); ax7.set_xlabel("Speed")
    ax7.yaxis.set_major_formatter(FMT_K); ax7.grid(alpha=0.3)

    ax8 = fig.add_subplot(gs[3, :2])
    r_b, c_b, _ = _RC_TABLE[best_s]
    for meta, col in zip(PRIMARY_METAS, ["#378ADD", "#1D9E75"]):
        res = simulate(r_b, c_b, best_s, meta, 2000, seed=7)
        kde = gaussian_kde(res["pnls"])
        xr  = np.linspace(res["pnls"].min(), res["pnls"].max(), 400)
        ax8.plot(xr, kde(xr), label=meta, color=col, lw=2)
        ax8.fill_between(xr, kde(xr), alpha=0.12, color=col)
    ax8.axvline(x=0, color="red", ls="--", lw=1.5, label="Break-even")
    ax8.set_title(f"PnL distribution at S={best_s} (primary metas, integer)")
    ax8.set_xlabel("PnL (XIRECs)"); ax8.set_ylabel("Density")
    ax8.xaxis.set_major_formatter(FMT_K); ax8.legend(); ax8.grid(alpha=0.3)

    ax9 = fig.add_subplot(gs[3, 2])
    for (meta, df), col in zip(sweep_dfs.items(), colors):
        if "sortino" in df.columns:
            ax9.plot(df["speed"], df["sortino"], color=col, lw=1.4, label=meta)
    _vline(ax9, best_s)
    ax9.set_title("Sortino"); ax9.set_xlabel("Speed"); ax9.legend(fontsize=6); ax9.grid(alpha=0.3)

    fig.suptitle("Trading Outpost v3 — Realistic Discrete Integer Analysis",
                 fontsize=15, fontweight="bold")
    path = os.path.join(outdir, "full_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")

def plot_convergence(trial_df, outdir):
    metrics = ["avg_pnl", "sharpe", "sortino", "cvar_5"]
    titles  = ["Avg PnL", "Sharpe", "Sortino", "CVaR-5%"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        for meta in trial_df["meta"].unique():
            sub = trial_df[trial_df["meta"]==meta]
            ax.plot(sub["iters"], sub[metric], label=meta, marker="o", ms=4)
        ax.set_title(title); ax.set_xlabel("Iterations"); ax.set_xscale("log")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if metric in ("avg_pnl","cvar_5"): ax.yaxis.set_major_formatter(FMT_K)
    fig.suptitle("Multi-trial convergence", fontsize=13)
    path = os.path.join(outdir, "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")

def plot_sensitivity(sens_df, best_s, outdir):
    metrics = ["avg_pnl", "sharpe", "sortino", "cvar_5"]
    titles  = [f"PnL (S={best_s})", "Sharpe", "Sortino", "CVaR-5%"]
    colors_ = ["#378ADD", "#1D9E75", "#D85A30", "#A32D2D"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, metric, title, col in zip(axes, metrics, titles, colors_):
        ax.bar(sens_df["delta"], sens_df[metric], color=col, alpha=0.8)
        ax.axvline(x=0, color="red", lw=2, ls="--")
        ax.set_title(title); ax.set_xlabel("Speed δ"); ax.grid(alpha=0.3)
        if metric in ("avg_pnl","cvar_5"): ax.yaxis.set_major_formatter(FMT_K)
    fig.suptitle("Sensitivity / robustness (±8 around recommendation)", fontsize=13)
    path = os.path.join(outdir, "sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")

def plot_bo_landscape(bo_results, outdir):
    n = len(bo_results)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1: axes = [axes]
    for ax, (meta, bo) in zip(axes, bo_results.items()):
        X = np.array(bo["X_obs"]); y = np.array(bo["y_obs"])
        sc = ax.scatter(X[:,0], X[:,1], c=y, cmap="RdYlGn", s=25, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Avg PnL")
        ax.scatter([bo["r"]], [bo["s"]], color="red", s=150, zorder=6,
                   marker="*", label=f"Best R={bo['r']} S={bo['s']}")
        ax.set_title(f"GP-BO [{meta}] — integer grid")
        ax.set_xlabel("Research"); ax.set_ylabel("Speed"); ax.legend(fontsize=8)
    fig.suptitle("Bayesian Optimisation Landscape (integer-rounded candidates)", fontsize=13)
    path = os.path.join(outdir, "bo_landscape.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")

def plot_heatmap(sweep_dfs, outdir):
    metas  = list(sweep_dfs.keys())
    speeds = sorted(sweep_dfs[metas[0]]["speed"].unique())
    mat = np.array([
        [float(sweep_dfs[m][sweep_dfs[m]["speed"]==s]["avg_pnl"].values[0])
         for s in speeds] for m in metas
    ])
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   extent=[speeds[0], speeds[-1], -0.5, len(metas)-0.5])
    ax.set_yticks(range(len(metas))); ax.set_yticklabels(metas, fontsize=9)
    ax.set_xlabel("Speed"); ax.set_title("PnL heatmap: meta × speed (integer, crowding-penalised)")
    plt.colorbar(im, ax=ax, label="Avg PnL")
    path = os.path.join(outdir, "heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  → Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("\n" + "="*70)
    print("  TRADING OUTPOST OPTIMIZER  v3 — Realistic Discrete Integer Model")
    print("="*70)
    print(f"  Opponents/sim  : {N_OPP:,}   |  Workers : {CONFIG['N_WORKERS']}")
    print(f"  MC iterations  : {CONFIG['MC_ITERATIONS']:,}   |  Sweep   : {CONFIG['SWEEP_ITERS']:,}/step")
    print(f"  Crowding pen.  : {CONFIG['CROWDING_PENALTY']}   |  Snipe % : {CONFIG['SNIPE_FRACTION']*100:.0f}%")
    print(f"  Integer-only   : YES — all speeds are ints, BO rounds before eval")
    print(f"  Metas          : {len(ALL_METAS)}   |  Output  : {CONFIG['OUTPUT_DIR']}/")
    print("="*70)

    # STEP 1 — sweep
    print("\n► STEP 1: Parallel integer speed sweep (all metas)")
    sweep_dfs = run_sweep_all(ALL_METAS, step=CONFIG["SPEED_STEP"],
                               iters=CONFIG["SWEEP_ITERS"])
    for meta, df in sweep_dfs.items():
        br = df.loc[df["avg_pnl"].idxmax()]
        print(f"  [{meta:<14}] S={int(br['speed']):>3}  R={int(br['research']):>3}  "
              f"C={int(br['scale']):>3}  PnL={br['avg_pnl']:>12,.0f}  "
              f"Sharpe={br['sharpe']:.3f}  CVaR5={br['cvar_5']:>12,.0f}")

    # STEP 2 — ensemble + primary
    print("\n► STEP 2: Weighted ensemble scoring")
    ensemble_df = ensemble_score(sweep_dfs)
    best_ens_s  = int(ensemble_df.loc[ensemble_df["weighted_pnl"].idxmax(), "speed"])
    print(f"  Ensemble optimal Speed: {best_ens_s}")
    primary_speeds = []
    for meta in PRIMARY_METAS:
        bs_ = int(sweep_dfs[meta].loc[sweep_dfs[meta]["avg_pnl"].idxmax(), "speed"])
        primary_speeds.append(bs_)
        print(f"  [{meta}] Best S = {bs_}")
    sweep_rec_s = int(round(np.mean(primary_speeds)))
    print(f"  Primary-meta avg: S={sweep_rec_s}")

    # STEP 3 — regret
    print("\n► STEP 3: MiniMax Regret analysis")
    regret_df = regret_analysis(sweep_dfs)
    minimax_s = int(regret_df.loc[regret_df["max_regret"].idxmin(), "speed"])
    print(f"  MiniMax Regret optimal Speed: {minimax_s}")

    # STEP 4 — GP-BO (integer)
    print("\n► STEP 4: Bayesian Optimisation (integer candidates)")
    bo_results = {}
    for meta in PRIMARY_METAS:
        bo = bayesian_optimise(meta)
        bo_results[meta] = bo
        print(f"  [{meta}] → R={bo['r']}  C={bo['c']}  S={bo['s']}  "
              f"PnL={bo['expected_pnl']:,.0f}  (sum={bo['r']+bo['c']+bo['s']})")
    bo_rec_s = int(round(np.mean([bo["s"] for bo in bo_results.values()])))
    print(f"  GP-BO avg recommendation: S={bo_rec_s}")

    # STEP 5 — convergence
    print("\n► STEP 5: Multi-trial convergence")
    r_rec, c_rec, _ = _RC_TABLE[sweep_rec_s]
    trial_df = run_multi_trials({"r": r_rec, "c": c_rec, "s": sweep_rec_s})
    trial_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "trial_convergence.csv"), index=False)

    # STEP 6 — sensitivity
    print("\n► STEP 6: Sensitivity analysis (±8)")
    sens_df = sensitivity_analysis(sweep_rec_s, meta="nash_approx", radius=8, iters=800)
    sens_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "sensitivity.csv"), index=False)

    # STEP 7 — full MC
    print(f"\n► STEP 7: Full MC validation (R={r_rec}, C={c_rec}, S={sweep_rec_s})")
    final_results = {}
    for meta in ALL_METAS:
        res = simulate(r_rec, c_rec, sweep_rec_s, meta,
                       CONFIG["MC_ITERATIONS"], seed=0)
        final_results[meta] = res
        print(f"  [{meta:<14}]  PnL={res['avg_pnl']:>12,.0f}  "
              f"Sharpe={res['sharpe']:>6.3f}  Sortino={res['sortino']:>6.3f}  "
              f"CVaR5={res['cvar_5']:>12,.0f}  Win={res['win_rate']*100:>5.1f}%  "
              f"AvgMult={res['avg_mult']:.3f}")

    # STEP 8 — save
    print("\n► STEP 8: Saving data")
    for meta, df in sweep_dfs.items():
        df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], f"sweep_{meta}.csv"), index=False)
    regret_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "regret.csv"), index=False)
    ensemble_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "ensemble.csv"), index=False)
    summary = [{k: res[k] for k in ["meta","avg_pnl","median_pnl","std_pnl","sharpe",
                                      "sortino","cvar_5","win_rate","avg_mult",
                                      "pnl_5th","pnl_95th"] if k in res}
               for res in final_results.values()]
    pd.DataFrame(summary).to_csv(
        os.path.join(CONFIG["OUTPUT_DIR"], "final_summary.csv"), index=False)

    # STEP 9 — plots
    print("\n► STEP 9: Generating plots")
    plot_sweeps(sweep_dfs, regret_df, ensemble_df, sweep_rec_s, CONFIG["OUTPUT_DIR"])
    plot_convergence(trial_df, CONFIG["OUTPUT_DIR"])
    plot_sensitivity(sens_df, sweep_rec_s, CONFIG["OUTPUT_DIR"])
    plot_bo_landscape(bo_results, CONFIG["OUTPUT_DIR"])
    plot_heatmap(sweep_dfs, CONFIG["OUTPUT_DIR"])
    plot_cluster_density(sweep_dfs, CONFIG["OUTPUT_DIR"])
    plot_rank_cliff(CONFIG["OUTPUT_DIR"])

    # FINAL
    elapsed = time.time() - t0
    signals   = sorted([sweep_rec_s, bo_rec_s, best_ens_s])
    final_s   = int(np.median(signals))
    final_r, final_c, _ = _RC_TABLE[final_s]
    assert final_r + final_c + final_s == 100, "Sum != 100"

    print("\n" + "="*70)
    print("  FINAL REPORT  (integer-only, crowding-penalised)")
    print("="*70)
    print(f"  Sweep (primary metas avg)  : S={sweep_rec_s}")
    print(f"  GP-BO (primary metas avg)  : S={bo_rec_s}")
    print(f"  Ensemble weighted score    : S={best_ens_s}")
    print(f"  MiniMax Regret             : S={minimax_s}")
    print(f"  FINAL (median of signals)  : S={final_s}")
    print()
    print(f"  ╔══════════════════════════════════════╗")
    print(f"  ║  SUBMIT:                             ║")
    print(f"  ║  Research = {final_r:>3}                     ║")
    print(f"  ║  Scale    = {final_c:>3}                     ║")
    print(f"  ║  Speed    = {final_s:>3}                     ║")
    print(f"  ║  Sum      = {final_r+final_c+final_s:>3}  ✓                ║")
    print(f"  ╚══════════════════════════════════════╝")
    print()
    for meta in PRIMARY_METAS:
        res = final_results[meta]
        print(f"  [{meta}]  PnL={res['avg_pnl']:,.0f}  "
              f"Sharpe={res['sharpe']:.3f}  CVaR5={res['cvar_5']:,.0f}  "
              f"Win={res['win_rate']*100:.1f}%")
    print(f"\n  Total runtime : {elapsed:.1f}s")
    print(f"  Results saved : {CONFIG['OUTPUT_DIR']}/")
    print("="*70 + "\n")

    rec = {
        "submission":  {"research": int(final_r), "scale": int(final_c), "speed": int(final_s)},
        "sweep_rec_s": int(sweep_rec_s), "bo_rec_s": int(bo_rec_s),
        "ensemble_s":  int(best_ens_s),  "minimax_s": int(minimax_s),
        "final_s":     int(final_s),     "signals":   [int(x) for x in signals],
        "crowding_penalty": CONFIG["CROWDING_PENALTY"],
        "snipe_fraction":   CONFIG["SNIPE_FRACTION"],
    }
    with open(os.path.join(CONFIG["OUTPUT_DIR"], "recommendation.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  Saved: {CONFIG['OUTPUT_DIR']}/recommendation.json")


if __name__ == "__main__":
    main()

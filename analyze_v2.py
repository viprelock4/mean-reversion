#!/usr/bin/env python3
"""
Enhanced analysis: test whether a SINGLE universal algorithm works across all TFs.
Key questions:
1. Does the same config (e.g., SMA(50) + ATR(N) + EMA(M)) work on ALL timeframes?
2. What's the best UNIVERSAL config (maximizing average R)?
3. How does the original's amplitude pattern match fixed vs scaled periods?
"""

import csv
import math
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def read_csv(filename):
    rows = []
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = int(row['time'])
                c = float(row['close'])
                p = float(row['Plot']) if row['Plot'] else None
                rows.append({'time': t, 'close': c, 'plot': p})
            except (ValueError, KeyError):
                continue
    return rows

def sma(values, period):
    result = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result

def wma(values, period):
    result = [None] * len(values)
    ws = period * (period + 1) / 2
    for i in range(period - 1, len(values)):
        total = 0.0
        for j in range(period):
            total += values[i - period + 1 + j] * (j + 1)
        result[i] = total / ws
    return result

def ema(values, period):
    result = [None] * len(values)
    alpha = 2.0 / (period + 1)
    started = False
    for i, v in enumerate(values):
        if v is not None and not started:
            started = True
            result[i] = v
        elif started and v is not None:
            result[i] = alpha * v + (1 - alpha) * (result[i-1] if result[i-1] is not None else v)
    return result

def rma(values, period):
    result = [None] * len(values)
    alpha = 1.0 / period
    started = False
    for i, v in enumerate(values):
        if v is not None and not started:
            started = True
            result[i] = v
        elif started and v is not None:
            result[i] = alpha * v + (1 - alpha) * (result[i-1] if result[i-1] is not None else v)
    return result

def stdev_roll(values, period):
    result = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        if None in window:
            continue
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / period
        result[i] = math.sqrt(var)
    return result

def true_range_from_close(closes):
    result = [None]
    for i in range(1, len(closes)):
        result.append(abs(closes[i] - closes[i-1]))
    return result

def linreg(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 10:
        return 0.0, 0.0, 0.0
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs) / n
    var = sum((a - mx) ** 2 for a, _ in pairs) / n
    sx = math.sqrt(var) if var > 0 else 0
    sy_var = sum((b - my) ** 2 for _, b in pairs) / n
    sy = math.sqrt(sy_var) if sy_var > 0 else 0
    if var < 1e-10:
        return 0.0, my, 0.0
    gain = cov / var
    bias = my - gain * mx
    r = cov / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0.0
    return gain, bias, r


def compute_oscillator(closes, base_type, base_period, norm_type, norm_period, smooth_type, smooth_period):
    """Compute the oscillator for a given config. Returns raw values (before gain/bias)."""
    n = len(closes)
    if base_period >= n - 20:
        return [None] * n

    if base_type == 'SMA':
        bl = sma(closes, base_period)
    elif base_type == 'WMA':
        bl = wma(closes, base_period)
    else:
        bl = ema(closes, base_period)

    deviation = [None] * n
    for i in range(n):
        if bl[i] is not None:
            deviation[i] = closes[i] - bl[i]

    if norm_type == 'none':
        normalized = deviation
    elif norm_type == 'atr_close':
        tr = true_range_from_close(closes)
        atr_vals = rma(tr, norm_period)
        normalized = [None] * n
        for i in range(n):
            if deviation[i] is not None and atr_vals[i] is not None and atr_vals[i] > 0.001:
                normalized[i] = deviation[i] / atr_vals[i]
    elif norm_type == 'stdev':
        sd = stdev_roll(closes, norm_period)
        normalized = [None] * n
        for i in range(n):
            if deviation[i] is not None and sd[i] is not None and sd[i] > 0.001:
                normalized[i] = deviation[i] / sd[i]
    elif norm_type == 'pct':
        normalized = [None] * n
        for i in range(n):
            if deviation[i] is not None and bl[i] is not None and abs(bl[i]) > 0.001:
                normalized[i] = deviation[i] / bl[i]
    else:
        normalized = deviation

    if smooth_type == 'ema' and smooth_period > 1:
        smoothed = ema(normalized, smooth_period)
    else:
        smoothed = normalized

    return smoothed


def test_config_on_all_tfs(config, datasets):
    """Test one config across all timeframes. Returns dict of {label: (gain, bias, r)}."""
    results = {}
    for label, data in datasets.items():
        closes = [r['close'] for r in data]
        plots = [r['plot'] for r in data]
        osc = compute_oscillator(
            closes,
            config['base_type'], config['base_period'],
            config['norm_type'], config['norm_period'],
            config['smooth_type'], config['smooth_period']
        )
        gain, bias, r = linreg(osc, plots)
        results[label] = {'gain': gain, 'bias': bias, 'r': r}
    return results


def main():
    # Load all datasets
    files = [
        ('tsla_daily.csv', 'Daily'),
        ('tsla_8h.csv',    '8H'),
        ('tsla_4h.csv',    '4H'),
        ('tsla_1h.csv',    '1H'),
        ('tsla_15m.csv',   '15M'),
    ]
    datasets = {}
    for filename, label in files:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            datasets[label] = read_csv(filename)

    print("=" * 80)
    print("  ENHANCED ANALYSIS: Universal Algorithm Search")
    print("  Testing whether ONE config works across ALL timeframes")
    print("=" * 80)

    # =========================================================================
    # PART 1: Original indicator amplitude analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 1: Original Indicator Output Characteristics")
    print("=" * 80)
    for label, data in datasets.items():
        plots = [r['plot'] for r in data if r['plot'] is not None]
        if plots:
            mean = sum(plots) / len(plots)
            sd = (sum((p - mean)**2 for p in plots) / len(plots)) ** 0.5
            print(f"  {label:>6}: range [{min(plots):>7.4f}, {max(plots):>7.4f}]  "
                  f"mean={mean:>7.4f}  stdev={sd:>6.4f}  bars={len(plots)}")

    # =========================================================================
    # PART 2: Test specific "universal" configs across all TFs
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 2: Universal Config Test — Same algorithm on every TF")
    print("=" * 80)

    # Candidate universal configs to test
    universal_configs = [
        # Our TradingView best (Daily R=0.9919)
        {'base_type': 'WMA', 'base_period': 50, 'norm_type': 'atr_close', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 9, 'label': 'WMA(50) ATR(50) EMA9 [TV best]'},
        # Python Daily best
        {'base_type': 'SMA', 'base_period': 50, 'norm_type': 'atr_close', 'norm_period': 20,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'SMA(50) ATR(20) EMA5 [Py best]'},
        # SMA variant
        {'base_type': 'SMA', 'base_period': 50, 'norm_type': 'atr_close', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 9, 'label': 'SMA(50) ATR(50) EMA9'},
        # SMA ATR(50) EMA5
        {'base_type': 'SMA', 'base_period': 50, 'norm_type': 'atr_close', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'SMA(50) ATR(50) EMA5'},
        # WMA ATR(20) variants
        {'base_type': 'WMA', 'base_period': 50, 'norm_type': 'atr_close', 'norm_period': 20,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'WMA(50) ATR(20) EMA5'},
        {'base_type': 'WMA', 'base_period': 50, 'norm_type': 'atr_close', 'norm_period': 20,
         'smooth_type': 'ema', 'smooth_period': 9, 'label': 'WMA(50) ATR(20) EMA9'},
        # Period 30 variants
        {'base_type': 'SMA', 'base_period': 30, 'norm_type': 'atr_close', 'norm_period': 20,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'SMA(30) ATR(20) EMA5'},
        # Raw (no norm) — tests if normalization even matters
        {'base_type': 'SMA', 'base_period': 50, 'norm_type': 'none', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'SMA(50) Raw EMA5 [no norm]'},
        {'base_type': 'WMA', 'base_period': 50, 'norm_type': 'none', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'WMA(50) Raw EMA5 [no norm]'},
        # StdDev norm
        {'base_type': 'SMA', 'base_period': 50, 'norm_type': 'stdev', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 5, 'label': 'SMA(50) StdDev(50) EMA5'},
        {'base_type': 'SMA', 'base_period': 50, 'norm_type': 'stdev', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 9, 'label': 'SMA(50) StdDev(50) EMA9'},
        {'base_type': 'WMA', 'base_period': 50, 'norm_type': 'stdev', 'norm_period': 50,
         'smooth_type': 'ema', 'smooth_period': 9, 'label': 'WMA(50) StdDev(50) EMA9'},
    ]

    print(f"\n  {'Config':<35} ", end="")
    for label in datasets:
        print(f"{'R_'+label:>10}", end="")
    print(f" {'Avg_R':>10} {'Min_R':>10} {'Gain_CV':>10}")
    print(f"  {'-'*35} ", end="")
    for _ in datasets:
        print(f" {'-'*9}", end="")
    print(f" {'-'*10} {'-'*10} {'-'*10}")

    best_avg_r = 0
    best_universal = None

    for cfg in universal_configs:
        results = test_config_on_all_tfs(cfg, datasets)
        rs = [results[l]['r'] for l in datasets]
        gains = [results[l]['gain'] for l in datasets]
        avg_r = sum(rs) / len(rs)
        min_r = min(rs)
        # Coefficient of variation of gains (lower = more consistent)
        mean_gain = sum(abs(g) for g in gains) / len(gains) if gains else 1
        gain_cv = (sum((abs(g) - mean_gain)**2 for g in gains) / len(gains))**0.5 / mean_gain if mean_gain > 0.001 else 99

        print(f"  {cfg['label']:<35} ", end="")
        for label in datasets:
            r = results[label]['r']
            print(f"  {r:>8.4f}", end="")
        print(f"  {avg_r:>8.4f}  {min_r:>8.4f}  {gain_cv:>8.4f}")

        if avg_r > best_avg_r:
            best_avg_r = avg_r
            best_universal = cfg

    # =========================================================================
    # PART 3: Fine-grained sweep around best candidates
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 3: Fine-Grained Sweep — Baseline period 30-70, ATR 10-60")
    print("  Testing SMA and WMA with ATR normalization + EMA(5) and EMA(9)")
    print("=" * 80)

    best_avg_r_fine = 0
    best_cfg_fine = None
    top_configs = []

    for base_type in ['SMA', 'WMA']:
        for base_period in range(30, 71, 5):
            for norm_period in range(10, 61, 5):
                for smooth_period in [3, 5, 7, 9]:
                    cfg = {
                        'base_type': base_type,
                        'base_period': base_period,
                        'norm_type': 'atr_close',
                        'norm_period': norm_period,
                        'smooth_type': 'ema',
                        'smooth_period': smooth_period,
                    }
                    results = test_config_on_all_tfs(cfg, datasets)
                    rs = [results[l]['r'] for l in datasets]
                    avg_r = sum(rs) / len(rs)
                    min_r = min(rs)
                    gains = [results[l]['gain'] for l in datasets]

                    top_configs.append((avg_r, min_r, base_type, base_period, norm_period, smooth_period, gains, rs))

                    if avg_r > best_avg_r_fine:
                        best_avg_r_fine = avg_r
                        best_cfg_fine = (base_type, base_period, norm_period, smooth_period)

    # Sort by avg R and show top 20
    top_configs.sort(key=lambda x: -x[0])
    print(f"\n  Top 20 configs by Average R across all TFs:")
    print(f"  {'Config':<35} ", end="")
    for label in datasets:
        print(f"{'R_'+label:>10}", end="")
    print(f" {'Avg_R':>10} {'Min_R':>10}")
    print(f"  {'-'*35} ", end="")
    for _ in datasets:
        print(f" {'-'*9}", end="")
    print(f" {'-'*10} {'-'*10}")

    for avg_r, min_r, bt, bp, np_, sp, gains, rs in top_configs[:20]:
        label = f"{bt}({bp}) ATR({np_}) EMA{sp}"
        print(f"  {label:<35} ", end="")
        for r in rs:
            print(f"  {r:>8.4f}", end="")
        print(f"  {avg_r:>8.4f}  {min_r:>8.4f}")

    # Also sort by MIN R (worst-case performance)
    top_configs.sort(key=lambda x: -x[1])
    print(f"\n  Top 20 configs by Minimum R (best worst-case):")
    print(f"  {'Config':<35} ", end="")
    for label in datasets:
        print(f"{'R_'+label:>10}", end="")
    print(f" {'Avg_R':>10} {'Min_R':>10}")
    print(f"  {'-'*35} ", end="")
    for _ in datasets:
        print(f" {'-'*9}", end="")
    print(f" {'-'*10} {'-'*10}")

    for avg_r, min_r, bt, bp, np_, sp, gains, rs in top_configs[:20]:
        label = f"{bt}({bp}) ATR({np_}) EMA{sp}"
        print(f"  {label:<35} ", end="")
        for r in rs:
            print(f"  {r:>8.4f}", end="")
        print(f"  {avg_r:>8.4f}  {min_r:>8.4f}")

    # =========================================================================
    # PART 4: Gain consistency analysis for the best config
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 4: Gain Analysis for Top Configs")
    print("  If gain is consistent across TFs, the algorithm is truly universal")
    print("=" * 80)

    for avg_r, min_r, bt, bp, np_, sp, gains, rs in top_configs[:5]:
        label = f"{bt}({bp}) ATR({np_}) EMA{sp}"
        print(f"\n  {label}:")
        for tf_label, g, r in zip(datasets.keys(), gains, rs):
            print(f"    {tf_label:>6}: Gain={g:>10.4f}  R={r:.4f}")
        mean_gain = sum(abs(g) for g in gains) / len(gains)
        gain_range = max(abs(g) for g in gains) - min(abs(g) for g in gains)
        print(f"    Range/Mean = {gain_range/mean_gain:.2f}  (lower = more consistent)")

    # =========================================================================
    # PART 5: StdDev normalization — may work better cross-TF
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 5: StdDev Normalization Sweep")
    print("  StdDev-based normalization may be more consistent than ATR across TFs")
    print("=" * 80)

    stdev_configs = []
    for base_type in ['SMA', 'WMA']:
        for base_period in [40, 45, 50, 55, 60]:
            for norm_period in [20, 30, 40, 50, 60]:
                for smooth_period in [3, 5, 7, 9, 12]:
                    cfg = {
                        'base_type': base_type,
                        'base_period': base_period,
                        'norm_type': 'stdev',
                        'norm_period': norm_period,
                        'smooth_type': 'ema',
                        'smooth_period': smooth_period,
                    }
                    results = test_config_on_all_tfs(cfg, datasets)
                    rs = [results[l]['r'] for l in datasets]
                    gains = [results[l]['gain'] for l in datasets]
                    avg_r = sum(rs) / len(rs)
                    min_r = min(rs)
                    stdev_configs.append((avg_r, min_r, base_type, base_period, norm_period, smooth_period, gains, rs))

    stdev_configs.sort(key=lambda x: -x[0])
    print(f"\n  Top 10 StdDev configs by Average R:")
    print(f"  {'Config':<35} ", end="")
    for label in datasets:
        print(f"{'R_'+label:>10}", end="")
    print(f" {'Avg_R':>10} {'Min_R':>10}")
    print(f"  {'-'*35} ", end="")
    for _ in datasets:
        print(f" {'-'*9}", end="")
    print(f" {'-'*10} {'-'*10}")

    for avg_r, min_r, bt, bp, np_, sp, gains, rs in stdev_configs[:10]:
        label = f"{bt}({bp}) SD({np_}) EMA{sp}"
        print(f"  {label:<35} ", end="")
        for r in rs:
            print(f"  {r:>8.4f}", end="")
        print(f"  {avg_r:>8.4f}  {min_r:>8.4f}")

    # Show gain consistency for top StdDev config
    avg_r, min_r, bt, bp, np_, sp, gains, rs = stdev_configs[0]
    label = f"{bt}({bp}) SD({np_}) EMA{sp}"
    print(f"\n  Gain analysis for best StdDev config: {label}")
    for tf_label, g, r in zip(datasets.keys(), gains, rs):
        print(f"    {tf_label:>6}: Gain={g:>10.4f}  R={r:.4f}")

    # =========================================================================
    # PART 6: Compare best ATR vs StdDev vs Raw across all TFs
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 6: Head-to-Head — Best ATR vs StdDev vs Raw")
    print("=" * 80)

    comparison_configs = []

    # Best ATR config (from Part 3)
    bt, bp, np_, sp = best_cfg_fine
    comparison_configs.append({
        'base_type': bt, 'base_period': bp, 'norm_type': 'atr_close', 'norm_period': np_,
        'smooth_type': 'ema', 'smooth_period': sp,
        'label': f'BEST ATR: {bt}({bp}) ATR({np_}) EMA{sp}'
    })

    # Best StdDev config (from Part 5)
    _, _, bt2, bp2, np2, sp2, _, _ = stdev_configs[0]
    comparison_configs.append({
        'base_type': bt2, 'base_period': bp2, 'norm_type': 'stdev', 'norm_period': np2,
        'smooth_type': 'ema', 'smooth_period': sp2,
        'label': f'BEST SD:  {bt2}({bp2}) SD({np2}) EMA{sp2}'
    })

    # Raw (no norm)
    for bt_raw in ['SMA', 'WMA']:
        for sp_raw in [5, 9]:
            comparison_configs.append({
                'base_type': bt_raw, 'base_period': 50, 'norm_type': 'none', 'norm_period': 50,
                'smooth_type': 'ema', 'smooth_period': sp_raw,
                'label': f'RAW:      {bt_raw}(50) Raw EMA{sp_raw}'
            })

    for cfg in comparison_configs:
        results = test_config_on_all_tfs(cfg, datasets)
        rs = [results[l]['r'] for l in datasets]
        gains = [results[l]['gain'] for l in datasets]
        avg_r = sum(rs) / len(rs)
        min_r = min(rs)

        print(f"\n  {cfg['label']}:")
        for tf_label in datasets:
            r = results[tf_label]['r']
            g = results[tf_label]['gain']
            b = results[tf_label]['bias']
            print(f"    {tf_label:>6}: R={r:.4f}  Gain={g:>10.4f}  Bias={b:>8.4f}")
        print(f"    Avg R={avg_r:.4f}  Min R={min_r:.4f}")

    # =========================================================================
    # PART 7: Residual pattern analysis for Daily
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART 7: Residual Analysis — What our best fit misses")
    print("=" * 80)

    daily_data = datasets.get('Daily', [])
    if daily_data:
        closes = [r['close'] for r in daily_data]
        plots = [r['plot'] for r in daily_data]
        n = len(closes)

        # Use the Python daily best: SMA(50) ATR(20) EMA5
        osc = compute_oscillator(closes, 'SMA', 50, 'atr_close', 20, 'ema', 5)
        gain, bias, r = linreg(osc, plots)

        residuals = []
        fitted_vals = []
        for i in range(n):
            if osc[i] is not None and plots[i] is not None:
                fitted = osc[i] * gain + bias
                residual = plots[i] - fitted
                residuals.append(residual)
                fitted_vals.append(fitted)

        if residuals:
            res_mean = sum(residuals) / len(residuals)
            res_std = (sum((r - res_mean)**2 for r in residuals) / len(residuals)) ** 0.5
            res_max = max(abs(r) for r in residuals)
            orig_std = (sum((p - sum(plots[i] for i in range(len(plots)) if plots[i] is not None)/sum(1 for p in plots if p is not None))**2
                        for p in plots if p is not None) / sum(1 for p in plots if p is not None)) ** 0.5

            print(f"  Daily: SMA(50) ATR(20) EMA5 → R={r:.4f}")
            print(f"  Residual: mean={res_mean:.4f}, stdev={res_std:.4f}, max={res_max:.4f}")
            print(f"  Original stdev={orig_std:.4f}")
            print(f"  Residual/Original ratio: {res_std/orig_std:.4f} (fraction unexplained)")
            print(f"  R² = {r**2:.4f} → {r**2*100:.1f}% variance explained")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n  Best Universal ATR Config: {best_cfg_fine}")
    print(f"  Best Universal ATR Avg R:  {best_avg_r_fine:.4f}")
    _, _, bt2, bp2, np2, sp2, _, _ = stdev_configs[0]
    print(f"  Best Universal SD Config:  ({bt2}, {bp2}, {np2}, {sp2})")
    print(f"  Best Universal SD Avg R:   {stdev_configs[0][0]:.4f}")

    # Check if the gains scale with some TF pattern
    print(f"\n  Note: Close-only ATR is a rough proxy. With proper OHLC data,")
    print(f"  ATR normalization would likely produce much more consistent gains")
    print(f"  across timeframes and higher R values on intraday TFs.")


if __name__ == '__main__':
    main()

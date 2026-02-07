#!/usr/bin/env python3
"""
Reverse-engineer IA-Mean-Reversion [3.6] from TradingView CSV exports.
Analyzes the relationship between price and the original indicator's plot values.
"""

import csv
import math
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def read_csv(filename):
    """Read TradingView CSV export. Returns list of dicts with time, close, plot."""
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
    """Simple Moving Average."""
    result = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result

def wma(values, period):
    """Weighted Moving Average."""
    result = [None] * len(values)
    weight_sum = period * (period + 1) / 2
    for i in range(period - 1, len(values)):
        total = 0.0
        for j in range(period):
            total += values[i - period + 1 + j] * (j + 1)
        result[i] = total / weight_sum
    return result

def ema(values, period):
    """Exponential Moving Average."""
    result = [None] * len(values)
    alpha = 2.0 / (period + 1)
    first_valid = None
    for i, v in enumerate(values):
        if v is not None and first_valid is None:
            first_valid = i
            result[i] = v
        elif first_valid is not None and v is not None:
            result[i] = alpha * v + (1 - alpha) * (result[i-1] if result[i-1] is not None else v)
    return result

def rma(values, period):
    """RMA (Wilder's smoothing / Running Moving Average)."""
    result = [None] * len(values)
    alpha = 1.0 / period
    first_valid = None
    for i, v in enumerate(values):
        if v is not None and first_valid is None:
            first_valid = i
            result[i] = v
        elif first_valid is not None and v is not None:
            result[i] = alpha * v + (1 - alpha) * (result[i-1] if result[i-1] is not None else v)
    return result

def stdev(values, period):
    """Rolling standard deviation."""
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
    """Approximate true range from close only: |close - close[1]|.
    This is a rough proxy when we don't have high/low."""
    result = [None]
    for i in range(1, len(closes)):
        result.append(abs(closes[i] - closes[i-1]))
    return result

def pearson_r(x, y):
    """Pearson correlation between two lists (skipping Nones)."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 10:
        return 0.0
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs) / n)
    sy = math.sqrt(sum((b - my) ** 2 for _, b in pairs) / n)
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return cov / (sx * sy)

def linreg(x, y):
    """Linear regression y = gain * x + bias. Returns (gain, bias, r)."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 10:
        return 0.0, 0.0, 0.0
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs) / n
    var = sum((a - mx) ** 2 for a, _ in pairs) / n
    if var < 1e-10:
        return 0.0, my, 0.0
    gain = cov / var
    bias = my - gain * mx
    r = pearson_r(x, y)
    return gain, bias, r

def analyze_timeframe(filename, label):
    """Analyze one timeframe's data."""
    rows = read_csv(filename)
    if not rows:
        print(f"\n{'='*60}")
        print(f"  {label}: NO DATA")
        return

    closes = [r['close'] for r in rows]
    plots = [r['plot'] for r in rows]
    n = len(closes)

    print(f"\n{'='*60}")
    print(f"  {label}  ({n} bars)")
    print(f"{'='*60}")

    # Basic stats on the original plot
    valid_plots = [p for p in plots if p is not None]
    if valid_plots:
        print(f"  Original Plot range: [{min(valid_plots):.4f}, {max(valid_plots):.4f}]")
        print(f"  Original Plot mean:  {sum(valid_plots)/len(valid_plots):.4f}")
        print(f"  Original Plot stdev: {(sum((p - sum(valid_plots)/len(valid_plots))**2 for p in valid_plots)/len(valid_plots))**0.5:.4f}")

    # Approximate true range from close-to-close (we don't have OHLC)
    tr_approx = true_range_from_close(closes)

    # Test multiple baseline types and periods
    print(f"\n  --- Baseline + Deviation Analysis ---")
    print(f"  {'Config':<45} {'Gain':>8} {'Bias':>8} {'R':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")

    best_r = 0.0
    best_config = ""
    best_detail = {}

    for base_type in ['SMA', 'WMA', 'EMA']:
        for base_period in [14, 20, 30, 50, 100]:
            if base_period >= n - 20:
                continue

            # Compute baseline
            if base_type == 'SMA':
                bl = sma(closes, base_period)
            elif base_type == 'WMA':
                bl = wma(closes, base_period)
            else:
                bl = ema(closes, base_period)

            # Raw deviation
            deviation = [None] * n
            for i in range(n):
                if bl[i] is not None:
                    deviation[i] = closes[i] - bl[i]

            # Test different normalization approaches
            for norm_type, norm_label in [
                ('none', 'Raw'),
                ('atr_close', 'ATR(close)'),
                ('stdev', 'StdDev'),
                ('pct', 'PctDev'),
            ]:
                for norm_period in [14, 20, 50]:
                    if norm_period >= n - 20:
                        continue

                    normalized = [None] * n

                    if norm_type == 'none':
                        normalized = deviation
                        if norm_period != 50:  # only test once
                            continue
                    elif norm_type == 'atr_close':
                        atr_vals = rma(tr_approx, norm_period)
                        for i in range(n):
                            if deviation[i] is not None and atr_vals[i] is not None and atr_vals[i] > 0.001:
                                normalized[i] = deviation[i] / atr_vals[i]
                    elif norm_type == 'stdev':
                        sd = stdev(closes, norm_period)
                        for i in range(n):
                            if deviation[i] is not None and sd[i] is not None and sd[i] > 0.001:
                                normalized[i] = deviation[i] / sd[i]
                    elif norm_type == 'pct':
                        for i in range(n):
                            if deviation[i] is not None and bl[i] is not None and abs(bl[i]) > 0.001:
                                normalized[i] = deviation[i] / bl[i]
                        if norm_period != 50:
                            continue

                    # Test different smoothing
                    for smooth_type, smooth_label in [('none', 'NoSm'), ('ema', 'EMA')]:
                        for smooth_period in [5, 7, 9, 12]:
                            if smooth_type == 'none' and smooth_period != 9:
                                continue

                            if smooth_type == 'none':
                                smoothed = normalized
                                sm_str = "NoSm"
                            else:
                                smoothed = ema(normalized, smooth_period)
                                sm_str = f"EMA{smooth_period}"

                            # Linear regression against original plot
                            gain, bias, r = linreg(smoothed, plots)

                            config = f"{base_type}({base_period}) {norm_label}({norm_period}) {sm_str}"

                            if abs(r) > abs(best_r):
                                best_r = r
                                best_config = config
                                best_detail = {
                                    'base_type': base_type,
                                    'base_period': base_period,
                                    'norm_type': norm_type,
                                    'norm_period': norm_period,
                                    'smooth_type': smooth_type,
                                    'smooth_period': smooth_period,
                                    'gain': gain,
                                    'bias': bias,
                                    'r': r,
                                }

                            if abs(r) > 0.90:
                                print(f"  {config:<45} {gain:>8.4f} {bias:>8.4f} {r:>8.4f}")

    print(f"\n  *** BEST MATCH: {best_config}")
    print(f"      Gain={best_detail.get('gain', 0):.4f}, Bias={best_detail.get('bias', 0):.4f}, R={best_detail.get('r', 0):.6f}")

    # Deep dive on the best config
    if best_detail:
        bd = best_detail
        print(f"\n  --- Deep Dive: {best_config} ---")

        # Recompute best config and show residual analysis
        if bd['base_type'] == 'SMA':
            bl = sma(closes, bd['base_period'])
        elif bd['base_type'] == 'WMA':
            bl = wma(closes, bd['base_period'])
        else:
            bl = ema(closes, bd['base_period'])

        deviation = [None] * n
        for i in range(n):
            if bl[i] is not None:
                deviation[i] = closes[i] - bl[i]

        if bd['norm_type'] == 'atr_close':
            atr_vals = rma(tr_approx, bd['norm_period'])
            normalized = [None] * n
            for i in range(n):
                if deviation[i] is not None and atr_vals[i] is not None and atr_vals[i] > 0.001:
                    normalized[i] = deviation[i] / atr_vals[i]
        elif bd['norm_type'] == 'stdev':
            sd = stdev(closes, bd['norm_period'])
            normalized = [None] * n
            for i in range(n):
                if deviation[i] is not None and sd[i] is not None and sd[i] > 0.001:
                    normalized[i] = deviation[i] / sd[i]
        elif bd['norm_type'] == 'pct':
            normalized = [None] * n
            for i in range(n):
                if deviation[i] is not None and bl[i] is not None and abs(bl[i]) > 0.001:
                    normalized[i] = deviation[i] / bl[i]
        else:
            normalized = deviation

        if bd['smooth_type'] == 'ema':
            smoothed = ema(normalized, bd['smooth_period'])
        else:
            smoothed = normalized

        # Show fitted vs original for last 20 bars
        print(f"\n  Last 10 bars (fitted vs original):")
        print(f"  {'Bar':<5} {'Fitted':>10} {'Original':>10} {'Error':>10}")
        for i in range(max(0, n-10), n):
            if smoothed[i] is not None and plots[i] is not None:
                fitted = smoothed[i] * bd['gain'] + bd['bias']
                error = fitted - plots[i]
                print(f"  {i:<5} {fitted:>10.4f} {plots[i]:>10.4f} {error:>10.4f}")

    return best_detail


def main():
    print("=" * 60)
    print("  IA-Mean-Reversion [3.6] — CSV Reverse Engineering")
    print("  NOTE: Only 'close' available (no OHLC), so ATR is")
    print("  approximated from close-to-close changes.")
    print("=" * 60)

    files = [
        ('tsla_daily.csv', 'TSLA DAILY'),
        ('tsla_8h.csv',    'TSLA 8H'),
        ('tsla_4h.csv',    'TSLA 4H'),
        ('tsla_1h.csv',    'TSLA 1H'),
        ('tsla_15m.csv',   'TSLA 15M'),
    ]

    results = {}
    for filename, label in files:
        if os.path.exists(os.path.join(DATA_DIR, filename)):
            result = analyze_timeframe(filename, label)
            if result:
                results[label] = result

    # Cross-timeframe comparison
    print(f"\n{'='*60}")
    print(f"  CROSS-TIMEFRAME SUMMARY")
    print(f"{'='*60}")
    print(f"  {'TF':<12} {'Best Config':<40} {'Gain':>8} {'Bias':>8} {'R':>8}")
    print(f"  {'-'*12} {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for tf, detail in results.items():
        config = f"{detail['base_type']}({detail['base_period']}) {detail['norm_type']}({detail['norm_period']}) {'EMA'+str(detail['smooth_period']) if detail['smooth_type']=='ema' else 'NoSm'}"
        print(f"  {tf:<12} {config:<40} {detail['gain']:>8.4f} {detail['bias']:>8.4f} {detail['r']:>8.4f}")

    # Key insight
    print(f"\n  KEY OBSERVATIONS:")
    gains = [d['gain'] for d in results.values()]
    if gains:
        print(f"  - Gain range across TFs: {min(gains):.4f} to {max(gains):.4f}")
        if max(gains) / max(min(gains), 0.001) > 2:
            print(f"  - Large gain variation → original likely uses TF-dependent scaling")
        else:
            print(f"  - Gains are similar → same algorithm across TFs")

        # Check if same base/norm config works everywhere
        configs = set()
        for d in results.values():
            configs.add(f"{d['base_type']}({d['base_period']}) {d['norm_type']}({d['norm_period']})")
        if len(configs) == 1:
            print(f"  - Same config wins on all TFs: {configs.pop()}")
        else:
            print(f"  - Different configs win on different TFs:")
            for tf, d in results.items():
                print(f"    {tf}: {d['base_type']}({d['base_period']}) {d['norm_type']}({d['norm_period']})")


if __name__ == '__main__':
    main()

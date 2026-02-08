#!/usr/bin/env python3
"""
V9 Analysis: Use OHLC data from data2/ to find the EXACT algorithm.
Now we have open, high, low, close AND the original's Plot values.
This means we can compute proper ATR (not the close-only approximation).
"""

import csv
import math
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data2')

def read_csv(filename):
    rows = []
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    # Find column indices
    shape_indices = [i for i, h in enumerate(header) if h == 'Shapes']
    for line in lines[1:]:
        vals = line.strip().split(',')
        try:
            r = {
                'time': int(vals[0]),
                'open': float(vals[1]),
                'high': float(vals[2]),
                'low': float(vals[3]),
                'close': float(vals[4]),
                'plot': float(vals[5]) if vals[5].strip() else None,
                'sell_shape': float(vals[shape_indices[0]]) if len(shape_indices) > 0 and vals[shape_indices[0]].strip() else None,
                'buy_shape': float(vals[shape_indices[1]]) if len(shape_indices) > 1 and vals[shape_indices[1]].strip() else None,
            }
            rows.append(r)
        except (ValueError, IndexError):
            continue
    return rows

# ─── Moving averages ──────────────────────────────────────────────────────────

def sma(values, period):
    result = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result

def wma(values, period):
    result = [None] * len(values)
    ws = period * (period + 1) / 2
    for i in range(period - 1, len(values)):
        total = sum(values[i - period + 1 + j] * (j + 1) for j in range(period))
        result[i] = total / ws
    return result

def ema_series(values, period):
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

def rma_series(values, period):
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

# ─── ATR computation with proper OHLC ─────────────────────────────────────────

def true_range_ohlc(rows):
    """Compute true range from OHLC data."""
    result = [None]
    for i in range(1, len(rows)):
        hl = rows[i]['high'] - rows[i]['low']
        hc = abs(rows[i]['high'] - rows[i-1]['close'])
        lc = abs(rows[i]['low'] - rows[i-1]['close'])
        result.append(max(hl, hc, lc))
    # First bar: just use high-low
    result[0] = rows[0]['high'] - rows[0]['low']
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
    sy_var = sum((b - my) ** 2 for _, b in pairs) / n
    sx = math.sqrt(var) if var > 0 else 0
    sy = math.sqrt(sy_var) if sy_var > 0 else 0
    if var < 1e-10:
        return 0.0, my, 0.0
    gain = cov / var
    bias = my - gain * mx
    r = cov / (sx * sy) if sx > 1e-10 and sy > 1e-10 else 0.0
    return gain, bias, r


def compute_oscillator(rows, base_type, base_period, atr_period, smooth_period, source='ohlc4'):
    n = len(rows)
    # Source
    if source == 'ohlc4':
        src = [(r['open'] + r['high'] + r['low'] + r['close']) / 4 for r in rows]
    elif source == 'hlc3':
        src = [(r['high'] + r['low'] + r['close']) / 3 for r in rows]
    elif source == 'hl2':
        src = [(r['high'] + r['low']) / 2 for r in rows]
    else:
        src = [r['close'] for r in rows]

    # Baseline
    if base_type == 'SMA':
        bl = sma(src, base_period)
    elif base_type == 'WMA':
        bl = wma(src, base_period)
    else:
        bl = ema_series(src, base_period)

    # ATR (RMA of true range)
    tr = true_range_ohlc(rows)
    atr = rma_series(tr, atr_period)

    # Deviation / ATR
    raw_z = [None] * n
    for i in range(n):
        if bl[i] is not None and atr[i] is not None and atr[i] > 0.001:
            raw_z[i] = (src[i] - bl[i]) / atr[i]

    # Smoothing
    if smooth_period > 1:
        smoothed = ema_series(raw_z, smooth_period)
    else:
        smoothed = raw_z

    return smoothed, raw_z, bl, atr


def main():
    files = [
        ('BATS_TSLA_V8_1D.csv', 'Daily'),
        ('BATS_TSLA_V8_8h.csv', '8H'),
        ('BATS_TSLA_V8_4h.csv', '4H'),
        ('BATS_TSLA_V8_1h.csv', '1H'),
        ('BATS_TSLA_V8_15m.csv', '15M'),
    ]
    datasets = {}
    for filename, label in files:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            datasets[label] = read_csv(filename)

    print("=" * 90)
    print("  V9 ANALYSIS: OHLC-based reverse engineering")
    print("  Now with PROPER ATR from high/low data!")
    print("=" * 90)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: Original indicator characteristics
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 1: Original Indicator Characteristics (from Plot column)")
    print("=" * 90)
    for label, data in datasets.items():
        plots = [r['plot'] for r in data if r['plot'] is not None]
        sells = [r for r in data if r['sell_shape'] is not None]
        buys  = [r for r in data if r['buy_shape'] is not None]
        if plots:
            mean = sum(plots) / len(plots)
            sd = (sum((p - mean)**2 for p in plots) / len(plots))**0.5
            print(f"  {label:>6}: range [{min(plots):>7.4f}, {max(plots):>7.4f}]  "
                  f"mean={mean:>7.4f}  stdev={sd:>6.4f}  bars={len(plots)}")
            print(f"          Sell dots: {len(sells)}  Buy dots: {len(buys)}")
            if sells:
                sell_vals = ['{:.3f}'.format(s['sell_shape']) for s in sells]
                print(f"          Sell values: {sell_vals}")
            if buys:
                buy_vals = ['{:.3f}'.format(b['buy_shape']) for b in buys]
                print(f"          Buy values: {buy_vals}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: Fine-grained OHLC parameter sweep
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 2: OHLC Parameter Sweep — Finding exact algorithm")
    print("  Testing SMA/WMA with proper ATR, ohlc4 source")
    print("=" * 90)

    all_results = []

    for base_type in ['SMA', 'WMA']:
        for base_period in [40, 45, 48, 50, 52, 55, 60]:
            for atr_period in [10, 14, 15, 20, 25, 30, 40, 50]:
                for smooth_period in [3, 5, 7, 9, 12]:
                    for source in ['ohlc4', 'close']:
                        tf_results = {}
                        for label, data in datasets.items():
                            plots = [r['plot'] for r in data]
                            osc, _, _, _ = compute_oscillator(data, base_type, base_period, atr_period, smooth_period, source)
                            gain, bias, r = linreg(osc, plots)
                            tf_results[label] = {'gain': gain, 'bias': bias, 'r': r}

                        rs = [tf_results[l]['r'] for l in datasets]
                        avg_r = sum(rs) / len(rs)
                        min_r = min(rs)
                        gains = [tf_results[l]['gain'] for l in datasets]
                        mean_gain = sum(abs(g) for g in gains) / len(gains) if gains else 1
                        gain_cv = (sum((abs(g) - mean_gain)**2 for g in gains) / len(gains))**0.5 / mean_gain if mean_gain > 0.001 else 99

                        all_results.append({
                            'base_type': base_type, 'base_period': base_period,
                            'atr_period': atr_period, 'smooth_period': smooth_period,
                            'source': source, 'tf_results': tf_results,
                            'avg_r': avg_r, 'min_r': min_r, 'gain_cv': gain_cv, 'rs': rs, 'gains': gains
                        })

    # Sort by avg R
    all_results.sort(key=lambda x: -x['avg_r'])

    print(f"\n  Top 25 configs by Average R (with OHLC ATR):")
    print(f"  {'Config':<40} {'Src':>5}", end="")
    for label in datasets:
        print(f" {'R_'+label:>8}", end="")
    print(f" {'AvgR':>8} {'MinR':>8} {'GnCV':>6}")
    print(f"  {'-'*40} {'-'*5}", end="")
    for _ in datasets:
        print(f" {'-'*8}", end="")
    print(f" {'-'*8} {'-'*8} {'-'*6}")

    for res in all_results[:25]:
        cfg = f"{res['base_type']}({res['base_period']}) ATR({res['atr_period']}) EMA{res['smooth_period']}"
        print(f"  {cfg:<40} {res['source']:>5}", end="")
        for r in res['rs']:
            print(f" {r:>8.4f}", end="")
        print(f" {res['avg_r']:>8.4f} {res['min_r']:>8.4f} {res['gain_cv']:>6.3f}")

    # Sort by min R
    all_results.sort(key=lambda x: -x['min_r'])
    print(f"\n  Top 15 configs by Minimum R (best worst-case):")
    print(f"  {'Config':<40} {'Src':>5}", end="")
    for label in datasets:
        print(f" {'R_'+label:>8}", end="")
    print(f" {'AvgR':>8} {'MinR':>8} {'GnCV':>6}")
    print(f"  {'-'*40} {'-'*5}", end="")
    for _ in datasets:
        print(f" {'-'*8}", end="")
    print(f" {'-'*8} {'-'*8} {'-'*6}")

    for res in all_results[:15]:
        cfg = f"{res['base_type']}({res['base_period']}) ATR({res['atr_period']}) EMA{res['smooth_period']}"
        print(f"  {cfg:<40} {res['source']:>5}", end="")
        for r in res['rs']:
            print(f" {r:>8.4f}", end="")
        print(f" {res['avg_r']:>8.4f} {res['min_r']:>8.4f} {res['gain_cv']:>6.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: Gain analysis for top configs
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 3: Gain/Bias Analysis for Top Configs")
    print("  Goal: find a config where gain is CONSISTENT across TFs")
    print("=" * 90)

    all_results.sort(key=lambda x: -x['avg_r'])
    for res in all_results[:8]:
        cfg = f"{res['base_type']}({res['base_period']}) ATR({res['atr_period']}) EMA{res['smooth_period']} [{res['source']}]"
        print(f"\n  {cfg}  (avg_R={res['avg_r']:.4f}, gain_CV={res['gain_cv']:.3f}):")
        for label, g in zip(datasets.keys(), res['gains']):
            b = res['tf_results'][label]['bias']
            r = res['tf_results'][label]['r']
            print(f"    {label:>6}: Gain={g:>10.4f}  Bias={b:>8.4f}  R={r:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 4: Band analysis (stdev of oscillator → red/green bands)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 4: Band Analysis — What are the red/green bands?")
    print("  Testing rolling stdev of the original's Plot values")
    print("=" * 90)

    for label, data in datasets.items():
        plots = [r['plot'] for r in data]
        valid = [p for p in plots if p is not None]
        if not valid:
            continue
        mean_p = sum(valid) / len(valid)
        std_p = (sum((p - mean_p)**2 for p in valid) / len(valid))**0.5

        # Test various rolling stdev windows
        print(f"\n  {label} — Overall stdev: {std_p:.4f}")
        for window in [20, 50, 100, 150, 200]:
            sd = stdev_roll(plots, window)
            # Look at last valid stdev values
            last_valid = [s for s in sd[-50:] if s is not None]
            if last_valid:
                avg_sd = sum(last_valid) / len(last_valid)
                print(f"    Stdev({window}) last 50 bars avg: {avg_sd:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 5: Signal dot analysis
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 5: Signal Dot Analysis — When do dots trigger?")
    print("=" * 90)

    for label, data in datasets.items():
        plots = [r['plot'] for r in data]
        n = len(data)
        print(f"\n  {label}:")

        for i in range(2, n - 1):
            if data[i]['sell_shape'] is not None:
                # Sell dot — should be at a local peak
                p_prev = plots[i-1] if plots[i-1] is not None else 0
                p_curr = plots[i] if plots[i] is not None else 0
                p_next = plots[i+1] if i+1 < n and plots[i+1] is not None else 0
                is_peak = p_curr > p_prev and p_curr > p_next
                # Check if it's the highest in N bars
                window_max = max(p for p in plots[max(0,i-30):i] if p is not None) if any(p is not None for p in plots[max(0,i-30):i]) else 0
                print(f"    SELL dot at bar {i}: plot={p_curr:.4f}, peak={is_peak}, "
                      f"prev={p_prev:.4f}, next={p_next:.4f}, "
                      f"30-bar max before={window_max:.4f}")

            if data[i]['buy_shape'] is not None:
                p_prev = plots[i-1] if plots[i-1] is not None else 0
                p_curr = plots[i] if plots[i] is not None else 0
                p_next = plots[i+1] if i+1 < n and plots[i+1] is not None else 0
                is_trough = p_curr < p_prev and p_curr < p_next
                window_min = min(p for p in plots[max(0,i-30):i] if p is not None) if any(p is not None for p in plots[max(0,i-30):i]) else 0
                print(f"    BUY  dot at bar {i}: plot={p_curr:.4f}, trough={is_trough}, "
                      f"prev={p_prev:.4f}, next={p_next:.4f}, "
                      f"30-bar min before={window_min:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 6: V8 vs Original comparison (fitted)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 6: V8 vs Original — How close is the current V8?")
    print("  Using WMA(50) ATR(50) EMA(9) ohlc4 with proper OHLC")
    print("=" * 90)

    for label, data in datasets.items():
        plots = [r['plot'] for r in data]
        osc, raw_z, bl, atr_vals = compute_oscillator(data, 'WMA', 50, 50, 9, 'ohlc4')
        gain, bias, r = linreg(osc, plots)

        # Also compute residuals
        residuals = []
        for s, p in zip(osc, plots):
            if s is not None and p is not None:
                fitted = s * gain + bias
                residuals.append(p - fitted)

        res_std = (sum(r**2 for r in residuals) / len(residuals))**0.5 if residuals else 0
        plot_std = (sum((p - sum(pp for pp in plots if pp is not None)/sum(1 for pp in plots if pp is not None))**2
                    for p in plots if p is not None) / sum(1 for p in plots if p is not None))**0.5

        print(f"\n  {label}: R={r:.4f}, Gain={gain:.4f}, Bias={bias:.4f}")
        print(f"    Residual stdev: {res_std:.4f} / Plot stdev: {plot_std:.4f} = {res_std/plot_std:.3f} unexplained")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 7: Best single universal config — detailed
    # ═══════════════════════════════════════════════════════════════════════════
    all_results.sort(key=lambda x: -x['avg_r'])
    best = all_results[0]

    print("\n" + "=" * 90)
    print(f"  PART 7: BEST UNIVERSAL CONFIG — DETAILED")
    print(f"  {best['base_type']}({best['base_period']}) ATR({best['atr_period']}) EMA{best['smooth_period']} [{best['source']}]")
    print("=" * 90)

    for label, data in datasets.items():
        plots = [r['plot'] for r in data]
        osc, _, _, _ = compute_oscillator(
            data, best['base_type'], best['base_period'],
            best['atr_period'], best['smooth_period'], best['source'])
        gain, bias, r = linreg(osc, plots)

        # Show last 10 bars
        n = len(data)
        print(f"\n  {label}: R={r:.4f}, Gain={gain:.4f}, Bias={bias:.4f}")
        print(f"    Last 10 bars:")
        print(f"    {'Bar':>5} {'Fitted':>10} {'Original':>10} {'Error':>10}")
        for i in range(max(0, n-10), n):
            if osc[i] is not None and plots[i] is not None:
                fitted = osc[i] * gain + bias
                error = fitted - plots[i]
                print(f"    {i:>5} {fitted:>10.4f} {plots[i]:>10.4f} {error:>10.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    all_results.sort(key=lambda x: -x['avg_r'])
    best = all_results[0]
    print(f"\n  Best by avg R:  {best['base_type']}({best['base_period']}) ATR({best['atr_period']}) EMA{best['smooth_period']} [{best['source']}] → avg_R={best['avg_r']:.4f}")

    all_results.sort(key=lambda x: -x['min_r'])
    best_min = all_results[0]
    print(f"  Best by min R:  {best_min['base_type']}({best_min['base_period']}) ATR({best_min['atr_period']}) EMA{best_min['smooth_period']} [{best_min['source']}] → min_R={best_min['min_r']:.4f}")

    # Find config with lowest gain CV (most consistent gain across TFs)
    all_results.sort(key=lambda x: x['gain_cv'])
    best_cv = all_results[0]
    print(f"  Most consistent: {best_cv['base_type']}({best_cv['base_period']}) ATR({best_cv['atr_period']}) EMA{best_cv['smooth_period']} [{best_cv['source']}] → gain_CV={best_cv['gain_cv']:.3f}")
    for label, g in zip(datasets.keys(), best_cv['gains']):
        print(f"    {label}: gain={g:.4f}")


if __name__ == '__main__':
    main()

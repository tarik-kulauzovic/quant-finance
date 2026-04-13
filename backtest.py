"""
BTC BACKTEST + MONTE CARLO SIMULATOR
=====================================
Temelji na signal logiki iz signal_tracker.py v1.2

Podatki  : yfinance (BTC-USD, 1h ali 1d)
Signal   : tocke >= +3 → LONG | tocke <= -3 → SHORT
SL/TP    : ATR-based (TF-specifični multiplikatorji iz v1.2)
Kapital  : 1000 EUR (privzeto), nastavljivo

Namestitev:
    pip install yfinance numpy scipy matplotlib

Zagon:
    python btc_backtest.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("NAPAKA: yfinance ni nameščen. Poženi: pip install yfinance")
    raise

# ════════════════════════════════════════════════════════
#  KONFIGURACIJA — nastavi po svojih željah
# ════════════════════════════════════════════════════════

ZACETNI_KAPITAL   = 1000.0          # EUR
POZICIJA_PCT      = 1.0             # 1.0 = 100% kapitala na trade (brez levered)
PROVIZIJA_PCT     = 0.0006          # 0.06% na vstop + izhod (Binance taker)

# Signal prag — isti kot tvoj tracker
SIGNAL_LONG_PRAG  = 4              # tocke >= tega → LONG
SIGNAL_SHORT_PRAG = -4             # tocke <= tega → SHORT

# ATR multiplikatorji po timeframeu (iz v1.2)
ATR_MULT_BY_TF = {
    "15m": {"sl": 2.0,  "tp": 4.0,  "be": 1.2},
    "1h":  {"sl": 2.5,  "tp": 5.0,  "be": 1.5},
    "1d":  {"sl": 1.5,  "tp": 3.0,  "be": 1.0},
}

# Rough Heston
HURST_CRYPTO = 0.08
RH_WINDOW    = 30

# Indikatorji
ATR_PERIOD   = 14
RSI_PERIOD   = 14
SMA_TREND    = 200
VOL_PERIOD   = 20

# ════════════════════════════════════════════════════════
#  1.  PRIDOBITEV PODATKOV
# ════════════════════════════════════════════════════════

TIMEFRAME_MENU = {
    "1": {"interval": "1h",  "period": "730d",  "label": "Urni (1h) — 2 leti"},
    "2": {"interval": "1d",  "period": "730d",  "label": "Dnevni (1d) — 2 leti"},
    "3": {"interval": "1h",  "period": "365d",  "label": "Urni (1h) — 1 leto"},
}

def pridobi_btc(interval="1h", period="730d"):
    print(f"\n  Nalagam BTC-USD ({interval}, zadnjih {period})...")
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(interval=interval, period=period)
    if df is None or len(df) < 50:
        raise ValueError(f"Premalo podatkov ({len(df) if df is not None else 0} svečk)")
    df = df.dropna(subset=["Close", "High", "Low"])
    print(f"  Naloženo: {len(df)} svečk  |  od {df.index[0].date()} do {df.index[-1].date()}")
    return df

# ════════════════════════════════════════════════════════
#  2.  INDIKATORJI  (iz signal_tracker.py)
# ════════════════════════════════════════════════════════

def calc_atr(closes, highs, lows, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i]  - closes[i-1])
        lpc = abs(lows[i]   - closes[i-1])
        trs.append(max(hl, hpc, lpc))
    atr = float(np.mean(trs[:period]))
    for x in trs[period:]:
        atr = (atr * (period - 1) + x) / period
    return atr

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    diffs  = np.diff(closes)
    gains  = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    ag = float(np.mean(gains[:period]))
    al = float(np.mean(losses[:period]))
    for g, l in zip(gains[period:], losses[period:]):
        ag = (ag * (period-1) + g) / period
        al = (al * (period-1) + l) / period
    if al == 0:
        return 100.0
    return round(100 - (100 / (1 + ag/al)), 2)

def calc_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, False, False
    def ema(data, n):
        k = 2/(n+1)
        e = float(np.mean(data[:n]))
        out = [e]
        for x in data[n:]:
            e = x*k + e*(1-k)
            out.append(e)
        return out
    fe = ema(closes, fast)
    se = ema(closes, slow)
    off = slow - fast
    ml = [f - s for f, s in zip(fe[off:], se)]
    if len(ml) < signal:
        return None, None, False, False
    sl_ = ema(ml, signal)
    m1, s1 = ml[-1], sl_[-1]
    m0 = ml[-2] if len(ml) > 1 else m1
    s0 = sl_[-2] if len(sl_) > 1 else s1
    cross_up   = m1 > s1 and m0 <= s0
    cross_down = m1 < s1 and m0 >= s0
    return m1, s1, cross_up, cross_down

def rough_heston_vol(closes, H=0.08, window=30):
    if len(closes) < window + 2:
        rets = np.diff(np.log(closes))
        return float(np.std(rets) * np.sqrt(252 if window > 50 else 8760))
    donosi = np.diff(np.log(closes))
    n = len(donosi)
    w = min(window, n)
    weights = np.array([(w - i) ** (H - 0.5) for i in range(w)])
    weights /= weights.sum()
    local_var = float(np.sum(weights * donosi[-w:] ** 2))
    # annualize: dnevni=252, urni=8760
    ann = 252 if window <= 30 else 8760
    sigma = float(np.sqrt(local_var * ann))
    return min(max(sigma, 0.03), 5.0)

def vol_rezim(closes, H=0.08, window=30):
    if len(closes) < window * 2 + 2:
        return "NORMAL", 1.0
    v1 = rough_heston_vol(closes[-window:],          H, window//2)
    v2 = rough_heston_vol(closes[-window*2:-window], H, window//2)
    if v2 == 0:
        return "NORMAL", 1.0
    ratio = v1 / v2
    if ratio > 1.4:   return "EXPANDING",   round(ratio, 3)
    elif ratio < 0.7: return "CONTRACTING", round(ratio, 3)
    return "NORMAL", round(ratio, 3)

# ════════════════════════════════════════════════════════
#  3.  SIGNAL LOGIKA  (iz signal_tracker.py v1.2)
# ════════════════════════════════════════════════════════

def izracunaj_tocke(closes, highs, lows, interval="1h"):
    n   = len(closes)
    S   = closes[-1]
    tocke = 0

    # 1. SMA200 trend
    sma200 = float(np.mean(closes[-200:])) if n >= 200 else float(np.mean(closes))
    if S > sma200: tocke += 2
    else:          tocke -= 2

    # 2. RSI
    rsi = calc_rsi(closes, RSI_PERIOD)
    if rsi is not None:
        if 40 <= rsi <= 65:   tocke += 2
        elif 20 < rsi < 40:   tocke += 1
        elif rsi > 65:        tocke -= 2
        elif rsi <= 20:       tocke -= 1

    # 3. MACD
    mv, sv, cu, cd = calc_macd(closes)
    if mv is not None:
        if cu:          tocke += 2
        elif mv > sv:   tocke += 1
        elif cd:        tocke -= 2
        elif mv < sv:   tocke -= 1

    # 4. Volatilnost (Rough Heston)
    sigma = rough_heston_vol(closes, H=HURST_CRYPTO, window=RH_WINDOW)
    if sigma > 0.60:   tocke -= 1
    elif sigma < 0.08: tocke += 1

    # 5. Vol rezim korekcija ATR
    rh_rezim, rh_ratio = vol_rezim(closes, H=HURST_CRYPTO, window=RH_WINDOW)
    if rh_rezim == "EXPANDING":
        atr_kor = 1.15
    elif rh_rezim == "CONTRACTING":
        atr_kor = 0.92 if interval == "1h" else 0.85
    else:
        atr_kor = 1.0

    return tocke, sigma, atr_kor, rh_rezim

# ════════════════════════════════════════════════════════
#  4.  BACKTEST ENGINE
# ════════════════════════════════════════════════════════

def run_backtest(df, interval="1h", kapital=1000.0, verbose=False):
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    dates  = df.index

    mult   = ATR_MULT_BY_TF.get(interval, ATR_MULT_BY_TF["1h"])
    SL_M   = mult["sl"]
    TP_M   = mult["tp"]

    WARM   = max(210, RH_WINDOW * 2 + 2)  # dovolj podatkov za SMA200 + RH

    trades  = []
    i       = WARM

    while i < len(closes) - 2:
        c_slice = closes[:i+1]
        h_slice = highs[:i+1]
        l_slice = lows[:i+1]

        tocke, sigma, atr_kor, rh_rezim = izracunaj_tocke(c_slice, h_slice, l_slice, interval)

        if tocke >= SIGNAL_LONG_PRAG:
            smer = "LONG"
        elif tocke <= SIGNAL_SHORT_PRAG:
            smer = "SHORT"
        else:
            i += 1
            continue

        # Vstop na naslednjem baru (close)
        vstop_idx   = i + 1
        vstop_cena  = closes[vstop_idx]
        vstop_datum = dates[vstop_idx]

        atr_raw = calc_atr(c_slice, h_slice, l_slice, ATR_PERIOD)
        if atr_raw is None:
            i += 1
            continue
        atr = atr_raw * atr_kor

        if smer == "LONG":
            sl = vstop_cena - atr * SL_M
            tp = vstop_cena + atr * TP_M
        else:
            sl = vstop_cena + atr * SL_M
            tp = vstop_cena - atr * TP_M

        # Simuliramo izhod
        izhod_cena  = None
        izhod_datum = None
        izhod_vzrok = "OPEN"

        for j in range(vstop_idx + 1, len(closes)):
            h, l = highs[j], lows[j]
            if smer == "LONG":
                if l <= sl:
                    izhod_cena  = sl
                    izhod_datum = dates[j]
                    izhod_vzrok = "SL"
                    break
                if h >= tp:
                    izhod_cena  = tp
                    izhod_datum = dates[j]
                    izhod_vzrok = "TP"
                    break
            else:
                if h >= sl:
                    izhod_cena  = sl
                    izhod_datum = dates[j]
                    izhod_vzrok = "SL"
                    break
                if l <= tp:
                    izhod_cena  = tp
                    izhod_datum = dates[j]
                    izhod_vzrok = "TP"
                    break

        if izhod_cena is None:
            izhod_cena  = closes[-1]
            izhod_datum = dates[-1]
            izhod_vzrok = "OPEN"

        # P&L z provizijo
        if smer == "LONG":
            raw_pnl_pct = (izhod_cena - vstop_cena) / vstop_cena
        else:
            raw_pnl_pct = (vstop_cena - izhod_cena) / vstop_cena
        pnl_pct = raw_pnl_pct - 2 * PROVIZIJA_PCT  # vstop + izhod

        trades.append({
            "vstop_datum":  vstop_datum,
            "izhod_datum":  izhod_datum,
            "smer":         smer,
            "vstop_cena":   vstop_cena,
            "izhod_cena":   izhod_cena,
            "sl":           sl,
            "tp":           tp,
            "tocke":        tocke,
            "atr":          atr,
            "sigma":        sigma,
            "rh_rezim":     rh_rezim,
            "pnl_pct":      pnl_pct * 100,
            "izhod_vzrok":  izhod_vzrok,
        })

        if verbose:
            emoji = "+" if pnl_pct > 0 else "-"
            print(f"  [{vstop_datum.date()}] {smer:5s}  vstop={vstop_cena:,.0f}  izhod={izhod_cena:,.0f}  "
                  f"tocke={tocke:+d}  P&L={pnl_pct*100:+.1f}%  [{izhod_vzrok}]")

        # Preskoči na dan po izhodu (preprečimo overlapping trades)
        izhod_idx = list(dates).index(izhod_datum) if izhod_datum in dates else vstop_idx + 1
        i = max(izhod_idx + 1, i + 1)

    return trades

def equity_krivulja(trades, kapital=1000.0):
    kap  = kapital
    krivulja = [kap]
    for t in trades:
        kap *= (1 + t["pnl_pct"] / 100 * POZICIJA_PCT)
        krivulja.append(kap)
    return np.array(krivulja)

def calc_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak * 100

def calc_statistike(trades, equity, kapital):
    if not trades:
        return {}
    pnls  = np.array([t["pnl_pct"] for t in trades])
    wins  = pnls[pnls > 0]
    loss  = pnls[pnls <= 0]
    dd    = calc_drawdown(equity)
    rr_avg = abs(np.mean(wins)) / abs(np.mean(loss)) if len(loss) > 0 and len(wins) > 0 else 0

    # Sharpe (po trade-ih, ne dnevni)
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))) if np.std(pnls) > 0 else 0

    longs  = [t for t in trades if t["smer"] == "LONG"]
    shorts = [t for t in trades if t["smer"] == "SHORT"]

    return {
        "skupaj_trades":   len(trades),
        "long_trades":     len(longs),
        "short_trades":    len(shorts),
        "win_rate":        len(wins) / len(pnls) * 100,
        "avg_win":         float(np.mean(wins)) if len(wins) > 0 else 0,
        "avg_loss":        float(np.mean(loss)) if len(loss) > 0 else 0,
        "rr_avg":          rr_avg,
        "sharpe":          sharpe,
        "skupni_donos_pct":(equity[-1] / kapital - 1) * 100,
        "skupni_donos_eur": equity[-1] - kapital,
        "max_dd_pct":       float(np.min(dd)),
        "konc_kapital":     equity[-1],
        "tp_trades":       sum(1 for t in trades if t["izhod_vzrok"] == "TP"),
        "sl_trades":       sum(1 for t in trades if t["izhod_vzrok"] == "SL"),
    }

# ════════════════════════════════════════════════════════
#  5.  MONTE CARLO
# ════════════════════════════════════════════════════════

def monte_carlo(trades, n_sim=100, kapital=1000.0, seed=42):
    """
    Premešaj vrstni red trade-ov n_sim krat.
    Vrne matriko equity krivulj (n_sim × n_trades+1).
    """
    rng     = np.random.default_rng(seed)
    pnls    = np.array([t["pnl_pct"] for t in trades])
    n       = len(pnls)
    matrix  = np.zeros((n_sim, n + 1))
    matrix[:, 0] = kapital

    for s in range(n_sim):
        shuffled = rng.permutation(pnls)
        kap = kapital
        for j, p in enumerate(shuffled):
            kap *= (1 + p / 100 * POZICIJA_PCT)
            matrix[s, j+1] = kap

    return matrix

# ════════════════════════════════════════════════════════
#  6.  VIZUALIZACIJA
# ════════════════════════════════════════════════════════

def narisi_graf(trades, equity, mc_matrix, stats, kapital, interval, n_sim):
    fig = plt.figure(figsize=(16, 14), facecolor="#0d0d0d")
    fig.suptitle(
        f"BTC/USD  ·  Backtest + Monte Carlo  ·  TF: {interval}  ·  "
        f"{len(trades)} trades  ·  začetni kapital: {kapital:.0f} EUR",
        color="#e0e0e0", fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97, top=0.94, bottom=0.06
    )

    BG    = "#0d0d0d"
    PANEL = "#161616"
    GRID  = "#222222"
    BLUE  = "#4a9eff"
    GREEN = "#4caf50"
    RED   = "#f44336"
    AMBER = "#ffb347"
    GRAY  = "#666666"
    WHITE = "#e0e0e0"
    MUTED = "#888888"

    def style_ax(ax, title=""):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
        if title:
            ax.set_title(title, color=WHITE, fontsize=9, pad=6, fontweight="normal")

    x_eq = np.arange(len(equity))

    # ── 1. Equity krivulja + MC  (zgornji pas, ves prostor) ──────────────
    ax_eq = fig.add_subplot(gs[0, :])
    style_ax(ax_eq, "Equity krivulja  +  Monte Carlo simulacije")

    # MC krivulje
    for i in range(min(n_sim, mc_matrix.shape[0])):
        ax_eq.plot(mc_matrix[i], color=GREEN, alpha=0.04, linewidth=0.6)

    # MC percentili
    p5  = np.percentile(mc_matrix, 5,  axis=0)
    p95 = np.percentile(mc_matrix, 95, axis=0)
    med = np.percentile(mc_matrix, 50, axis=0)
    x_mc = np.arange(mc_matrix.shape[1])
    ax_eq.fill_between(x_mc, p5, p95, color=GREEN, alpha=0.08, label="MC 5–95 percentil")
    ax_eq.plot(x_mc, med, color=GREEN, linewidth=1.2, alpha=0.55,
               linestyle="--", label="MC median")

    # Dejanska equity
    ax_eq.plot(x_eq, equity, color=BLUE, linewidth=2.0, zorder=5, label="Dejanska equity")

    # Začetni kapital — referenčna črta
    ax_eq.axhline(kapital, color=GRAY, linewidth=0.7, linestyle=":", alpha=0.7, label=f"Začetni kapital ({kapital:.0f} EUR)")

    ax_eq.set_ylabel("Kapital (EUR)", color=MUTED, fontsize=8)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_eq.set_xlabel("Trade #", color=MUTED, fontsize=8)

    leg = ax_eq.legend(fontsize=7.5, loc="upper left",
                       facecolor="#1a1a1a", edgecolor=GRID, labelcolor=WHITE)

    # Označi win/loss trade-e
    for idx, t in enumerate(trades):
        col = GREEN if t["pnl_pct"] > 0 else RED
        ax_eq.axvline(idx + 1, color=col, alpha=0.10, linewidth=0.5)

    # ── 2. Drawdown ──────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1, :2])
    style_ax(ax_dd, "Drawdown (%)")
    dd = calc_drawdown(equity)
    ax_dd.fill_between(x_eq, dd, 0, color=RED, alpha=0.35)
    ax_dd.plot(x_eq, dd, color=RED, linewidth=1.0)
    ax_dd.set_ylabel("%", color=MUTED, fontsize=8)
    ax_dd.set_xlabel("Trade #", color=MUTED, fontsize=8)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dd.axhline(stats.get("max_dd_pct", 0), color=AMBER, linewidth=0.8,
                  linestyle="--", alpha=0.7)
    ax_dd.text(1, stats.get("max_dd_pct", 0) + 0.3,
               f"max DD: {stats.get('max_dd_pct',0):.1f}%",
               color=AMBER, fontsize=7.5)

    # ── 3. P&L distribucija ──────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 2])
    style_ax(ax_dist, "P&L distribucija po trade-ih")
    pnls = np.array([t["pnl_pct"] for t in trades])
    bins = np.linspace(pnls.min() - 1, pnls.max() + 1, 28)
    colors_hist = [GREEN if (b + (bins[1]-bins[0])/2) > 0 else RED for b in bins[:-1]]
    ax_dist.hist(pnls, bins=bins, color=colors_hist, edgecolor=PANEL, linewidth=0.3)
    ax_dist.axvline(0, color=WHITE, linewidth=0.8, alpha=0.5)
    ax_dist.axvline(np.mean(pnls), color=AMBER, linewidth=1.0,
                    linestyle="--", alpha=0.8, label=f"avg {np.mean(pnls):.1f}%")
    ax_dist.set_xlabel("P&L %", color=MUTED, fontsize=8)
    ax_dist.set_ylabel("Št. trade-ov", color=MUTED, fontsize=8)
    ax_dist.legend(fontsize=7, facecolor="#1a1a1a", edgecolor=GRID, labelcolor=WHITE)

    # ── 4. Statistike (besedilo) ─────────────────────────────────────────
    ax_stat = fig.add_subplot(gs[2, 0])
    style_ax(ax_stat, "Ključne statistike")
    ax_stat.axis("off")

    donos_col = GREEN if stats.get("skupni_donos_pct",0) >= 0 else RED
    dd_col    = RED

    stat_lines = [
        ("Skupaj trades",    f"{stats.get('skupaj_trades',0)}"),
        ("  Long",           f"{stats.get('long_trades',0)}"),
        ("  Short",          f"{stats.get('short_trades',0)}"),
        ("Win rate",         f"{stats.get('win_rate',0):.1f}%"),
        ("TP zadetki",       f"{stats.get('tp_trades',0)}"),
        ("SL zadetki",       f"{stats.get('sl_trades',0)}"),
        ("Avg win",          f"+{stats.get('avg_win',0):.2f}%"),
        ("Avg loss",         f"{stats.get('avg_loss',0):.2f}%"),
        ("Avg R:R",          f"1 : {stats.get('rr_avg',0):.2f}"),
        ("Sharpe",           f"{stats.get('sharpe',0):.2f}"),
        ("Max drawdown",     f"{stats.get('max_dd_pct',0):.1f}%"),
        ("Skupni donos",     f"{stats.get('skupni_donos_pct',0):+.1f}%"),
        ("Končni kapital",   f"{stats.get('konc_kapital',0):.2f} EUR"),
        ("Zaslužek",         f"{stats.get('skupni_donos_eur',0):+.2f} EUR"),
    ]

    col_map = {
        "Skupni donos":  donos_col,
        "Končni kapital": donos_col,
        "Zaslužek":       donos_col,
        "Max drawdown":   dd_col,
        "Win rate":       GREEN if stats.get("win_rate",0) >= 50 else RED,
    }

    for row, (k, v) in enumerate(stat_lines):
        vc = col_map.get(k, WHITE)
        ax_stat.text(0.02, 1 - row * 0.072, k + ":",
                     color=MUTED, fontsize=8.2, transform=ax_stat.transAxes, va="top")
        ax_stat.text(0.62, 1 - row * 0.072, v,
                     color=vc, fontsize=8.2, fontweight="bold",
                     transform=ax_stat.transAxes, va="top")

    # ── 5. MC končni kapital distribucija ───────────────────────────────
    ax_mc = fig.add_subplot(gs[2, 1])
    style_ax(ax_mc, "MC — distribucija končnega kapitala")
    final_caps = mc_matrix[:, -1]
    bins_mc = np.linspace(final_caps.min()*0.98, final_caps.max()*1.02, 30)
    col_mc = [GREEN if b >= kapital else RED for b in bins_mc[:-1]]
    ax_mc.hist(final_caps, bins=bins_mc, color=col_mc, edgecolor=PANEL, linewidth=0.3)
    ax_mc.axvline(kapital, color=WHITE, linewidth=0.8, linestyle=":",
                  alpha=0.7, label=f"Začetek ({kapital:.0f})")
    ax_mc.axvline(np.median(final_caps), color=AMBER, linewidth=1.0,
                  linestyle="--", label=f"Mediana ({np.median(final_caps):.0f})")
    ax_mc.axvline(np.percentile(final_caps, 5), color=RED, linewidth=0.8,
                  linestyle="--", alpha=0.8, label=f"5.p ({np.percentile(final_caps,5):.0f})")
    ax_mc.axvline(np.percentile(final_caps, 95), color=GREEN, linewidth=0.8,
                  linestyle="--", alpha=0.8, label=f"95.p ({np.percentile(final_caps,95):.0f})")
    pct_profitable = np.sum(final_caps > kapital) / len(final_caps) * 100
    ax_mc.set_title(
        f"MC — končni kapital  ({pct_profitable:.0f}% simulacij v dobičku)",
        color=WHITE, fontsize=9, pad=6
    )
    ax_mc.set_xlabel("EUR", color=MUTED, fontsize=8)
    ax_mc.set_ylabel("Št. simulacij", color=MUTED, fontsize=8)
    ax_mc.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_mc.legend(fontsize=7, facecolor="#1a1a1a", edgecolor=GRID, labelcolor=WHITE)

    # ── 6. Trade timeline (LONG/SHORT/win/loss) ──────────────────────────
    ax_tl = fig.add_subplot(gs[2, 2])
    style_ax(ax_tl, "P&L po trade-ih (kronološki)")
    idxs  = np.arange(len(pnls))
    cols  = [GREEN if p > 0 else RED for p in pnls]
    bars  = ax_tl.bar(idxs, pnls, color=cols, width=0.8, edgecolor="none")
    ax_tl.axhline(0, color=WHITE, linewidth=0.6, alpha=0.5)
    ax_tl.set_xlabel("Trade #", color=MUTED, fontsize=8)
    ax_tl.set_ylabel("P&L %", color=MUTED, fontsize=8)
    ax_tl.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Legenda signalov
    legend_els = [
        Line2D([0],[0], color=GREEN, lw=2, label="Dobiček"),
        Line2D([0],[0], color=RED,   lw=2, label="Izguba"),
    ]
    ax_tl.legend(handles=legend_els, fontsize=7,
                 facecolor="#1a1a1a", edgecolor=GRID, labelcolor=WHITE)

    plt.savefig("btc_backtest_rezultat.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    print("\n  Graf shranjen: btc_backtest_rezultat.png")

# ════════════════════════════════════════════════════════
#  7.  IZPIS ZADNJIH TRADE-OV
# ════════════════════════════════════════════════════════

def izpisi_trades(trades, n=20):
    print(f"\n{'═'*80}")
    print(f"  ZADNJIH {min(n, len(trades))} TRADE-OV")
    print(f"{'═'*80}")
    header = f"  {'Datum vstopa':<14} {'Smer':<7} {'Vstop':>10} {'Izhod':>10} {'Tocke':>7} {'P&L':>8} {'Izhod':<8}"
    print(header)
    print(f"  {'-'*74}")
    for t in trades[-n:]:
        emoji = "✓" if t["pnl_pct"] > 0 else "✗"
        print(
            f"  {str(t['vstop_datum'].date()):<14} "
            f"{t['smer']:<7} "
            f"{t['vstop_cena']:>10,.0f} "
            f"{t['izhod_cena']:>10,.0f} "
            f"{t['tocke']:>+7d} "
            f"{t['pnl_pct']:>+7.1f}% "
            f"{t['izhod_vzrok']:<6} {emoji}"
        )

# ════════════════════════════════════════════════════════
#  8.  MAIN
# ════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*58)
    print("  BTC BACKTEST + MONTE CARLO  v1.0")
    print("  Signal logika: signal_tracker.py v1.2")
    print("═"*58)

    # ── Timeframe ────────────────────────────────────────
    print("\n  TIMEFRAME:")
    for k, v in TIMEFRAME_MENU.items():
        print(f"  {k}. {v['label']}")
    tf_izbira = input("\n  Izberi [1/2/3, privzeto=1]: ").strip()
    tf_cfg    = TIMEFRAME_MENU.get(tf_izbira, TIMEFRAME_MENU["1"])
    interval  = tf_cfg["interval"]
    period    = tf_cfg["period"]
    print(f"  Izbran: {tf_cfg['label']}")

    # ── Monte Carlo N ────────────────────────────────────
    n_sim_vnos = input("\n  Koliko Monte Carlo simulacij? [privzeto=200]: ").strip()
    try:
        n_sim = int(n_sim_vnos)
        n_sim = max(10, min(n_sim, 5000))
    except ValueError:
        n_sim = 200
    print(f"  MC simulacij: {n_sim}")

    # ── Kapital ──────────────────────────────────────────
    kap_vnos = input(f"\n  Začetni kapital EUR [privzeto={ZACETNI_KAPITAL:.0f}]: ").strip()
    try:
        kapital = float(kap_vnos.replace(",", "."))
        if kapital <= 0: raise ValueError
    except ValueError:
        kapital = ZACETNI_KAPITAL
    print(f"  Kapital: {kapital:.2f} EUR")

    # ── Verbose ──────────────────────────────────────────
    verb_vnos = input("\n  Prikaži vsak trade med izvajanjem? [d/N, privzeto=N]: ").strip().lower()
    verbose   = verb_vnos in ("d", "da", "y", "yes")

    # ── Podatki ──────────────────────────────────────────
    df = pridobi_btc(interval=interval, period=period)

    # ── Backtest ─────────────────────────────────────────
    print(f"\n  Izvajam backtest (prag signala: long≥+{SIGNAL_LONG_PRAG}, short≤{SIGNAL_SHORT_PRAG})...")
    trades = run_backtest(df, interval=interval, kapital=kapital, verbose=verbose)

    if not trades:
        print("\n  OPOZORILO: Ni bilo generiranih signalov v izbranem obdobju.")
        print("  Poskusi z daljšim periodom ali nižjim signalnim pragom.")
        return

    print(f"  Najdenih: {len(trades)} trade-ov")

    equity = equity_krivulja(trades, kapital)
    stats  = calc_statistike(trades, equity, kapital)

    # ── Statistike ───────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  REZULTATI  ({interval}, {kapital:.0f} EUR)")
    print(f"{'═'*58}")
    print(f"  Skupaj trades  : {stats['skupaj_trades']}  (Long: {stats['long_trades']}, Short: {stats['short_trades']})")
    print(f"  Win rate       : {stats['win_rate']:.1f}%")
    print(f"  TP / SL        : {stats['tp_trades']} / {stats['sl_trades']}")
    print(f"  Avg win        : +{stats['avg_win']:.2f}%")
    print(f"  Avg loss       : {stats['avg_loss']:.2f}%")
    print(f"  Avg R:R        : 1 : {stats['rr_avg']:.2f}")
    print(f"  Sharpe         : {stats['sharpe']:.2f}")
    print(f"  Max drawdown   : {stats['max_dd_pct']:.1f}%")
    print(f"  Skupni donos   : {stats['skupni_donos_pct']:+.1f}%")
    print(f"  Začetni kapital: {kapital:.2f} EUR")
    print(f"  Končni kapital : {stats['konc_kapital']:.2f} EUR")
    print(f"  Zaslužek       : {stats['skupni_donos_eur']:+.2f} EUR")
    print(f"{'═'*58}")

    izpisi_trades(trades, n=20)

    # ── Monte Carlo ──────────────────────────────────────
    print(f"\n  Izvajam {n_sim} Monte Carlo simulacij...")
    mc_matrix = monte_carlo(trades, n_sim=n_sim, kapital=kapital)

    final = mc_matrix[:, -1]
    print(f"  MC mediana končnega kapitala : {np.median(final):,.2f} EUR")
    print(f"  MC 5. percentil              : {np.percentile(final, 5):,.2f} EUR")
    print(f"  MC 95. percentil             : {np.percentile(final, 95):,.2f} EUR")
    print(f"  % simulacij v dobičku        : {np.sum(final > kapital)/len(final)*100:.1f}%")

    # ── Graf ─────────────────────────────────────────────
    print("\n  Rišem graf...")
    narisi_graf(trades, equity, mc_matrix, stats, kapital, interval, n_sim)


if __name__ == "__main__":
    main()
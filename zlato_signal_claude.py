"""
MULTI-TICKER SIGNAL TRACKER
Delnice | Kripto | Surovine | Indeksi | Forex

NAMESTITEV (enkrat):
    pip install yfinance numpy scipy requests
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime
import csv
import os
import time
import requests

# yfinance - zanesljivejse od raw Yahoo requests
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("  OPOZORILO: yfinance ni namesescen. Pozeni: pip install yfinance")

# ==============================================================
# TICKER KONFIGURACIJSKI SLOVAR
# ==============================================================

TICKER_CONFIG = {
    # --- PLEMENITE KOVINE ---
    "XAUUSD": {"yahoo_symbol": "GC=F",    "gold_api": "XAU", "opis": "Zlato (USD/unca)",          "min_cena": 1000,   "max_cena": 5000},
    "XAGUSD": {"yahoo_symbol": "SI=F",    "gold_api": "XAG", "opis": "Srebro (USD/unca)",          "min_cena": 5,      "max_cena": 500},
    # --- ENERGIJA ---
    "WTI":    {"yahoo_symbol": "CL=F",                       "opis": "WTI Nafta (USD/sod)",        "min_cena": 10,     "max_cena": 300},
    "BRENT":  {"yahoo_symbol": "BZ=F",                       "opis": "Brent Nafta (USD/sod)",      "min_cena": 10,     "max_cena": 300},
    "NGAS":   {"yahoo_symbol": "NG=F",                       "opis": "Naravni plin (USD/MMBtu)",   "min_cena": 0.5,    "max_cena": 50},
    # --- KRIPTO ---
    "BTCUSD": {"yahoo_symbol": "BTC-USD",                    "opis": "Bitcoin (USD)",               "min_cena": 1000,   "max_cena": 500000},
    "ETHUSD": {"yahoo_symbol": "ETH-USD",                    "opis": "Ethereum (USD)",              "min_cena": 10,     "max_cena": 50000},
    "SOLUSD": {"yahoo_symbol": "SOL-USD",                    "opis": "Solana (USD)",                "min_cena": 0.1,    "max_cena": 5000},
    "XRPUSD": {"yahoo_symbol": "XRP-USD",                    "opis": "XRP (USD)",                   "min_cena": 0.0001, "max_cena": 1000},
    # --- INDEKSI ---
    "SPX":    {"yahoo_symbol": "^GSPC",                      "opis": "S&P 500",                     "min_cena": 500,    "max_cena": 20000},
    "NDX":    {"yahoo_symbol": "^IXIC",                      "opis": "NASDAQ Composite",            "min_cena": 500,    "max_cena": 50000},
    "DAX":    {"yahoo_symbol": "^GDAXI",                     "opis": "DAX (Frankfurt)",             "min_cena": 1000,   "max_cena": 30000},
    # --- DELNICE ---
    "AAPL":   {"yahoo_symbol": "AAPL",                       "opis": "Apple Inc.",                  "min_cena": 1,      "max_cena": 10000},
    "TSLA":   {"yahoo_symbol": "TSLA",                       "opis": "Tesla Inc.",                  "min_cena": 1,      "max_cena": 10000},
    "NVDA":   {"yahoo_symbol": "NVDA",                       "opis": "NVIDIA Corp.",                "min_cena": 1,      "max_cena": 10000},
    "MSFT":   {"yahoo_symbol": "MSFT",                       "opis": "Microsoft Corp.",             "min_cena": 1,      "max_cena": 10000},
    "META":   {"yahoo_symbol": "META",                       "opis": "Meta Platforms",              "min_cena": 1,      "max_cena": 10000},
    # --- FOREX ---
    "EURUSD": {"yahoo_symbol": "EURUSD=X",                   "opis": "EUR/USD",                     "min_cena": 0.5,    "max_cena": 3.0},
    "GBPUSD": {"yahoo_symbol": "GBPUSD=X",                   "opis": "GBP/USD",                     "min_cena": 0.5,    "max_cena": 3.0},
}

TIMEFRAME_OPTIONS = {
    "1": {"interval": "15m", "period": "5d",  "opis": "15-minutni (kratkorocni intraday)"},
    "2": {"interval": "1h",  "period": "30d", "opis": "Urni (intraday/swing)"},
    "3": {"interval": "1d",  "period": "90d", "opis": "Dnevni (swing/position)"},
}

ATR_MULT_SL   = 1.5   # swing: 1.5x ATR stop
ATR_MULT_TP   = 3.0   # swing: 3.0x ATR target -> R:R = 1:2
ATR_BREAKEVEN = 1.0   # move SL to breakeven once price moves 1x ATR in profit

# --- Adaptivni ATR period ---
# V konsolidaciji uporabi krajsi period (7) -> manjsi, bolj realisticni SL/TP
# V trendu ostane standardni 14
ATR_PERIOD       = 14  # trend
ATR_PERIOD_TREND = 14
ATR_PERIOD_CONS  = 7   # konsolidacija - bolj odziven na trenutni range

# --- Konsolidacijski filter (ADX + BB width) ---
# ADX < 20  -> sideways, ne trguj
# BB width / cena < BB_WIDTH_MIN -> ozek range, ne trguj
ADX_PERIOD      = 14
ADX_CONS_THRESH = 20    # pod tem = konsolidacija
BB_PERIOD       = 20    # Bollinger Band period
BB_STD          = 2.0   # standardni odklon za BB
BB_WIDTH_MIN    = 0.02  # BB sirina / cena < 2% = konsolidacija

# Swing-specific MA periods
SMA_TREND   = 200   # long-term trend filter
SMA_ENTRY   = 50    # medium-term entry context
VOL_PERIOD  = 20    # volume average lookback
VOL_THRESH  = 1.2   # volume must be 1.2x the 20-day average

# RSI settings
RSI_PERIOD      = 14
RSI_LONG_MIN    = 40   # RSI must be above this for a long
RSI_LONG_MAX    = 65   # RSI must be below this (not overbought)
RSI_SHORT_MIN   = 55   # RSI must be below this for a short
RSI_SHORT_MAX   = 75   # RSI must be above this (not oversold)

# ==============================================================
# POMOZNE FUNKCIJE
# ==============================================================

def _get(url, timeout=10):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    return requests.get(url, headers=headers, timeout=timeout)


def izberi_ticker():
    print("\n" + "=" * 58)
    print("  RAZPOLOZLJIVI TICKERJI")
    print("=" * 58)
    tickerji = list(TICKER_CONFIG.keys())
    for i, t in enumerate(tickerji, 1):
        print(f"  {i:2}. {t:<10}  {TICKER_CONFIG[t]['opis']}")
    print()
    print("  Vnesi stevilko, simbol iz seznama, ali lasten Yahoo simbol")
    print("  (npr. MSFT, DOGE-USD, NG=F, ^VIX, AMZN ...)")
    print()
    vnos = input("  Tvoj izbor: ").strip().upper()

    if vnos.isdigit() and 1 <= int(vnos) <= len(tickerji):
        ticker = tickerji[int(vnos) - 1]
        return ticker, TICKER_CONFIG[ticker]

    if vnos in TICKER_CONFIG:
        return vnos, TICKER_CONFIG[vnos]

    print(f"\n  '{vnos}' ni v seznamu - poskusam direktno kot Yahoo simbol...")
    cfg = {
        "yahoo_symbol": vnos,
        "opis":         vnos,
        "min_cena":     0.000001,
        "max_cena":     100_000_000,
    }
    return vnos, cfg


def izberi_timeframe():
    print()
    print("  TIMEFRAME:")
    for k, v in TIMEFRAME_OPTIONS.items():
        print(f"  {k}. {v['opis']}")
    print()
    izbira = input("  Izberi [1/2/3, privzeto=1]: ").strip()
    tf = TIMEFRAME_OPTIONS.get(izbira, TIMEFRAME_OPTIONS["1"])
    print(f"  Izbran: {tf['opis']}")
    return tf["interval"], tf["period"]


# ==============================================================
# 1. PRIDOBITEV PODATKOV
# ==============================================================

def pridobi_gold_api(simbol):
    """Real-time spot cena za plemenite kovine."""
    try:
        r = _get(f"https://api.gold-api.com/price/{simbol}")
        S = float(r.json()["price"])
        print(f"  [gold-api] Real-time spot: {S:.4g} USD")
        return S
    except Exception as e:
        print(f"  [gold-api] NAPAKA: {e}")
        return None


def pridobi_yfinance(yahoo_symbol, interval, period):
    """
    Pridobi OHLC podatke prek yfinance knjiznice.
    Knjiznica sama upravlja cookie/crumb session z Yahoo Finance.
    """
    if not YF_AVAILABLE:
        return None
    try:
        ticker_obj = yf.Ticker(yahoo_symbol)
        df = ticker_obj.history(interval=interval, period=period)

        if df is None or len(df) < ATR_PERIOD + 2:
            n = 0 if df is None else len(df)
            print(f"  [yfinance] Premalo podatkov ({n} sveck)")
            return None

        cene   = list(df["Close"].dropna())
        high   = list(df["High"].dropna())
        low    = list(df["Low"].dropna())
        volume = list(df["Volume"]) if "Volume" in df.columns else []

        n = min(len(cene), len(high), len(low))
        cene, high, low = cene[-n:], high[-n:], low[-n:]
        if volume:
            volume = volume[-n:]

        if len(cene) < ATR_PERIOD + 2:
            print(f"  [yfinance] Premalo podatkov po cistenju ({len(cene)} sveck)")
            return None

        donosi = np.diff(np.log(cene))
        sigma  = float(np.std(donosi) * np.sqrt(252))
        sigma  = min(max(sigma, 0.03), 3.0)

        # Spot cena - fast_info je hitrejsi od .info
        spot = None
        try:
            spot = ticker_obj.fast_info.last_price
        except Exception:
            spot = cene[-1]

        print(
            f"  [yfinance] {yahoo_symbol} ({interval}, {len(cene)} sveck): "
            f"spot = {spot:.6g} USD  |  vol = {sigma*100:.1f}%"
        )
        return sigma, cene, high, low, volume, spot

    except Exception as e:
        print(f"  [yfinance] NAPAKA: {e}")
        return None


def pridobi_yahoo_raw(yahoo_symbol, interval, range_str):
    """Fallback: raw Yahoo Finance v8 chart API."""
    try:
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            f"?interval={interval}&range={range_str}"
        )
        r    = _get(url)
        data = r.json()

        if "chart" not in data or not data["chart"].get("result"):
            print(f"  [yahoo-raw] Prazen odgovor (morda rate-limit ali neveljaven simbol)")
            return None

        result = data["chart"]["result"][0]
        meta   = result.get("meta", {})
        quote  = result["indicators"]["quote"][0]

        closes = quote.get("close") or []
        highs  = quote.get("high")  or []
        lows   = quote.get("low")   or []

        combined = [
            (c, h, l)
            for c, h, l in zip(closes, highs, lows)
            if c is not None and h is not None and l is not None
        ]

        if len(combined) < ATR_PERIOD + 2:
            print(f"  [yahoo-raw] Premalo sveck ({len(combined)})")
            return None

        cene   = [x[0] for x in combined]
        high   = [x[1] for x in combined]
        low    = [x[2] for x in combined]
        volume = list(quote.get("volume") or [])
        # align volume length
        if len(volume) != len(cene):
            volume = []

        donosi = np.diff(np.log(cene))
        sigma  = float(np.std(donosi) * np.sqrt(252))
        sigma  = min(max(sigma, 0.03), 3.0)

        spot = meta.get("regularMarketPrice") or cene[-1]

        print(
            f"  [yahoo-raw] {yahoo_symbol} ({interval}, {len(cene)} sveck): "
            f"spot = {spot:.6g} USD  |  vol = {sigma*100:.1f}%"
        )
        return sigma, cene, high, low, volume, spot

    except Exception as e:
        print(f"  [yahoo-raw] NAPAKA: {e}")
        return None


def pridobi_vse(ticker, cfg, interval, period):
    print(f"\n  Pridobivam podatke za {ticker} ({cfg['opis']})...\n")

    S        = None
    hist_ret = None

    # 1. Real-time spot za plemenite kovine (gold-api)
    if "gold_api" in cfg:
        S = pridobi_gold_api(cfg["gold_api"])

    # 2. yfinance - primarna metoda
    hist_ret = pridobi_yfinance(cfg["yahoo_symbol"], interval, period)

    # 3. Raw Yahoo API fallback
    if hist_ret is None:
        print("  Poskusam fallback (raw Yahoo API)...")
        time.sleep(1.5)
        range_map = {"5d": "5d", "30d": "1mo", "90d": "3mo"}
        range_str = range_map.get(period, period)
        hist_ret  = pridobi_yahoo_raw(cfg["yahoo_symbol"], interval, range_str)

    print()

    if hist_ret:
        sigma, cene_hist, high_hist, low_hist, vol_hist, spot_yf = hist_ret
        if S is None:
            S = spot_yf or cene_hist[-1]
    else:
        sigma     = 0.15
        cene_hist = []
        high_hist = []
        low_hist  = []
        vol_hist  = []
        print("  OPOZORILO: Zgodovinski podatki niso na voljo. Privzeta vol = 15%")

    avto = S is not None
    return S, sigma, cene_hist, high_hist, low_hist, vol_hist, avto


# ==============================================================
# 2. ROCNI VNOS
# ==============================================================

def vnesi_rocno(ticker, cfg):
    print(f"\n  Vnesi trenutno ceno za {ticker} ({cfg['opis']}).")
    min_c = cfg.get("min_cena", 0.000001)
    max_c = cfg.get("max_cena", 100_000_000)
    while True:
        try:
            vnos = input("  Cena: ").strip().replace(",", ".")
            S    = float(vnos)
            if min_c <= S <= max_c:
                break
            print(f"  Vrednost mora biti med {min_c} in {max_c}.")
        except ValueError:
            print("  Napaka - vnesi samo stevilko.")
    return S


# ==============================================================
# 3. BLACK-SCHOLES
# ==============================================================

def black_scholes(S, K, T, r, sigma, tip="call"):
    if T <= 0:
        return (max(S - K, 0), 0, 0, 0, 0) if tip == "call" else (max(K - S, 0), 0, 0, 0, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tip == "call":
        cena  = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        cena  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if tip == "call" else -d2)
    ) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return cena, delta, gamma, theta, vega


# ==============================================================
# 4. ATR (Wilder)
# ==============================================================

def izracunaj_atr(cene, high, low, period=14):
    if not cene or len(cene) < period + 1:
        return None
    tr = []
    for i in range(1, len(cene)):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - cene[i - 1])
        lpc = abs(low[i]  - cene[i - 1])
        tr.append(max(hl, hpc, lpc))
    atr = float(np.mean(tr[:period]))
    for x in tr[period:]:
        atr = (atr * (period - 1) + x) / period
    return atr


# ==============================================================
# 5. INDIKATORJI
# ==============================================================

def izracunaj_rsi(cene, period=14):
    """Wilder RSI."""
    if len(cene) < period + 1:
        return None
    diffs  = np.diff(cene)
    gains  = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_g  = float(np.mean(gains[:period]))
    avg_l  = float(np.mean(losses[:period]))
    for g, l in zip(gains[period:], losses[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        return 100.0
    rs  = avg_g / avg_l
    return round(100 - (100 / (1 + rs)), 2)


def izracunaj_macd(cene, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    if len(cene) < slow + signal:
        return None, None, None

    def ema(data, n):
        k   = 2 / (n + 1)
        val = float(np.mean(data[:n]))
        for x in data[n:]:
            val = x * k + val * (1 - k)
        return val

    def ema_series(data, n):
        k      = 2 / (n + 1)
        result = [float(np.mean(data[:n]))]
        for x in data[n:]:
            result.append(x * k + result[-1] * (1 - k))
        return result

    fast_ema = ema_series(cene, fast)
    # align slow ema to same length as fast_ema
    slow_ema = ema_series(cene, slow)
    # macd line starts where both EMAs exist
    offset   = slow - fast
    macd_line = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]
    if len(macd_line) < signal:
        return None, None, None
    sig_line  = ema_series(macd_line, signal)
    macd_val  = macd_line[-1]
    sig_val   = sig_line[-1]
    hist_val  = macd_val - sig_val
    # crossover: macd crossed above signal in the last candle?
    crossed_up   = (macd_line[-1] > sig_line[-1]) and (macd_line[-2] <= sig_line[len(sig_line)-2]) if len(macd_line) > 1 and len(sig_line) > 1 else False
    crossed_down = (macd_line[-1] < sig_line[-1]) and (macd_line[-2] >= sig_line[len(sig_line)-2]) if len(macd_line) > 1 and len(sig_line) > 1 else False
    return round(macd_val, 6), round(sig_val, 6), round(hist_val, 6), crossed_up, crossed_down


def izracunaj_volume_ratio(vol_hist, period=20):
    """Ratio of last volume vs N-period average. Returns None if no volume data."""
    clean = [v for v in vol_hist if v is not None and v > 0]
    if len(clean) < period + 1:
        return None
    avg = float(np.mean(clean[-period - 1:-1]))  # average of previous N candles
    if avg == 0:
        return None
    return round(clean[-1] / avg, 2)



def izracunaj_adx(cene, high, low, period=14):
    """Wilder ADX - meri mocnost trenda (ne smeri).
    ADX < 20 = sibak trend / konsolidacija
    ADX > 25 = jasen trend
    """
    if len(cene) < period * 2 + 1:
        return None
    # True Range
    tr_list = []
    dm_plus  = []
    dm_minus = []
    for i in range(1, len(cene)):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - cene[i-1])
        lpc = abs(low[i]  - cene[i-1])
        tr_list.append(max(hl, hpc, lpc))
        up   = high[i] - high[i-1]
        down = low[i-1] - low[i]
        dm_plus.append(up   if up > down and up > 0   else 0.0)
        dm_minus.append(down if down > up and down > 0 else 0.0)

    # Wilder smoothing
    def wilder_smooth(data, n):
        val = float(sum(data[:n]))
        result = [val]
        for x in data[n:]:
            val = val - val/n + x
            result.append(val)
        return result

    atr14   = wilder_smooth(tr_list, period)
    dmp14   = wilder_smooth(dm_plus,  period)
    dmm14   = wilder_smooth(dm_minus, period)

    di_plus  = [100 * p / a if a > 0 else 0 for p, a in zip(dmp14, atr14)]
    di_minus = [100 * m / a if a > 0 else 0 for m, a in zip(dmm14, atr14)]

    dx_list = []
    for p, m in zip(di_plus, di_minus):
        s = p + m
        dx_list.append(100 * abs(p - m) / s if s > 0 else 0)

    if len(dx_list) < period:
        return None

    adx = float(sum(dx_list[:period]) / period)
    for x in dx_list[period:]:
        adx = (adx * (period - 1) + x) / period

    return round(adx, 2)


def izracunaj_bb_width(cene, period=20, std_mult=2.0):
    """Bollinger Band sirina relativno na ceno.
    Nizka sirina = konsolidacija, visoka sirina = trend/volatilnost.
    Vrne: (bb_width_pct, zgornji_band, spodnji_band, srednji_band)
    """
    if len(cene) < period:
        return None, None, None, None
    recent = cene[-period:]
    mid    = float(np.mean(recent))
    std    = float(np.std(recent, ddof=1))
    upper  = mid + std_mult * std
    lower  = mid - std_mult * std
    width_pct = (upper - lower) / mid if mid > 0 else 0
    return round(width_pct, 6), round(upper, 6), round(lower, 6), round(mid, 6)


def zazna_konsolidacijo(cene, high, low, S):
    """Vrne (je_konsolidacija, adx, bb_width_pct, opis).
    je_konsolidacija = True ce sta oba pogoja izpolnjena:
      - ADX < ADX_CONS_THRESH  (sibak trend)
      - BB sirina < BB_WIDTH_MIN (ozek range)
    """
    adx = izracunaj_adx(cene, high, low, period=ADX_PERIOD)
    bb_width, bb_upper, bb_lower, bb_mid = izracunaj_bb_width(cene, period=BB_PERIOD, std_mult=BB_STD)

    adx_sideways = (adx is not None) and (adx < ADX_CONS_THRESH)
    bb_sideways  = (bb_width is not None) and (bb_width < BB_WIDTH_MIN)

    # Konsolidacija = oba pogoja
    je_kons = adx_sideways and bb_sideways

    # Ce samo en pogoj: sibka konsolidacija
    sibka_kons = adx_sideways or bb_sideways

    if je_kons:
        opis = f"KONSOLIDACIJA (ADX {adx}, BB-width {bb_width*100:.2f}%)"
    elif sibka_kons:
        opis = f"SIBKA KONSOLIDACIJA (ADX {adx}, BB-width {bb_width*100:.2f}% ce ni None)"
    else:
        opis = f"TREND (ADX {adx}, BB-width {bb_width*100:.2f}% ce ni None)"

    return je_kons, sibka_kons, adx, bb_width, bb_upper, bb_lower, opis

# ==============================================================
# 6. SIGNAL LOGIKA (swing-optimised)
# ==============================================================

def izracunaj_signal(S, sigma, cene_hist, high_hist, low_hist, vol_hist, interval):
    N = len(cene_hist)

    # --- Konsolidacijski filter + adaptivni ATR ---
    je_kons, sibka_kons, adx_val, bb_width, bb_upper, bb_lower, kons_opis = (
        zazna_konsolidacijo(cene_hist, high_hist, low_hist, S)
        if N >= ADX_PERIOD * 2 + 1
        else (False, False, None, None, None, None, "Premalo podatkov za ADX/BB")
    )
    # Adaptivni ATR period: v konsolidaciji uporabi krajsega
    atr_period_used = ATR_PERIOD_CONS if je_kons else ATR_PERIOD_TREND

    # --- Moving averages ---
    sma200 = float(np.mean(cene_hist[-200:])) if N >= 200 else float(np.mean(cene_hist)) if N > 0 else S
    sma50  = float(np.mean(cene_hist[-50:]))  if N >= 50  else float(np.mean(cene_hist)) if N > 0 else S
    sma20  = float(np.mean(cene_hist[-20:]))  if N >= 20  else float(np.mean(cene_hist)) if N > 0 else S

    # Long-term trend: price vs SMA200
    trend_gor = S > sma200
    trend_dol = S < sma200

    # Entry context: SMA50 slope (positive = rising)
    sma50_slope = (cene_hist[-1] - cene_hist[-6]) / cene_hist[-6] * 100 if N >= 6 else 0.0

    # --- RSI ---
    rsi = izracunaj_rsi(cene_hist, RSI_PERIOD) if N >= RSI_PERIOD + 1 else None

    # RSI zone checks for long/short
    rsi_long_ok  = (rsi is not None) and (RSI_LONG_MIN  <= rsi <= RSI_LONG_MAX)
    rsi_short_ok = (rsi is not None) and (55 <= rsi <= 75)

    # --- MACD ---
    macd_ret = izracunaj_macd(cene_hist) if N >= 35 else (None, None, None, False, False)
    if macd_ret[0] is not None:
        macd_val, sig_val, hist_val, macd_crossed_up, macd_crossed_down = macd_ret
    else:
        macd_val = sig_val = hist_val = None
        macd_crossed_up = macd_crossed_down = False

    macd_bullish = (macd_val is not None) and (macd_val > sig_val)
    macd_bearish = (macd_val is not None) and (macd_val < sig_val)

    # --- Volume ---
    vol_ratio = izracunaj_volume_ratio(vol_hist, VOL_PERIOD)
    vol_ok    = (vol_ratio is None) or (vol_ratio >= VOL_THRESH)  # pass if no data

    # --- ATR (adaptiven period) ---
    atr_raw = izracunaj_atr(cene_hist, high_hist, low_hist, period=atr_period_used)
    if atr_raw is not None:
        atr, atr_vir = atr_raw, interval
    else:
        atr     = S * sigma * np.sqrt(1 / 365) * 0.5
        atr_vir = "vol-fallback"
    atr_period_label = f"{atr_period_used}p{'(kons)' if je_kons else ''}"
    print(f"  ATR ({atr_vir}, {atr_period_label}): {atr:.6g} USD")

    # --- Scoring (swing-optimised) ---
    # Max possible: +6 (trend+2, RSI+2, MACD+1, volume+1)
    # Min possible: -5 (trend-2, RSI-2, MACD-1)
    tocke = 0
    razlogi = []

    # 1. Trend filter — SMA200 (+/-2, most important)
    if trend_gor:
        tocke += 2
        razlogi.append(f"Cena > SMA200 ({fmt(sma200)}) +2")
    elif trend_dol:
        tocke -= 2
        razlogi.append(f"Cena < SMA200 ({fmt(sma200)}) -2")

    # 2. RSI momentum (+/-2)
    if rsi is not None:
        if rsi_long_ok:
            tocke += 2
            razlogi.append(f"RSI {rsi} v bull coni ({RSI_LONG_MIN}-{RSI_LONG_MAX}) +2")
        elif rsi < RSI_LONG_MIN and rsi > 20:
            tocke += 1
            razlogi.append(f"RSI {rsi} nizek — morebitna priložnost +1")
        elif rsi > RSI_LONG_MAX:
            tocke -= 2
            razlogi.append(f"RSI {rsi} precenjen (>{RSI_LONG_MAX}) -2")
        elif rsi <= 20:
            tocke -= 1
            razlogi.append(f"RSI {rsi} ekstremno nizek -1")
        else:
            razlogi.append(f"RSI {rsi} (nevtralen)")
    else:
        razlogi.append("RSI: premalo podatkov")

    # 3. MACD (+/-1)
    if macd_val is not None:
        if macd_crossed_up:
            tocke += 2
            razlogi.append(f"MACD krizanje navzgor (crossover) +2")
        elif macd_bullish:
            tocke += 1
            razlogi.append(f"MACD > signal ({macd_val:.4f} > {sig_val:.4f}) +1")
        elif macd_crossed_down:
            tocke -= 2
            razlogi.append(f"MACD krizanje navzdol (crossover) -2")
        elif macd_bearish:
            tocke -= 1
            razlogi.append(f"MACD < signal ({macd_val:.4f} < {sig_val:.4f}) -1")
    else:
        razlogi.append("MACD: premalo podatkov")

    # 4. Volume confirmation (+1, no penalty — absence = neutral)
    if vol_ratio is not None:
        if vol_ratio >= VOL_THRESH:
            tocke += 1
            razlogi.append(f"Volumen {vol_ratio:.2f}x povprecje +1")
        else:
            razlogi.append(f"Volumen {vol_ratio:.2f}x povprecje (ni potrditve)")
    else:
        razlogi.append("Volumen: ni podatkov (kripto/forex)")

    # 5. Volatility penalty (high vol = risky entry)
    if sigma > 0.60:
        tocke -= 1
        razlogi.append(f"Vol {sigma*100:.0f}% zelo visoka -1")
    elif sigma < 0.08:
        tocke += 1
        razlogi.append(f"Vol {sigma*100:.0f}% nizka (stabilen trend) +1")

    # 6. Konsolidacijski filter (-2 ce je trg sideways)
    razlogi.append(f"Trg: {kons_opis}")
    if je_kons:
        tocke -= 2
        razlogi.append("KONSOLIDACIJA zaznana -> -2, SL/TP zmanjsana")
    elif sibka_kons:
        tocke -= 1
        razlogi.append("Sibka konsolidacija -> -1, previdnost")

    # --- Decision ---
    if tocke >= 4:
        odlocitev, jakost, emoji = "KUPI",              "mocen signal",           "ZELENA"
    elif tocke >= 2:
        odlocitev, jakost, emoji = "KUPI",              "sibek signal",           "RUMENA"
    elif tocke <= -4:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "mocen signal",           "RDECA"
    elif tocke <= -2:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "sibek signal - pocakaj", "ORANZNA"
    else:
        odlocitev, jakost, emoji = "CAKAJ",             "trg je nevtralen",       "BELA"

    # Block entry if volume not confirmed on a buy signal
    if odlocitev == "KUPI" and vol_ratio is not None and vol_ratio < VOL_THRESH:
        jakost = jakost + " (nizek volumen!)"

    # --- SL / TP / Trailing breakeven ---
    if odlocitev == "KUPI":
        stop_loss      = S - atr * ATR_MULT_SL
        take_profit    = S + atr * ATR_MULT_TP
        breakeven_trig = S + atr * ATR_BREAKEVEN   # move SL to entry once hit
    elif odlocitev == "PRODAJ / NE KUPUJ":
        stop_loss      = S + atr * ATR_MULT_SL
        take_profit    = S - atr * ATR_MULT_TP
        breakeven_trig = S - atr * ATR_BREAKEVEN
    else:
        stop_loss      = S - atr * ATR_MULT_SL
        take_profit    = S + atr * ATR_MULT_TP
        breakeven_trig = S + atr * ATR_BREAKEVEN

    return {
        "S": S, "sigma": sigma,
        "sma200": sma200, "sma50": sma50, "sma20": sma20,
        "sma50_slope": sma50_slope,
        "rsi": rsi,
        "macd_val": macd_val, "sig_val": sig_val, "hist_val": hist_val,
        "macd_crossed_up": macd_crossed_up, "macd_crossed_down": macd_crossed_down,
        "vol_ratio": vol_ratio,
        "je_kons": je_kons, "sibka_kons": sibka_kons,
        "adx_val": adx_val, "bb_width": bb_width,
        "bb_upper": bb_upper, "bb_lower": bb_lower,
        "atr_period_used": atr_period_used,
        "tocke": tocke, "razlogi": razlogi,
        "odlocitev": odlocitev, "jakost": jakost, "emoji": emoji,
        "atr": atr, "atr_vir": atr_vir,
        "stop_loss": stop_loss, "take_profit": take_profit,
        "breakeven_trig": breakeven_trig,
    }


# ==============================================================
# 7. IZPIS
# ==============================================================

def fmt(x):
    if x is None:        return "N/A"
    if abs(x) < 0.0001:  return f"{x:.8f}"
    if abs(x) < 0.01:    return f"{x:.6f}"
    if abs(x) < 1:       return f"{x:.4f}"
    if abs(x) < 100:     return f"{x:.2f}"
    return f"{x:,.2f}"


EMOJI_MAP = {"ZELENA": "🟢", "RUMENA": "🟡", "RDECA": "🔴", "ORANZNA": "🟠", "BELA": "⚪"}


def izpisi(r, ticker, cfg, interval):
    cas   = datetime.now().strftime("%d.%m.%Y %H:%M")
    sep   = "=" * 62
    sep2  = "-" * 62
    ikona = EMOJI_MAP.get(r["emoji"], "")

    print(f"\n{sep}")
    print(f"  {ticker} — SWING SIGNAL  [{interval} TF]")
    print(f"  {cfg['opis']}")
    print(f"  {cas}")
    print(sep)

    # --- Cena & trend ---
    trend_str = "UPTREND ▲" if r["S"] > r["sma200"] else "DOWNTREND ▼"
    print(f"\n  Cena (USD)        : {fmt(r['S'])}")
    print(f"  Volatilnost       : {r['sigma']*100:.1f}%")
    print(f"\n  --- TREND (SMA) ---")
    print(f"  SMA 200           : {fmt(r['sma200'])}  →  {trend_str}")
    print(f"  SMA 50            : {fmt(r['sma50'])}  (naklon: {r['sma50_slope']:+.2f}%)")
    print(f"  SMA 20            : {fmt(r['sma20'])}")

    # --- Konsolidacijski filter ---
    print(f"\n  --- TRZNA STRUKTURA ---")
    kons_ikona = "⚠️  SIDEWAYS" if r["je_kons"] else ("~ sibka kons." if r["sibka_kons"] else "OK trend")
    adx_str = f"{r['adx_val']:.1f}" if r["adx_val"] is not None else "N/A"
    bb_str  = f"{r['bb_width']*100:.2f}%" if r["bb_width"] is not None else "N/A"
    print(f"  ADX ({ADX_PERIOD})          : {adx_str}  (prag konsolidacije: <{ADX_CONS_THRESH})")
    print(f"  BB sirina         : {bb_str}  (prag: <{BB_WIDTH_MIN*100:.0f}%)")
    print(f"  Struktura         : {kons_ikona}")
    if r["je_kons"]:
        print(f"  !! SIDEWAYS TRG — signal oslabljen, ATR period={ATR_PERIOD_CONS} (adaptiven)")
        if r["bb_upper"] is not None:
            print(f"  BB zgornji        : {fmt(r['bb_upper'])}")
            print(f"  BB spodnji        : {fmt(r['bb_lower'])}")
    else:
        print(f"  ATR period        : {r['atr_period_used']} (standardni trend)")

    # --- RSI ---
    print(f"\n  --- MOMENTUM ---")
    rsi_str = f"{r['rsi']}" if r["rsi"] is not None else "N/A"
    if r["rsi"] is not None:
        if r["rsi"] > 70:   rsi_str += "  (precenjen)"
        elif r["rsi"] < 30: rsi_str += "  (podcenjen)"
        elif RSI_LONG_MIN <= r["rsi"] <= RSI_LONG_MAX:
            rsi_str += "  ✓ bull cona"
    print(f"  RSI ({RSI_PERIOD})          : {rsi_str}")

    # --- MACD ---
    if r["macd_val"] is not None:
        cross_str = ""
        if r["macd_crossed_up"]:   cross_str = "  ⬆ CROSSOVER NAVZGOR"
        elif r["macd_crossed_down"]: cross_str = "  ⬇ CROSSOVER NAVZDOL"
        print(f"  MACD              : {r['macd_val']:.5f}  |  Signal: {r['sig_val']:.5f}{cross_str}")
        print(f"  MACD histogram    : {r['hist_val']:+.5f}")
    else:
        print(f"  MACD              : N/A (premalo podatkov)")

    # --- Volume ---
    print(f"\n  --- VOLUMEN ---")
    if r["vol_ratio"] is not None:
        vol_ikona = "✓" if r["vol_ratio"] >= VOL_THRESH else "✗"
        print(f"  Vol / avg(20)     : {r['vol_ratio']:.2f}x  {vol_ikona}  (prag: {VOL_THRESH}x)")
    else:
        print(f"  Vol / avg(20)     : N/A (kripto/forex nima vol. podatkov)")

    # --- Signal ---
    print(f"\n  --- SIGNAL ---")
    print(f"  Tocke             : {r['tocke']:+d}")
    print(f"  Signal            : {ikona}  {r['odlocitev']}")
    print(f"  Jakost            : {r['jakost']}")

    # --- Razlogi ---
    print(f"\n  Razlogi:")
    for raz in r["razlogi"]:
        print(f"    • {raz}")

    # --- ATR & SL/TP ---
    print(f"\n  --- ATR & NIVOJI ---")
    print(f"  ATR ({r['atr_vir']}, {r['atr_period_used']}p) : {fmt(r['atr'])} USD")
    print(f"  SL faktor         : {ATR_MULT_SL}x ATR = {fmt(r['atr'] * ATR_MULT_SL)} USD")
    print(f"  TP faktor         : {ATR_MULT_TP}x ATR = {fmt(r['atr'] * ATR_MULT_TP)} USD")
    print(f"  Breakeven trigger : {ATR_BREAKEVEN}x ATR v profit = premakni SL na vhod")

    if r["stop_loss"] is not None:
        rr        = abs(r["take_profit"] - r["S"]) / abs(r["stop_loss"] - r["S"])
        sl_usd    = abs(r["stop_loss"]   - r["S"])
        tp_usd    = abs(r["take_profit"] - r["S"])
        be_usd    = abs(r["breakeven_trig"] - r["S"])

        # Tiki: broker tick size je obicajno 0.01 USD za vecino instrumentov
        # (BTC, XAU, WTI, delnice). Za forex (EUR/USD) je 0.00001.
        S = r["S"]
        if S < 5:           tick = 0.00001   # forex (EUR/USD, GBP/USD)
        elif S < 50:        tick = 0.0001    # nizki kripto / penny
        else:               tick = 0.01      # BTC, XAU, delnice, nafta

        sl_tiki = round(sl_usd / tick)
        tp_tiki = round(tp_usd / tick)
        be_tiki = round(be_usd / tick)

        print(f"\n{sep2}")
        print(f"  {ticker}  —  TRADINGVIEW / BROKER")
        print(sep2)
        print(f"  Smer              : {r['odlocitev']}")
        print(f"  Tocke             : {r['tocke']:+d}")
        print(f"  Signal            : {ikona}  {r['odlocitev']}")
        print(f"  Vstop (close)     : {fmt(r['S'])}")
        print(f"  Stop Loss         : {fmt(r['stop_loss'])}  ({sl_usd:.2f} USD  |  {sl_tiki:,} tikov)")
        print(f"  Take Profit       : {fmt(r['take_profit'])}  ({tp_usd:.2f} USD  |  {tp_tiki:,} tikov)")
        print(f"  Breakeven trigger : {fmt(r['breakeven_trig'])}  ({be_usd:.2f} USD  |  {be_tiki:,} tikov od vstopa)")
        print(f"  R:R razmerje      : 1 : {rr:.2f}")
        print(f"  Tick size (privzet): {tick} USD/tick")
        print(sep2)
        print(f"\n  >> Vnesi v broker/TradingView:")
        print(f"     Stop Loss   = {sl_tiki:,} tikov")
        print(f"     Take Profit = {tp_tiki:,} tikov")
        print(f"     (breakeven ko je trade +{be_tiki:,} tikov v profit)")
        print(sep2)


# ==============================================================
# 8. CSV
# ==============================================================

#def shrani_csv(r, ticker, interval):
#    ime     = "zgodovina_signalov.csv"
#    obstaja = os.path.exists(ime)
#    with open(ime, "a", newline="", encoding="utf-8") as f:
#        w = csv.writer(f)
#        if not obstaja:
#            w.writerow([
#                "datum", "ticker", "interval", "cena_usd", "volatilnost", "je_kons", "adx", "bb_width",
#                "sma200", "sma50", "sma20", "sma50_slope",
#                "rsi", "macd", "macd_signal", "macd_hist",
#                "vol_ratio",
#                "tocke", "signal", "jakost",
#                "atr", "atr_vir", "sl_faktor", "tp_faktor",
#                "stop_loss", "take_profit", "breakeven_trig",
#            ])
#       w.writerow([
#            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#            ticker, interval,
#            r["S"], round(r["sigma"], 4), r["je_kons"], r["adx_val"], r["bb_width"],
#            round(r["sma200"], 6), round(r["sma50"], 6), round(r["sma20"], 6),
#            round(r["sma50_slope"], 4),
#            r["rsi"],
#            r["macd_val"], r["sig_val"], r["hist_val"],
#            r["vol_ratio"],
#            r["tocke"], r["odlocitev"], r["jakost"],
#            round(r["atr"], 8), r["atr_vir"],
#            ATR_MULT_SL, ATR_MULT_TP,
#            r["stop_loss"], r["take_profit"], r["breakeven_trig"],
#        ])
#    print(f"\n  Signal shranjen -> {ime}")


# ==============================================================
# 8. MAIN
# ==============================================================

def main():
    print("\n" + "=" * 58)
    print("  MULTI-TICKER SIGNAL TRACKER")
    print("  Delnice | Kripto | Surovine | Indeksi | Forex")
    if YF_AVAILABLE:
        import yfinance as yf
        print(f"  Podatki: yfinance {yf.__version__} + gold-api.com (kovine)")
    else:
        print("  OPOZORILO: yfinance NI namesescen!")
        print("  Namesti z: pip install yfinance")
        print("  Fallback: raw Yahoo API (manj zanesljiv)")
    print("=" * 58)

    while True:
        ticker, cfg      = izberi_ticker()
        interval, period = izberi_timeframe()

        S, sigma, cene_hist, high_hist, low_hist, vol_hist, avto = pridobi_vse(
            ticker, cfg, interval, period
        )

        if not avto or not S:
            print("  Avtomatski prenos ni uspel.")
            S = vnesi_rocno(ticker, cfg)

        rezultat = izracunaj_signal(S, sigma, cene_hist, high_hist, low_hist, vol_hist, interval)
        izpisi(rezultat, ticker, cfg, interval)
        #shrani_csv(rezultat, ticker, interval)

        print()
        nadaljevanje = input("  Analiziraj drug ticker? [d/n]: ").strip().lower()
        if nadaljevanje not in ("d", "da", "y", "yes"):
            print("\n  Konec. Na svidenje!\n")
            break


if __name__ == "__main__":
    main()
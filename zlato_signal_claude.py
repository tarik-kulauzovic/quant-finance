import numpy as np
from scipy.stats import norm
from datetime import datetime
import csv
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}


def _get(url, timeout=8):
    import requests
    return requests.get(url, headers=HEADERS, timeout=timeout)


# ══════════════════════════════════════════════════════════════
# 1. PRIDOBITEV PODATKOV  (vse v USD/unca)
# ══════════════════════════════════════════════════════════════

def pridobi_spot_usd():
    # Poskusi Yahoo Finance GC=F (futures, bližje TradingView)
    try:
        r = _get(
            "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            "?interval=1m&range=1d"
        )
        closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        cena = [x for x in closes if x is not None][-1]
        print(f"  ✓ Yahoo GC=F spot: {cena:.2f} USD/unca")
        return float(cena)
    except Exception as e:
        print(f"  ✗ Yahoo GC=F ni uspel: {e}")

    # Fallback: gold-api.com
    try:
        r = _get("https://api.gold-api.com/price/XAU")
        cena = float(r.json()["price"])
        print(f"  ✓ gold-api.com: {cena:.2f} USD/unca (spot, možen delay)")
        return cena
    except Exception as e:
        print(f"  ✗ gold-api.com ni uspel: {e}")
        return None


def pridobi_zgodovinske_usd():
    """
    30-dnevne dnevne closing cene GC=F (USD/unca) z Yahooja.
    Vrne (sigma, cene) ali None.
    """
    try:
        r = _get(
            "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            "?interval=1d&range=30d"
        )
        closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        cene = [x for x in closes if x is not None][-20:]

        if len(cene) < 5:
            print("  ✗ Yahoo GC=F: premalo podatkov")
            return None

        donosi = np.diff(np.log(cene))
        sigma = float(np.std(donosi) * np.sqrt(252))
        sigma = min(max(sigma, 0.10), 0.22)

        print(f"  ✓ Yahoo GC=F: {len(cene)} dni, "
              f"zadnja = {cene[-1]:.2f} USD/unca, "
              f"vol = {sigma*100:.1f}%")
        return sigma, cene
    except Exception as e:
        print(f"  ✗ Yahoo GC=F ni uspel: {e}")
        return None


def pridobi_vse():
    print("\n  Pridobivam podatke...\n")

    S        = pridobi_spot_usd()
    hist_ret = pridobi_zgodovinske_usd()

    print()

    avto = S is not None

    if hist_ret:
        sigma, cene_hist = hist_ret
    else:
        sigma     = 0.15
        cene_hist = []
        print("  ℹ Privzeta volatilnost: 15%")

    return S, sigma, cene_hist, avto


# ══════════════════════════════════════════════════════════════
# 2. ROCNI VNOS
# ══════════════════════════════════════════════════════════════

def vnesi_rocno():
    print()
    print("  Kje najdes ceno XAUUSD:")
    print("  -> TradingView: XAUUSD")
    print("  -> Pepperstone / Forex.com: Gold Spot / U.S. Dollar")
    print()

    while True:
        try:
            vnos = input("  Vpisi trenutno ceno [USD/unca]: ").strip().replace(",", ".")
            S = float(vnos)
            if 1500 < S < 5000:
                break
            print("  Vrednost mora biti med 1500 in 5000.")
        except ValueError:
            print("  Napaka - vnesi samo stevilko")

    return S


# ══════════════════════════════════════════════════════════════
# 3. BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, tip="call"):
    if T <= 0:
        return (max(S-K, 0) if tip == "call" else max(K-S, 0)), 0, 0, 0, 0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if tip == "call":
        cena  = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        cena  = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2 if tip=="call" else -d2)) / 365
    vega  = S*norm.pdf(d1)*np.sqrt(T) / 100

    return cena, delta, gamma, theta, vega


# ══════════════════════════════════════════════════════════════
# 4. ATR IZRACUN  (Wilder's smoothed, iz closing cen)
# ══════════════════════════════════════════════════════════════

def izracunaj_atr(cene, period=14):
    """
    Aproksimacija ATR iz dnevnih closing cen (brez high/low).
    True range = |close[i] - close[i-1]|
    """
    if not cene or len(cene) < period + 1:
        return None

    tr = [abs(cene[i] - cene[i-1]) for i in range(1, len(cene))]

    atr = float(np.mean(tr[:period]))
    for x in tr[period:]:
        atr = (atr * (period - 1) + x) / period

    return atr


# ══════════════════════════════════════════════════════════════
# 5. SIGNAL LOGIKA
# ══════════════════════════════════════════════════════════════

def izracunaj_signal(S, sigma, cene_hist, T_dni=3, r=0.025,
                     atr_period=14, atr_mult_sl=1.5, atr_mult_tp=2.5):

    K = round(S, 2)
    T = T_dni / 365.0

    _, delta_call, gamma, theta, vega = black_scholes(S, K, T, r, sigma, "call")
    _, delta_put,  *_                 = black_scholes(S, K, T, r, sigma, "put")

    if cene_hist and len(cene_hist) >= 5:
        sma5  = float(np.mean(cene_hist[-5:]))
        sma20 = float(np.mean(cene_hist))
        trend_gor = S > sma5 > sma20
        trend_dol = S < sma5 < sma20
        momentum  = (cene_hist[-1] - cene_hist[0]) / cene_hist[0] * 100
    else:
        sma5 = sma20 = S
        trend_gor = trend_dol = False
        momentum  = 0.0

    tocke = 0

    if delta_call > 0.60:    tocke += 2
    elif delta_call > 0.52:  tocke += 1
    elif delta_call < 0.40:  tocke -= 2
    elif delta_call < 0.48:  tocke -= 1

    if trend_gor:   tocke += 2
    elif trend_dol: tocke -= 2

    if momentum > 1.5:    tocke += 1
    elif momentum < -1.5: tocke -= 1

    if sigma > 0.20:    tocke -= 1
    elif sigma < 0.12:  tocke += 1

    if tocke >= 3:
        odlocitev, jakost, emoji = "KUPI",              "mocen signal",              "🟢"
    elif tocke >= 1:
        odlocitev, jakost, emoji = "KUPI",              "sibek signal - manj. poz.", "🟡"
    elif tocke <= -3:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "mocen signal",              "🔴"
    elif tocke <= -1:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "sibek signal - pocakaj",    "🟠"
    else:
        odlocitev, jakost, emoji = "CAKAJ",             "trg je nevtralen",          "⚪"

    pricakovan_premik = S * sigma * np.sqrt(T)

    atr_raw = izracunaj_atr(cene_hist, period=atr_period)
    if atr_raw is not None:
        atr     = atr_raw
        atr_vir = "GC=F"
    else:
        atr     = pricakovan_premik * 0.5
        atr_vir = "vol-fallback"

    print(f"  ATR ({atr_vir}, {atr_period}-period): {atr:.2f} USD/unca")

    if odlocitev == "KUPI":
        stop_loss   = round(S - atr * atr_mult_sl, 2)
        take_profit = round(S + atr * atr_mult_tp, 2)
    elif odlocitev == "PRODAJ / NE KUPUJ":
        stop_loss   = round(S + atr * atr_mult_sl, 2)
        take_profit = round(S - atr * atr_mult_tp, 2)
    else:
        stop_loss = take_profit = None

    return {
        "S":                 S,
        "sigma":             sigma,
        "T_dni":             T_dni,
        "delta_call":        delta_call,
        "sma5":              sma5,
        "sma20":             sma20,
        "momentum":          momentum,
        "tocke":             tocke,
        "odlocitev":         odlocitev,
        "jakost":            jakost,
        "emoji":             emoji,
        "pricakovan_premik": pricakovan_premik,
        "atr":               atr,
        "atr_vir":           atr_vir,
        "atr_mult_sl":       atr_mult_sl,
        "atr_mult_tp":       atr_mult_tp,
        "stop_loss":         stop_loss,
        "take_profit":       take_profit,
    }


# ══════════════════════════════════════════════════════════════
# 6. IZPIS
# ══════════════════════════════════════════════════════════════

def izpisi(r):
    cas = datetime.now().strftime("%d.%m.%Y %H:%M")

    print(f"\n{'='*55}")
    print("  XAUUSD - ZLATO INVESTICIJSKI SIGNAL")
    print(f"  {cas}")
    print(f"{'='*55}\n")

    print(f"  Cena (USD/unca)   : {r['S']:.2f}")
    print(f"  Volatilnost       : {r['sigma']*100:.1f}%")
    print(f"  SMA5              : {r['sma5']:.2f}")
    print(f"  SMA20             : {r['sma20']:.2f}")
    print(f"  Momentum          : {r['momentum']:+.2f}%")
    print(f"  Delta CALL        : {r['delta_call']:.4f}")
    print(f"  Tocke             : {r['tocke']:+d}")
    print()
    print(f"  Signal            : {r['emoji']}  {r['odlocitev']}")
    print(f"  Jakost            : {r['jakost']}")
    print()
    print(f"  Pricakovan premik : +/-{r['pricakovan_premik']:.2f} USD")
    print(f"  ATR ({r['atr_vir']}, {r['atr_mult_sl']}x/{r['atr_mult_tp']}x) : {r['atr']:.2f} USD/unca")

    if r['stop_loss'] is not None:
        print()
        print(f"{'─'*55}")
        print("  TRADINGVIEW / PEPPERSTONE  -  XAUUSD (USD/unca)")
        print(f"{'─'*55}")
        print(f"  Buy cena    : {r['S']:.2f}")
        print(f"  Stop Loss   : {r['stop_loss']:.2f}")
        print(f"  Take Profit : {r['take_profit']:.2f}")
        print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════
# 7. CSV SHRANJEVANJE
# ══════════════════════════════════════════════════════════════

def shrani_csv(r):
    ime = "zgodovina_signalov.csv"
    obstaja = os.path.exists(ime)

    with open(ime, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not obstaja:
            w.writerow([
                "datum", "cena_usd_unca", "volatilnost",
                "delta_call", "sma5", "sma20", "momentum",
                "tocke", "signal", "jakost",
                "atr", "atr_vir", "stop_loss", "take_profit"
            ])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            r['S'],
            round(r['sigma'], 4),
            round(r['delta_call'], 4),
            round(r['sma5'], 2),
            round(r['sma20'], 2),
            round(r['momentum'], 2),
            r['tocke'],
            r['odlocitev'],
            r['jakost'],
            round(r['atr'], 2),
            r['atr_vir'],
            r['stop_loss'],
            r['take_profit'],
        ])

    print(f"\n  ✓ Signal shranjen v {ime}")


# ══════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n  Zaganjam XAUUSD signal...\n")

    S, sigma, cene_hist, avto = pridobi_vse()

    if not avto or not S:
        print("  Avtomatski prenos ni uspel.")
        S = vnesi_rocno()

    rezultat = izracunaj_signal(
        S, sigma, cene_hist,
        atr_period=14,    # ATR obdobje (dni)
        atr_mult_sl=1.5,  # Stop Loss  = cena +/- ATR x 1.5
        atr_mult_tp=2.5,  # Take Profit = cena +/- ATR x 2.5
    )
    izpisi(rezultat)
    shrani_csv(rezultat)


if __name__ == "__main__":
    main()
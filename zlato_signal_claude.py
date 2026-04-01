"""
╔══════════════════════════════════════════════════════════════╗
║         ZLATO INVESTICIJSKI SIGNAL – XAUEUR                  ║
║         Black-Scholes + Tehnična analiza                     ║
╚══════════════════════════════════════════════════════════════╝

NAMESTITEV (enkrat):
    pip install numpy scipy requests

ZAGON:
    python zlato_signal.py

VIRI PODATKOV:
    Cena zlata : gold-api.com  (brezplačno, brez registracije, brez ključa)
    EUR tečaj  : frankfurter.app (Evropska centralna banka, brez ključa)
    Volatilnost: Yahoo Finance GC=F v USD (čisto, brez EUR konverzije)
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}

def _get(url, timeout=8):
    import requests
    return requests.get(url, headers=HEADERS, timeout=timeout)


# ══════════════════════════════════════════════════════════════
# 1. PRIDOBITEV PODATKOV
# ══════════════════════════════════════════════════════════════

def pridobi_spot_usd():
    """
    gold-api.com — brezplačen, brez API ključa, brez omejitev.
    Vrne spot ceno XAU v USD za troy unco.
    """
    try:
        r = _get("https://api.gold-api.com/price/XAU")
        data = r.json()
        cena_usd = data["price"]
        print(f"  ✓ gold-api.com: {cena_usd:.2f} USD/unca")
        return float(cena_usd)
    except Exception as e:
        print(f"  ✗ gold-api.com ni uspel: {e}")
        return None


def pridobi_eurusd():
    """
    Frankfurter.app — uradni ECB tečaj, brezplačen, brez ključa.
    """
    try:
        r = _get("https://api.frankfurter.app/latest?from=USD&to=EUR")
        eurusd = r.json()["rates"]["EUR"]
        print(f"  ✓ ECB tečaj (frankfurter.app): 1 USD = {eurusd:.4f} EUR")
        return float(eurusd)
    except Exception as e:
        print(f"  ✗ Frankfurter ni uspel: {e}")
        # Rezervni vir
        try:
            r = _get("https://open.er-api.com/v6/latest/USD")
            eurusd = r.json()["rates"]["EUR"]
            print(f"  ✓ Rezervni tečaj (open.er-api): 1 USD = {eurusd:.4f} EUR")
            return float(eurusd)
        except Exception:
            return None


def pridobi_volatilnost_usd():
    """
    Yahoo Finance GC=F — zadnjih 20 dnevnih zaprtij v USD.
    Volatilnost računamo SAMO v USD, da se izognemo šumu EUR konverzije.
    Volatilnost zlata je skoraj identična v USD in EUR (razlika < 0.5%).
    """
    try:
        r = _get(
            "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            "?interval=1d&range=30d"
        )
        closes = r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        cene = [x for x in closes if x is not None][-20:]
        if len(cene) < 5:
            return None
        donosi = np.diff(np.log(cene))
        sigma = float(np.std(donosi) * np.sqrt(252))
        # Realen razpon za zlato: 10%–22%
        sigma = min(max(sigma, 0.10), 0.22)
        print(f"  ✓ Volatilnost (Yahoo GC=F USD, {len(cene)} dni): {sigma*100:.1f}%")
        return sigma, cene
    except Exception as e:
        print(f"  ✗ Yahoo volatilnost ni uspela: {e}")
        return None


def pridobi_vse():
    """Pridobi vse podatke in vrne S (EUR/gram), sigma, cene_hist."""
    print("\n  Pridobivam podatke...\n")

    xauusd  = pridobi_spot_usd()
    eurusd  = pridobi_eurusd()
    vol_ret = pridobi_volatilnost_usd()

    print()

    if xauusd and eurusd:
        S = round((xauusd * eurusd) / 31.1035, 2)
        print(f"  Preračun: {xauusd:.2f} USD/unca × {eurusd:.4f} EUR/USD ÷ 31.1035 = {S} EUR/gram")
        avto = True
    else:
        S = None
        avto = False

    if vol_ret:
        sigma, cene_hist = vol_ret
    else:
        sigma = 0.15
        cene_hist = []
        print(f"  ℹ  Privzeta volatilnost: 15%")

    return S, sigma, cene_hist, avto


# ══════════════════════════════════════════════════════════════
# 2. ROČNI VNOS
# ══════════════════════════════════════════════════════════════

def vnesi_rocno():
    print()
    print("  Kje najdeš ceno XAUEUR/gram:")
    print("  → TradingView: tradingview.com  (išči XAUEUR, deli z 31.1035 za gram)")
    print("  → Google: 'gold price EUR per gram'")
    print()
    while True:
        try:
            vnos = input("  Vpiši trenutno ceno [EUR/gram]: ").strip().replace(",", ".")
            S = float(vnos)
            if 50 < S < 600:
                break
            print("  Vrednost mora biti med 50 in 600.")
        except ValueError:
            print("  Napaka – vnesi samo številko, npr. 132.18")
    return S


# ══════════════════════════════════════════════════════════════
# 3. BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, tip="call"):
    if T <= 0:
        return (max(S-K,0) if tip=="call" else max(K-S,0)), 0, 0, 0, 0
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
# 4. SIGNAL LOGIKA
# ══════════════════════════════════════════════════════════════

def izracunaj_signal(S, sigma, cene_hist, T_dni=30, r=0.025):
    K = round(S, 2)
    T = T_dni / 365.0

    _, delta_call, gamma, theta, vega = black_scholes(S, K, T, r, sigma, "call")
    _, delta_put, *_                  = black_scholes(S, K, T, r, sigma, "put")

    # Tehnična analiza iz USD cen (proporcionalno enako kot EUR)
    if cene_hist and len(cene_hist) >= 5:
        # Normaliziramo USD cene na zadnjo vrednost da dobimo trend
        faktor = S / cene_hist[-1]
        cene_eur = [c * faktor for c in cene_hist]
        sma5  = float(np.mean(cene_eur[-5:]))
        sma20 = float(np.mean(cene_eur))
        trend_gor = S > sma5 > sma20
        trend_dol = S < sma5 < sma20
        momentum  = (cene_hist[-1] - cene_hist[0]) / cene_hist[0] * 100
    else:
        sma5 = sma20 = S
        trend_gor = trend_dol = False
        momentum = 0.0

    tocke = 0
    if delta_call > 0.60:    tocke += 2
    elif delta_call > 0.52:  tocke += 1
    elif delta_call < 0.40:  tocke -= 2
    elif delta_call < 0.48:  tocke -= 1

    if trend_gor:             tocke += 2
    elif trend_dol:           tocke -= 2

    if momentum > 1.5:        tocke += 1
    elif momentum < -1.5:     tocke -= 1

    if sigma > 0.20:          tocke -= 1
    elif sigma < 0.12:        tocke += 1

    if tocke >= 3:
        odlocitev, jakost, emoji = "KUPI",              "močan signal",                   "🟢"
    elif tocke >= 1:
        odlocitev, jakost, emoji = "KUPI",              "šibek signal – manjša pozicija", "🟡"
    elif tocke <= -3:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "močan signal",                   "🔴"
    elif tocke <= -1:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "šibek signal – počakaj",         "🟠"
    else:
        odlocitev, jakost, emoji = "ČAKAJ",             "trg je nevtralen",               "⚪"

    return dict(
        S=S, K=K, sigma=sigma, T_dni=T_dni,
        delta_call=delta_call, delta_put=delta_put,
        gamma=gamma, theta=theta, vega=vega,
        sma5=sma5, sma20=sma20, momentum=momentum,
        tocke=tocke, odlocitev=odlocitev, jakost=jakost, emoji=emoji,
        trend_gor=trend_gor, trend_dol=trend_dol,
    )


# ══════════════════════════════════════════════════════════════
# 5. IZPIS
# ══════════════════════════════════════════════════════════════

def izpisi(r):
    cas = datetime.now().strftime("%d.%m.%Y  %H:%M")
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         XAUEUR – ZLATO INVESTICIJSKI SIGNAL                 ║")
    print(f"║         {cas}                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"""
  VHODNI PODATKI
  ──────────────────────────────────────────────────
  Cena zlata (S)   : {r['S']:.2f} EUR/gram
  Strike (K)       : {r['K']:.2f} EUR/gram
  Volatilnost (σ)  : {r['sigma']*100:.1f}%  ← realna (USD GC=F, brez EUR šuma)
  Horizont         : {r['T_dni']} dni
  Obrestna mera    : 2.5% (ECB)

  BLACK-SCHOLES GREEKS
  ──────────────────────────────────────────────────
  Delta CALL  : {r['delta_call']:.4f}   {'▲ bullish' if r['delta_call'] > 0.52 else ('▼ bearish' if r['delta_call'] < 0.48 else '─ nevtralno')}
  Delta PUT   : {r['delta_put']:.4f}
  Gamma       : {r['gamma']:.6f}
  Theta       : {r['theta']:.4f} EUR/dan  (dnevna časovna erozija)
  Vega        : {r['vega']:.4f}           (občutljivost na vol.)

  TEHNIČNA ANALIZA
  ──────────────────────────────────────────────────
  SMA 5-dni   : {r['sma5']:.2f} EUR   {'✓ cena nad SMA5' if r['S'] > r['sma5'] else '✗ cena pod SMA5'}
  SMA 20-dni  : {r['sma20']:.2f} EUR   {'✓ bullish trend' if r['trend_gor'] else ('✗ bearish trend' if r['trend_dol'] else '─ nevtralno')}
  Momentum    : {r['momentum']:+.2f}%  {'↑ pozitiven' if r['momentum'] > 0 else '↓ negativen'}

  TOČKOVANJE   {r['tocke']:+d} / max 6
  (pozitivno = bullish, negativno = bearish)
""")
    print("  ╔══════════════════════════════════════════╗")
    print(f"  ║  {r['emoji']}  {r['odlocitev']:<40}║")
    print(f"  ║     {r['jakost']:<43}║")
    print("  ╚══════════════════════════════════════════╝")
    print()
    print("  ⚠  Informativno orodje – ni finančni nasvet.")
    print()
    print("  ══════════════════════════════════════════")
    print(f"  KONČNI ODGOVOR:  {r['odlocitev']}")
    print("  ══════════════════════════════════════════")
    print()


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   ZLATO SIGNAL – zaganjam...         ║")
    print("  ╚══════════════════════════════════════╝")

    S, sigma, cene_hist, avto = pridobi_vse()

    if not avto or not S:
        print("  Avtomatski prenos cene ni uspel.")
        S = vnesi_rocno()

    rezultat = izracunaj_signal(S, sigma, cene_hist)
    izpisi(rezultat)


if __name__ == "__main__":
    main()
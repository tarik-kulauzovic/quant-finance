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
# 1. PRIDOBITEV PODATKOV
# ══════════════════════════════════════════════════════════════

def pridobi_spot_usd():
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
    try:
        r = _get("https://api.frankfurter.app/latest?from=USD&to=EUR")
        eurusd = r.json()["rates"]["EUR"]
        print(f"  ✓ ECB tečaj (frankfurter.app): 1 USD = {eurusd:.4f} EUR")
        return float(eurusd)
    except Exception as e:
        print(f"  ✗ Frankfurter ni uspel: {e}")
        try:
            r = _get("https://open.er-api.com/v6/latest/USD")
            eurusd = r.json()["rates"]["EUR"]
            print(f"  ✓ Rezervni tečaj (open.er-api): 1 USD = {eurusd:.4f} EUR")
            return float(eurusd)
        except Exception:
            return None


def pridobi_volatilnost_usd():
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
        sigma = min(max(sigma, 0.10), 0.22)

        print(f"  ✓ Volatilnost (Yahoo GC=F USD, {len(cene)} dni): {sigma*100:.1f}%")
        return sigma, cene
    except Exception as e:
        print(f"  ✗ Yahoo volatilnost ni uspela: {e}")
        return None


def pridobi_vse():
    print("\n  Pridobivam podatke...\n")

    xauusd = pridobi_spot_usd()
    eurusd = pridobi_eurusd()
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
        print("  ℹ Privzeta volatilnost: 15%")

    return S, sigma, cene_hist, avto


# ══════════════════════════════════════════════════════════════
# 2. ROČNI VNOS
# ══════════════════════════════════════════════════════════════

def vnesi_rocno():
    print()
    print("  Kje najdeš ceno XAUEUR/gram:")
    print("  → TradingView: tradingview.com")
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
            print("  Napaka – vnesi samo številko")

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
        cena = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        cena = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2 if tip=="call" else -d2)) / 365
    vega = S*norm.pdf(d1)*np.sqrt(T) / 100

    return cena, delta, gamma, theta, vega


# ══════════════════════════════════════════════════════════════
# 4. SIGNAL LOGIKA
# ══════════════════════════════════════════════════════════════

def izracunaj_signal(S, sigma, cene_hist, T_dni=3, r=0.025):
    K = round(S, 2)
    T = T_dni / 365.0

    _, delta_call, gamma, theta, vega = black_scholes(S, K, T, r, sigma, "call")
    _, delta_put, *_ = black_scholes(S, K, T, r, sigma, "put")

    if cene_hist and len(cene_hist) >= 5:
        faktor = S / cene_hist[-1]
        cene_eur = [c * faktor for c in cene_hist]

        sma5 = float(np.mean(cene_eur[-5:]))
        sma20 = float(np.mean(cene_eur))

        trend_gor = S > sma5 > sma20
        trend_dol = S < sma5 < sma20
        momentum = (cene_hist[-1] - cene_hist[0]) / cene_hist[0] * 100
    else:
        sma5 = sma20 = S
        trend_gor = trend_dol = False
        momentum = 0.0

    tocke = 0

    if delta_call > 0.60:
        tocke += 2
    elif delta_call > 0.52:
        tocke += 1
    elif delta_call < 0.40:
        tocke -= 2
    elif delta_call < 0.48:
        tocke -= 1

    if trend_gor:
        tocke += 2
    elif trend_dol:
        tocke -= 2

    if momentum > 1.5:
        tocke += 1
    elif momentum < -1.5:
        tocke -= 1

    if sigma > 0.20:
        tocke -= 1
    elif sigma < 0.12:
        tocke += 1

    if tocke >= 3:
        odlocitev, jakost, emoji = "KUPI", "močan signal", "🟢"
    elif tocke >= 1:
        odlocitev, jakost, emoji = "KUPI", "šibek signal – manjša pozicija", "🟡"
    elif tocke <= -3:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "močan signal", "🔴"
    elif tocke <= -1:
        odlocitev, jakost, emoji = "PRODAJ / NE KUPUJ", "šibek signal – počakaj", "🟠"
    else:
        odlocitev, jakost, emoji = "ČAKAJ", "trg je nevtralen", "⚪"

    pricakovan_premik = S * sigma * np.sqrt(T)

    if odlocitev == "KUPI":
        stop_loss = round(S - pricakovan_premik * 0.25, 2)
        take_profit = round(S + pricakovan_premik * 0.5, 2)
    elif odlocitev == "PRODAJ / NE KUPUJ":
        stop_loss = round(S + pricakovan_premik * 0.25, 2)
        take_profit = round(S - pricakovan_premik * 0.5, 2)
    else:
        stop_loss = None
        take_profit = None

    spot_buy = round(S, 2)

    if odlocitev == "KUPI":
        spot_sell_target = take_profit
    elif odlocitev == "PRODAJ / NE KUPUJ":
        spot_sell_target = round(S, 2)
    else:
        spot_sell_target = None

    spot_buy_unca = round(S * 31.1035, 2)

    if stop_loss is not None:
        stop_loss_unca = round(stop_loss * 31.1035, 2)
    else:
        stop_loss_unca = None

    if take_profit is not None:
        take_profit_unca = round(take_profit * 31.1035, 2)
    else:
        take_profit_unca = None

    return {
        "S": S,
        "K": K,
        "sigma": sigma,
        "T_dni": T_dni,
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "sma5": sma5,
        "sma20": sma20,
        "momentum": momentum,
        "tocke": tocke,
        "odlocitev": odlocitev,
        "jakost": jakost,
        "emoji": emoji,
        "trend_gor": trend_gor,
        "trend_dol": trend_dol,
        "pricakovan_premik": pricakovan_premik,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "spot_buy": spot_buy,
        "spot_sell_target": spot_sell_target,
        "spot_buy_unca": spot_buy_unca,
        "stop_loss_unca": stop_loss_unca,
        "take_profit_unca": take_profit_unca
    }


# ══════════════════════════════════════════════════════════════
# 5. IZPIS
# ══════════════════════════════════════════════════════════════

def izpisi(r):
    cas = datetime.now().strftime("%d.%m.%Y %H:%M")

    print(f"\n{'='*60}")
    print("XAUEUR – ZLATO INVESTICIJSKI SIGNAL")
    print(cas)
    print(f"{'='*60}\n")

    print(f"Cena zlata      : {r['S']:.2f} EUR/gram")
    print(f"Volatilnost     : {r['sigma']*100:.1f}%")
    print(f"SMA5            : {r['sma5']:.2f}")
    print(f"SMA20           : {r['sma20']:.2f}")
    print(f"Momentum        : {r['momentum']:+.2f}%")
    print(f"Delta CALL      : {r['delta_call']:.4f}")
    print(f"Točke           : {r['tocke']:+d}")
    print()
    print(f"Signal          : {r['emoji']} {r['odlocitev']}")
    print(f"Jakost          : {r['jakost']}")
    print()
    print(f"Pričakovan premik : ±{r['pricakovan_premik']:.2f} EUR")

    if r['stop_loss'] is not None:
        print(f"Stop Loss         : {r['stop_loss']:.2f} EUR")
        print(f"Take Profit       : {r['take_profit']:.2f} EUR")
        print()
        print("SPOT TRADING")
        print(f"Spot buy cena     : {r['spot_buy']:.2f} EUR")
        
        if r['spot_sell_target'] is not None:
            print(f"Spot sell target  : {r['spot_sell_target']:.2f} EUR")

        print()
        print("TRADINGVIEW / FOREX.COM XAUEUR (EUR na unčo)")
        print(f"Buy cena          : {r['spot_buy_unca']:.2f}")

        if r['stop_loss_unca'] is not None:
            print(f"Stop Loss         : {r['stop_loss_unca']:.2f}")

        if r['take_profit_unca'] is not None:
            print(f"Take Profit       : {r['take_profit_unca']:.2f}")


# ══════════════════════════════════════════════════════════════
# 6. CSV SHRANJEVANJE
# ══════════════════════════════════════════════════════════════

def shrani_csv(r):
    ime_datoteke = "zgodovina_signalov.csv"
    obstaja = os.path.exists(ime_datoteke)

    with open(ime_datoteke, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not obstaja:
            writer.writerow([
                "datum",
                "cena_eur_g",
                "volatilnost",
                "delta_call",
                "sma5",
                "sma20",
                "momentum",
                "tocke",
                "signal",
                "jakost",
                "stop_loss",
                "take_profit"
            ])

        writer.writerow([
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
            r['stop_loss'],
            r['take_profit']
        ])

    print(f"\n✓ Signal shranjen v {ime_datoteke}")


# ══════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\nZaganjam zlato signal...\n")

    S, sigma, cene_hist, avto = pridobi_vse()

    if not avto or not S:
        print("Avtomatski prenos ni uspel.")
        S = vnesi_rocno()

    rezultat = izracunaj_signal(S, sigma, cene_hist)
    izpisi(rezultat)
    shrani_csv(rezultat)


if __name__ == "__main__":
    main()

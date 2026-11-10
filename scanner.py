"""
Scanner forex con TUTTE le strategie:
  1) RANGE IN DOWNTREND (SELL)
  2) RANGE IN UPTREND (BUY)
  3) STRATEGIA SUI LIVELLI (BUY e SELL)
Controlla 28 coppie e manda un messaggio Telegram quando trova un segnale
sull'ultima candela CHIUSA. Rischio/rendimento fisso (RR).
"""
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- IMPOSTAZIONI GENERALI ----------------
TF_HOURS = 1        # 1 = H1, 4 = H4
EMA_LEN = 50
ATR_LEN = 14
SOGLIA = 1.3        # soglia momentum (prova 1.2 / 1.3 / 1.5)
RR = 1.5            # rischio/rendimento
FINESTRA_MIN = 58   # segnala solo candele chiuse da meno di questi minuti

# ---------------- STRATEGIE RANGE ----------------
RANGE_LEN = 20      # lunghezza range (prova 15 / 20 / 30)
STOP_ATR = 0.5      # stop oltre l'estremo del range
BODY_ATR = 1.5      # corpo candela "grande"

# ---------------- STRATEGIA LIVELLI ----------------
MIN_RUN = 3         # candele consecutive minime del movimento
MIN_AMP = 2.0       # ampiezza minima del movimento in ATR
LVL_STOP = 0.75     # stop oltre il primo livello (0.5 - 1 ATR)
LVL_MAX_AGE = 30    # il livello scade dopo queste candele
LVL_MAX_TESTS = 2   # o dopo questi test
USA_LIVELLI = True   # True = range + livelli | False = solo strategie range

PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
    "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "EURJPY",
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF", "GBPJPY",
    "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
    "NZDCAD", "NZDCHF", "NZDJPY",
    "CADCHF", "CADJPY", "CHFJPY",
]

ETICHETTA = "[NUOVO RR 1,5]"   # compare all'inizio di ogni messaggio

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")


def telegram(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )


def get_data(pair):
    period = "30d" if TF_HOURS == 1 else "60d"
    df = yf.Ticker(pair + "=X").history(period=period, interval="1h")
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close"]]
    df.columns = ["open", "high", "low", "close"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    if TF_HOURS > 1:
        df = df.resample(f"{TF_HOURS}h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
    now = datetime.now(timezone.utc)
    # tieni solo le candele gia' chiuse
    df = df[df.index + pd.Timedelta(hours=TF_HOURS) <= now]
    return df


def indicators(df):
    h, l, c = df["high"], df["low"], df["close"]
    ema = c.ewm(span=EMA_LEN, adjust=False).mean()
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()  # RMA come TradingView
    atr_avg = atr.rolling(20).mean()
    return ema, atr, atr_avg


def is_fresh(df, now):
    """True se l'ultima candela e' appena chiusa (evita doppioni)."""
    chiusura = df.index[-1] + pd.Timedelta(hours=TF_HOURS)
    return (now - chiusura).total_seconds() / 60 <= FINESTRA_MIN


# ====================== STRATEGIE RANGE ======================
def check_range(df, ema, atr, atr_avg):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng_high = h.rolling(RANGE_LEN).max().shift(1)
    rng_low = l.rolling(RANGE_LEN).min().shift(1)

    A = atr.iloc[-1]
    RH, RL = rng_high.iloc[-1], rng_low.iloc[-1]
    if pd.isna(A) or pd.isna(RH) or pd.isna(RL) or pd.isna(atr_avg.iloc[-1]):
        return None

    trend_up = c.iloc[-1] > ema.iloc[-1] and ema.iloc[-1] > ema.iloc[-2]
    trend_dn = c.iloc[-1] < ema.iloc[-1] and ema.iloc[-1] < ema.iloc[-2]
    ampiezza = (RH - RL) / A
    range_ok = 2 <= ampiezza <= 5
    accel = A > atr_avg.iloc[-1] * SOGLIA
    big_body = abs(c.iloc[-1] - o.iloc[-1]) > BODY_ATR * A
    if not range_ok or accel or big_body:
        return None

    entry = c.iloc[-1]
    if trend_dn and h.iloc[-1] >= RH and entry < RH:
        stop = RH + STOP_ATR * A
        return {"tipo": "RANGE", "dir": "SELL", "entry": entry, "stop": stop,
                "tp": entry - RR * (stop - entry)}
    if trend_up and l.iloc[-1] <= RL and entry > RL:
        stop = RL - STOP_ATR * A
        return {"tipo": "RANGE", "dir": "BUY", "entry": entry, "stop": stop,
                "tp": entry + RR * (entry - stop)}
    return None


# ====================== STRATEGIA LIVELLI ======================
def check_levels(df, ema, atr, atr_avg):
    """
    Ripercorre le candele ricostruendo i livelli (come fa lo script TradingView).
    Segnala solo se il ritorno sul livello 1 con candela di conferma
    avviene sull'ULTIMA candela chiusa.
    """
    o, h = df["open"].values, df["high"].values
    l, c = df["low"].values, df["close"].values
    e, a, av = ema.values, atr.values, atr_avg.values
    n = len(df)

    g = r = 0
    up = dn = None  # livelli attivi: dict con l1, l2, bar, tests
    segnale = None

    for i in range(1, n):
        g_prev, r_prev = g, r
        g = g + 1 if c[i] > o[i] else 0
        r = r + 1 if c[i] < o[i] else 0

        # --- formazione di un movimento al rialzo valido ---
        if g == 0 and g_prev >= MIN_RUN and i - g_prev >= 0 and a[i - 1] > 0:
            k = g_prev
            l1 = o[i - k]
            l2 = h[i - k:i].max()
            if c[i] > e[i] and (l2 - l1) / a[i - 1] >= MIN_AMP:
                up = {"l1": l1, "l2": l2, "bar": i, "tests": 0}

        # --- formazione di un movimento al ribasso valido ---
        if r == 0 and r_prev >= MIN_RUN and i - r_prev >= 0 and a[i - 1] > 0:
            k = r_prev
            l1 = o[i - k]
            l2 = l[i - k:i].min()
            if c[i] < e[i] and (l1 - l2) / a[i - 1] >= MIN_AMP:
                dn = {"l1": l1, "l2": l2, "bar": i, "tests": 0}

        # --- scadenza dei livelli ---
        if up and (h[i] > up["l2"] or i - up["bar"] > LVL_MAX_AGE
                   or up["tests"] >= LVL_MAX_TESTS):
            up = None
        if dn and (l[i] < dn["l2"] or i - dn["bar"] > LVL_MAX_AGE
                   or dn["tests"] >= LVL_MAX_TESTS):
            dn = None

        acc = (not np.isnan(av[i])) and a[i] > av[i] * SOGLIA

        # --- ritorno sul livello (BUY) con candela di conferma ---
        if up and not acc and l[i] <= up["l1"] and c[i] > up["l1"] and c[i] > o[i]:
            if i == n - 1:
                stop = up["l1"] - LVL_STOP * a[i]
                tp = up["l2"]
                rischio = c[i] - stop
                if rischio > 0 and (tp - c[i]) / rischio >= RR:
                    segnale = {"tipo": "LIVELLI", "dir": "BUY", "entry": c[i],
                               "stop": stop, "tp": tp, "livello": up["l1"]}
            up["tests"] += 1

        # --- ritorno sul livello (SELL) con candela di conferma ---
        if dn and not acc and h[i] >= dn["l1"] and c[i] < dn["l1"] and c[i] < o[i]:
            if i == n - 1:
                stop = dn["l1"] + LVL_STOP * a[i]
                tp = dn["l2"]
                rischio = stop - c[i]
                if rischio > 0 and (c[i] - tp) / rischio >= RR:
                    segnale = {"tipo": "LIVELLI", "dir": "SELL", "entry": c[i],
                               "stop": stop, "tp": tp, "livello": dn["l1"]}
            dn["tests"] += 1

    return segnale


def fmt(pair, x):
    dec = 3 if "JPY" in pair else 5
    return f"{x:.{dec}f}"


def rr_reale(s):
    rischio = abs(s["entry"] - s["stop"])
    return abs(s["tp"] - s["entry"]) / rischio if rischio else 0


def messaggio(pair, tf, s):
    righe = [f"{ETICHETTA} {s['dir']} {pair} {tf} - strategia {s['tipo']}"]
    if s["tipo"] == "LIVELLI":
        righe.append(f"Livello 1 (entrata): {fmt(pair, s['livello'])}")
    righe += [
        f"Entrata (chiusura): {fmt(pair, s['entry'])}",
        f"Stop: {fmt(pair, s['stop'])}",
        f"Target: {fmt(pair, s['tp'])}  (rischio/rendimento {rr_reale(s):.2f})",
        "Controlla il grafico su TradingView prima di entrare.",
    ]
    return "\n".join(righe)


def main():
    tf = f"H{TF_HOURS}"
    now = datetime.now(timezone.utc)
    segnali, errori = [], 0

    for p in PAIRS:
        try:
            df = get_data(p)
            if df is None or len(df) < 120:
                errori += 1
                continue
            if not is_fresh(df, now):
                continue
            ema, atr, atr_avg = indicators(df)
            for s in (check_range(df, ema, atr, atr_avg),
                      check_levels(df, ema, atr, atr_avg) if USA_LIVELLI else None):
                if s:
                    segnali.append((p, s))
        except Exception as ex:
            errori += 1
            print(p, "errore:", ex)

    for p, s in segnali:
        telegram(messaggio(p, tf, s))

    # avviso se i dati non funzionano (cosi' sai se si e' fermato)
    if errori > len(PAIRS) // 2:
        telegram(f"ATTENZIONE: scanner con {errori} errori su {len(PAIRS)} coppie. "
                 "Probabile problema con i dati.")

    # lanciato a mano: conferma che funziona
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        telegram(f"{ETICHETTA} Scanner attivo ({tf}) - livelli: {'si' if USA_LIVELLI else 'no'}. "
                 f"Coppie controllate: {len(PAIRS) - errori}. "
                 f"Segnali trovati ora: {len(segnali)}.")


if __name__ == "__main__":
    main()

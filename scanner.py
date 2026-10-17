"""
Scanner forex: strategie RANGE IN DOWNTREND / RANGE IN UPTREND
Controlla 28 coppie e manda un messaggio Telegram quando trova un segnale
sull'ultima candela CHIUSA.
"""
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

# ---------------- IMPOSTAZIONI ----------------
TF_HOURS = 1        # 1 = H1, 4 = H4
EMA_LEN = 50
ATR_LEN = 14
RANGE_LEN = 20      # lunghezza range (prova 15 / 20 / 30)
SOGLIA = 1.3        # soglia momentum (prova 1.2 / 1.3 / 1.5)
STOP_ATR = 0.5      # stop oltre l'estremo del range
BODY_ATR = 1.5      # corpo candela "grande"
FINESTRA_MIN = 58   # segnala solo candele chiuse da meno di questi minuti

PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
    "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "EURJPY",
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF", "GBPJPY",
    "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
    "NZDCAD", "NZDCHF", "NZDJPY",
    "CADCHF", "CADJPY", "CHFJPY",
]

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


def check(df, now=None):
    """Ritorna un dizionario con il segnale, oppure None."""
    if df is None or len(df) < 120:
        return None
    now = now or datetime.now(timezone.utc)

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ema = c.ewm(span=EMA_LEN, adjust=False).mean()
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()  # RMA come TradingView
    atr_avg = atr.rolling(20).mean()
    rng_high = h.rolling(RANGE_LEN).max().shift(1)
    rng_low = l.rolling(RANGE_LEN).min().shift(1)

    # la candela deve essere appena chiusa (evita doppioni)
    chiusura = df.index[-1] + pd.Timedelta(hours=TF_HOURS)
    if (now - chiusura).total_seconds() / 60 > FINESTRA_MIN:
        return None

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

    meta = (RH + RL) / 2
    if trend_dn and h.iloc[-1] >= RH and c.iloc[-1] < RH:
        return {"dir": "SELL", "entry": c.iloc[-1], "stop": RH + STOP_ATR * A,
                "t1": meta, "t2": RL, "atr": A}
    if trend_up and l.iloc[-1] <= RL and c.iloc[-1] > RL:
        return {"dir": "BUY", "entry": c.iloc[-1], "stop": RL - STOP_ATR * A,
                "t1": meta, "t2": RH, "atr": A}
    return None


def fmt(pair, x):
    dec = 3 if "JPY" in pair else 5
    return f"{x:.{dec}f}"


def main():
    tf = f"H{TF_HOURS}"
    segnali, errori = [], 0
    for p in PAIRS:
        try:
            s = check(get_data(p))
            if s:
                segnali.append((p, s))
        except Exception as e:
            errori += 1
            print(p, "errore:", e)

    for p, s in segnali:
        telegram(
            f"{s['dir']} {p} {tf}\n"
            f"Entrata (chiusura): {fmt(p, s['entry'])}\n"
            f"Stop: {fmt(p, s['stop'])}\n"
            f"Target 1 (meta' range): {fmt(p, s['t1'])}\n"
            f"Target 2 (estremo opposto): {fmt(p, s['t2'])}\n"
            f"Controlla il grafico su TradingView prima di entrare."
        )

    # avviso se i dati non funzionano (cosi' sai se si e' fermato)
    if errori > len(PAIRS) // 2:
        telegram(f"ATTENZIONE: scanner con {errori} errori su {len(PAIRS)} coppie. "
                 "Probabile problema con i dati.")

    # lanciato a mano: conferma che funziona
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        telegram(f"Scanner attivo ({tf}). Coppie controllate: {len(PAIRS) - errori}. "
                 f"Segnali trovati ora: {len(segnali)}.")


if __name__ == "__main__":
    main()

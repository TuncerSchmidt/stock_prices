from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
import math

app = FastAPI()

def safe_val(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else x

def calculate_indicators(symbol: str):
    ticker = yf.Ticker(symbol)

    info = ticker.info if hasattr(ticker, "info") else ticker.get_info()
    hist = ticker.history(period="6mo")

    price = safe_val(info.get("currentPrice") or info.get("regularMarketPrice"))
    volume = safe_val(info.get("volume"))
    market_cap = safe_val(info.get("marketCap"))
    pe_ratio = safe_val(info.get("trailingPE"))
    peg_ratio = safe_val(info.get("pegRatio"))
    pb_ratio = safe_val(info.get("priceToBook"))
    currency = info.get("currency")

    hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
    hist["EMA_200"] = hist["Close"].ewm(span=200, adjust=False).mean()

    sma_50 = safe_val(hist["SMA_50"].iloc[-1])
    ema_200 = safe_val(hist["EMA_200"].iloc[-1])

    delta = hist["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = safe_val(rsi.iloc[-1])

    ema_12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = hist["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": currency,
        "volume": volume,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "peg_ratio": peg_ratio,
        "pb_ratio": pb_ratio,
        "sma_50": sma_50,
        "ema_200": ema_200,
        "rsi_14": rsi_14,
        "macd": safe_val(macd.iloc[-1]),
        "signal": safe_val(signal.iloc[-1])
    }

@app.get("/indicators/{symbol}")
def indicators_endpoint(symbol: str):
    try:
        return calculate_indicators(symbol)
    except Exception as e:
        return {"error": str(e)}

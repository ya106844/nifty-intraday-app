from flask import Flask, jsonify
from nsetools import Nse
import requests
import numpy as np

app = Flask(__name__)
nse = Nse()

# Function to calculate RSI
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    rs = gains / losses if losses != 0 else 0
    return 100 - (100 / (1 + rs))

# Dummy historical data fetcher (Replace with live OHLC API if available)
def get_last_day_ohlc():
    return {"high": 24850, "low": 24500, "close": 24680}

# Option chain PCR fetcher
def get_pcr():
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {"User-Agent": "Mozilla/5.0"}
        data = requests.get(url, headers=headers).json()
        ce_oi = sum([x['CE']['openInterest'] for x in data['records']['data'] if 'CE' in x])
        pe_oi = sum([x['PE']['openInterest'] for x in data['records']['data'] if 'PE' in x])
        return round(pe_oi / ce_oi, 2) if ce_oi > 0 else None
    except:
        return None

@app.route("/")
def nifty_analysis():
    try:
        # Get live price
        quote = nse.get_index_quote("NIFTY 50")
        price = quote.get("last")
        vwap = quote.get("vwap")

        # Get support & resistance
        ohlc = get_last_day_ohlc()
        support = round(ohlc['low'], 2)
        resistance = round(ohlc['high'], 2)

        # Calculate RSI (using dummy small series for demo)
        prices = [price - i*5 for i in range(20)]
        rsi = calculate_rsi(prices)

        # Get PCR
        pcr = get_pcr()

        # Decision logic
        trend = "Bullish" if price > vwap else "Bearish"
        strength = "Strong" if (trend == "Bullish" and rsi and rsi < 70 and pcr and pcr > 1) else "Weak"
        action = "Buy" if trend == "Bullish" and strength == "Strong" else ("Sell" if trend == "Bearish" and strength == "Strong" else "Wait")

        return jsonify({
            "price": price,
            "vwap": vwap,
            "support": support,
            "resistance": resistance,
            "RSI": rsi,
            "PCR": pcr,
            "trend": trend,
            "strength": strength,
            "action": action
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

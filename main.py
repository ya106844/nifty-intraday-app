# Create a deploy-ready backend ZIP for Render/GitHub with advanced signals + option recommendation
import os, zipfile, textwrap, json, pathlib

backend_dir = "/mnt/data/nifty_backend_advanced"
os.makedirs(backend_dir, exist_ok=True)

main_py = textwrap.dedent(r"""
from flask import Flask, jsonify, request
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import math

app = Flask(__name__)

# Configurable parameters
TICKER = "^NSEI"  # Yahoo Finance symbol for Nifty 50
SHORT_MA = 3      # 15-min moving average (3 x 5min)
LONG_MA = 6       # 30-min moving average (6 x 5min)
RSI_PERIOD = 14
RISK_FREE_RATE = 0.06  # 6% annual risk-free, used for BS approx
TRADING_INTERVALS_PER_DAY = 78  # approx number of 5-min intervals in trading day (6.5*60/5)
TRADING_DAYS_PER_YEAR = 252

def get_intraday_df(period='1d', interval='5m'):
    # Fetch recent intraday data for ticker
    try:
        df = yf.download(TICKER, period=period, interval=interval, progress=False)
        return df
    except Exception as e:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def annualized_vol_from_returns(returns, interval_minutes=5):
    # returns: pandas Series of log returns for intraday intervals
    # annualize: std * sqrt(number of intervals per year)
    intervals_per_day = TRADING_INTERVALS_PER_DAY
    intervals_per_year = intervals_per_day * TRADING_DAYS_PER_YEAR
    vol = returns.std() * math.sqrt(intervals_per_year)
    return vol

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    # S: spot, K: strike, T: time to expiry in years, r: risk-free, sigma: vol
    if T <= 0 or sigma <= 0:
        # option intrinsic value approximation
        if option_type == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return price

def nearest_strike(price, step=50):
    return round(price / step) * step

@app.route('/signal', methods=['GET'])
def signal():
    """
    Returns:
      - signal: Bullish / Bearish / Neutral
      - probability: heuristic percent
      - last_price: latest close
      - rsi: latest RSI
      - short_ma, long_ma
    """
    df = get_intraday_df(period='1d', interval='5m')
    if df is None or df.empty:
        return jsonify({'error': 'data_fetch_failed'}), 500

    close = df['Close']
    last_price = float(close.iloc[-1])

    short_ma = float(close.rolling(window=SHORT_MA).mean().iloc[-1])
    long_ma = float(close.rolling(window=LONG_MA).mean().iloc[-1])

    rsi_series = calculate_rsi(close, RSI_PERIOD)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

    # Simple rule-based signal
    score_up = 0
    score_down = 0

    if short_ma > long_ma:
        score_up += 30
    elif short_ma < long_ma:
        score_down += 30

    if rsi is not None:
        if rsi < 40:
            score_up += 20
        elif rsi > 60:
            score_down += 20

    # recent momentum: last return
    returns = np.log(close / close.shift(1)).dropna()
    last_ret = float(returns.iloc[-1])
    if last_ret > 0:
        score_up += 10
    else:
        score_down += 10

    # normalize to probabilities
    base = 30
    up_score = base + score_up
    down_score = base + score_down
    flat_score = max(0, 100 - (up_score + down_score - base))
    total = up_score + down_score + flat_score
    prob_up = round((up_score / total) * 100, 1)
    prob_down = round((down_score / total) * 100, 1)
    prob_flat = round((flat_score / total) * 100, 1)

    # Prepare response
    resp = {
        'signal': 'Bullish' if prob_up > prob_down and prob_up > prob_flat else ('Bearish' if prob_down > prob_up and prob_down > prob_flat else 'Neutral'),
        'probabilities': {'up': prob_up, 'down': prob_down, 'flat': prob_flat},
        'last_price': last_price,
        'rsi': round(rsi,2) if rsi is not None else None,
        'short_ma': round(short_ma,2),
        'long_ma': round(long_ma,2)
    }
    return jsonify(resp)

@app.route('/option_reco', methods=['GET'])
def option_reco():
    """
    Returns intraday option-selling recommendation for nearest weekly expiry.
    Query params:
      days_to_expiry (int) - optional (default 2)
      step (int) - strike step (default 50)
    """
    days_to_expiry = int(request.args.get('days_to_expiry', 2))
    step = int(request.args.get('step', 50))

    df = get_intraday_df(period='1d', interval='5m')
    if df is None or df.empty:
        return jsonify({'error': 'data_fetch_failed'}), 500

    close = df['Close']
    last_price = float(close.iloc[-1])

    # volatility estimate from intraday returns
    returns = np.log(close / close.shift(1)).dropna()
    sigma = annualized_vol_from_returns(returns)

    # time to expiry in years
    T = max(1, days_to_expiry) / TRADING_DAYS_PER_YEAR

    # heuristic bias from signal endpoint
    sig_resp = app.test_client().get('/signal')
    sig_json = sig_resp.get_json()
    bias = sig_json.get('signal', 'Neutral')

    # choose strike
    if bias == 'Bearish':
        # sell Call (OTM above spot)
        strike = nearest_strike(last_price + 100, step)
        side = 'sell_call'
    elif bias == 'Bullish':
        strike = nearest_strike(last_price - 100, step)
        side = 'sell_put'
    else:
        # if neutral, sell both slightly OTM (strangle) - but we will recommend waiting
        strike = nearest_strike(last_price + 100, step)
        side = 'wait'

    # Estimate premium using Black-Scholes (approx)
    if side == 'sell_call':
        premium = black_scholes_price(last_price, strike, T, RISK_FREE_RATE, sigma, option_type='call')
    elif side == 'sell_put':
        premium = black_scholes_price(last_price, strike, T, RISK_FREE_RATE, sigma, option_type='put')
    else:
        premium = None

    response = {
        'last_price': last_price,
        'vol_annual': round(float(sigma),4),
        'days_to_expiry': days_to_expiry,
        'bias': bias,
        'recommendation': {
            'side': side,
            'strike': int(strike),
            'estimated_premium': round(float(premium),2) if premium is not None else None,
            'probability_edge': sig_json.get('probabilities') if sig_json else None
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
""").lstrip()

requirements = textwrap.dedent("""
Flask
yfinance
pandas
numpy
""").strip()

readme = textwrap.dedent("""
Nifty Intraday Advanced Backend
===============================

Files:
- main.py   (Flask app with /signal and /option_reco endpoints)
- requirements.txt

How to deploy to Render:
1. Create a public GitHub repo and push these files.
2. On Render, create a new Web Service and connect the repo.
3. Set the build command to: pip install -r requirements.txt
4. Set the start command to: python main.py
5. Deploy — the service will provide a public URL like https://your-app.onrender.com
6. Use /signal and /option_reco endpoints from the Flutter app.

Endpoints:
- GET /signal  -> returns signal, probabilities, last_price, rsi, short_ma, long_ma
- GET /option_reco?days_to_expiry=2&step=50 -> returns option recommendation and estimated premium
""").strip()

# write files
with open(os.path.join(backend_dir, "main.py"), "w") as f:
    f.write(main_py)
with open(os.path.join(backend_dir, "requirements.txt"), "w") as f:
    f.write(requirements)
with open(os.path.join(backend_dir, "README.md"), "w") as f:
    f.write(readme)

# create zip
zip_path = "/mnt/data/nifty_backend_advanced.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, _, files in os.walk(backend_dir):
        for file in files:
            fullpath = os.path.join(folder, file)
            arcname = os.path.relpath(fullpath, backend_dir)
            z.write(fullpath, arcname)

zip_path


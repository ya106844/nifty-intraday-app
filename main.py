from flask import Flask, jsonify
import requests, time, math
import numpy as np

app = Flask(__name__)

# ----------------------------
# Global HTTP session & headers
# ----------------------------
SESSION = requests.Session()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "accept": "application/json, text/plain, */*",
    "referer": "https://www.nseindia.com/"
}

NSE_HOME = "https://www.nseindia.com"
NSE_ALL_INDICES = "https://www.nseindia.com/api/allIndices"
NSE_OPTION_CHAIN = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
YAHOO_INTRADAY = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=1d&interval=5m"

# Last known fallbacks so API glitches don't break responses
CACHE = {
    "pcr": None,
    "vix": None,
    "nifty_last": None,
    "nifty_prev_close": None,
    "rsi": None,
    "support": None,
    "resistance": None,
    "max_pain": None,
    "ts": 0
}

# ----------------------------
# Utility: warm-up cookies
# ----------------------------
def warmup():
    try:
        SESSION.get(NSE_HOME, headers=HEADERS, timeout=6)
    except Exception as e:
        print("Warmup warn:", e)

# ----------------------------
# Safe JSON GET with warm-up + fallback
# ----------------------------
def json_get(url, timeout=8):
    try:
        # warm-up cookie if needed (simple throttle)
        if time.time() - CACHE["ts"] > 30:
            warmup()
            CACHE["ts"] = time.time()
        r = SESSION.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            print(f"⚠ HTTP {r.status_code} for {url}")
            return None
        try:
            return r.json()
        except Exception as e:
            print("⚠ JSON parse error:", e)
            return None
    except Exception as e:
        print("⚠ json_get error:", e)
        return None

# ----------------------------
# Fetch Nifty & VIX from allIndices
# ----------------------------
def fetch_nifty_and_vix():
    data = json_get(NSE_ALL_INDICES)
    last = CACHE["nifty_last"]
    prev_close = CACHE["nifty_prev_close"]
    vix = CACHE["vix"]

    if not data or "data" not in data:
        return last, prev_close, vix

    try:
        for row in data["data"]:
            name = (row.get("index") or row.get("indexSymbol") or "").upper()
            if "NIFTY 50" in name:
                last = float(row.get("last", last or 0)) if row.get("last") is not None else last
                prev_close = float(row.get("previousClose", prev_close or 0)) if row.get("previousClose") is not None else prev_close
            if "INDIA VIX" in name:
                vix = float(row.get("last", vix or 0)) if row.get("last") is not None else vix
    except Exception as e:
        print("⚠ parse allIndices:", e)

    CACHE["nifty_last"] = last
    CACHE["nifty_prev_close"] = prev_close
    CACHE["vix"] = vix
    return last, prev_close, vix

# ----------------------------
# Option chain → PCR, OI levels, max pain
# ----------------------------
def fetch_option_chain():
    data = json_get(NSE_OPTION_CHAIN)
    if not data or "records" not in data:
        return None
    return data

def compute_pcr(oc):
    try:
        ce_oi = 0
        pe_oi = 0
        for row in oc["records"]["data"]:
            if "CE" in row and row["CE"].get("openInterest") is not None:
                ce_oi += int(row["CE"]["openInterest"])
            if "PE" in row and row["PE"].get("openInterest") is not None:
                pe_oi += int(row["PE"]["openInterest"])
        if ce_oi > 0:
            return round(pe_oi / ce_oi, 2)
    except Exception as e:
        print("⚠ compute_pcr:", e)
    return None

def compute_oi_levels(oc, spot):
    """Return (support, resistance) using nearest strong PE/CE OI strikes around spot."""
    support = CACHE["support"]
    resistance = CACHE["resistance"]
    try:
        pe_levels = []
        ce_levels = []
        for row in oc["records"]["data"]:
            strike = row.get("strikePrice")
            if strike is None:
                continue
            if "PE" in row and row["PE"].get("openInterest") is not None:
                pe_levels.append((strike, int(row["PE"]["openInterest"])))
            if "CE" in row and row["CE"].get("openInterest") is not None:
                ce_levels.append((strike, int(row["CE"]["openInterest"])))

        # strongest below/above spot
        pe_below = [x for x in pe_levels if x[0] <= (spot or 0)]
        ce_above = [x for x in ce_levels if x[0] >= (spot or 0)]

        pe_below.sort(key=lambda x: (abs((spot or 0) - x[0]), -x[1]))  # nearest with high OI
        ce_above.sort(key=lambda x: (abs(x[0] - (spot or 0)), -x[1]))

        if pe_below:
            support = pe_below[0][0]
        if ce_above:
            resistance = ce_above[0][0]
    except Exception as e:
        print("⚠ compute_oi_levels:", e)

    CACHE["support"] = support
    CACHE["resistance"] = resistance
    return support, resistance

def compute_max_pain(oc):
    try:
        oi_by_strike = {}
        for row in oc["records"]["data"]:
            strike = row.get("strikePrice")
            if strike is None:
                continue
            ce = int(row["CE"]["openInterest"]) if "CE" in row and row["CE"].get("openInterest") else 0
            pe = int(row["PE"]["openInterest"]) if "PE" in row and row["PE"].get("openInterest") else 0
            oi_by_strike[strike] = oi_by_strike.get(strike, 0) + ce + pe
        if not oi_by_strike:
            return CACHE["max_pain"]
        # strike with max total OI
        max_strike = max(oi_by_strike, key=lambda k: oi_by_strike[k])
        CACHE["max_pain"] = max_strike
        return max_strike
    except Exception as e:
        print("⚠ compute_max_pain:", e)
        return CACHE["max_pain"]

# ----------------------------
# Yahoo intraday → RSI(14)
# ----------------------------
def fetch_yahoo_intraday_closes():
    data = json_get(YAHOO_INTRADAY, timeout=8)
    if not data or "chart" not in data or not data["chart"].get("result"):
        return None
    try:
        result = data["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        # filter None
        closes = [c for c in closes if c is not None]
        return closes if len(closes) >= 15 else None
    except Exception as e:
        print("⚠ parse yahoo:", e)
        return None

def rsi(series, period=14):
    if not series or len(series) <= period:
        return None
    arr = np.array(series, dtype=float)
    deltas = np.diff(arr)
    ups = np.where(deltas > 0, deltas, 0.0)
    downs = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(ups[-period:]) if np.any(ups) else 0.0
    avg_loss = np.mean(downs[-period:]) if np.any(downs) else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def fetch_rsi():
    try:
        closes = fetch_yahoo_intraday_closes()
        value = rsi(closes, 14) if closes else None
        if value is not None:
            CACHE["rsi"] = value
        return CACHE["rsi"]
    except Exception as e:
        print("⚠ fetch_rsi:", e)
        return CACHE["rsi"]

# ----------------------------
# Signal logic
# ----------------------------
def build_signal(spot, pcr, vix, rsi_val, support, resistance, max_pain):
    # Heuristics with confidence scoring
    score = 0
    reasons = []

    if spot and support and resistance:
        if support < spot < resistance:
            reasons.append("Within OI S/R range (neutral).")
        if spot > resistance:
            score += 2; reasons.append("Spot above OI resistance (bullish breakout).")
        if spot < support:
            score -= 2; reasons.append("Spot below OI support (bearish breakdown).")
        # distance to max pain
        if max_pain:
            dist_mp = spot - max_pain
            if abs(dist_mp) < 50:
                reasons.append("Near max pain (mean-revert risk).")
            elif dist_mp > 0:
                score += 1; reasons.append("Spot above max pain (bullish bias).")
            else:
                score -= 1; reasons.append("Spot below max pain (bearish bias).")

    if pcr is not None:
        if pcr > 1.3:
            score += 1; reasons.append(f"High PCR {pcr} (bullish).")
        elif pcr < 0.7:
            score -= 1; reasons.append(f"Low PCR {pcr} (bearish).")
        else:
            reasons.append(f"Neutral PCR {pcr}.")

    if rsi_val is not None:
        if 45 <= rsi_val <= 55:
            reasons.append(f"RSI {rsi_val} neutral.")
        elif rsi_val < 35:
            score -= 1; reasons.append(f"RSI {rsi_val} oversold → bearish momentum risk.")
        elif rsi_val > 65:
            score += 1; reasons.append(f"RSI {rsi_val} strong → bullish momentum.")

    if vix is not None:
        if vix >= 16:
            reasons.append(f"High VIX {vix} (volatility risk).")
            # dampen confidence
            score = score * 0.8

    # Final label
    label = "Neutral"
    if score >= 1.5:
        label = "Bullish"
    elif score <= -1.5:
        label = "Bearish"

    # Confidence (0-100) from |score|
    conf = int(max(0, min(100, 50 + (score * 15))))
    return label, conf, reasons

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def root():
    return jsonify({
        "status": "OK",
        "routes": ["/signal", "/pcr", "/levels", "/vix", "/health"]
    })

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/pcr")
def pcr_route():
    oc = fetch_option_chain()
    pcr = compute_pcr(oc) if oc else None
    if pcr is None:
        pcr = CACHE["pcr"]
    else:
        CACHE["pcr"] = pcr
    return jsonify({"pcr": pcr, "note": "Fallback to last known value if live unavailable."})

@app.route("/vix")
def vix_route():
    _, _, vix = fetch_nifty_and_vix()
    return jsonify({"vix": vix})

@app.route("/levels")
def levels_route():
    spot, _, _ = fetch_nifty_and_vix()
    oc = fetch_option_chain()
    support, resistance = (None, None)
    max_pain = None
    if oc:
        support, resistance = compute_oi_levels(oc, spot)
        max_pain = compute_max_pain(oc)
    # update cache if new
    if support is not None: CACHE["support"] = support
    if resistance is not None: CACHE["resistance"] = resistance
    if max_pain is not None: CACHE["max_pain"] = max_pain
    return jsonify({
        "spot": spot,
        "support": support,
        "resistance": resistance,
        "max_pain": max_pain
    })

@app.route("/signal")
def signal_route():
    spot, prev_close, vix = fetch_nifty_and_vix()
    oc = fetch_option_chain()
    pcr = compute_pcr(oc) if oc else CACHE["pcr"]
    if pcr is not None:
        CACHE["pcr"] = pcr

    support, resistance = CACHE["support"], CACHE["resistance"]
    max_pain = CACHE["max_pain"]
    if oc:
        s2, r2 = compute_oi_levels(oc, spot)
        mp2 = compute_max_pain(oc)
        support = s2 or support
        resistance = r2 or resistance
        max_pain = mp2 or max_pain

    rsi_val = fetch_rsi()

    label, conf, reasons = build_signal(spot, pcr, vix, rsi_val, support, resistance, max_pain)
    return jsonify({
        "spot": spot,
        "prev_close": prev_close,
        "vix": vix,
        "pcr": pcr,
        "rsi14": rsi_val,
        "support": support,
        "resistance": resistance,
        "max_pain": max_pain,
        "signal": label,
        "confidence": conf,
        "reasons": reasons,
        "note": "All endpoints are resilient to NSE/Yahoo hiccups; values may be cached if live feed blocks."
    })

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # initial warmup to avoid first-hit failures
    warmup()
    app.run(host="0.0.0.0", port=5000)

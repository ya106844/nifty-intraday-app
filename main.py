from flask import Flask, jsonify, Response
import requests, time
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

CACHE = {
    "pcr": None,
    "vix": None,
    "nifty_last": None,
    "nifty_prev_close": None,
    "rsi": None,
    "support": None,
    "resistance": None,
    "max_pain": None,
    "closes": None,
    "ts": 0
}

# ----------------------------
# Utility functions
# ----------------------------

def warmup():
    try:
        SESSION.get("https://www.nseindia.com", headers=HEADERS, timeout=6)
    except Exception as e:
        print("Warmup warning:", e)

def json_get(url, timeout=8):
    try:
        if time.time() - CACHE["ts"] > 30:
            warmup()
            CACHE["ts"] = time.time()
        r = SESSION.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            print(f"HTTP {r.status_code} for {url}")
            return None
        return r.json()
    except Exception as e:
        print("json_get error:", e)
        return None

# ----------------------------
# Live data fetchers
# ----------------------------

def get_nifty_spot():
    try:
        data = json_get("https://www.nseindia.com/api/marketStatus")
        for idx in data.get("marketState", []):
            if idx.get("market") == "NIFTY 50":
                return float(idx.get("last"))
    except Exception as e:
        print("Error fetching spot:", e)
    return 24585.0  # fallback

def get_vix():
    try:
        data = json_get("https://www.nseindia.com/api/allIndices")
        for idx in data.get("data", []):
            if idx.get("index") == "India VIX":
                return float(idx.get("last"))
    except Exception as e:
        print("Error fetching VIX:", e)
    return 12.5  # fallback

def fetch_option_chain():
    data = json_get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY")
    return data

def compute_pcr(oc):
    try:
        ce_oi = sum(int(r["CE"]["openInterest"]) for r in oc["records"]["data"] if "CE" in r and r["CE"].get("openInterest"))
        pe_oi = sum(int(r["PE"]["openInterest"]) for r in oc["records"]["data"] if "PE" in r and r["PE"].get("openInterest"))
        if ce_oi > 0: return round(pe_oi / ce_oi, 2)
    except: return None

def compute_oi_levels(oc, spot):
    support = CACHE["support"]
    resistance = CACHE["resistance"]
    try:
        pe_levels = [(r["strikePrice"], int(r["PE"]["openInterest"])) for r in oc["records"]["data"] if "PE" in r and r["PE"].get("openInterest")]
        ce_levels = [(r["strikePrice"], int(r["CE"]["openInterest"])) for r in oc["records"]["data"] if "CE" in r and r["CE"].get("openInterest")]
        pe_below = sorted([x for x in pe_levels if x[0] <= (spot or 0)], key=lambda x: (abs((spot or 0)-x[0]), -x[1]))
        ce_above = sorted([x for x in ce_levels if x[0] >= (spot or 0)], key=lambda x: (abs(x[0]-(spot or 0)),-x[1]))
        if pe_below: support = pe_below[0][0]
        if ce_above: resistance = ce_above[0][0]
    except: pass
    CACHE["support"] = support
    CACHE["resistance"] = resistance
    return support, resistance

def compute_max_pain(oc):
    try:
        oi_by_strike = {}
        for r in oc["records"]["data"]:
            strike = r.get("strikePrice")
            if strike is None: continue
            ce = int(r["CE"]["openInterest"]) if "CE" in r and r["CE"].get("openInterest") else 0
            pe = int(r["PE"]["openInterest"]) if "PE" in r and r["PE"].get("openInterest") else 0
            oi_by_strike[strike] = oi_by_strike.get(strike,0) + ce + pe
        if not oi_by_strike: return CACHE["max_pain"]
        max_strike = max(oi_by_strike, key=lambda k: oi_by_strike[k])
        CACHE["max_pain"] = max_strike
        return max_strike
    except: return CACHE["max_pain"]

def fetch_yahoo_intraday_closes():
    try:
        url="https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=1d&interval=5m"
        data = json_get(url)
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        closes = [c for c in closes if c is not None]
        if closes: CACHE["closes"]=closes
        return CACHE["closes"]
    except: return CACHE["closes"]

def rsi(series, period=14):
    if not series or len(series)<=period: return None
    arr=np.array(series,dtype=float)
    deltas=np.diff(arr)
    ups = np.where(deltas>0,deltas,0.0)
    downs = np.where(deltas<0,-deltas,0.0)
    avg_gain = np.mean(ups[-period:]) if np.any(ups) else 0.0
    avg_loss = np.mean(downs[-period:]) if np.any(downs) else 0.0
    if avg_loss==0: return 100.0
    rs = avg_gain/avg_loss
    return round(100-(100/(1+rs)),2)

def fetch_rsi():
    closes = fetch_yahoo_intraday_closes()
    val = rsi(closes,14) if closes else None
    if val is not None: CACHE["rsi"]=val
    return CACHE["rsi"]

def build_signal(spot, pcr, vix, rsi_val, support, resistance, max_pain):
    score = 0
    reasons=[]
    if spot and support and resistance:
        if spot>resistance: score+=2; reasons.append("Above OI resistance (breakout).")
        elif spot<support: score-=2; reasons.append("Below OI support (breakdown).")
        else: reasons.append("Inside OI S/R range (neutral).")
        if max_pain:
            dist_mp=spot-max_pain
            if abs(dist_mp)<50: reasons.append("Near max pain (mean-revert).")
            elif dist_mp>0: score+=1; reasons.append("Above max pain (bullish bias).")
            else: score-=1; reasons.append("Below max pain (bearish bias).")
    if pcr is not None:
        if pcr>1.3: score+=1; reasons.append(f"PCR {pcr} high (bullish).")
        elif pcr<0.7: score-=1; reasons.append(f"PCR {pcr} low (bearish).")
        else: reasons.append(f"PCR {pcr} neutral.")
    if rsi_val is not None:
        if rsi_val>65: score+=1; reasons.append(f"RSI {rsi_val} strong.")
        elif rsi_val<35: score-=1; reasons.append(f"RSI {rsi_val} weak.")
        else: reasons.append(f"RSI {rsi_val} neutral.")
    if vix is not None and vix>=16:
        reasons.append(f"High VIX {vix} (volatility risk).")
        score=score*0.8
    label="Neutral"
    if score>=1.5: label="Bullish"
    elif score<=-1.5: label="Bearish"
    conf=int(max(0,min(100,50+(score*15))))
    return label, conf, reasons

# ----------------------------
# Routes
# ----------------------------

@app.route("/health")
def health(): return jsonify({"ok": True})

@app.route("/pcr")
def pcr_route():
    oc = fetch_option_chain()
    pcr = compute_pcr(oc) if oc else CACHE["pcr"]
    if pcr is not None: CACHE["pcr"]=pcr
    return jsonify({"pcr": pcr})

@app.route("/vix")
def vix_route():
    v = get_vix()
    return jsonify({"vix": v})

@app.route("/levels")
def levels_route():
    spot = get_nifty_spot()
    oc = fetch_option_chain()
    support, resistance = compute_oi_levels(oc, spot) if oc else (CACHE["support"], CACHE["resistance"])
    max_pain = compute_max_pain(oc) if oc else CACHE["max_pain"]
    return jsonify({"spot": spot, "support": support, "resistance": resistance, "max_pain": max_pain})

@app.route("/chart")
def chart_route():
    closes = fetch_yahoo_intraday_closes()
    return jsonify({"closes": closes[-60:] if closes else None})

@app.route("/signal")
def signal_route():
    spot = get_nifty_spot()
    vix = get_vix()
    oc = fetch_option_chain()
    pcr = compute_pcr(oc) if oc else CACHE["pcr"]
    support, resistance = compute_oi_levels(oc, spot) if oc else (CACHE["support"], CACHE["resistance"])
    max_pain = compute_max_pain(oc) if oc else CACHE["max_pain"]
    rsi_val = fetch_rsi()
    label, conf, reasons = build_signal(spot, pcr, vix, rsi_val, support, resistance, max_pain)
    return jsonify({
        "spot": spot, "vix": vix, "pcr": pcr, "rsi14": rsi_val,
        "support": support, "resistance": resistance, "max_pain": max_pain,
        "signal": label, "confidence": conf, "reasons": reasons
    })

# ----------------------------
# Dashboard UI
# ----------------------------
DASHBOARD_HTML = """<HTML+JS content same as previous code>"""  # keep your previous dashboard HTML here

@app.route("/")
def root():
    return Response(DASHBOARD_HTML, mimetype="text/html")

# ----------------------------
if __name__ == "__main__":
    warmup()
    app.run(host="0.0.0.0", port=5000)

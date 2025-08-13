from flask import Flask, jsonify, Response
import requests, time
import numpy as np
import json

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

def warmup():
    try:
        SESSION.get(NSE_HOME, headers=HEADERS, timeout=6)
    except Exception as e:
        print("Warmup warn:", e)

def json_get(url, timeout=8):
    try:
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
    support = CACHE["support"]
    resistance = CACHE["resistance"]
    try:
        pe_levels, ce_levels = [], []
        for row in oc["records"]["data"]:
            strike = row.get("strikePrice")
            if strike is None: continue
            if "PE" in row and row["PE"].get("openInterest") is not None:
                pe_levels.append((strike, int(row["PE"]["openInterest"])))
            if "CE" in row and row["CE"].get("openInterest") is not None:
                ce_levels.append((strike, int(row["CE"]["openInterest"])))
        pe_below = [x for x in pe_levels if x[0] <= (spot or 0)]
        ce_above = [x for x in ce_levels if x[0] >= (spot or 0)]
        pe_below.sort(key=lambda x: (abs((spot or 0) - x[0]), -x[1]))
        ce_above.sort(key=lambda x: (abs(x[0] - (spot or 0)), -x[1]))
        if pe_below: support = pe_below[0][0]
        if ce_above: resistance = ce_above[0][0]
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
            if strike is None: continue
            ce = int(row["CE"]["openInterest"]) if "CE" in row and row["CE"].get("openInterest") else 0
            pe = int(row["PE"]["openInterest"]) if "PE" in row and row["PE"].get("openInterest") else 0
            oi_by_strike[strike] = oi_by_strike.get(strike, 0) + ce + pe
        if not oi_by_strike:
            return CACHE["max_pain"]
        max_strike = max(oi_by_strike, key=lambda k: oi_by_strike[k])
        CACHE["max_pain"] = max_strike
        return max_strike
    except Exception as e:
        print("⚠ compute_max_pain:", e)
        return CACHE["max_pain"]

def fetch_yahoo_intraday_closes():
    data = json_get(YAHOO_INTRADAY, timeout=8)
    if not data or "chart" not in data or not data["chart"].get("result"):
        return CACHE["closes"]
    try:
        result = data["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        closes = [c for c in closes if c is not None]
        if closes and len(closes) >= 15:
            CACHE["closes"] = closes
        return CACHE["closes"]
    except Exception as e:
        print("⚠ parse yahoo:", e)
        return CACHE["closes"]

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

def build_signal(spot, pcr, vix, rsi_val, support, resistance, max_pain):
    score = 0
    reasons = []
    if spot and support and resistance:
        if spot > resistance: score += 2; reasons.append("Above OI resistance (breakout).")
        elif spot < support: score -= 2; reasons.append("Below OI support (breakdown).")
        else: reasons.append("Inside OI S/R range (neutral).")
        if max_pain:
            dist_mp = spot - max_pain
            if abs(dist_mp) < 50: reasons.append("Near max pain (mean-revert risk).")
            elif dist_mp > 0: score += 1; reasons.append("Above max pain (bullish bias).")
            else: score -= 1; reasons.append("Below max pain (bearish bias).")
    if pcr is not None:
        if pcr > 1.3: score += 1; reasons.append(f"PCR {pcr} high (bullish).")
        elif pcr < 0.7: score -= 1; reasons.append(f"PCR {pcr} low (bearish).")
        else: reasons.append(f"PCR {pcr} neutral.")
    if rsi_val is not None:
        if rsi_val > 65: score += 1; reasons.append(f"RSI {rsi_val} strong.")
        elif rsi_val < 35: score -= 1; reasons.append(f"RSI {rsi_val} weak.")
        else: reasons.append(f"RSI {rsi_val} neutral.")
    if vix is not None and vix >= 16:
        reasons.append(f"High VIX {vix} (volatility risk).")
        score = score * 0.8  # dampen confidence
    label = "Neutral"
    if score >= 1.5: label = "Bullish"
    elif score <= -1.5: label = "Bearish"
    conf = int(max(0, min(100, 50 + (score * 15))))
    return label, conf, reasons

# ----------------------------
# API Routes
# ----------------------------
@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/pcr")
def pcr_route():
    oc = fetch_option_chain()
    pcr = compute_pcr(oc) if oc else CACHE["pcr"]
    if pcr is not None: CACHE["pcr"] = pcr
    return jsonify({"pcr": pcr, "note": "Fallback to last known value if live unavailable."})

@app.route("/vix")
def vix_route():
    _, _, vix = fetch_nifty_and_vix()
    return jsonify({"vix": vix})

@app.route("/levels")
def levels_route():
    spot, _, _ = fetch_nifty_and_vix()
    oc = fetch_option_chain()
    support, resistance = CACHE["support"], CACHE["resistance"]
    max_pain = CACHE["max_pain"]
    if oc:
        s2, r2 = compute_oi_levels(oc, spot)
        mp2 = compute_max_pain(oc)
        support = s2 or support
        resistance = r2 or resistance
        max_pain = mp2 or max_pain
    if support is not None: CACHE["support"] = support
    if resistance is not None: CACHE["resistance"] = resistance
    if max_pain is not None: CACHE["max_pain"] = max_pain
    return jsonify({"spot": spot, "support": support, "resistance": resistance, "max_pain": max_pain})

@app.route("/chart")
def chart_route():
    closes = fetch_yahoo_intraday_closes()
    return jsonify({"closes": closes[-60:] if closes else None})

@app.route("/signal")
def signal_route():
    spot, prev_close, vix = fetch_nifty_and_vix()
    oc = fetch_option_chain()
    pcr = compute_pcr(oc) if oc else CACHE["pcr"]
    if pcr is not None: CACHE["pcr"] = pcr
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
        "spot": spot, "prev_close": prev_close, "vix": vix, "pcr": pcr, "rsi14": rsi_val,
        "support": support, "resistance": resistance, "max_pain": max_pain,
        "signal": label, "confidence": conf, "reasons": reasons
    })

# ----------------------------
# Web Dashboard (root "/")
# ----------------------------
DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Nifty Intraday Dashboard</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root { color-scheme: dark light; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#0f1220; color:#e6e8f0; }
  .container { max-width: 1100px; margin: 0 auto; padding: 14px; }
  .grid { display: grid; gap: 12px; grid-template-columns: 1fr; }
  @media (min-width: 900px) { .grid { grid-template-columns: 2fr 1fr; } }
  .card { background:#151837; border:1px solid #262a57; border-radius:16px; padding:14px; box-shadow: 0 6px 20px rgba(0,0,0,.25); }
  .title { font-size: 18px; margin: 0 0 8px; opacity:.9 }
  .big { font-size: 28px; margin: 6px 0; font-weight: 700; }
  .row { display:flex; gap: 10px; flex-wrap: wrap; }
  .pill { padding:6px 10px; border-radius:999px; background:#1e2352; border:1px solid #2d3470; font-size: 13px; }
  .good { background:#103f2d; border-color:#1c6b4c; }
  .bad { background:#4a1a22; border-color:#7b2a36; }
  .neutral { background:#2e2e2e; border-color:#454545; }
  .signal { font-size: 20px; font-weight: 800; }
  .reason { font-size: 13px; opacity:.9; margin: 2px 0; }
  canvas { width:100%; height:260px; }
  a { color:#8ab4ff; text-decoration: none; }
  footer { opacity:.6; font-size:12px; text-align:center; margin-top: 12px; }
</style>
</head>
<body>
<div class="container">
  <div class="grid">
    <div class="card">
      <div class="title">Nifty Intraday — Live</div>
      <div class="row">
        <div class="pill" id="spot">Spot: —</div>
        <div class="pill" id="pcr">PCR: —</div>
        <div class="pill" id="vix">VIX: —</div>
        <div class="pill" id="rsi">RSI(14): —</div>
        <div class="pill" id="sr">S/R: —</div>
        <div class="pill" id="mp">Max Pain: —</div>
      </div>
      <div class="big signal" id="signalText">Signal: —</div>
      <div class="row" id="reasons"></div>
      <canvas id="chart"></canvas>
    </div>

    <div class="card">
      <div class="title">Quick Links</div>
      <div class="row">
        <a class="pill" href="/signal">/signal</a>
        <a class="pill" href="/levels">/levels</a>
        <a class="pill" href="/pcr">/pcr</a>
        <a class="pill" href="/vix">/vix</a>
        <a class="pill" href="/health">/health</a>
      </div>
      <p style="margin-top:10px; opacity:.8">Auto-refreshes every 60s. If any feed hiccups, values fall back to last known — the page will never crash.</p>
    </div>
  </div>
  <footer>Built for intraday use. Stay disciplined: size small, respect stops.</footer>
</div>

<script>
let lineChart = null;

function cls(name){ return name.toLowerCase()==="bullish"?"good":(name.toLowerCase()==="bearish"?"bad":"neutral"); }

async function fetchJSON(path){
  try{
    const r = await fetch(path, {cache:"no-cache"});
    return await r.json();
  }catch(e){ return {}; }
}

async function refresh(){
  const sig = await fetchJSON('/signal');
  const lev = await fetchJSON('/levels');
  const chart = await fetchJSON('/chart');

  // Pills
  document.getElementById('spot').textContent = 'Spot: ' + (sig.spot ?? '—');
  document.getElementById('pcr').textContent = 'PCR: ' + (sig.pcr ?? '—');
  document.getElementById('vix').textContent = 'VIX: ' + (sig.vix ?? '—');
  document.getElementById('rsi').textContent = 'RSI(14): ' + (sig.rsi14 ?? '—');
  document.getElementById('sr').textContent = 'S/R: ' + ((sig.support ?? lev.support ?? '—') + ' / ' + (sig.resistance ?? lev.resistance ?? '—'));
  document.getElementById('mp').textContent = 'Max Pain: ' + (sig.max_pain ?? '—');

  // Signal
  const signalText = document.getElementById('signalText');
  signalText.textContent = 'Signal: ' + (sig.signal ?? '—') + '  (' + (sig.confidence ?? 0) + '%)';
  signalText.className = 'big signal ' + cls(sig.signal || 'neutral');

  // Reasons
  const reasonsDiv = document.getElementById('reasons');
  reasonsDiv.innerHTML = '';
  (sig.reasons || []).slice(0, 5).forEach(r=>{
    const div = document.createElement('div');
    div.className = 'pill';
    div.textContent = r;
    reasonsDiv.appendChild(div);
  });

  // Chart
  const closes = (chart.closes || []);
  const labels = closes.map((_,i)=>i+1);
  const ctx = document.getElementById('chart').getContext('2d');
  if(lineChart) { lineChart.destroy(); }
  lineChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ data: closes, fill:false, tension:0.2 }] },
    options: {
      responsive:true,
      plugins:{ legend:{ display:false } },
      scales:{ x:{ display:false }, y:{ display:true } }
    }
  });
}

refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>
"""

@app.route("/")
def root():
    return Response(DASHBOARD_HTML, mimetype="text/html")

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    warmup()
    app.run(host="0.0.0.0", port=5000)

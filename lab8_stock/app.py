from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime, timedelta
app = Flask(__name__)
AV_KEY = "854fe4aedf93f63d5b29c9a7207e52d5"
AV_BASE = "https://www.alphavantage.co/query"
POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOGL", "name": "Alphabet (Google)"},
    {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "META", "name": "Meta Platforms"},
    {"symbol": "NFLX", "name": "Netflix"},
]
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/popular")
def popular():
    return jsonify({"stocks": POPULAR_STOCKS})
@app.route("/quote", methods=["POST"])
def get_quote():
    symbol = request.json.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Please enter a stock symbol."}), 400
    try:
        resp = requests.get(AV_BASE, params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": AV_KEY
        }, timeout=10)
        data = resp.json()
        if "Note" in data:
            return jsonify({"error": "API rate limit reached. Please wait 1 minute."}), 429
        gq = data.get("Global Quote", {})
        if not gq or not gq.get("05. price"):
            return jsonify({"error": f"Symbol '{symbol}' not found. Try AAPL, MSFT, TSLA, etc."}), 404
        price = float(gq["05. price"])
        change = float(gq["09. change"])
        change_pct = gq["10. change percent"].replace("%", "")
        change_pct_f = float(change_pct)
        return jsonify({
            "symbol": gq["01. symbol"],
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct_f, 2),
            "open": round(float(gq["02. open"]), 2),
            "high": round(float(gq["03. high"]), 2),
            "low": round(float(gq["04. low"]), 2),
            "prev_close": round(float(gq["08. previous close"]), 2),
            "volume": int(gq["06. volume"]),
            "latest_day": gq["07. latest trading day"],
            "direction": "up" if change >= 0 else "down"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/history", methods=["POST"])
def get_history():
    symbol = request.json.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    try:
        resp = requests.get(AV_BASE, params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": AV_KEY
        }, timeout=15)
        data = resp.json()
        if "Note" in data:
            return jsonify({"error": "API rate limit. Please wait."}), 429
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return jsonify({"error": "No historical data found."}), 404
        dates = sorted(ts.keys())[-30:]  
        history = []
        for d in dates:
            history.append({
                "date": d,
                "open": float(ts[d]["1. open"]),
                "high": float(ts[d]["2. high"]),
                "low": float(ts[d]["3. low"]),
                "close": float(ts[d]["4. close"]),
                "volume": int(ts[d]["5. volume"])
            })
        return jsonify({"symbol": symbol, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/search-symbol", methods=["POST"])
def search_symbol():
    keywords = request.json.get("keywords", "").strip()
    try:
        resp = requests.get(AV_BASE, params={
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "apikey": AV_KEY
        }, timeout=10)
        data = resp.json()
        matches = []
        for m in data.get("bestMatches", [])[:6]:
            matches.append({
                "symbol": m.get("1. symbol", ""),
                "name": m.get("2. name", ""),
                "region": m.get("4. region", ""),
                "type": m.get("3. type", "")
            })
        return jsonify({"matches": matches})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)

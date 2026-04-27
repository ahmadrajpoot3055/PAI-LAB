from flask import Flask, render_template, request, jsonify
from groq import Groq
import os
app = Flask(__name__)
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    notes = data.get("notes", "").strip()
    summary_length = data.get("length", "Short")
    summary_format = data.get("format", "Bullet Points")
    
    if not notes:
        return jsonify({"error": "Notes cannot be empty"}), 400
        
    system_prompt = f"You are an expert AI assistant specializing in text summarization. Transform the user's lengthy notes into concise, clear, and meaningful key points. Output length: {summary_length}. Output format: {summary_format}."
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": notes}
            ],
            temperature=0.7
        )
        
        summary = response.choices[0].message.content
        return jsonify({"summary": summary})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5002)

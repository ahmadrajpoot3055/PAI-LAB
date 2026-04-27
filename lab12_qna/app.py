"""
Lab 12 – QnA Bot: Study & Academic Knowledge Base
Batch 2
Pipeline: Preprocess → MiniLM Embeddings → FAISS → Flask UI
Install:
  pip install flask sentence-transformers faiss-cpu numpy
"""
STUDY_QA = [
    ("What is machine learning?",
     "Machine learning is a branch of AI that enables computers to learn patterns from data and make decisions or predictions without being explicitly programmed for each task."),
    ("What is the difference between supervised and unsupervised learning?",
     "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Unsupervised learning finds hidden patterns in unlabeled data without predefined categories."),
    ("What is overfitting in machine learning?",
     "Overfitting occurs when a model learns the training data too well, including noise and outliers, causing it to perform poorly on new, unseen data. It memorizes rather than generalizes."),
    ("What is gradient descent?",
     "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that minimizes the loss function, using the gradient (slope) to guide each step."),
    ("What is a neural network?",
     "A neural network is a computing system inspired by the human brain, consisting of layers of interconnected nodes (neurons). Each layer transforms the input, and the final layer produces a prediction."),
    ("What is backpropagation?",
     "Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss with respect to each weight by applying the chain rule from output layer back to input."),
    ("What is the difference between precision and recall?",
     "Precision measures what fraction of positive predictions were actually correct. Recall measures what fraction of actual positives were correctly identified. There's often a tradeoff between the two."),
    ("What is a transformer model?",
     "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data in parallel rather than step by step. It is the foundation of modern LLMs like GPT and BERT."),
    ("What is regularization in machine learning?",
     "Regularization adds a penalty to the model's loss function to discourage complexity, preventing overfitting. Common methods include L1 (Lasso), L2 (Ridge), and Dropout for neural networks."),
    ("What is cross-validation?",
     "Cross-validation is a technique to evaluate model performance by splitting data into k folds, training on k-1 folds and testing on the remaining fold, repeating k times and averaging results."),
    ("What is the bias-variance tradeoff?",
     "Bias is error from overly simple models that underfit. Variance is error from overly complex models that overfit. The tradeoff is finding the right model complexity that minimizes total error."),
    ("What is natural language processing?",
     "NLP is a field of AI focused on enabling computers to understand, interpret, and generate human language. It powers applications like chatbots, translation, sentiment analysis, and text summarization."),
    ("What is reinforcement learning?",
     "Reinforcement learning trains an agent to make sequential decisions by rewarding good actions and penalizing bad ones. The agent learns a policy that maximizes cumulative reward over time."),
    ("What is the difference between bagging and boosting?",
     "Bagging trains multiple models in parallel on random subsets and averages predictions (e.g., Random Forest). Boosting trains models sequentially, each correcting errors of the previous one (e.g., XGBoost)."),
    ("What is transfer learning?",
     "Transfer learning uses a model pre-trained on a large dataset and fine-tunes it for a specific task with a smaller dataset. It saves computation and works well when task-specific data is limited."),
]
from flask import Flask, render_template_string, request, jsonify
app = Flask(__name__)
try:
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np
    print("Loading MiniLM model...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    questions = [qa[0] for qa in STUDY_QA]
    answers   = [qa[1] for qa in STUDY_QA]
    embeddings = model.encode(questions).astype('float32')
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(f"FAISS ready: {index.ntotal} Q&A pairs indexed ({d} dims)")
    READY = True
except ImportError as e:
    print(f"Missing libraries: {e}\nInstall: pip install sentence-transformers faiss-cpu")
    READY = False
def search(query, k=3):
    if not READY:
        return [{"question": "Library Error", "answer": "pip install sentence-transformers faiss-cpu", "similarity": 0}]
    q_vec = model.encode([query]).astype('float32')
    D, I = index.search(q_vec, k)
    return [{"question": questions[i], "answer": answers[i],
             "similarity": round(max(0, 1 - float(d) / 10), 2)}
            for d, i in zip(D[0], I[0])]
HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StudyBot QnA – AI Learning Assistant</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',sans-serif;background:#f0fdf4;color:#1a2e1a;min-height:100vh}
  header{background:linear-gradient(135deg,#14532d,#166534);padding:18px 28px}
  header h1{color:#bbf7d0;font-size:1.4rem}
  header p{color:#86efac;font-size:.82rem;margin-top:3px}
  .main{max-width:800px;margin:0 auto;padding:28px 18px}
  .search-box{background:#fff;border:1px solid #d1fae5;border-radius:14px;padding:22px;margin-bottom:22px;box-shadow:0 2px 10px rgba(0,0,0,.05)}
  .search-box h2{color:#166534;font-size:.9rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
  .row{display:flex;gap:10px}
  .row input{flex:1;padding:12px 16px;background:#f0fdf4;border:1.5px solid #d1fae5;border-radius:10px;color:#1a2e1a;font-size:.93rem;outline:none;transition:border .3s}
  .row input:focus{border-color:#22c55e}
  .row button{padding:12px 22px;background:linear-gradient(135deg,#16a34a,#15803d);border:none;border-radius:10px;color:#fff;font-weight:700;cursor:pointer;transition:transform .2s}
  .row button:hover{transform:translateY(-1px)}
  .suggestions{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
  .sug{padding:5px 12px;background:#dcfce7;border:none;border-radius:18px;color:#166534;font-size:.76rem;cursor:pointer;font-weight:600;transition:background .2s}
  .sug:hover{background:#bbf7d0}
  .result-card{background:#fff;border:1px solid #d1fae5;border-radius:12px;padding:18px;margin-bottom:12px;border-left:3px solid #22c55e;box-shadow:0 2px 8px rgba(0,0,0,.04)}
  .res-q{font-size:.82rem;color:#22c55e;margin-bottom:8px;font-style:italic;font-weight:600}
  .res-a{font-size:.9rem;color:#374151;line-height:1.65}
  .sim-bar{height:5px;background:#22c55e;border-radius:3px;margin-top:10px;transition:width .5s}
  .sim-label{font-size:.72rem;color:#6b7280;margin-top:4px}
  .empty{text-align:center;padding:40px;color:#86efac;font-size:1rem}
  .loading{text-align:center;padding:20px;color:#166534;display:none;font-weight:600}
</style></head>
<body>
<header><h1>📚 StudyBot QnA</h1><p>AI & Machine Learning Knowledge Base — Powered by MiniLM + FAISS</p></header>
<div class="main">
  <div class="search-box">
    <h2>Ask a Study Question</h2>
    <div class="row">
      <input id="q" type="text" placeholder="e.g. What is overfitting?">
      <button onclick="ask()">Search</button>
    </div>
    <div class="suggestions">
      <button class="sug" onclick="fill('What is machine learning?')">Machine Learning</button>
      <button class="sug" onclick="fill('What is overfitting?')">Overfitting</button>
      <button class="sug" onclick="fill('Explain neural networks')">Neural Networks</button>
      <button class="sug" onclick="fill('What is gradient descent?')">Gradient Descent</button>
      <button class="sug" onclick="fill('What is transfer learning?')">Transfer Learning</button>
    </div>
  </div>
  <div class="loading" id="loading">🔍 Searching knowledge base...</div>
  <div id="results"><div class="empty">✨ Ask any AI/ML question above to get started</div></div>
</div>
<script>
  document.getElementById('q').addEventListener('keypress',e=>{if(e.key==='Enter')ask()});
  function fill(t){document.getElementById('q').value=t;ask()}
  async function ask(){
    const q=document.getElementById('q').value.trim();
    if(!q)return;
    document.getElementById('loading').style.display='block';
    document.getElementById('results').innerHTML='';
    const res=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
    const data=await res.json();
    document.getElementById('loading').style.display='none';
    document.getElementById('results').innerHTML=data.results.map(r=>`
      <div class="result-card">
        <div class="res-q">📖 Related: "${r.question}"</div>
        <div class="res-a">${r.answer}</div>
        <div class="sim-bar" style="width:${r.similarity*100}%"></div>
        <div class="sim-label">Relevance: ${(r.similarity*100).toFixed(0)}%</div>
      </div>`).join('');
  }
</script>
</body></html>"""
@app.route("/")
def index(): return render_template_string(HTML)
@app.route("/search", methods=["POST"])
def search_route():
    query = request.json.get("query", "")
    return jsonify({"results": search(query)})
if __name__ == "__main__":
    app.run(debug=True)

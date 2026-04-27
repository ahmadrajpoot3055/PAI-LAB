import nltk, re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def preprocess(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return " ".join(stemmer.stem(w) for w in word_tokenize(text) if w not in stop_words and len(w) > 2)
print("=" * 55)
print("1. TEXT PREPROCESSING")
print("=" * 55)
raw = "CONGRATULATIONS! You WON a FREE iPhone. Click HERE now!!!"
print(f"Raw:       {raw}")
print(f"Processed: {preprocess(raw)}\n")
spam = [
    "Win a free iPhone click here now claim your prize",
    "URGENT you have been selected for a cash prize 10000",
    "Free entry win competition call to claim reward today",
    "You are winner claim your prize immediately limited time",
    "Get rich quick make money from home no experience needed",
    "Exclusive offer buy now get 90 percent off limited deal",
    "Your account suspended verify immediately click link now",
    "Win holiday trip free text YES to 8877 today only",
    "Earn 5000 weekly working from home guaranteed income",
    "Free gift card limited stock click to redeem now"
]
ham = [
    "Hey can we meet tomorrow for the project discussion",
    "The lecture notes have been uploaded to the portal",
    "Please review the attached document and give feedback",
    "I will be late to meeting by about fifteen minutes",
    "Can you send me the report when you get a chance",
    "The team dinner is scheduled for Friday evening at seven",
    "Your order has been shipped and will arrive on Thursday",
    "Reminder your dentist appointment is next Monday at three",
    "Please find attached the invoice for this month review",
    "Looking forward to seeing you at the conference next week"
]
all_msgs = spam + ham
all_labels = ["spam"] * len(spam) + ["ham"] * len(ham)
processed = [preprocess(m) for m in all_msgs]
print("=" * 55)
print("2. SPAM DETECTION (Naive Bayes vs LinearSVC)")
print("=" * 55)
X_train, X_test, y_train, y_test = train_test_split(processed, all_labels, test_size=0.25, random_state=7)
vec = TfidfVectorizer()
X_tr = vec.fit_transform(X_train)
X_te = vec.transform(X_test)
nb = MultinomialNB(); nb.fit(X_tr, y_train)
svm = LinearSVC(); svm.fit(X_tr, y_train)
print(f"Naive Bayes accuracy : {accuracy_score(y_test, nb.predict(X_te))*100:.1f}%")
print(f"LinearSVC accuracy   : {accuracy_score(y_test, svm.predict(X_te))*100:.1f}%\n")
test_msgs = ["Claim your free prize by clicking this link today", "Can you bring your notes to the study session tonight"]
for msg in test_msgs:
    p = vec.transform([preprocess(msg)])
    print(f"  '{msg[:50]}...'  NB={nb.predict(p)[0].upper()}")
print("\n" + "=" * 55)
print("3. EXTRACTIVE TEXT SUMMARIZATION")
print("=" * 55)
def summarize(text, n=2):
    sents = sent_tokenize(text)
    words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in stop_words]
    freq = Counter(words)
    max_f = max(freq.values()) if freq else 1
    scores = {}
    for i, s in enumerate(sents):
        for w in word_tokenize(s.lower()):
            if w in freq: scores[i] = scores.get(i, 0) + freq[w] / max_f
    top = sorted(scores, key=scores.get, reverse=True)[:n]
    return " ".join(sents[i] for i in sorted(top))
article = """
Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.
NLP is used in many real-world applications like chatbots, translation, and spam detection.
Deep learning models like BERT and GPT have revolutionized the field.
These models use transformers with attention mechanisms to process text contextually.
NLP continues to grow rapidly with new models being released every year.
"""
print("Original word count:", len(article.split()))
print("Summary:", summarize(article))

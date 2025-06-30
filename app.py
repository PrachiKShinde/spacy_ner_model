import spacy
from flask import Flask, request, jsonify

# Load your local trained model
nlp = spacy.load("custom_ner_model")

app = Flask(__name__)

@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    data = request.get_json()
    text = data.get("text", "")
    doc = nlp(text)
    results = [(ent.text, ent.label_) for ent in doc.ents]
    return jsonify({"entities": results})

@app.route('/')
def health():
    return "✅ Custom NER API is up"


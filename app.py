from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)
nlp = spacy.load("custom_ner_model")  # Your trained model folder

@app.route("/extract_entities", methods=["POST"])
def extract_entities():
    data = request.get_json()
    text = data.get("text", "")
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return jsonify({"entities": entities})


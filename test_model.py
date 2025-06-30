import spacy

# Load your custom NER model (make sure the model is already trained and saved in this folder)
nlp = spacy.load("custom_ner_model")

# Sample input for testing
text = "Helped Ayush test WeatherApp for 1 hour on 10 June by running multiple device checks."

# Process the input text
doc = nlp(text)

# Print entities found
if not doc.ents:
    print("❌ No entities found.")
else:
    for ent in doc.ents:
        print(f"{ent.text} -> {ent.label_}")


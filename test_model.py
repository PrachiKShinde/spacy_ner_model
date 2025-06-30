import spacy

nlp = spacy.load("custom_ner_model")  # Use your actual model folder name
text = "Helped Ayush test WeatherApp for 1 hour on 10 June by running multiple device checks."

doc = nlp(text)

if not doc.ents:
    print("❌ No entities found.")
else:
    for ent in doc.ents:
        print(f"{ent.text} -> {ent.label_}")


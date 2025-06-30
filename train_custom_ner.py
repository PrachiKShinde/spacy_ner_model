import spacy
from spacy.training.example import Example
from training_data import TRAIN_DATA
import random
import os

# Load blank English model
nlp = spacy.blank("en")

# Add NER pipe
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add custom labels to NER
labels = set()
for text, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        labels.add(ent[2])

for label in labels:
    ner.add_label(label)

# Disable other pipes for training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  
    optimizer = nlp.begin_training()
    for itn in range(50):  # Number of training iterations
        print(f"🔁 Iteration {itn + 1}/50")
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3, losses=losses)
        print(f"Losses: {losses}")

# Save the model to disk
output_dir = "custom_ner_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nlp.to_disk(output_dir)
print(f"✅ Model training complete. Saved to '{output_dir}'")


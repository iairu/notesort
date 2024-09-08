from transformers import pipeline
import json

# classifier = pipeline("sentiment-analysis", model="trained_model_1")
classifier = pipeline("text-classification", model="trained_model_1")

with open('infer_input.md', 'r') as file:
    paragraphs = file.read().split('\n\n')

results = []
for paragraph in paragraphs:
    if paragraph.strip():
        classification = classifier(paragraph)
        results.append({
            'text': paragraph,
            'result': classification[0]  # Get the first (and only) classification result
        })

with open('infer_output.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)

print(f"Classification complete. Results written to infer_output.json")

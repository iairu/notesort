from transformers import pipeline
import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="DistilBERT Inference Script")
parser.add_argument('-c', '--no-normalize', action='store_true', help="Do not normalize scores")
parser.add_argument('-l', '--top-label', action='store_true', help="Only output the top label name")
args = parser.parse_args()

# Load the classifier with top_k=None to get all label probabilities
classifier = pipeline("text-classification", model="trained_model_1", top_k=None)

# Load the labels with explicit words
with open('train_labels.json', 'r') as f:
    labels_data = json.load(f)

with open('infer_input.md', 'r') as file:
    paragraphs = file.read().split('\n\n')

results = []
for paragraph in paragraphs:
    if paragraph.strip():
        classifications = classifier(paragraph)  # Get the list of all label probabilities
        # Adjust scores based on explicit words
        for class_list in classifications:
            for classification in class_list:
                label = classification['label']
                label_index = next((index for index, data in labels_data.items() if data['label'] == label), None)
                if label_index is not None:
                    label_data = labels_data[label_index]
                    increase_if_words = label_data['increase_if']
                    decrease_if_words = label_data['decrease_if']
                    must_have_words = label_data['must_have']
                    
                    # Check for must_have words
                    if must_have_words:
                        if not any(word.lower() in paragraph.lower() for word in must_have_words):
                            classification['score'] = 0  # Set score to 0 if no must_have word is present
                    
                    # Check for increase_if words
                    if increase_if_words:
                        if any(word.lower() in paragraph.lower() for word in increase_if_words):
                            classification['score'] *= 1.3  # Increase score by 30%
                    
                    # Check for decrease_if words
                    if decrease_if_words:
                        if any(word.lower() in paragraph.lower() for word in decrease_if_words):
                            classification['score'] *= 0.5  # Decrease score by 50%

        # Normalize scores if -c argument is not passed
        if not args.no_normalize:
            for class_list in classifications:
                if class_list:  # Check if the list is not empty
                    scores = [classification['score'] for classification in class_list]
                    min_score = min(scores)
                    max_score = max(scores)
                    
                    # Avoid division by zero
                    if max_score != min_score:
                        for classification in class_list:
                            normalized_score = (classification['score'] - min_score) / (max_score - min_score) * 100.0
                            classification['score'] = normalized_score
                    else:
                        # If all scores are the same, set them to 100.0
                        for classification in class_list:
                            classification['score'] = 100.0

        # Sort the classifications by normalized score in descending order
        sorted_classifications = sorted(classifications[0], key=lambda x: x['score'], reverse=True)
        if args.top_label:
            results.append({
                'text': paragraph,
                'label': sorted_classifications[0]['label']  # Only include the top label without score
            })
        else:
            results.append({
                'text': paragraph,
                'results': sorted_classifications  # Include all sorted label probabilities
            })

with open('infer_output.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)

print(f"Classification complete. Results written to infer_output.json")

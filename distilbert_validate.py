# this script will validate the output of the distilbert_infer.py script
# it will compare the assigned labels in output in infer_output.json to the train_input.json file and print the results
# it will print the precision, recall, and f1 score for top label (if "results" list has more than one label or just "label" is present instead of "results" list)
# it will also print the overall precision, recall, and f1 score

import json
from sklearn.metrics import precision_recall_fscore_support
import warnings

def get_top_label(item):
    if 'results' in item:
        return item['results'][0]['label']
    return item['label']

with open('train_input.json', 'r') as file:
    train_dataset = json.load(file)

with open('infer_output.json', 'r') as file:
    infer_dataset = json.load(file)

with open('train_labels.json', 'r') as file:
    label_mapping = {str(k): v['label'] for k, v in json.load(file).items()}

# compare the labels and print mistakes
true_labels = []
predicted_labels = []
mistakes = []

# Create a dictionary to map text to labels from train_dataset
train_text_to_label = {item['text']: label_mapping[str(item['label'])] for item in train_dataset}

for infer_item in infer_dataset:
    text = infer_item['text']
    if text in train_text_to_label:
        true_label = train_text_to_label[text]
        predicted_label = get_top_label(infer_item)
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        
        if true_label != predicted_label:
            text_preview = text[:100] + ('...' if len(text) > 100 else '')
            mistakes.append((text_preview, predicted_label, true_label))

print("Mistakes:")
for text, pred_label, true_label in mistakes:
    print(f"Text: {text}")
    print(f"Predicted: {pred_label}")
    print(f"Correct: {true_label}")
    print()

# calculate metrics
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', zero_division=0)

# print the results
print("Top Label Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# calculate overall metrics
all_true_labels = []
all_predicted_labels = []

for infer_item in infer_dataset:
    text = infer_item['text']
    if text in train_text_to_label:
        true_label = train_text_to_label[text]
        if 'results' in infer_item:
            predicted_labels = [result['label'] for result in infer_item['results']]
        else:
            predicted_labels = [infer_item['label']]
        
        all_true_labels.extend([true_label] * len(predicted_labels))
        all_predicted_labels.extend(predicted_labels)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

print("\nOverall Metrics:")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall: {overall_recall:.4f}")
print(f"F1 Score: {overall_f1:.4f}")

# Calculate the percentage of mistakes compared to the amount of text in train dataset
total_train_texts = len(train_text_to_label)
total_mistakes = len(mistakes)

mistake_percentage = (total_mistakes / total_train_texts) * 100
correct_percentage = 100 - mistake_percentage

print("\nPrediction Accuracy:")
print(f"Correct: {correct_percentage:.2f}%")
print(f"Incorrect: {mistake_percentage:.2f}%")
print(f"Total mistakes: {total_mistakes}")
print(f"Total texts in train dataset: {total_train_texts}")

import json
import os
import shutil
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

with open('train_labels.json', 'r') as f:
    data = json.load(f)

id2label = {int(k): v['label'] for k, v in data.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

with open('train_input.json', 'r') as file:
    train_dataset = json.load(file)

# Convert the list of dictionaries to separate lists for text and labels
train_dict = {
    'text': [item['text'] for item in train_dataset],
    'label': [item['label'] for item in train_dataset]
}

dataset = Dataset.from_dict(train_dict)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", clean_up_tokenization_spaces=True)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Delete existing model if it exists
output_dir = "trained_model_1"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", 
    num_labels=num_labels, 
    id2label=id2label, 
    label2id=label2id,
    problem_type="single_label_classification"
)

training_args = TrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    save_total_limit=1,  # Keep only the last checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=16,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# Move the last checkpoint to the main save directory and remove the subfolder
last_checkpoint_dir = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")], key=os.path.getmtime)
for filename in os.listdir(last_checkpoint_dir):
    shutil.move(os.path.join(last_checkpoint_dir, filename), output_dir)
shutil.rmtree(last_checkpoint_dir)

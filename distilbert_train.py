import json
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

with open('train_labels.json', 'r') as f:
    data = json.load(f)

id2label = {int(k): v['label'] for k, v in data.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

with open('train_input.json', 'r') as file:
    train_dataset = json.load(file)

dataset = DatasetDict({
    'train': Dataset.from_dict(train_dataset),
    'validation': Dataset.from_dict(train_dataset),
    'test': Dataset.from_dict(train_dataset)
})

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=num_labels, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="trained_model_1",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

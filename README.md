# NoteSort: Simple DistilBERT Model Training and Inferring Workflow for Paragraph Sorting with a Labeling Interface

Ondrej Špánik &copy; 2024-09

## Package Setup

After cloning this repository and navigating to it, make sure to create and activate a virtual environment using `python3 -m venv venv` and `source venv/bin/activate` in order to use Python packages locally within the repository's context, then install them using `pip3 install -r requirements.txt`.

## Stage 1: Training Dataset (Manual Labeling)

In this stage, we input a manually labeled dataset to the model for training, enabling it to infer labels for an automated dataset in the second stage.

### Required Files:
- `train_labels.json`
- `train_input.json`

### Steps:

1. Create `train_labels.json` manually in the following format:
   ```json
   {
        "0": {"label": "poetry", "color": "#00FF00", "increase_if": [], "decrease_if": [], "must_have": []},
        "1": {"label": "description", "color": "#FFA500", "increase_if": [], "decrease_if": [], "must_have": []},
        "2": {"label": "spiritual", "color": "#FFD700", "increase_if": [], "decrease_if": [], "must_have": ["god",  "jesus", "religion"]},
        "3": {"label": "sadness", "color": "#4169E1", "increase_if": [], "decrease_if": [], "must_have": []},
        "4": {"label": "psychology", "color": "#800080", "increase_if": [], "decrease_if": [], "must_have": []}
   }
   ```

2. Prepare text with paragraphs for manual sorting (training).

3. Use `labeling.html` to load `train_labels.json` and sort the paragraphs into buckets.
   Launch command for Google Chrome: `chrome labeling.html`

4. Export the buckets as `train_input.json`.

5. Run `distilbert_train.py` to train the model on `train_input.json`. This will generate the model's directory. Note that the training is with validation/testing on the same data (edit the script yourself for improvements).
   Launch command: `py distilbert_train.py`

## Stage 2: Inferred Dataset (Automated Labeling)

After training the model, we use it to infer labels for an unlabeled dataset, which can be the rest of your data that hasn't been manually labeled yet.

### Required File:
- `infer_input.md`

### Steps:

1. Prepare text with paragraphs for automated (inferred) sorting.

2. Input the text in a clear format into `infer_input.md`.

3. Run `distilbert_infer.py` to execute the model. This will generate the output file `infer_output.json`.
   Launch command: `py distilbert_infer.py [options]`

   Available options:
   - `-c` or `--no-normalize`: Do not normalize scores
   - `-l` or `--top-label`: Only output the top label name
   - `-s` or `--skip-long`: Automatically skip paragraphs exceeding maximum length

   Example:
   - To run with default settings: `py distilbert_infer.py`
   - To run without score normalization: `py distilbert_infer.py -c`
   - To output only the top label: `py distilbert_infer.py -l`
   - To combine options: `py distilbert_infer.py -c -l`
   - To skip long paragraphs: `py distilbert_infer.py -s`
   ß
## Stage 3: Validation (Optional)

In this stage, we validate the output of the inference process by comparing it to the original training data and analyze the distribution of predicted labels.

### Required Files:
- `train_input.json`
- `infer_output.json`
- `train_labels.json`

### Steps:

1. Ensure you have the required files from the previous stages.

2. Run `distilbert_validate.py` to perform the validation. This script will:
   - Compare the assigned labels in `infer_output.json` to the labels in `train_input.json`.
   - Print any mistakes found during the comparison.
   - Calculate and display metrics including precision, recall, and F1 score for the top label.
   - Calculate and display overall metrics for all labels.
   - Show the percentage of correct and incorrect predictions.
   - Calculate and display the percentage distribution of predicted labels.
   Launch command: `py distilbert_validate.py`

3. Review the output, which will include:
   - A list of mistakes (if any) showing the text, predicted label, and correct label.
   - Top Label Metrics (Precision, Recall, F1 Score).
   - Overall Metrics (Precision, Recall, F1 Score).
   - Prediction Accuracy (percentage of correct and incorrect predictions).
   - Total number of mistakes and total number of texts in the training dataset.
   - Label Percentages showing the distribution of predicted labels.

This validation stage helps assess the model's performance, identify areas for improvement in the training process, and understand the distribution of labels in the inferred dataset.

## Useful Links

- [DistilBERT Documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [Transformers Library](https://huggingface.co/transformers/)
- [JSON Format Guide](https://www.json.org/json-en.html)
- [Markdown Syntax](https://www.markdownguide.org/basic-syntax/)

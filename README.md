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
       "0": {"label": "poetry", "color": "#00FF00"},
       "1": {"label": "family", "color": "#FFA500"},
       "2": {"label": "religious", "color": "#FFD700"},
       "3": {"label": "sadness", "color": "#4169E1"},
       "4": {"label": "psychology", "color": "#800080"}
   }
   ```

2. Prepare text with paragraphs for manual sorting (training).

3. Use `labeling.html` to load `train_labels.json` and sort the paragraphs into buckets.

4. Export the buckets as `train_input.json`.

5. Run `distilbert_train.py` to train the model on `train_input.json`. This will generate the model's directory. Note that the training is with validation/testing on the same data (edit the script yourself for improvements).

## Stage 2: Inferred Dataset (Automated Labeling)

After training the model, we use it to infer labels for an unlabeled dataset, which can be the rest of your data that hasn't been manually labeled yet.

### Required File:
- `infer_input.md`

### Steps:

1. Prepare text with paragraphs for automated (inferred) sorting.

2. Input the text in a clear format into `infer_input.md`.

3. Run `distilbert_infer.py` to execute the model. This will generate the output file `infer_output.json`.

## Useful Links

- [DistilBERT Documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [Transformers Library](https://huggingface.co/transformers/)
- [JSON Format Guide](https://www.json.org/json-en.html)
- [Markdown Syntax](https://www.markdownguide.org/basic-syntax/)

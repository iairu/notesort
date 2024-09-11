import json

# Load the JSON data from train_input.json
with open('train_input.json', 'r') as file:
    data = json.load(file)

# Extract the "text" content and join them with double newlines
text_content = "\n\n".join(item['text'] for item in data)

# Write the content to infer_input.md
with open('infer_input.md', 'w') as file:
    file.write(text_content)
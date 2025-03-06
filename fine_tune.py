# This code loads a pre-trained tokenizer and model for question answering

import json

# Load class_data.json before using it
with open("class_data.json", "r") as f:
    class_data = json.load(f)

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "bert-base-uncased"  # Switch from DistilBERT to BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

print("Loaded pre-trained model and tokenizer!")

from datasets import load_dataset

# Load your dataset from the JSON file
dataset = load_dataset('json', data_files={'train': 'class_data.json'})
print(dataset)  # Debugging step to check dataset structure
print("Train dataset before split:", dataset["train"])

# Split dataset (80% train, 20% validation)
# Extract the train dataset from DatasetDict
full_dataset = dataset["train"]

# Now split the dataset into 80% train and 20% validation
train_test_split = full_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define a function to tokenize each example
def preprocess_function(examples):
    inputs = tokenizer(
        examples['question'], examples['context'], 
        truncation="only_second",  
        max_length=384, 
        stride=128, 
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")
    
    start_positions = []
    end_positions = []

    for i in range(len(sample_mapping)):
        sample_index = sample_mapping[i]

        # 🛠️ Fix: Ensure correct data structure handling
        if "answers" in examples and isinstance(examples["answers"], list):
            if len(examples["answers"][sample_index]["text"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            answer_text = examples["answers"][sample_index]["text"][0]
            start_char = examples["answers"][sample_index]["answer_start"][0]
            end_char = start_char + len(answer_text)
        else:
            start_positions.append(0)
            end_positions.append(0)
            continue

        sequence_ids = inputs.sequence_ids(i)
        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1
        
        token_end_index = len(sequence_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offset_mapping[i][token_start_index][0] <= start_char and offset_mapping[i][token_end_index][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            found_start = False
            for idx in range(token_start_index, token_end_index + 1):
                if offset_mapping[i][idx][0] <= start_char and offset_mapping[i][idx][1] > start_char:
                    start_positions.append(idx)
                    found_start = True
                    break
            if not found_start:
                start_positions.append(0)

            found_end = False
            for idx in range(token_end_index, token_start_index - 1, -1):
                if offset_mapping[i][idx][1] >= end_char:
                    end_positions.append(idx)
                    found_end = True
                    break
            if not found_end:
                end_positions.append(0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
# Apply the preprocessing function to your dataset
tokenized_dataset = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

print("Dataset tokenized!")

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=3e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# Save your fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Test the model with a sample input
context = "In today's class, we learned about the water cycle which includes evaporation, condensation, and precipitation."
question = "What are the main processes in the water cycle?"

inputs = tokenizer(question, context, return_tensors="pt")
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

predicted_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
print("Predicted answer:", predicted_answer)

print("\nInteractive Q&A session started! Type 'exit' to quit.")

while True:
    question = input("\nEnter your question (or type 'exit' to quit'): ")
    if question.lower() == 'exit':
        break

    # Find the most relevant context from training data
    best_context = ""
    for entry in class_data:
        if question.lower() in entry["question"].lower():
            best_context = entry["context"]
            break

    if best_context == "":
        print("No relevant context found. Please provide context manually.")
        best_context = input("Enter the context for the question: ")

    # Tokenize and run inference
    inputs = tokenizer(question, best_context, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    predicted_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    print("Predicted answer:", predicted_answer)
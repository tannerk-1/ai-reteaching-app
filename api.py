from flask import Flask, request, jsonify
import os
import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load class data from JSON file
with open("class_data.json", "r") as f:
    class_data = json.load(f)

# Get Hugging Face token from environment variable
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

model_name = "tanner01/ai-reteaching-model"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token=huggingface_token)

# Set device (MPS for Mac, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Initialize Flask app
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "").strip().lower()

    # Step 1: First, check if the answer exists in class_data.json
    for item in class_data:
        if question in item["question"].lower():
            return jsonify({"answer": item["answers"]["text"][0]})

    # Step 2: If not found in JSON, use the model
    context = data.get("context", "").strip()
    if not question or not context:
        return jsonify({"error": "Please provide both question and context"}), 400

    # Tokenize input efficiently
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run optimized model inference
    with torch.inference_mode():  # More efficient than torch.no_grad()
        outputs = model(**inputs)

    # Extract answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    predicted_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    return jsonify({"answer": predicted_answer})

# Run the Flask API
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Get Render's assigned port or default to 10000
    app.run(host="0.0.0.0", port=port)
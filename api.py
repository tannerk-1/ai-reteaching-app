from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load fine-tuned model & tokenizer from Hugging Face
model_name = "tanner01/ai-reteaching-model"  # Hugging Face model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Set device (MPS for Mac, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Initialize Flask app
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")
    context = data.get("context")

    if not question or not context:
        return jsonify({"error": "Please provide both question and context"}), 400

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    predicted_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    return jsonify({"answer": predicted_answer})

# Run the Flask API
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Get Render's assigned port or default to 5000
    app.run(host="0.0.0.0", port=port)
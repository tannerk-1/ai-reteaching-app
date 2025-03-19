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
    
    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    # Search for a matching lesson in class_data.json
    for item in class_data:
        if question in item["question"].lower():
            return jsonify({"answer": item["answers"]["text"][0]})
    
    return jsonify({"error": "No matching lesson found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Get Render's assigned port or default to 10000
    app.run(host="0.0.0.0", port=port)
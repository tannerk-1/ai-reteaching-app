import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load fine-tuned model & tokenizer
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Streamlit UI
st.title("AI-Powered Reteaching Assistant 🎓")
st.subheader("Ask a question based on class content")

# User input for question & context
question = st.text_input("Enter your question:")
context = st.text_area("Enter the relevant class content:")

if st.button("Get Answer"):
    if question and context:
        # Tokenize input
        inputs = tokenizer(question, context, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract answer
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        predicted_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )

        st.success(f"AI Answer: {predicted_answer}")
    else:
        st.warning("Please enter both a question and context.")

st.markdown("---")
st.subheader("📤 Upload Lesson Data (Teachers Only)")
uploaded_file = st.file_uploader("Upload a JSON file with class content", type="json")

if uploaded_file:
    st.success("File uploaded successfully! (Processing feature can be added)")
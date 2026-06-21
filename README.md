# Text Summarizer using HuggingFace & FastAPI 🚀

A web application that takes long dialogues or paragraphs and generates a concise summary using a fine-tuned T5 Transformer model from Hugging Face.

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **Model:** Hugging Face T5 (Fine-tuned)
- **Deep Learning Framework:** PyTorch

## ⚙️ How to Run Locally

1. Clone this repository or download the code.
2. Put your fine-tuned model folder inside the project directory and name it `saved_summary_model`.
3. Install the required dependencies:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch

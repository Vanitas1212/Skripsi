from flask import Flask, render_template, request
import re
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
import numpy as np
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    request.form = request.form.copy()  # Refresh the input to ensure it always gets the newest data
    name = request.form.get('input')
    name_cleaned = clean_text(name)  # Panggil fungsi untuk membersihkan teks
    prediction = predict_sentiment(name_cleaned)

    print(f"Model prediction: {prediction}")
    
    return render_template('index.html', result=prediction)  # Kembalikan hasil prediksi ke template

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def predict_sample(sample_text, model, tokenizer):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)  # Move model to the appropriate device
#     model.eval()
#     inputs = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
#     inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device

#     # Prediksi sentimen
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Konversi hasil ke probabilitas
#     predictions = torch.nn.functional.softmax(outputs[0], dim=-1)
#     predictions = predictions.cpu().detach().numpy()
#     # Access logits from the model output
#     logits = outputs.logits

#     # Convert logits to probabilities
#     predictions = torch.nn.functional.softmax(logits, dim=-1)
#     predictions = predictions.cpu().detach().numpy()
#     # Ambil kelas dengan probabilitas tertinggi
#     predicted_label = np.argmax(predictions)

#     # Tampilkan hasil prediksi
#     sentiment_label = "Not Cyberbullying" if predicted_label == 1 else "Cyberbullying"

#     # Print hasil prediksi
#     print(f"Text: {sample_text}")
#     print(f"\n📢 Predicted Sentiment: {sentiment_label}")
#     return sentiment_label

# def predict_sample(sample_text, model, tokenizer):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     inputs = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         predictions = torch.nn.functional.softmax(logits, dim=-1)
#         predicted_label = torch.argmax(predictions, dim=-1).item()

#     sentiment_label = "Not Cyberbullying" if predicted_label == 1 else "Cyberbullying"

#     print(f"Text: {sample_text}")
#     print(f"\n📢 Predicted Sentiment: {sentiment_label}")
#     return sentiment_label

def predict_sentiment(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load fine-tuned model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("FinalModel")
    model = BertForSequenceClassification.from_pretrained("FinalModel")
    model.to(device)
    model.eval()

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    label = np.argmax(probs)
    label_name = "Not Cyberbully" if label == 1 else "Cyberbully"
    print(f"\n📢 Text: {text}")
    print(f"Predicted: {label_name} ({probs[0][label]:.4f})")
    return label_name


def clean_text(text):
    # text = text.lower()  # Konversi ke lowercase
    text = re.sub(r"(?i)<\s*username\s*>", "", text)  # Hapus tag username
    text = re.sub(r"http\S+|www\S+", "", text)  # Hapus URL
    text = re.sub(r"\s*@\S+", "", text) # Hapus mention
    text = re.sub(r"[^\w\s]", " ", text)  # Hapus karakter khusus (kecuali spasi)
    text = re.sub(r"\s+", " ", text).strip()  # Hapus spasi berlebih
    return text



if __name__ == '__main__':
    set_seed(42)  # Set seed for reproducibility
    app.run(debug=True)
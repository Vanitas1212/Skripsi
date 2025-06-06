import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download("stopwords")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import re
from sklearn.model_selection import KFold
import numpy as np
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random


def visualize_sentiment_distribution(dataset):
    plt.figure(figsize=(6,4))
    sns.countplot(x=dataset["Label"], palette="coolwarm")
    plt.xticks(ticks=[0,1], labels=["Negative", "Positive"])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Distribution of Sentiment Labels")
    plt.show()

# nltk.download("stopwords")

# lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # text = text.lower()  # Konversi ke lowercase
    text = re.sub(r"(?i)<\s*username\s*>", "", text)  # Hapus tag username
    text = re.sub(r"http\S+|www\S+", "", text)  # Hapus URL
    text = re.sub(r"\s*@\S+", "", text) # Hapus mention
    text = re.sub(r"[^\w\s]", " ", text)  # Hapus karakter khusus (kecuali spasi)
    text = re.sub(r"\s+", " ", text).strip()  # Hapus spasi berlebih

    # Hapus stopwords
    # stop_words = set(stopwords.words("indonesian"))
    # words = text.split()
    # filtered_words = [word for word in words if word not in stop_words]
    # print(filtered_words)

    # # Lakukan lemmatization pada setiap kata Inggris
    # lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # # Gabungkan kembali kata-kata hasil stemming
    # text = " ".join(lemmatized_words)

    # #Tokenization kalimat (pisah per kata)
    # tokens = word_tokenize(text)
    return text
    # return " ".join(filtered_words)

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="coolwarm").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.show()

def normalize_reduplicated_words(text):
    text = re.sub(r'(\w+)[2"\'’](\w*)', r'\1-\1\2', text)  # Ubah kata2nya -> kata-katanya
    return text

def normalize_suffix(text):
    """Menormalisasi akhiran tidak baku (-ny, -xa, -’y) menjadi -nya."""
    text = re.sub(r'(\w+)(-ny|-xa|-’y|-\'y)', r'\1-nya', text)  # rumahxa -> rumahnya
    return text

def clean_token(token):
    """Menghapus hanya koma dan spasi dari token, menjaga tanda baca lainnya."""
    return token.replace(",", "").strip()

def tokenize_text(text):
    """Melakukan tokenisasi dan membersihkan token dari hanya koma serta spasi."""
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    cleaned_tokens = [clean_token(token) for token in tokens if clean_token(token)]  # Hapus token kosong
    return cleaned_tokens

def load_model():
    # Load dataset
    df = pd.read_excel("D:\Skripsi-20250317T163434Z-001\Skripsi\skirpsi\Code_Try\How to normalize code mixing\Dataset Raw\Dataset_Gabungan With syntethic data.xlsx", index_col=False)

    # Hapus kolom Id
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    # Konversi label sentimen ke numerik
    df["Label"] = df["Sentiment"].map({"negative": 0, "positive": 1})
    # df["Label"] = df["Label"]


    # Bersihkan teks komentar
    # df["Cleaned_Text"] = df["Translated"].apply(clean_text)
    # df["Cleaned_Text"] = df["Cleaned_Text"].apply(normalize_reduplicated_words)
    # df["Cleaned_Text"] = df["Cleaned_Text"].apply(normalize_suffix) 
    # df["Cleaned_Text"] = df["Cleaned_Text"].apply(clean_text)
    # df["Cleaned_Text"] = df["Cleaned_Text"].apply(tokenize_text)
    # df["Cleaned_Text"] = df["Cleaned_Text"].apply()

    df["Cleaned_Text"] = df["Instagram Comment Text"].apply(clean_text)
    df["Cleaned_Text"] = df["Cleaned_Text"].apply(normalize_reduplicated_words)
    df["Cleaned_Text"] = df["Cleaned_Text"].apply(normalize_suffix) 
    # df["Cleaned_Text"] = df["Cleaned_Text"].apply(clean_text)
    # print(df["Label"])
    # Buat dataset final
    dataset = pd.DataFrame({"Text": df["Cleaned_Text"], "Label": df["Label"]})
    cyberbullying_text = " ".join(df[df["Label"] == 0]["Cleaned_Text"])
    non_cyberbullying_text = " ".join(df[df["Label"] == 1]["Cleaned_Text"])
    print(dataset)
    generate_wordcloud(cyberbullying_text, "Word Cloud - Cyberbullying Comments")
    generate_wordcloud(non_cyberbullying_text, "Word Cloud - Non-Cyberbullying Comments")


    return dataset

def load_indoBERT():
    model_name = "indobenchmark/indobert-base-p2"
    #declare nama model yang kita mau pakai
    tokenizer = BertTokenizer.from_pretrained(model_name)
    #ambil tokenizer dari indoBERT
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, from_tf=True)
    #ambil pretrained model nya, pakai num_labels=2 buat input berupa sequence
    return tokenizer,model
    #return tokenizer dan juga modlenya

def prepare_data(dataset, tokenizer):
    X = list(dataset["Text"])
    y = list(dataset["Label"])

    # Tokenize the entire dataset at once
    tokenized_data = tokenizer(X, padding=True, truncation=True, max_length=512)

    # Convert tokenized data to Dataset format
    dataset = Dataset.from_dict({
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": y
    })

    return dataset

def evaluation_metrics(eval_pred):
    #Ini function buat evaluation metrice
    # print(type(p))
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def predict_sample(sample_text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the appropriate device

    inputs = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device
    model.save_pretrained("saved_model/")
    tokenizer.save_pretrained("saved_model/")
    # Check if the model is already fine-tuned
    if not hasattr(model.config, "fine_tuned") or not model.config.fine_tuned:
        print("The model is not fine-tuned. Please fine-tune the model before saving.")
        model.config.fine_tuned = True  # Mark the model as fine-tuned
    else:
        print("The model is already fine-tuned. di sini kah")
    # Prediksi sentimen
    with torch.no_grad():
        outputs = model(**inputs)

    # Konversi hasil ke probabilitas
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    # Ambil kelas dengan probabilitas tertinggi
    predicted_label = np.argmax(predictions)

    # Tampilkan hasil prediksi
    sentiment_label = "Positive" if predicted_label == 1 else "Negative"

    # Print hasil prediksi
    print(f"Text: {sample_text}")
    print(f"\n📢 Predicted Sentiment: {sentiment_label}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def cross_val_train(dataset, model, tokenizer, k=5):
    set_seed(42)  # Set seed for reproducibility
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n========== Fold {fold + 1} ==========")

        # Split dataset into train and validation sets
        train_subset = dataset.select(train_idx)
        val_subset = dataset.select(val_idx)

        # Training arguments
        args = TrainingArguments(
            output_dir=f"output_fold{fold + 1}",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"./logs_fold{fold + 1}",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            run_name=f"IndoBERT-CyberbullyingDetection-Fold{fold+1}",
            report_to="none",
            seed=42  # Set seed for reproducibility
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            compute_metrics=evaluation_metrics
        )

        # Training
        trainer.train()

        # Evaluation
        eval_results = trainer.evaluate()
        predictions = trainer.predict(val_subset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        # Generate and plot confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold + 1}")
        plt.show()
        fold_results.append(eval_results)

        print("\n--- Evaluation Results ---")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")

        # Save the model for this fold
        trainer.save_model(f'CustomModel_Fold{fold+1}')

    # Aggregate Results
    print("\n===== Cross-Validation Results =====")
    avg_results = {key: np.mean([fold[key] for fold in fold_results]) for key in fold_results[0]}
    for key, value in avg_results.items():
        print(f"{key}: {value:.4f}")

    # Aggregate confusion matrix
    all_true_labels = []
    all_pred_labels = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        val_subset = dataset.select(val_idx)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=f"output_fold{fold + 1}"),
            train_dataset=None,
            eval_dataset=val_subset,
            tokenizer=tokenizer
        )
        predictions = trainer.predict(val_subset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)

    # Generate and plot the aggregated confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Aggregated Confusion Matrix")
    plt.show()
    # Example prediction using the trained model
    sample_text = "Aura kamu tuh always positive vibes."
    predict_sample(sample_text, model, tokenizer)
    
    if not hasattr(model.config, "fine_tuned") or not model.config.fine_tuned:
        print("The model is not fine-tuned. Please fine-tune the model before saving.")
        model.config.fine_tuned = True  # Mark the model as fine-tuned
    else:
        print("The model is already fine-tuned.DI SINI KAH")
    # Prediksi sentimen
    print("Final model and tokenizer have been saved to the 'FinalModel' directory. dimana ya ini")
    print(f"Prediction for sample text '{sample_text}' completed.")
    # Save the final model after cross-validation
    model.save_pretrained("FinalModel")
    tokenizer.save_pretrained("FinalModel")
    return avg_results


if __name__ == "__main__":
    set_seed(42)  # Set seed for reproducibility
    dataset = load_model()

    # visualisasi proporsi jumlah data
    visualize_sentiment_distribution(dataset)

    tokenizer,model = load_indoBERT()
    # sample = ["Bodoh"]
    # nyehe = tokenizer(sample, padding=True, truncation=True, max_length=512)
    # print(nyehe)
    # X_train, X_val, y_train, y_val = prepare_data(dataset,tokenizer)
    data = prepare_data(dataset,tokenizer)
    average = cross_val_train(data,model,tokenizer)
    print(average)
    # train_dataset = Dataset(X_train,y_train)
    # test_dataset = Dataset(X_val,y_val)
    # train_data(train_dataset,test_dataset,model,tokenizer)



    # model_2 = BertForSequenceClassification.from_pretrained("CustomModel")
    # text = "goblok lu"
    # inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
    # outputs = model_2(**inputs)
    # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # predictions = predictions.cpu().detach().numpy()
    # predictions
import random
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
from seqeval.metrics import precision_score, recall_score, f1_score

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     # Convert numeric labels back to original text labels
#     label_list = ["O", "Place", "Person", "Organisation"]  # Make sure this matches your label_map
#     true_labels = [[label_list[label] for label in label_seq if label != -100] for label_seq in labels]
#     pred_labels = [[label_list[label] for label in pred_seq if label != -100] for pred_seq in predictions]

#     return {
#         "precision": precision_score(true_labels, pred_labels),
#         "recall": recall_score(true_labels, pred_labels),
#         "f1": f1_score(true_labels, pred_labels),
#     }

def load_ner_data():
    sentences = []
    entities = []
    words = []
    labels = []
    split_ratio=0.8
    with open('D:\\Skripsi-20250317T163434Z-001\\Skripsi\\skirpsi\\Code_Try\\How to normalize code mixing\\20k_dee_withblankline.tsv', "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line == "":  # New sentence
                if words:
                    print(words)
                    # words = words.lower()
                    words = [kata.lower() for kata in words]
                    sentences.append(words)
                    entities.append(labels)
                    words, labels = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    word, label = parts
                    word = word.lower()
                    words.append(word)
                    labels.append(label)
    
    # Ensure last sentence is added
    if words:
        sentences.append(words)
        entities.append(labels)

    # Shuffle data
    data = list(zip(sentences, entities))
    random.shuffle(data)

    # Split into train and test sets (80:20)
    split_idx = int(len(data) * split_ratio)
    train_data, test_data = data[:split_idx], data[split_idx:]

    train_sentences, train_labels = zip(*train_data)
    test_sentences, test_labels = zip(*test_data)

    return train_sentences, train_labels, test_sentences, test_labels

def tokenize_and_align_labels(examples, tokenizer, label_map):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        padding=True,  # ✅ Enable padding
        truncation=True,  # ✅ Enable truncation
        max_length=128,  # ✅ Set max length (adjust as needed)
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore this token
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])  # Convert label to integer
            else:
                label_ids.append(-100)  # Ignore subword tokens

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def predict_ner(text, tokenizer, model, label_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect device
    model.to(device) 
    
    tokens = tokenizer(
        text.split(),  # ✅ Ensuring input is a list of words
        padding=True,
        truncation=True,
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt"  # ✅ Convert to PyTorch tensor
    )

    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    for token, label_id in zip(tokens_list, predictions):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:  # ✅ Ignore special tokens
            print(f"{token}: {label_list[label_id]}")


if __name__ == "__main__":
    # Load and split data
    train_sentences, train_labels, test_sentences, test_labels = load_ner_data()
    print(train_sentences)
    # Convert to Hugging Face dataset format
    train_dataset = Dataset.from_dict({"tokens": train_sentences, "ner_tags": train_labels})
    test_dataset = Dataset.from_dict({"tokens": test_sentences, "ner_tags": test_labels})

    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    label_map = {"O": 0, "Place": 1, "Person": 2, "Organisation": 3}

    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_map), batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,  # Increase batch size
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,  # Mixed precision for speed
        gradient_accumulation_steps=2,  # Simulate larger batch size
        warmup_steps=500,  # Smooth learning rate
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        save_total_limit=2,  # Save only 2 best models
        logging_steps=100,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    tokenizer=tokenizer
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print("\n--- Evaluation Results ---")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # Example sentence
    label_list = ["O", "Place", "Person", "Organisation"]
    predict_ner("Barack Obama mengunjungi Jakarta dan bertemu dengan Joko Widodo.",tokenizer,model,label_list)
    # predict_ner("Barack Obama visited Jakarta and met with Joko Widodo.",tokenizer)
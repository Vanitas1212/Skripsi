import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


def get_synonyms(word):
    """Retrieve synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalpha() or char.isspace()])
            synonyms.add(synonym)
    synonyms.discard(word)  # Remove the original word from the synonyms
    return list(synonyms)


def extract_en_tokens(row):
    """Extract tokens labeled as 'en' from a row."""
    tokens = eval(row["Token"]) if isinstance(row["Token"], str) else row["Token"]
    langs = eval(row["Lang"]) if isinstance(row["Lang"], str) else row["Lang"]
    return [token for token, lang in zip(tokens, langs) if lang == "en"]


def process_row(row):
    """Process a single row to generate synonyms for 'en'-labeled tokens."""
    en_tokens = extract_en_tokens(row)
    return {word: get_synonyms(word) for word in en_tokens}


def process_dataset(filepath):
    """Load and process the dataset to generate synonyms for 'en'-labeled tokens."""
    # Load dataset
    data = pd.read_excel(filepath)
    print("Columns in dataset:", data.columns)

    # Drop unnecessary columns
    columns_to_drop = ['Id', 'Text']
    data.drop([col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)

    # Filter and balance rows with labels 0 and 1
    if 'Label' in data.columns:
        label_0_data = data[data['Label'] == 0]
        label_1_data = data[data['Label'] == 1]

        # Balance the dataset to have 47 rows for each label
        label_0_data = label_0_data.sample(n=47, random_state=42)
        label_1_data = label_1_data.sample(n=47, random_state=42)

        # Combine the balanced data
        data = pd.concat([label_0_data, label_1_data]).reset_index(drop=True)
        label_counts = data['Label'].value_counts()
        print("Balanced label counts:\n", label_counts)

    # Generate synonyms for each row
    data["Synonyms"] = data.apply(process_row, axis=1)
    return data


def concatenate_tokens(row):
    """Concatenate the tokens into a single string."""
    tokens = eval(row["Token"]) if isinstance(row["Token"], str) else row["Token"]
    return " ".join(tokens)


if __name__ == "__main__":
    # Define the dataset path
    dataset_path = 'D:\Skripsi-20250317T163434Z-001\Skripsi\skirpsi\Code_Try\How to normalize code mixing\Dataset Raw\Dataset English on Dataset Mixed.xlsx'

    # Process the dataset
    processed_data = process_dataset(dataset_path)

    # Perform data augmentation
    augmented_rows = []
    for _, row in processed_data.iterrows():
        tokens = eval(row["Token"]) if isinstance(row["Token"], str) else row["Token"]
        langs = eval(row["Lang"]) if isinstance(row["Lang"], str) else row["Lang"]
        synonyms = row["Synonyms"]

        augmented_tokens_list = [tokens]  # Start with the original tokens
        for i, token in enumerate(tokens):
            if langs[i] == "en" and token in synonyms and synonyms[token]:
                for synonym in synonyms[token]:
                    augmented_tokens = tokens.copy()
                    augmented_tokens[i] = synonym
                    augmented_tokens_list.append(augmented_tokens)

        for augmented_tokens in augmented_tokens_list:
            augmented_rows.append({
                "Token": str(augmented_tokens),  
                "Lang": str(langs),  
                "Label": row["Label"]  # Include the label in the augmented data
            })

    # Create a new DataFrame for augmented data
    augmented_data = pd.DataFrame(augmented_rows)

    # Add a column with concatenated tokens
    augmented_data["Concatenated_Tokens"] = augmented_data.apply(concatenate_tokens, axis=1)

    # Replicate the augmented data to twice the size
    replicated_data = pd.concat([augmented_data] * 2, ignore_index=True)

    # Save the replicated dataset
    replicated_data.to_excel("Replicated_Augmented_Dataset.xlsx", index=False)

    print("Data augmentation and replication completed. Replicated dataset saved as 'Replicated_Augmented_Dataset.xlsx'.")
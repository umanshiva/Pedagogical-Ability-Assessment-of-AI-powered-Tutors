import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

## This script is used to load and flatten the JSON data from the Tutor dataset.
def load_and_flatten(json_path):
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for instance in data:
        convo_id = instance["conversation_id"]
        history = instance["conversation_history"]

        for tutor_id, tutor_data in instance["tutor_responses"].items():
            row = {
                "conversation_id": convo_id,
                "tutor_id": tutor_id,
                "conversation_history": history,
                "tutor_response": tutor_data["response"],
                "Mistake_Identification": tutor_data["annotation"]["Mistake_Identification"],
                "Mistake_Location": tutor_data["annotation"]["Mistake_Location"],
                "Pedagogical_Guidance": tutor_data["annotation"]["Providing_Guidance"],
                "Actionability": tutor_data["annotation"]["Actionability"]
            }
            rows.append(row)

    return pd.DataFrame(rows)

## This function is used to load the data from the CSV file and add a new column with the input text for the model.
def build_input_text(row):
    return f"Context:\n{row['conversation_history']}\n\nTutor Response:\n{row['tutor_response']}"

# Encode labels
EXACT_MAP = {"Yes": 0, "To some extent": 1, "No": 2}
LINEANT_MAP = {"Yes": 1, "To some extent": 1, "No": 0}  # For lenient setting

## This function is used to encode the labels for the tasks.
def encode_labels(df):
    for task in ["Mistake_Identification", "Mistake_Location", "Pedagogical_Guidance", "Actionability"]:
        df[f"{task}_label"] = df[task].map(EXACT_MAP)
        df[f"{task}_binary"] = df[task].map(LINEANT_MAP)
    return df

# Tokenize inputs
def tokenize_inputs(tokenizer, texts, max_length=256):
    return tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

class TutorEvalMultiTaskDataset(Dataset):
    def __init__(self, encodings, label_dict):
        self.encodings = encodings
        self.label_dict = label_dict

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        for task, labels in self.label_dict.items():
            item[task] = labels[idx]
        return item

    def __len__(self):
        return len(self.label_dict['Mistake_Identification_label'])


def preprocess_dataset(json_path, tokenizer, is_leniant = False,max_length=256, mode = None, task = None):
    df = load_and_flatten(json_path)
    df["input_text"] = df.apply(build_input_text, axis=1)
    df = encode_labels(df)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["Mistake_Identification_label"], random_state=42)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_enc = tokenize_inputs(tokenizer, train_df["input_text"].tolist(), max_length=max_length)
    val_enc = tokenize_inputs(tokenizer, val_df["input_text"].tolist(), max_length=max_length)

    if mode == 'balanced' and task is not None:
        # Use SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        y = train_df[f"{task}_label"].values
        if is_leniant:
            y = train_df[f"{task}_binary"].values
        X1 = train_enc['input_ids']
        X2 = train_enc['attention_mask']
        X = torch.concatenate((X1, X2), axis=1)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        train_enc['input_ids'] = torch.tensor(X_resampled[:, :X1.shape[1]])
        train_enc['attention_mask'] = torch.tensor(X_resampled[:, X1.shape[1]:])
        train_labels = {task + "_label": torch.tensor(y_resampled)}
        val_labels = {task + "_label": torch.tensor(val_df[f"{task}_label"].tolist())}
        if is_leniant:
            val_labels = {task + "_label": torch.tensor(val_df[f"{task}_binary"].tolist())}
        train_dataset = TutorEvalMultiTaskDataset(train_enc, train_labels)
        val_dataset = TutorEvalMultiTaskDataset(val_enc, val_labels)
        return train_dataset, val_dataset, tokenizer, df
        
    if is_leniant:
        # Use lenient labels for training and validation sets
        train_labels = {task + "_label": torch.tensor(train_df[task + "_binary"].tolist()) for task in ["Mistake_Identification", "Mistake_Location", "Pedagogical_Guidance", "Actionability"]}
        val_labels = {task + "_label": torch.tensor(val_df[task + "_binary"].tolist()) for task in ["Mistake_Identification", "Mistake_Location", "Pedagogical_Guidance", "Actionability"]}
    else:
        # Label dicts for multi-task 
        train_labels = {task + "_label": torch.tensor(train_df[task + "_label"].tolist()) for task in ["Mistake_Identification", "Mistake_Location", "Pedagogical_Guidance", "Actionability"]}
        val_labels = {task + "_label": torch.tensor(val_df[task + "_label"].tolist()) for task in ["Mistake_Identification", "Mistake_Location", "Pedagogical_Guidance", "Actionability"]}
        

    train_dataset = TutorEvalMultiTaskDataset(train_enc, train_labels)
    val_dataset = TutorEvalMultiTaskDataset(val_enc, val_labels)
    
    return train_dataset, val_dataset, tokenizer, df


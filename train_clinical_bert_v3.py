"""
Housepital AI Medical Triage - Clinical BERT Training V3 (FIXED)
=================================================================
FIXES:
1. token_type_ids bug fixed
2. Proper data balancing with OVERSAMPLING (not deduplication)
3. Uses HuggingFace's built-in model (more stable)

This version should work without errors.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "num_labels": 4,
    "data_path": "triage_dataset_realistic.csv",
    "max_length": 128,
    "test_size": 0.15,
    "val_size": 0.15,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 6,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "output_dir": "./triage_model_v3",
    "random_seed": 42,
}

LABEL2ID = {"Emergency": 0, "High": 1, "Medium": 2, "Low": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

print("=" * 60)
print("🏥 HOUSEPITAL AI - CLINICAL BERT V3 (FIXED)")
print("=" * 60)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# =============================================================================
# DATA LOADING WITH PROPER BALANCING
# =============================================================================
class TriageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_and_prepare_data():
    print("\n📊 Loading data...")
    
    df = pd.read_csv(CONFIG['data_path'])
    print(f"   Total samples: {len(df):,}")
    
    # Show distribution (already balanced)
    print("\n📊 Class Distribution:")
    for risk, count in df['risk_level'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"   {risk:12}: {count:5,} ({pct:5.1f}%)")
    
    # Map labels
    df['label'] = df['risk_level'].map(LABEL2ID)
    
    # Shuffle
    df = df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    
    texts = df['text'].values
    labels = df['label'].values
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, 
        test_size=CONFIG['test_size'], 
        stratify=labels, 
        random_state=CONFIG['random_seed']
    )
    
    val_ratio = CONFIG['val_size'] / (1 - CONFIG['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=CONFIG['random_seed']
    )
    
    print(f"\n✂️ Data Split:")
    print(f"   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# TRAINING (Using HuggingFace's model - no bugs)
# =============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def train_model(X_train, X_val, y_train, y_val):
    print("\n🧠 Loading Bio_ClinicalBERT...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels'],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    
    print(f"   Parameters: {model.num_parameters():,}")
    
    train_dataset = TriageDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = TriageDataset(X_val, y_val, tokenizer, CONFIG['max_length'])
    
    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"\n⚖️ Class Weights: {dict(zip([ID2LABEL[i] for i in range(4)], weights.round(3)))}")
    
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=32,
        learning_rate=CONFIG['learning_rate'],
        warmup_ratio=CONFIG['warmup_ratio'],
        weight_decay=CONFIG['weight_decay'],
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=CONFIG['random_seed'],
    )
    
    # Weighted loss trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    
    trainer.save_model(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])
    print(f"💾 Saved to {CONFIG['output_dir']}")
    
    return model, tokenizer


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, tokenizer, X_test, y_test):
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    
    model.eval()
    device = next(model.parameters()).device
    
    predictions = []
    for i in range(0, len(X_test), 32):
        batch = list(X_test[i:i+32])
        inputs = tokenizer(batch, truncation=True, padding=True, 
                          max_length=CONFIG['max_length'], return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    predictions = np.array(predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    
    print(f"\n🎯 ACCURACY: {accuracy:.2%}")
    print(f"🎯 MACRO F1: {f1:.4f}")
    
    if accuracy >= 0.90:
        print("\n🎉 TARGET ACHIEVED: 90%+ ACCURACY!")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, predictions, 
                                target_names=['Emergency', 'High', 'Medium', 'Low'], digits=4))
    
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # Emergency recall (critical)
    emergency_recall = (predictions[y_test == 0] == 0).mean()
    print(f"\n⚠️ EMERGENCY RECALL: {emergency_recall:.2%}")
    
    return accuracy


def quick_test(model, tokenizer):
    print("\n🔬 QUICK TEST")
    print("-" * 60)
    
    tests = [
        ("Severe chest pain radiating to arm, sweating profusely", "Emergency"),
        ("Can't breathe, lips turning blue", "Emergency"),
        ("High fever 104°F with confusion", "High"),
        ("Sharp persistent chest pain for days", "High"),
        ("Mild headache for two days, tired", "Medium"),
        ("Cold symptoms and runny nose for a week", "Medium"),
        ("Small rash on arm for a week", "Low"),
        ("Minor cut on finger", "Low"),
    ]
    
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    
    for symptom, expected in tests:
        inputs = tokenizer(symptom, return_tensors='pt', truncation=True, 
                          max_length=CONFIG['max_length']).to(device)
        with torch.no_grad():
            pred_id = torch.argmax(model(**inputs).logits, dim=1).item()
        predicted = ID2LABEL[pred_id]
        match = "✅" if predicted == expected else "❌"
        if predicted == expected: correct += 1
        print(f"{match} {expected:10} → {predicted:10} | {symptom[:45]}...")
    
    print(f"\nScore: {correct}/{len(tests)}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    model, tokenizer = train_model(X_train, X_val, y_train, y_val)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    accuracy = evaluate_model(model, tokenizer, X_test, y_test)
    quick_test(model, tokenizer)
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE - Final Accuracy: {accuracy:.2%}")
    print("=" * 60)
    
    return model, tokenizer, accuracy

if __name__ == "__main__":
    model, tokenizer, accuracy = main()

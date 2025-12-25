"""
FINAL TRAINING ON CORRECTED DATASET
====================================
Now that labels are medically correct, simple ML should work much better.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

print("=" * 70)
print("🏥 TRAINING ON CORRECTED DATASET")
print("=" * 70)

# Load corrected data
df = pd.read_csv('triage_dataset_corrected.csv')
print(f"Samples: {len(df):,}")
print(f"Distribution: {df['risk_level'].value_counts().to_dict()}")

# Prepare data
texts = df['text'].values
labels = df['risk_level'].values

# Map labels
label_map = {'Emergency': 0, 'High': 1, 'Medium': 2, 'Low': 3}
reverse_map = {v: k for k, v in label_map.items()}
y = np.array([label_map[l] for l in labels])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# TF-IDF
print("\n📊 Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"   Features: {X_train_tfidf.shape[1]:,}")

# Try multiple classifiers
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=42),
}

best_acc = 0
best_model = None
best_name = None

print("\n🧠 Training models...")
for name, clf in classifiers.items():
    print(f"\n   Training {name}...")
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"   {name}: {acc:.2%}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = clf
        best_name = name

# Evaluate best model
print("\n" + "=" * 70)
print(f"📊 BEST MODEL: {best_name} with {best_acc:.2%} accuracy")
print("=" * 70)

y_pred = best_model.predict(X_test_tfidf)

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, 
                            target_names=['Emergency', 'High', 'Medium', 'Low'],
                            digits=4))

print("🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Quick test
print("\n" + "=" * 70)
print("🔬 QUICK TEST")
print("=" * 70)

test_cases = [
    ("I'm having severe chest pain radiating to my left arm and sweating profusely", "Emergency"),
    ("I can't breathe and my lips are turning blue", "Emergency"),
    ("I've been diagnosed with metastatic cancer", "High"),  
    ("Blood in my stool for the past few days", "High"),
    ("I have a mild cold with runny nose", "Low"),
    ("Minor cut on my finger from cooking", "Low"),
    ("Persistent cough for a week, not too severe", "Medium"),
    ("Stomach pain after eating, comes and goes", "Medium"),
]

correct = 0
for symptom, expected in test_cases:
    X = vectorizer.transform([symptom])
    pred_id = best_model.predict(X)[0]
    predicted = reverse_map[pred_id]
    match = "✅" if predicted == expected else "❌"
    if predicted == expected:
        correct += 1
    print(f"{match} {expected:10} → {predicted:10} | {symptom[:50]}...")

print(f"\n📊 Quick Test: {correct}/{len(test_cases)}")

# Save
print("\n💾 Saving model...")
with open('triage_model_corrected.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'model': best_model,
        'label_map': label_map,
        'reverse_map': reverse_map
    }, f)
print("   Saved to triage_model_corrected.pkl")

if best_acc >= 0.90:
    print("\n🎉 " + "=" * 60)
    print("🎉 TARGET ACHIEVED: 90%+ ACCURACY!")
    print("🎉 " + "=" * 60)

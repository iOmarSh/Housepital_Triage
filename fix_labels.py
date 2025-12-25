"""
FIX DATASET LABELS BASED ON MEDICAL KNOWLEDGE
==============================================
The original data has INCORRECT labels!
- Metastatic cancer labeled as "Low"
- Poisoning labeled as "Low"
- ALS labeled as "Low"

This script relabels based on ACTUAL medical triage protocols.
"""

import pandas as pd

# Load data
df = pd.read_csv('generated_symptom_texts_clean.csv')
print(f"Original samples: {len(df):,}")
print(f"Original distribution: {df['risk_level'].value_counts().to_dict()}")

# =============================================================================
# CORRECT MEDICAL LABELS
# =============================================================================

# EMERGENCY: Life-threatening, requires immediate attention
EMERGENCY_DISEASES = [
    # From current "Emergency" that are correct
    'diabetic ketoacidosis', 'malignant hypertension', 'heart block',
    'poisoning due to antimicrobial drugs', 'hemiplegia', 'subarachnoid hemorrhage',
    'hypertensive heart disease', 'acute myocardial infarction', 'cardiac arrest',
    'anaphylaxis', 'status epilepticus', 'respiratory failure', 'shock',
    'pulmonary embolism', 'stroke', 'meningitis', 'sepsis',
    
    # From "Low" that should be Emergency
    'poisoning due to analgesics', 'poisoning due to antihypertensives',
    'poisoning', 'drug overdose',
    
    # Heart-related emergencies
    'heart attack', 'acute coronary syndrome', 'ventricular fibrillation',
    'aortic dissection', 'cardiac tamponade',
    
    # Respiratory emergencies
    'acute respiratory distress', 'tension pneumothorax', 'airway obstruction',
]

# HIGH: Serious, needs urgent care
HIGH_DISEASES = [
    # Current High diseases (mostly correct)
    'hypertrophic obstructive cardiomyopathy', 'vertebrobasilar insufficiency',
    'kidney disease due to longstanding hypertension', 'congenital heart defect',
    'hypothermia', 'pneumoconiosis', 'aspergillosis', 'malaria',
    'high blood pressure', 'abscess of the lung',
    
    # From "Low" that should be High
    'metastatic cancer', 'amyotrophic lateral sclerosis', 'huntington disease',
    'adrenal cancer', 'kidney cancer', 'heart contusion',
    
    # General high-risk conditions
    'acute pancreatitis', 'appendicitis', 'bowel obstruction',
    'deep vein thrombosis', 'severe infection', 'kidney failure',
    'liver failure', 'diabetic emergency', 'severe dehydration',
]

# LOW: Minor, routine care acceptable
LOW_DISEASES = [
    # Minor conditions only
    'common cold', 'minor wound', 'minor cut', 'bruise', 'mild headache',
    'seasonal allergies', 'mild rash', 'acne', 'minor skin irritation',
    'dry skin', 'minor muscle pain', 'routine checkup', 'wellness question',
    'prescription refill', 'vitamin question', 'diet advice',
    'minor sprain', 'minor burn', 'insect bite', 'sunburn',
]

# Everything else is MEDIUM by default

def categorize_disease(disease):
    """Categorize disease based on medical knowledge."""
    disease_lower = disease.lower()
    
    # Check emergency
    for e in EMERGENCY_DISEASES:
        if e.lower() in disease_lower or disease_lower in e.lower():
            return 'Emergency'
    
    # Check high-risk keywords
    emergency_keywords = ['poisoning', 'overdose', 'hemorrhage', 'cardiac arrest',
                          'respiratory failure', 'shock', 'sepsis', 'anaphylaxis']
    for kw in emergency_keywords:
        if kw in disease_lower:
            return 'Emergency'
    
    # Check high
    for h in HIGH_DISEASES:
        if h.lower() in disease_lower or disease_lower in h.lower():
            return 'High'
    
    # High-risk keywords
    high_keywords = ['cancer', 'tumor', 'malignant', 'failure', 'als',
                     'huntington', 'parkinson', 'liver disease', 'kidney disease']
    for kw in high_keywords:
        if kw in disease_lower:
            return 'High'
    
    # Check low
    for l in LOW_DISEASES:
        if l.lower() in disease_lower or disease_lower in l.lower():
            return 'Low'
    
    # Low-risk keywords
    low_keywords = ['minor', 'mild', 'common cold', 'routine', 'wellness']
    for kw in low_keywords:
        if kw in disease_lower:
            return 'Low'
    
    # Default to Medium
    return 'Medium'

# Apply corrections
print("\n🔧 Correcting labels based on medical knowledge...")
df['corrected_risk'] = df['disease'].apply(categorize_disease)

# Count changes
changes = (df['risk_level'] != df['corrected_risk']).sum()
print(f"   Changed {changes:,} labels ({changes/len(df)*100:.1f}%)")

# Show examples of changes
print("\n📊 Example corrections:")
changed_df = df[df['risk_level'] != df['corrected_risk']].head(20)
for _, row in changed_df.iterrows():
    print(f"   {row['disease'][:40]:40} | {row['risk_level']:10} → {row['corrected_risk']}")

# Apply corrections
df['risk_level'] = df['corrected_risk']
df = df.drop('corrected_risk', axis=1)

# Show new distribution
print(f"\n📊 NEW distribution:")
print(df['risk_level'].value_counts().sort_index())

# Balance the dataset
print("\n📊 Balancing dataset...")
balanced_dfs = []
target = 2500

for risk in ['Emergency', 'High', 'Medium', 'Low']:
    subset = df[df['risk_level'] == risk]
    if len(subset) >= target:
        balanced_dfs.append(subset.sample(n=target, random_state=42))
    else:
        # Oversample
        multiplier = (target // len(subset)) + 1
        oversampled = pd.concat([subset] * multiplier).head(target)
        balanced_dfs.append(oversampled)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Balanced: {len(df_balanced):,} samples")
print(df_balanced['risk_level'].value_counts().sort_index())

# Save
df_balanced.to_csv('triage_dataset_corrected.csv', index=False)
print(f"\n💾 Saved to triage_dataset_corrected.csv")

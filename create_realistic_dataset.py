"""
Create realistic training dataset:
- Use ORIGINAL data (real variation)
- Only augment the underrepresented High class
- Result: ~12-15K samples with REAL data patterns
"""

import pandas as pd
import random
import hashlib

random.seed(42)

# Load original dataset
df = pd.read_csv('generated_symptom_texts_clean.csv')
print(f"Original dataset: {len(df):,} samples")

# Check distribution
print("\nOriginal distribution:")
for risk, count in df['risk_level'].value_counts().sort_index().items():
    print(f"  {risk}: {count}")

# Target: Match the minimum of Emergency or Medium (around 2300-3800)
# We'll augment High to ~3000 samples
target_high = 3000

# Get High samples
high_df = df[df['risk_level'] == 'High'].copy()
original_high_count = len(high_df)
print(f"\nHigh class: {original_high_count} samples")

# Diverse augmentation for High class
def augment_high_text(text):
    """Create diverse variations of text."""
    variations = []
    
    # Contraction variations
    v1 = text.replace("I'm", "I am").replace("I've", "I have").replace("can't", "cannot")
    if v1 != text:
        variations.append(v1)
    
    # Reverse contractions
    v2 = text.replace("I am", "I'm").replace("I have", "I've").replace("cannot", "can't")
    if v2 != text and v2 not in variations:
        variations.append(v2)
    
    # Add context starters
    starters = [
        "I need help, ", "Please help, ", "I'm worried because ",
        "Something is wrong, ", "This is concerning, ",
        "I've been experiencing something worrying: ",
    ]
    for starter in starters[:2]:  # Only use 2 starters
        v = starter + text[0].lower() + text[1:]
        if v not in variations:
            variations.append(v)
    
    # Add urgency enders
    enders = [
        " What should I do?", " Is this serious?", " I need advice.",
        " Should I be concerned?", " Please advise.",
    ]
    for ender in enders[:2]:
        v = text.rstrip('.') + ender
        if v not in variations:
            variations.append(v)
    
    return variations[:4]  # Max 4 variations per original

# Augment High class
print(f"Augmenting High class to ~{target_high} samples...")

augmented_rows = []
used_hashes = set()

# First, add all originals
for _, row in high_df.iterrows():
    text_hash = hashlib.md5(row['text'].lower().encode()).hexdigest()
    if text_hash not in used_hashes:
        used_hashes.add(text_hash)
        augmented_rows.append(row.to_dict())

# Then add augmented versions until we reach target
for _, row in high_df.iterrows():
    if len(augmented_rows) >= target_high:
        break
    
    variations = augment_high_text(row['text'])
    for var in variations:
        if len(augmented_rows) >= target_high:
            break
        
        var_hash = hashlib.md5(var.lower().encode()).hexdigest()
        if var_hash not in used_hashes:
            used_hashes.add(var_hash)
            augmented_rows.append({
                'disease': row['disease'],
                'text': var,
                'risk_level': 'High'
            })

print(f"Augmented High: {len(augmented_rows)} samples")

# Create augmented High dataframe
augmented_high_df = pd.DataFrame(augmented_rows)

# Combine: Keep original Emergency, Medium, Low + Augmented High
other_df = df[df['risk_level'] != 'High'].copy()
final_df = pd.concat([other_df, augmented_high_df], ignore_index=True)

# Shuffle
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset: {len(final_df):,} samples")
print("\nFinal distribution:")
for risk, count in final_df['risk_level'].value_counts().sort_index().items():
    pct = count / len(final_df) * 100
    print(f"  {risk:12}: {count:5,} ({pct:.1f}%)")

# Save
final_df.to_csv('triage_dataset_realistic.csv', index=False)
print(f"\nSaved to triage_dataset_realistic.csv")

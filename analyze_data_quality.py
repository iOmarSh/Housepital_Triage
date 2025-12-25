"""
Data Quality Analysis - Find labeling issues
============================================
Identifies:
1. Similar texts with different labels (inconsistent labeling)
2. Keywords that appear in multiple risk levels (confusion)
3. Potential mislabeled samples
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
import re

# Load data
df = pd.read_csv('generated_symptom_texts_clean.csv')
print("=" * 70)
print("DATA QUALITY ANALYSIS")
print("=" * 70)
print(f"Total samples: {len(df):,}")

# =============================================================================
# 1. FIND KEYWORD OVERLAP BETWEEN CLASSES
# =============================================================================
print("\n" + "=" * 70)
print("1. KEYWORD ANALYSIS - Words that appear in multiple risk levels")
print("=" * 70)

# Extract important medical keywords
def extract_keywords(text):
    text = text.lower()
    # Common medical symptoms/terms
    keywords = []
    patterns = [
        r'\bchest pain\b', r'\bheadache\b', r'\bfever\b', r'\bdizzy\b', r'\bdizziness\b',
        r'\bnausea\b', r'\bvomiting\b', r'\bbreathing\b', r'\bbreath\b', r'\bcough\b',
        r'\bpain\b', r'\bweak\b', r'\bweakness\b', r'\btired\b', r'\bfatigue\b',
        r'\bswell\b', r'\bswelling\b', r'\bswollen\b', r'\bblood\b', r'\bbleeding\b',
        r'\bheart\b', r'\bstomach\b', r'\bback\b', r'\bleg\b', r'\barm\b',
        r'\bconfusion\b', r'\bconfused\b', r'\bunconscious\b', r'\bseizure\b',
        r'\bchills\b', r'\bsweating\b', r'\brash\b', r'\binfection\b',
        r'\bsevere\b', r'\bmild\b', r'\bchronic\b', r'\bsudden\b'
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            keywords.append(pattern.replace(r'\b', '').replace('\\', ''))
    return keywords

# Build keyword -> risk level mapping
keyword_risks = defaultdict(lambda: defaultdict(int))
for _, row in df.iterrows():
    keywords = extract_keywords(row['text'])
    for kw in keywords:
        keyword_risks[kw][row['risk_level']] += 1

# Find confusing keywords (appear significantly in 3+ risk levels)
print("\nKeywords with mixed risk levels (potential confusion):")
print("-" * 70)
confusing_keywords = []
for kw, risks in sorted(keyword_risks.items()):
    if len(risks) >= 3:
        total = sum(risks.values())
        if total >= 50:  # Minimum frequency
            risk_str = ", ".join([f"{r}:{c}" for r, c in sorted(risks.items(), key=lambda x: -x[1])])
            print(f"  '{kw}': {risk_str}")
            confusing_keywords.append(kw)

# =============================================================================
# 2. FIND SIMILAR TEXTS WITH DIFFERENT LABELS
# =============================================================================
print("\n" + "=" * 70)
print("2. SIMILAR TEXTS WITH DIFFERENT LABELS (Inconsistent Labeling)")
print("=" * 70)

def text_similarity(t1, t2):
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio()

# Sample pairs to check (full comparison too slow)
samples_per_class = 200
inconsistencies = []

for risk1 in ['Emergency', 'High', 'Medium', 'Low']:
    for risk2 in ['Emergency', 'High', 'Medium', 'Low']:
        if risk1 >= risk2:  # Avoid duplicates
            continue
        
        df1 = df[df['risk_level'] == risk1].sample(n=min(samples_per_class, len(df[df['risk_level'] == risk1])), random_state=42)
        df2 = df[df['risk_level'] == risk2].sample(n=min(samples_per_class, len(df[df['risk_level'] == risk2])), random_state=42)
        
        for _, r1 in df1.iterrows():
            for _, r2 in df2.iterrows():
                sim = text_similarity(r1['text'], r2['text'])
                if sim > 0.7:  # Very similar texts
                    inconsistencies.append({
                        'text1': r1['text'][:80],
                        'label1': risk1,
                        'text2': r2['text'][:80],
                        'label2': risk2,
                        'similarity': sim
                    })

print(f"\nFound {len(inconsistencies)} similar text pairs with different labels:")
print("-" * 70)
for inc in sorted(inconsistencies, key=lambda x: -x['similarity'])[:15]:
    print(f"\nSimilarity: {inc['similarity']:.0%}")
    print(f"  [{inc['label1']:10}] {inc['text1']}...")
    print(f"  [{inc['label2']:10}] {inc['text2']}...")

# =============================================================================
# 3. FIND POTENTIAL MISLABELS
# =============================================================================
print("\n" + "=" * 70)
print("3. POTENTIAL MISLABELS (Emergency keywords in Low, etc.)")
print("=" * 70)

emergency_keywords = ['chest pain', "can't breathe", 'cannot breathe', 'unconscious', 
                      'seizure', 'stroke', 'heart attack', 'bleeding profusely',
                      'turning blue', 'crushing', 'radiating to arm']

high_keywords = ['high fever', 'fever 103', 'fever 104', 'blood in', 'vomiting blood',
                 'severe pain', 'spreading infection', 'difficulty breathing']

potential_mislabels = []

for _, row in df.iterrows():
    text_lower = row['text'].lower()
    
    # Emergency keywords in Low or Medium
    if row['risk_level'] in ['Low', 'Medium']:
        for kw in emergency_keywords:
            if kw in text_lower:
                potential_mislabels.append({
                    'text': row['text'][:80],
                    'label': row['risk_level'],
                    'issue': f"Has emergency keyword '{kw}' but labeled {row['risk_level']}"
                })
                break
    
    # High keywords in Low
    if row['risk_level'] == 'Low':
        for kw in high_keywords:
            if kw in text_lower:
                potential_mislabels.append({
                    'text': row['text'][:80],
                    'label': row['risk_level'],
                    'issue': f"Has high-risk keyword '{kw}' but labeled Low"
                })
                break

print(f"\nFound {len(potential_mislabels)} potential mislabels:")
print("-" * 70)
for ml in potential_mislabels[:20]:
    print(f"\n  [{ml['label']:10}] {ml['text']}...")
    print(f"  ISSUE: {ml['issue']}")

# =============================================================================
# 4. CLASS DISTRIBUTION BY DISEASE
# =============================================================================
print("\n" + "=" * 70)
print("4. DISEASES WITH INCONSISTENT RISK LEVELS")
print("=" * 70)

disease_risks = df.groupby(['disease', 'risk_level']).size().unstack(fill_value=0)
inconsistent_diseases = []

for disease in disease_risks.index:
    non_zero = (disease_risks.loc[disease] > 0).sum()
    if non_zero >= 2:  # Disease appears in 2+ risk levels
        total = disease_risks.loc[disease].sum()
        if total >= 20:  # Minimum samples
            inconsistent_diseases.append({
                'disease': disease,
                'counts': disease_risks.loc[disease].to_dict(),
                'total': total
            })

print(f"\nDiseases appearing in multiple risk levels ({len(inconsistent_diseases)}):")
print("-" * 70)
for d in sorted(inconsistent_diseases, key=lambda x: -x['total'])[:20]:
    counts = ", ".join([f"{k}:{v}" for k, v in d['counts'].items() if v > 0])
    print(f"  {d['disease'][:40]:40}: {counts}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY - Data Quality Issues")
print("=" * 70)
print(f"""
1. Confusing Keywords: {len(confusing_keywords)} keywords appear across 3+ risk levels
2. Similar Texts Different Labels: {len(inconsistencies)} pairs found
3. Potential Mislabels: {len(potential_mislabels)} samples may be mislabeled
4. Inconsistent Diseases: {len(inconsistent_diseases)} diseases span multiple risk levels

RECOMMENDATION:
These inconsistencies explain why the model struggles to exceed ~75-80% accuracy.
The same symptoms/keywords lead to different labels, confusing the model.

Options:
A) Clean the data manually (time-consuming but most effective)
B) Use keyword-based rules to override model for obvious cases
C) Accept ~80% accuracy as the ceiling for this dataset quality
""")

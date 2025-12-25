"""
Triage Model - Inference & API
==============================
1. Test with your own text
2. Run as Flask API endpoint
"""

import pickle
import re
from flask import Flask, request, jsonify

# =============================================================================
# LOAD MODEL
# =============================================================================
print("🔄 Loading model...")
with open('triage_production_model.pkl', 'rb') as f:
    data = pickle.load(f)
    vectorizer = data['vectorizer']
    model = data['model']
    label_map = data['label_map']
    reverse_map = data['reverse_map']
print("✅ Model loaded!")

# =============================================================================
# KEYWORD RULES (for real-world generalization)
# =============================================================================
EMERGENCY_PATTERNS = [
    r"(can'?t|cannot|unable to)\s*(breathe|breath)",
    r"(lips?|fingers?)\s*(turn|turning)\s*(blue)",
    r"chest\s*pain.*(radiat|arm|jaw)",
    r"heart\s*attack",
    r"(severe|heavy)\s*(bleed|blood)",
    r"(unconscious|passed out|fainted)",
    r"(seizure|convulsion)",
    r"stroke|paralysis",
    r"poison|overdose",
    r"anaphyla|throat\s*(clos|swell)",
]

HIGH_PATTERNS = [
    r"(cancer|tumor|malignant)",
    r"blood\s*in\s*(stool|urine)",
    r"(vomit|cough)\s*(blood)",
    r"(high|severe)\s*fever.*(confus)",
    r"fever\s*(10[3-5])",
    r"(severe|unbearable|excruciating)\s*pain",
]

LOW_PATTERNS = [
    r"(minor|small|slight)\s*(cut|bruise)",
    r"(mild|slight)\s*(cold|cough|headache)",
    r"(runny|stuffy)\s*nose",
    r"(routine|general)\s*(checkup|question)",
    r"(vitamin|supplement)\s*(question|advice)",
]

def apply_rules(text):
    """Apply keyword rules."""
    text_lower = text.lower()
    
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, text_lower):
            return 'Emergency', pattern
    
    for pattern in HIGH_PATTERNS:
        if re.search(pattern, text_lower):
            return 'High', pattern
    
    for pattern in LOW_PATTERNS:
        if re.search(pattern, text_lower):
            return 'Low', pattern
    
    return None, None

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_triage(text):
    """
    Predict triage level for symptom text.
    Returns: dict with prediction details
    """
    # Try rules first
    rule_pred, rule_pattern = apply_rules(text)
    
    if rule_pred:
        return {
            'risk_level': rule_pred,
            'source': 'RULE',
            'confidence': 1.0,
            'rule_matched': rule_pattern,
        }
    
    # Fall back to model
    X = vectorizer.transform([text])
    pred_id = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = proba[pred_id]
    
    return {
        'risk_level': reverse_map[pred_id],
        'source': 'MODEL',
        'confidence': float(confidence),
        'rule_matched': None,
    }

# =============================================================================
# FLASK API
# =============================================================================
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for triage prediction.
    
    POST /predict
    Body: {"text": "symptom description"}
    Returns: {"risk_level": "Emergency/High/Medium/Low", "confidence": 0.95, ...}
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request body'}), 400
    
    result = predict_triage(data['text'])
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': 'triage_production'})

# =============================================================================
# INTERACTIVE TESTING
# =============================================================================
def interactive_test():
    """Interactive testing mode."""
    print("\n" + "=" * 60)
    print("🏥 HOUSEPITAL TRIAGE - Interactive Test")
    print("=" * 60)
    print("Enter symptoms to get triage level (type 'quit' to exit)\n")
    
    while True:
        text = input("Symptoms: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if not text:
            continue
        
        result = predict_triage(text)
        
        print(f"\n🎯 Risk Level: {result['risk_level']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"📌 Source: {result['source']}")
        if result['rule_matched']:
            print(f"🔧 Rule: {result['rule_matched']}")
        print()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--api':
            # Run as API server
            print("\n🚀 Starting API server on http://localhost:5000")
            print("   POST /predict with {'text': 'symptoms'}")
            app.run(host='0.0.0.0', port=5000, debug=False)
        elif sys.argv[1] == '--test':
            # Interactive test mode
            interactive_test()
        else:
            # Quick prediction
            text = ' '.join(sys.argv[1:])
            result = predict_triage(text)
            print(f"\n🎯 {result['risk_level']} ({result['confidence']:.0%}) [{result['source']}]")
    else:
        # Default: interactive mode
        interactive_test()

#!/usr/bin/env python3
"""
DGA Domain Detector - Command Line Interface
Standalone script for detecting DGA domains
"""

import sys
import argparse
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

def load_model_artifacts():
    """Load the trained model, tokenizer, and configuration"""
    try:
        print("Loading model artifacts...")
        model = keras.models.load_model('dga_detector_model.h5')
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        print("✓ Model loaded successfully\n")
        return model, tokenizer, config
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please train the model first.")
        print(f"   Run the Jupyter notebook: dga_detection_training.ipynb")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)

def predict_domain(domain, model, tokenizer, max_length):
    """
    Predict if a domain is legitimate or DGA-generated
    
    Args:
        domain: Domain name to classify
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        max_length: Maximum sequence length
    
    Returns:
        label: Classification label (Legitimate/DGA)
        confidence: Confidence score (0-1)
        raw_score: Raw prediction score
    """
    # Tokenize domain at character level
    char_tokens = ' '.join(list(domain.lower()))
    sequence = tokenizer.texts_to_sequences([char_tokens])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    raw_score = model.predict(padded, verbose=0)[0][0]
    
    # Interpret result
    if raw_score > 0.5:
        label = "DGA"
        confidence = raw_score
        icon = "⚠️"
    else:
        label = "Legitimate"
        confidence = 1 - raw_score
        icon = "✅"
    
    return label, confidence, raw_score, icon

def calculate_entropy(domain):
    """Calculate Shannon entropy of domain"""
    prob = [float(domain.count(c)) / len(domain) for c in set(domain)]
    entropy = -sum([p * np.log2(p) for p in prob if p > 0])
    return entropy

def display_prediction(domain, label, confidence, raw_score, icon, verbose=False):
    """Display prediction results"""
    print("=" * 70)
    print(f"Domain: {domain}")
    print("=" * 70)
    print(f"{icon} Classification: {label}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   DGA Score: {raw_score:.4f}")
    print(f"   Legitimate Score: {1-raw_score:.4f}")
    
    if verbose:
        entropy = calculate_entropy(domain)
        print(f"\nDomain Features:")
        print(f"   Length: {len(domain)}")
        print(f"   Unique Characters: {len(set(domain))}")
        print(f"   Entropy: {entropy:.2f}")
        print(f"   Digits: {sum(c.isdigit() for c in domain)}")
        print(f"   Vowels: {sum(c.lower() in 'aeiou' for c in domain)}")
    
    print("=" * 70)
    print()

def analyze_file(file_path, model, tokenizer, max_length):
    """Analyze domains from a file"""
    import pandas as pd
    
    try:
        # Read file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'domain' not in df.columns:
                print("❌ Error: CSV must have a 'domain' column")
                return
            domains = df['domain'].tolist()
        else:
            # Text file, one domain per line
            with open(file_path, 'r') as f:
                domains = [line.strip() for line in f if line.strip()]
        
        print(f"Analyzing {len(domains)} domains from {file_path}...\n")
        
        results = []
        dga_count = 0
        legit_count = 0
        
        for domain in domains:
            label, confidence, raw_score, icon = predict_domain(
                domain, model, tokenizer, max_length
            )
            results.append({
                'domain': domain,
                'prediction': label,
                'confidence': f"{confidence*100:.2f}%",
                'dga_score': f"{raw_score:.4f}"
            })
            
            if label == "DGA":
                dga_count += 1
            else:
                legit_count += 1
        
        # Display results
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total Domains: {len(domains)}")
        print(f"DGA Domains: {dga_count} ({dga_count/len(domains)*100:.1f}%)")
        print(f"Legitimate Domains: {legit_count} ({legit_count/len(domains)*100:.1f}%)")
        print("=" * 70)
        
        # Save results
        output_file = file_path.replace('.csv', '_results.csv').replace('.txt', '_results.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description='DGA Domain Detector - Detect malicious domain names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py google
  python predict.py xjfkdslfjkdslfj --verbose
  python predict.py --file domains.txt
  python predict.py --file domains.csv
  python predict.py --interactive
        """
    )
    
    parser.add_argument('domain', nargs='?', help='Domain name to analyze')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Show detailed analysis')
    parser.add_argument('-f', '--file', help='Analyze domains from a file (CSV or TXT)')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode - analyze multiple domains')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, config = load_model_artifacts()
    
    # File mode
    if args.file:
        analyze_file(args.file, model, tokenizer, config['max_length'])
        return
    
    # Interactive mode
    if args.interactive:
        print("🛡️  DGA Domain Detector - Interactive Mode")
        print("Enter domain names to analyze (type 'quit' to exit)\n")
        
        while True:
            domain = input("Enter domain: ").strip()
            if domain.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if domain:
                label, confidence, raw_score, icon = predict_domain(
                    domain, model, tokenizer, config['max_length']
                )
                display_prediction(domain, label, confidence, raw_score, icon, args.verbose)
        return
    
    # Single domain mode
    if args.domain:
        label, confidence, raw_score, icon = predict_domain(
            args.domain, model, tokenizer, config['max_length']
        )
        display_prediction(args.domain, label, confidence, raw_score, icon, args.verbose)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

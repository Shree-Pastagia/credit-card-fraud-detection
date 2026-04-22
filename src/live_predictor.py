"""Live Prediction - CLI interface for fraud detection"""
import numpy as np


def cli_live_input(model):
    """CLI interface for live prediction"""
    print("\n" + "="*50)
    print("      READY FOR USER INPUT      ")
    print("="*50)
    print("\nEnter 30 comma-separated feature values")
    print("Order: Time, V1-V28, Amount")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("Paste 30 values (or 'exit'): ").strip()
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!\n")
            break

        try:
            values = [float(v.strip()) for v in user_input.split(',') if v.strip()]
            
            if len(values) != 30:
                print(f"❌ Need 30 values, got {len(values)}\n")
                continue

            test_data = np.array(values).reshape(1, -1)
            prediction = model.predict(test_data)[0]
            prob = model.predict_proba(test_data)[0]
            confidence = prob[1] if prediction == 1 else prob[0]

            result = "🚨 FRAUDULENT" if prediction == 1 else "✅ LEGITIMATE"
            print(f"\n{result} ({confidence*100:.2f}%)\n")
            
        except ValueError:
            print("❌ Error: Enter only numbers separated by commas\n")

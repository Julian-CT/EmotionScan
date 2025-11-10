#!/usr/bin/env python3
"""
Analyze Model Results - Compare with expected performance
Based on the classification reports shown in images
"""

print("=" * 70)
print("ANALYSIS: Comparing Test Results with Expected Performance")
print("=" * 70)

# Expected results from Image 1 (smaller dataset - 373 samples)
image1_results = {
    "AUSENTE": {"precision": 0.86, "recall": 0.73, "f1": 0.79, "support": 115},
    "RAIVA": {"precision": 0.63, "recall": 0.69, "f1": 0.66, "support": 95},
    "TRISTEZA": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 20},
    "MEDO": {"precision": 0.76, "recall": 0.71, "f1": 0.73, "support": 89},
    "CONFIANÇA": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 19},
    "ALEGRIA": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 29},
    "AMOR": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 6},
}

# Expected results from Image 2 (larger dataset with threshold 0.5 - 1870 samples)
image2_results = {
    "AUSENTE": {"precision": 0.93, "recall": 0.88, "f1": 0.91, "support": 564},
    "RAIVA": {"precision": 0.79, "recall": 0.89, "f1": 0.83, "support": 487},
    "TRISTEZA": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 109},
    "MEDO": {"precision": 0.84, "recall": 0.82, "f1": 0.83, "support": 441},
    "CONFIANÇA": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 88},
    "ALEGRIA": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 148},
    "AMOR": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 33},
}

print("\n1. KEY OBSERVATIONS FROM IMAGES:")
print("   " + "-" * 65)
print("\n   Image 1 (373 samples):")
print("   - Good performance: AUSENTE (f1=0.79), MEDO (f1=0.73), RAIVA (f1=0.66)")
print("   - Zero performance: TRISTEZA, CONFIANCA, ALEGRIA, AMOR")
print("\n   Image 2 (1870 samples, threshold=0.5):")
print("   - Better overall: AUSENTE (f1=0.91), RAIVA (f1=0.83), MEDO (f1=0.83)")
print("   - Still zero: TRISTEZA, CONFIANCA, ALEGRIA, AMOR")

print("\n2. MODEL BIAS ANALYSIS:")
print("   " + "-" * 65)
print("\n   Working Emotions (Positive Performance):")
working_emotions = ["AUSENTE", "RAIVA", "MEDO"]
for emotion in working_emotions:
    img1 = image1_results[emotion]
    img2 = image2_results[emotion]
    print(f"   - {emotion}:")
    print(f"     Image 1: f1={img1['f1']:.2f}, support={img1['support']}")
    print(f"     Image 2: f1={img2['f1']:.2f}, support={img2['support']}")

print("\n   Failing Emotions (Zero Performance):")
failing_emotions = ["TRISTEZA", "CONFIANÇA", "ALEGRIA", "AMOR"]
for emotion in failing_emotions:
    img1 = image1_results[emotion]
    img2 = image2_results[emotion]
    print(f"   - {emotion}:")
    print(f"     Image 1: f1={img1['f1']:.2f}, support={img1['support']}")
    print(f"     Image 2: f1={img2['f1']:.2f}, support={img2['support']}")

print("\n3. IMPLICATIONS FOR FEELING CLASSIFICATION:")
print("   " + "-" * 65)

# Map emotions to feelings
def emotion_to_feeling(emotion):
    if emotion in ["ALEGRIA", "AMOR", "CONFIANÇA"]:
        return "Positivo"
    elif emotion in ["RAIVA", "TRISTEZA", "MEDO"]:
        return "Negativo"
    else:
        return "Neutro"

print("\n   Feeling Category Breakdown:")
print("\n   Positivo (should come from ALEGRIA, AMOR, CONFIANÇA):")
positive_emotions = ["ALEGRIA", "AMOR", "CONFIANÇA"]
for emo in positive_emotions:
    img2 = image2_results[emo]
    total = img2['support']
    print(f"     - {emo}: {total} samples, but f1=0.00 (model fails completely)")
print("     CONCLUSION: Positivo category will have issues!")

print("\n   Negativo (should come from RAIVA, TRISTEZA, MEDO):")
negative_emotions = ["RAIVA", "TRISTEZA", "MEDO"]
neg_total = 0
neg_working = 0
for emo in negative_emotions:
    img2 = image2_results[emo]
    total = img2['support']
    neg_total += total
    if img2['f1'] > 0:
        neg_working += total
        print(f"     - {emo}: {total} samples, f1={img2['f1']:.2f} (WORKING)")
    else:
        print(f"     - {emo}: {total} samples, f1=0.00 (FAILING)")
print(f"     CONCLUSION: Negativo category mostly works ({neg_working}/{neg_total} samples working)")

print("\n   Neutro (should come from AUSENTE):")
neutro_emotions = ["AUSENTE"]
for emo in neutro_emotions:
    img2 = image2_results[emo]
    print(f"     - {emo}: {img2['support']} samples, f1={img2['f1']:.2f} (WORKING)")
print("     CONCLUSION: Neutro category works well!")

print("\n4. EXPECTED FEELING PERFORMANCE:")
print("   " + "-" * 65)
print("\n   Based on emotion performance:")

# Calculate expected feeling metrics from image 2 (larger dataset)
positivo_support = sum([image2_results[e]['support'] for e in positive_emotions])
negativo_support = sum([image2_results[e]['support'] for e in negative_emotions])
neutro_support = image2_results['AUSENTE']['support']

print(f"\n   Positivo:")
print(f"     - Total samples: {positivo_support}")
print(f"     - Expected: POOR (all emotions have f1=0.00)")
print(f"     - Issue: Model cannot detect positive emotions")

print(f"\n   Negativo:")
print(f"     - Total samples: {negativo_support}")
print(f"     - Expected: MODERATE to GOOD")
print(f"     - RAIVA and MEDO work well (f1=0.83 each)")
print(f"     - TRISTEZA fails (f1=0.00), but represents {image2_results['TRISTEZA']['support']}/{negativo_support} samples")

print(f"\n   Neutro:")
print(f"     - Total samples: {neutro_support}")
print(f"     - Expected: EXCELLENT (f1=0.91)")
print(f"     - AUSENTE detection works very well")

print("\n5. COMPARISON WITH OUR TEST RESULTS:")
print("   " + "-" * 65)
print("\n   Our Test Results (7 sample cases):")
print("   - Positivo: precision=1.00, recall=0.67, f1=0.80 (GOOD)")
print("   - Negativo: precision=0.60, recall=1.00, f1=0.75 (GOOD)")
print("   - Neutro: precision=0.00, recall=0.00, f1=0.00 (FAILING)")
print("\n   WARNING: Our test used only 7 samples - not representative!")
print("\n   Expected Results (based on images with 1870 samples):")
print("   - Positivo: Should be POOR (f1 near 0.00) - all positive emotions fail")
print("   - Negativo: Should be MODERATE-GOOD (f1 ~0.60-0.75) - RAIVA and MEDO work")
print("   - Neutro: Should be EXCELLENT (f1 ~0.91) - AUSENTE works very well")

print("\n6. CONCLUSION:")
print("   " + "-" * 65)
print("\n   The model shows significant bias:")
print("     - Only detects: AUSENTE, RAIVA, MEDO")
print("     - Fails completely on: TRISTEZA, CONFIANCA, ALEGRIA, AMOR")
print("\n   For Feeling Classification:")
print("     - Neutro should work EXCELLENT (from AUSENTE)")
print("     - Negativo should work MODERATELY (from RAIVA, MEDO, but not TRISTEZA)")
print("     - Positivo will have MAJOR ISSUES (all positive emotions fail)")
print("\n   WARNING: Our small test (7 samples) does NOT match expected results.")
print("   WARNING: Need to test with full dataset (1870+ samples) to confirm.")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)


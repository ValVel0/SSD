#!/usr/bin/env python3
"""
Display Brain Tumor SSD Results
"""
import pickle

# Load results
with open('brain_tumor_ssd_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("🧠 BRAIN TUMOR SSD CLASSIFICATION RESULTS")
print("="*50)

print(f"\n📋 Algorithm: {results['algorithm']}")
print(f"📋 Dataset: {results['dataset']}")
print(f"📋 Classes: {results['classes']}")

perf = results['performance']
print(f"\n📊 PERFORMANCE METRICS:")
print(f"   Final Accuracy: {perf['final_accuracy']:.4f} ({perf['final_accuracy']*100:.2f}%)")
print(f"   Balanced Accuracy: {perf['balanced_accuracy']:.4f} ({perf['balanced_accuracy']*100:.2f}%)")
print(f"   F1-Score: {perf['f1_score']:.4f}")

print(f"\n🎯 FEATURE SELECTION:")
print(f"   Selected Features: {perf['selected_features']:,}")
print(f"   Total Iterations: {perf['total_iterations']}")
print(f"   Final Diversity: {perf.get('final_diversity', 0):.4f}")

print(f"\n⏱️  RUNTIME:")
print(f"   Total Time: {perf['runtime_minutes']:.1f} minutes")

print(f"\n🔧 OPTIMIZATIONS USED:")
for opt in results['optimizations']:
    print(f"   ✅ {opt}")

print(f"\n🎉 Brain tumor classification completed successfully!")
print(f"   4-class classification: glioma, meningioma, notumor, pituitary")
print(f"   Paper-faithful SSD with NO PCA preprocessing")
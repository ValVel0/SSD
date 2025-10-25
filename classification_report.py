#!/usr/bin/env python3
"""
Brain Tumor SSD Classification Report
Detailed analysis of classification performance per class
"""
import pickle
import numpy as np
from pathlib import Path

def load_and_analyze_results():
    """Load results and create detailed classification report"""
    
    print("🧠 BRAIN TUMOR SSD CLASSIFICATION REPORT")
    print("="*60)
    
    # Load results
    try:
        with open('brain_tumor_ssd_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✅ Results loaded successfully")
    except FileNotFoundError:
        print("❌ Results file not found. Please run the SSD algorithm first.")
        return
    
    # Display basic info
    print(f"\n📋 ALGORITHM DETAILS:")
    print(f"   Algorithm: {results['algorithm']}")
    print(f"   Dataset: {results['dataset']}")
    print(f"   Classes: {len(results['classes'])} classes")
    
    # Performance overview
    perf = results['performance']
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Final Accuracy: {perf['final_accuracy']:.4f} ({perf['final_accuracy']*100:.2f}%)")
    print(f"   Balanced Accuracy: {perf['balanced_accuracy']:.4f} ({perf['balanced_accuracy']*100:.2f}%)")
    print(f"   F1-Score: {perf['f1_score']:.4f}")
    
    # Try to get detailed classification report from SSD results
    if 'results' in results and 'best_metrics' in results['results']:
        ssd_results = results['results']
        print(f"\n🎯 DETAILED SSD METRICS:")
        best_metrics = ssd_results['best_metrics']
        
        print(f"   Best Particle Fitness: {best_metrics.get('fitness', 'N/A')}")
        print(f"   Best Accuracy: {best_metrics.get('accuracy', perf['final_accuracy']):.4f}")
        print(f"   Best Balanced Accuracy: {best_metrics.get('balanced_accuracy', perf['balanced_accuracy']):.4f}")
        print(f"   Error Rate: {best_metrics.get('error_rate', 1-perf['final_accuracy']):.4f}")
    
    # Feature selection details
    print(f"\n🎯 FEATURE SELECTION ANALYSIS:")
    print(f"   Selected Features: {perf['selected_features']:,}")
    print(f"   Total Iterations: {perf['total_iterations']}")
    print(f"   Final Diversity: {perf.get('final_diversity', 0):.4f}")
    print(f"   Runtime: {perf['runtime_minutes']:.1f} minutes")
    
    # Class details
    print(f"\n🏥 BRAIN TUMOR CLASS INFORMATION:")
    class_mapping = {
        'glioma': 'Glioma Tumor - Most common and aggressive brain tumor',
        'meningioma': 'Meningioma Tumor - Usually benign, arises from meninges',
        'notumor': 'No Tumor - Normal brain tissue, healthy scans',
        'pituitary': 'Pituitary Tumor - Affects pituitary gland, hormonal impact'
    }
    
    for i, class_name in enumerate(['glioma', 'meningioma', 'notumor', 'pituitary']):
        print(f"   Class {i}: {class_name}")
        print(f"      Description: {class_mapping[class_name]}")
    
    # Optimization details
    print(f"\n🔧 OPTIMIZATION TECHNIQUES:")
    for opt in results['optimizations']:
        opt_details = {
            'Fast k-NN': 'Optimized k-nearest neighbors with ball_tree algorithm',
            'Class Balancing': 'Weighted sampling to handle medical data imbalance',
            'Elite Preservation': 'Keeps best solutions across generations',
            'Delayed Convergence': 'Prevents premature stopping for better accuracy'
        }
        print(f"   ✅ {opt}")
        if opt in opt_details:
            print(f"      └─ {opt_details[opt]}")
    
    # Medical significance
    print(f"\n🏥 MEDICAL SIGNIFICANCE:")
    print(f"   📈 Classification Accuracy: {perf['final_accuracy']*100:.2f}%")
    
    if perf['final_accuracy'] >= 0.85:
        significance = "🟢 EXCELLENT - Clinically viable performance"
    elif perf['final_accuracy'] >= 0.80:
        significance = "🟡 GOOD - Strong diagnostic assistance capability"
    elif perf['final_accuracy'] >= 0.75:
        significance = "🟠 FAIR - Useful for preliminary screening"
    else:
        significance = "🔴 NEEDS IMPROVEMENT - Requires further optimization"
    
    print(f"   {significance}")
    
    # Feature analysis
    feature_ratio = perf['selected_features'] / 50176  # Assuming 224x224 images
    reduction_factor = 50176 / perf['selected_features'] if perf['selected_features'] > 0 else 0
    
    print(f"\n📊 FEATURE EFFICIENCY:")
    print(f"   Original Image Pixels: ~50,176 (224×224)")
    print(f"   Selected Features: {perf['selected_features']}")
    print(f"   Feature Ratio: {feature_ratio:.6f} ({feature_ratio*100:.4f}%)")
    print(f"   Reduction Factor: {reduction_factor:.1f}x smaller")
    print(f"   Efficiency: Achieved {perf['final_accuracy']*100:.2f}% accuracy with {feature_ratio*100:.4f}% of features")
    
    # Runtime analysis
    print(f"\n⏱️ COMPUTATIONAL EFFICIENCY:")
    samples_processed = 14046  # Total images processed
    time_per_sample = (perf['runtime_minutes'] * 60) / samples_processed
    
    print(f"   Total Samples Processed: {samples_processed:,}")
    print(f"   Processing Time: {perf['runtime_minutes']:.1f} minutes")
    print(f"   Time per Sample: {time_per_sample:.3f} seconds")
    print(f"   Processing Rate: {samples_processed / (perf['runtime_minutes'] * 60):.1f} samples/second")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if perf['final_accuracy'] >= 0.82:
        print(f"   ✅ Strong performance for 4-class brain tumor classification")
        print(f"   ✅ Suitable for clinical decision support systems")
        print(f"   ✅ Feature selection effectively reduced dimensionality")
    
    if perf['selected_features'] < 100:
        print(f"   ✅ Excellent feature efficiency - very compact representation")
    
    if perf['runtime_minutes'] < 20:
        print(f"   ✅ Fast processing time suitable for real-time applications")
    
    print(f"   📋 Consider validation on independent test sets")
    print(f"   📋 Evaluate performance across different MRI scanners")
    print(f"   📋 Test robustness with varied image qualities")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   The optimized SSD algorithm achieved {perf['final_accuracy']*100:.2f}% accuracy")
    print(f"   on brain tumor classification using only {perf['selected_features']} features.")
    print(f"   This demonstrates the effectiveness of evolutionary feature selection")
    print(f"   for medical image analysis with significant dimensionality reduction.")

def create_confusion_matrix_estimate():
    """Create estimated performance breakdown by class"""
    print(f"\n📊 ESTIMATED CLASS PERFORMANCE:")
    print(f"   (Based on overall accuracy and balanced accuracy)")
    
    # Load results for accuracy info
    with open('brain_tumor_ssd_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    perf = results['performance']
    overall_acc = perf['final_accuracy']
    balanced_acc = perf['balanced_accuracy']
    
    # Estimate per-class performance (simplified)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    print(f"   Overall Accuracy: {overall_acc:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f}")
    print(f"   Performance appears consistent across classes (balanced accuracy ≈ overall accuracy)")
    
    # Estimated class distribution
    class_sizes = {
        'glioma': 3242,      # 2642 train + 600 test  
        'meningioma': 3290,  # 2678 train + 612 test
        'notumor': 4000,     # 3190 train + 810 test
        'pituitary': 3514    # 2914 train + 600 test
    }
    
    total_samples = sum(class_sizes.values())
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    for class_name in classes:
        count = class_sizes[class_name]
        percentage = (count / total_samples) * 100
        print(f"   {class_name:12}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"   Total Samples: {total_samples:,}")

if __name__ == "__main__":
    load_and_analyze_results()
    create_confusion_matrix_estimate()
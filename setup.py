#!/usr/bin/env python3
"""
Brain Tumor MRI Classification with SSD Algorithm
Demo script showing how to set up and run the optimized SSD on brain tumor data

Download the dataset from:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected folder structure:
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
"""

import os
import sys
from pathlib import Path

def setup_brain_tumor_classification():
    """Setup and run brain tumor classification with SSD"""
    
    print("🧠 Brain Tumor MRI Classification Setup")
    print("="*50)
    
    # Step 1: Dataset Setup Options
    print("\n📁 Step 1: Dataset Setup Options")
    print("Choose how to get the brain tumor dataset:")
    print("1. 📥 Auto-download with KaggleHub (recommended)")
    print("2. 📂 Use existing local dataset path")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Auto-download option
            print("\n📥 Auto-download selected")
            print("Requirements:")
            print("- KaggleHub installed: pip install kagglehub")
            print("- Kaggle API credentials configured")
            
            try:
                import kagglehub
                print("✅ KaggleHub is available")
                
                confirm = input("Download dataset automatically? (y/n): ").strip().lower()
                if confirm == 'y':
                    return "auto_download"
                else:
                    print("Switching to manual path option...")
                    choice = "2"
                    
            except ImportError:
                print("❌ KaggleHub not installed")
                install = input("Install KaggleHub now? (y/n): ").strip().lower()
                if install == 'y':
                    try:
                        import subprocess
                        subprocess.check_call(["pip", "install", "kagglehub"])
                        print("✅ KaggleHub installed successfully")
                        return "auto_download"
                    except Exception as e:
                        print(f"❌ Failed to install KaggleHub: {e}")
                        print("Switching to manual path option...")
                        choice = "2"
                else:
                    print("Switching to manual path option...")
                    choice = "2"
        
        if choice == "2":
            # Manual path option
            print("\n📂 Manual dataset path selected")
            print("Please download the brain tumor dataset from:")
            print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            
            # Get dataset path from user
            dataset_path = input("\n📂 Enter the path to your downloaded brain-tumor-mri-dataset folder: ").strip()
            dataset_path = Path(dataset_path.strip('"\''))  # Remove quotes if present
            
            if not dataset_path.exists():
                print(f"❌ Dataset path not found: {dataset_path}")
                print("Please download the dataset and provide the correct path.")
                return False
            
            return str(dataset_path)
        
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    # Step 2: Verify dataset structure
    print(f"\n🔍 Step 2: Verifying Dataset Structure")
    training_path = dataset_path / "Training"
    testing_path = dataset_path / "Testing"
    
    expected_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    if training_path.exists() and testing_path.exists():
        print("✅ Found Training/Testing folder structure")
        
        # Check class folders
        missing_classes = []
        for class_name in expected_classes:
            train_class_path = training_path / class_name
            test_class_path = testing_path / class_name
            
            if not train_class_path.exists():
                missing_classes.append(f"Training/{class_name}")
            if not test_class_path.exists():
                missing_classes.append(f"Testing/{class_name}")
        
        if missing_classes:
            print(f"⚠️  Missing folders: {', '.join(missing_classes)}")
        else:
            print("✅ All class folders found")
    else:
        print("⚠️  Training/Testing folders not found")
        print("Checking for direct class folders...")
        
        missing_classes = []
        for class_name in expected_classes:
            class_path = dataset_path / class_name
            if not class_path.exists():
                missing_classes.append(class_name)
        
        if missing_classes:
            print(f"❌ Missing class folders: {', '.join(missing_classes)}")
            return False
        else:
            print("✅ Found direct class folders")
    
    # Step 3: Update the SSD script with correct path
    print(f"\n🔧 Step 3: Updating SSD Script")
    
    # Read the current SSD script
    ssd_script_path = Path("true_paper_ssd.py")
    if not ssd_script_path.exists():
        print(f"❌ SSD script not found: {ssd_script_path}")
        return False
    
    # Update the dataset path in the script
    with open(ssd_script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the placeholder path
    old_path = r'brain_tumor_path = Path(r"C:\path\to\brain-tumor-mri-dataset")'
    new_path = f'brain_tumor_path = Path(r"{dataset_path}")'
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        
        with open(ssd_script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated dataset path in {ssd_script_path}")
        print(f"   Path set to: {dataset_path}")
    else:
        print(f"⚠️  Could not auto-update path. Please manually edit:")
        print(f"   File: {ssd_script_path}")
        print(f"   Change: brain_tumor_path = Path(r\"{dataset_path}\")")
    
    # Step 4: Ready to run
    print(f"\n🚀 Step 4: Ready to Run!")
    print(f"You can now run the brain tumor classification:")
    print(f"python true_paper_ssd.py")
    
    return True

def estimate_runtime():
    """Provide runtime estimates"""
    print(f"\n⏱️  Expected Runtime Estimates:")
    print(f"   Dataset size: ~3,000-7,000 images")
    print(f"   Image processing: 2-5 minutes")
    print(f"   SSD optimization: 15-45 minutes")
    print(f"   Total time: 20-50 minutes")
    print(f"   (depends on hardware and dataset size)")

def main():
    """Main setup function"""
    print("🧠 Brain Tumor MRI Classification with SSD")
    print("Optimized TRUE Paper-Faithful Social Ski-Driver Implementation")
    print("="*60)
    
    dataset_option = setup_brain_tumor_classification()
    
    if dataset_option:
        estimate_runtime()
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Install dependencies: pip install -r requirements.txt")
        if dataset_option == "auto_download":
            print(f"2. Dataset will be auto-downloaded during classification")
        else:
            print(f"2. Dataset path configured: {dataset_option}")
        print(f"3. Run classification: python true_paper_ssd.py")
        print(f"4. Results will be saved to: brain_tumor_ssd_results.pkl")
        
        run_now = input(f"\n▶️  Run classification now? (y/n): ").strip().lower()
        
        if run_now == 'y':
            print(f"\n🚀 Starting Brain Tumor Classification...")
            
            # Update the SSD script with the chosen option
            if dataset_option != "auto_download":
                update_ssd_script_path(dataset_option)
            
            try:
                from brain_tumor_ssd import run_enhanced_true_paper_ssd
                results = run_enhanced_true_paper_ssd()
                
                if results:
                    print(f"\n🎉 Classification completed successfully!")
                else:
                    print(f"\n❌ Classification encountered issues")
                    
            except Exception as e:
                print(f"\n❌ Error running classification: {e}")
                print(f"You may need to install dependencies first:")
                print(f"pip install -r requirements.txt")
    
    else:
        print(f"\n❌ Setup failed. Please check the dataset path and structure.")

def update_ssd_script_path(dataset_path):
    """Update the SSD script with the manual dataset path"""
    ssd_script_path = Path("true_paper_ssd.py")
    if not ssd_script_path.exists():
        return
    
    try:
        with open(ssd_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the placeholder path
        old_path = r'brain_tumor_path = Path(r"C:\path\to\brain-tumor-mri-dataset")'
        new_path = f'brain_tumor_path = Path(r"{dataset_path}")'
        
        if old_path in content:
            content = content.replace(old_path, new_path)
            
            with open(ssd_script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Updated dataset path in {ssd_script_path}")
    except Exception as e:
        print(f"⚠️  Could not update script: {e}")

if __name__ == "__main__":
    main()
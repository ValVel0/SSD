#!/usr/bin/env python3
"""
Brain Tumor MRI Dataset Loader
For Kaggle dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset Structure:
- Training/
  - glioma_tumor/
  - meningioma_tumor/  
  - no_tumor/
  - pituitary_tumor/
- Testing/
  - glioma_tumor/
  - meningioma_tumor/
  - no_tumor/
  - pituitary_tumor/
"""

import numpy as np
import os
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# KaggleHub for automatic dataset download
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("⚠️  KaggleHub not available. Manual dataset path required.")

class BrainTumorDataLoader:
    """Load and preprocess brain tumor MRI dataset"""
    
    def __init__(self, data_path, target_size=(224, 224)):
        self.data_path = Path(data_path)
        self.target_size = target_size
        # Updated class names to match the actual dataset structure
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def load_images_from_folder(self, folder_path, class_label):
        """Load images from a specific folder"""
        images = []
        labels = []
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder}")
            return images, labels
            
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        print(f"   Found {len(image_files)} image files in {folder.name}")
        
        loaded_count = 0
        for img_path in image_files:
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"     ⚠️ Could not load: {img_path.name}")
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img = cv2.resize(img, self.target_size)
                
                # Convert to grayscale for consistency with paper
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                images.append(img_gray)
                labels.append(class_label)
                loaded_count += 1
                
            except Exception as e:
                print(f"     ⚠️ Error loading {img_path.name}: {e}")
                continue
        
        print(f"   ✅ Successfully loaded {loaded_count} images from {folder.name}")
        return np.array(images), np.array(labels)
    
    def load_dataset(self):
        """Load the complete brain tumor dataset"""
        print(f"\n📁 Loading Brain Tumor MRI Dataset...")
        print(f"   Path: {self.data_path}")
        print(f"   Target size: {self.target_size}")
        print(f"   Classes: {self.class_names}")
        
        # Validate dataset path exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.data_path}\n"
                f"Please check the path or use auto_download=True"
            )
        
        all_images = []
        all_labels = []
        
        # Check for Training/Testing structure or direct class folders
        training_path = self.data_path / "Training"
        testing_path = self.data_path / "Testing"
        
        if training_path.exists() and testing_path.exists():
            print(f"\n📂 Found Training/Testing structure")
            
            # Load training data
            train_images = []
            train_labels = []
            
            for class_name in self.class_names:
                class_folder = training_path / class_name
                if class_folder.exists():
                    class_images, class_labels = self.load_images_from_folder(
                        class_folder, self.class_to_idx[class_name]
                    )
                    train_images.extend(class_images)
                    train_labels.extend(class_labels)
            
            # Load testing data
            test_images = []
            test_labels = []
            
            for class_name in self.class_names:
                class_folder = testing_path / class_name
                if class_folder.exists():
                    class_images, class_labels = self.load_images_from_folder(
                        class_folder, self.class_to_idx[class_name]
                    )
                    test_images.extend(class_images)
                    test_labels.extend(class_labels)
            
            train_images = np.array(train_images)
            train_labels = np.array(train_labels)
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            
            # Validate that images were loaded
            if len(train_images) == 0 and len(test_images) == 0:
                raise ValueError(
                    f"No images found in dataset!\n"
                    f"Expected Training/Testing structure:\n"
                    f"  {training_path}/glioma_tumor/\n"
                    f"  {training_path}/meningioma_tumor/\n"
                    f"  {training_path}/no_tumor/\n"
                    f"  {training_path}/pituitary_tumor/\n"
                    f"  {testing_path}/ (same structure)\n"
                    f"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff"
                )
            
        else:
            print(f"\n📂 Looking for direct class folders")
            
            # Load all data from class folders
            for class_name in self.class_names:
                class_folder = self.data_path / class_name
                if class_folder.exists():
                    class_images, class_labels = self.load_images_from_folder(
                        class_folder, self.class_to_idx[class_name]
                    )
                    all_images.extend(class_images)
                    all_labels.extend(class_labels)
            
            all_images = np.array(all_images)
            all_labels = np.array(all_labels)
            
            # Check if any images were loaded
            if len(all_images) == 0:
                raise ValueError(
                    f"No images found in dataset!\n"
                    f"Expected folder structure:\n"
                    f"  {self.data_path}/glioma_tumor/\n"
                    f"  {self.data_path}/meningioma_tumor/\n"
                    f"  {self.data_path}/no_tumor/\n"
                    f"  {self.data_path}/pituitary_tumor/\n"
                    f"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff"
                )
            
            # Split into train/test
            print(f"\n🔄 Splitting dataset (80% train, 20% test)...")
            train_images, test_images, train_labels, test_labels = train_test_split(
                all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
            )
        
        # Print dataset statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Training samples: {len(train_images):,}")
        print(f"   Testing samples: {len(test_images):,}")
        print(f"   Image shape: {train_images[0].shape if len(train_images) > 0 else 'N/A'}")
        
        # Class distribution
        unique_train, counts_train = np.unique(train_labels, return_counts=True)
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        
        print(f"\n   Training class distribution:")
        for i, (class_idx, count) in enumerate(zip(unique_train, counts_train)):
            class_name = self.class_names[class_idx]
            print(f"     {class_name}: {count:,} samples")
        
        print(f"\n   Testing class distribution:")
        for i, (class_idx, count) in enumerate(zip(unique_test, counts_test)):
            class_name = self.class_names[class_idx]
            print(f"     {class_name}: {count:,} samples")
        
        # Calculate class imbalance ratio
        max_count = max(counts_train)
        min_count = min(counts_train)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n   Class imbalance ratio: {imbalance_ratio:.1f}:1")
        
        return train_images, train_labels, test_images, test_labels
    
    def preprocess_for_ssd(self, images):
        """Preprocess images for SSD algorithm"""
        # Flatten and normalize
        processed = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
        
        # Feature dimensionality management
        original_features = processed.shape[1]
        
        if original_features > 5000:
            print(f"\n🔧 Managing feature dimensionality:")
            print(f"   Original features: {original_features:,}")
            
            # Intelligent feature sampling - take every nth feature
            target_features = min(5000, original_features)
            sample_ratio = target_features / original_features
            
            # Create sampling indices
            step = max(1, original_features // target_features)
            sample_indices = np.arange(0, original_features, step)[:target_features]
            
            processed = processed[:, sample_indices]
            print(f"   Sampled features: {processed.shape[1]:,}")
            print(f"   Reduction ratio: {original_features / processed.shape[1]:.1f}x")
        
        return processed
    
    def diagnose_dataset_structure(self):
        """Diagnose and report dataset structure issues"""
        print(f"\n🔍 Dataset Structure Diagnosis:")
        print(f"   Root path: {self.data_path}")
        
        if not self.data_path.exists():
            print(f"   ❌ Root path does not exist!")
            return False
        
        print(f"   ✅ Root path exists")
        
        # List all items in root
        items = list(self.data_path.iterdir())
        print(f"   📁 Items in root directory:")
        for item in items:
            if item.is_dir():
                print(f"     📁 {item.name}/")
            else:
                print(f"     📄 {item.name}")
        
        # Check for Training/Testing structure
        training_path = self.data_path / "Training"
        testing_path = self.data_path / "Testing"
        
        if training_path.exists() and testing_path.exists():
            print(f"\n   ✅ Found Training/Testing structure")
            self._check_class_folders(training_path, "Training")
            self._check_class_folders(testing_path, "Testing")
            return True
        
        # Check for direct class folders
        print(f"\n   🔍 Checking for direct class folders:")
        found_classes = []
        for class_name in self.class_names:
            class_path = self.data_path / class_name
            if class_path.exists():
                image_count = self._count_images(class_path)
                print(f"     ✅ {class_name}: {image_count} images")
                found_classes.append(class_name)
            else:
                print(f"     ❌ {class_name}: not found")
        
        if len(found_classes) > 0:
            print(f"   ✅ Found {len(found_classes)} class folders")
            return True
        
        print(f"   ❌ No valid dataset structure found!")
        return False
    
    def _check_class_folders(self, base_path, folder_type):
        """Check class folders within Training or Testing"""
        for class_name in self.class_names:
            class_path = base_path / class_name
            if class_path.exists():
                image_count = self._count_images(class_path)
                print(f"     ✅ {folder_type}/{class_name}: {image_count} images")
            else:
                print(f"     ❌ {folder_type}/{class_name}: not found")
    
    def _count_images(self, folder_path):
        """Count images in a folder"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        count = 0
        for ext in extensions:
            count += len(list(folder_path.glob(f'*{ext}')))
            count += len(list(folder_path.glob(f'*{ext.upper()}')))
        return count

def download_brain_tumor_dataset():
    """
    Automatically download the brain tumor dataset using KaggleHub
    
    Returns:
        str: Path to the downloaded dataset
    """
    if not KAGGLEHUB_AVAILABLE:
        raise ImportError(
            "KaggleHub is not installed. Install it with: pip install kagglehub\n"
            "Or manually download the dataset from: "
            "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
        )
    
    print("📥 Downloading brain tumor dataset using KaggleHub...")
    print("   Dataset: masoudnickparvar/brain-tumor-mri-dataset")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        print(f"✅ Dataset downloaded successfully!")
        print(f"   Path: {path}")
        return path
    
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print(f"   Please ensure you have Kaggle API credentials configured")
        print(f"   Or manually download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        raise

def load_brain_tumor_dataset(data_path=None, target_size=(224, 224), auto_download=False):
    """
    Convenience function to load brain tumor dataset
    
    Args:
        data_path: Path to the brain tumor dataset (optional if auto_download=True)
        target_size: Target image size (height, width)
        auto_download: If True, automatically download dataset using KaggleHub
    
    Returns:
        train_x, train_y, test_x, test_y: Preprocessed data ready for SSD
    """
    
    # Auto-download if requested
    if auto_download:
        if data_path is not None:
            print("⚠️  Both auto_download=True and data_path provided. Using auto_download.")
        data_path = download_brain_tumor_dataset()
    
    # Validate data path
    if data_path is None:
        raise ValueError(
            "Please provide either:\n"
            "1. data_path to existing dataset, or\n"
            "2. set auto_download=True to download automatically"
        )
    
    loader = BrainTumorDataLoader(data_path, target_size)
    
    # Load raw images
    try:
        train_images, train_labels, test_images, test_labels = loader.load_dataset()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ Dataset loading failed: {e}")
        print(f"\n🔍 Running dataset structure diagnosis...")
        loader.diagnose_dataset_structure()
        
        print(f"\n💡 Troubleshooting Tips:")
        print(f"   1. Verify dataset path is correct")
        print(f"   2. Check folder structure matches expected format")
        print(f"   3. Ensure image files are in supported formats")
        print(f"   4. Try using auto_download=True for automatic setup")
        raise
    
    if len(train_images) == 0:
        print(f"\n❌ No images loaded! Running diagnosis...")
        loader.diagnose_dataset_structure()
        raise ValueError("No images found! Please check the dataset path and structure.")
    
    # Preprocess for SSD
    print(f"\n🔧 Preprocessing for SSD algorithm...")
    train_x = loader.preprocess_for_ssd(train_images)
    test_x = loader.preprocess_for_ssd(test_images)
    
    print(f"✅ Dataset ready for SSD:")
    print(f"   Training: {train_x.shape}")
    print(f"   Testing: {test_x.shape}")
    
    return train_x, train_labels, test_x, test_labels

if __name__ == "__main__":
    # Example usage
    dataset_path = r"C:\path\to\brain-tumor-dataset"
    
    try:
        train_x, train_y, test_x, test_y = load_brain_tumor_dataset(data_path=dataset_path)
        print(f"\n🎯 Successfully loaded brain tumor dataset!")
        print(f"   Classes: {len(np.unique(train_y))}")
        print(f"   Features: {train_x.shape[1]:,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
import random
import hashlib
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImprovedFaceDataProcessor:
    """Enhanced face emotion data processor with better augmentation and validation"""
    
    def __init__(self, data_dir, target_size=(96, 96), balance_strategy='adaptive'):
        self.data_dir = data_dir
        self.target_size = target_size  # Increased from 48x48
        self.balance_strategy = balance_strategy
        
        self.emotion_to_label = {
            "anger": 0, "contempt": 1, "disgust": 2, "fear": 3,
            "happiness": 4, "neutral": 5, "sadness": 6, "surprise": 7
        }
        self.label_to_emotion = {v: k for k, v in self.emotion_to_label.items()}
        self.EMOTION_LABELS = [self.label_to_emotion[i].title() for i in range(8)]
        
        # Enhanced augmentation pipeline
        self.train_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(5, 25), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def hash_image(self, img):
        """Create hash for duplicate detection"""
        return hashlib.sha256(img.tobytes()).hexdigest()
    
    def load_and_validate_image(self, img_path):
        """Load and validate image with quality checks"""
        try:
            # Load as grayscale first for consistency
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None
            
            # Check image quality
            if gray.shape[0] < 32 or gray.shape[1] < 32:
                return None
            
            # Check if image is too dark or too bright
            mean_intensity = np.mean(gray)
            if mean_intensity < 10 or mean_intensity > 245:
                return None
            
            # Convert to RGB
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return rgb
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("🔍 Analyzing dataset...")
        
        class_counts = {label: 0 for label in self.emotion_to_label.values()}
        quality_issues = []
        
        for split in ["train", "test"]:
            split_path = os.path.join(self.data_dir, split)
            if not os.path.exists(split_path):
                continue
                
            for emotion in os.listdir(split_path):
                if emotion not in self.emotion_to_label:
                    continue
                    
                emotion_path = os.path.join(split_path, emotion)
                if not os.path.isdir(emotion_path):
                    continue
                
                label = self.emotion_to_label[emotion]
                valid_count = 0
                
                for img_file in os.listdir(emotion_path):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    
                    img_path = os.path.join(emotion_path, img_file)
                    img = self.load_and_validate_image(img_path)
                    
                    if img is not None:
                        valid_count += 1
                    else:
                        quality_issues.append(img_path)
                
                class_counts[label] += valid_count
                print(f"📊 {split}/{emotion}: {valid_count} valid images")
        
        # Analysis summary
        total_samples = sum(class_counts.values())
        print(f"\n📈 Dataset Analysis Summary:")
        print(f"Total valid samples: {total_samples}")
        print(f"Quality issues found: {len(quality_issues)}")
        
        for label in sorted(class_counts):
            count = class_counts[label]
            percentage = (count / total_samples) * 100
            print(f"{self.label_to_emotion[label]}: {count} ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return class_counts, quality_issues
    
    def load_dataset_with_splits(self, test_size=0.2, val_size=0.2):
        """Load dataset with proper train/val/test splits and deduplication"""
        print("📂 Loading dataset with enhanced preprocessing...")
        
        X, y = [], []
        hash_set = set()
        duplicate_count = 0
        
        # Load all data first
        for split in ["train", "test"]:
            split_path = os.path.join(self.data_dir, split)
            if not os.path.exists(split_path):
                continue
                
            for emotion in os.listdir(split_path):
                if emotion not in self.emotion_to_label:
                    continue
                    
                label = self.emotion_to_label[emotion]
                emotion_path = os.path.join(split_path, emotion)
                
                if not os.path.isdir(emotion_path):
                    continue
                
                img_files = [f for f in os.listdir(emotion_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_file in tqdm(img_files, desc=f"Loading {split}/{emotion}"):
                    img_path = os.path.join(emotion_path, img_file)
                    img = self.load_and_validate_image(img_path)
                    
                    if img is None:
                        continue
                    
                    # Duplicate detection
                    img_hash = self.hash_image(img)
                    if img_hash in hash_set:
                        duplicate_count += 1
                        continue
                    
                    hash_set.add(img_hash)
                    X.append(img)
                    y.append(label)
        
        print(f"🧹 Removed {duplicate_count} duplicate images")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Create stratified splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
        )
        
        # Calculate class weights
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        class_weights = {
            label: total_samples / (len(class_counts) * count)
            for label, count in class_counts.items()
        }
        
        print(f"\n✅ Dataset splits created:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test), class_weights

def main():
    """Main preprocessing function"""
    data_dir = os.path.join("data", "fer2013plus", "fer2013")
    
    processor = ImprovedFaceDataProcessor(
        data_dir=data_dir,
        target_size=(96, 96),  # Increased resolution
        balance_strategy='adaptive'
    )
    
    # Analyze dataset
    class_counts, quality_issues = processor.analyze_dataset()
    
    # Load and split dataset
    (X_train, X_val, X_test), (y_train, y_val, y_test), class_weights = processor.load_dataset_with_splits()
    
    # Save processed data
    np.save("X_face_train_96x96.npy", X_train)
    np.save("X_face_val_96x96.npy", X_val)
    np.save("X_face_test_96x96.npy", X_test)
    np.save("y_face_train.npy", y_train)
    np.save("y_face_val.npy", y_val)
    np.save("y_face_test.npy", y_test)
    np.save("face_class_weights.npy", class_weights)
    
    print("\n💾 Saved processed dataset files:")
    print("- X_face_train_96x96.npy, X_face_val_96x96.npy, X_face_test_96x96.npy")
    print("- y_face_train.npy, y_face_val.npy, y_face_test.npy")
    print("- face_class_weights.npy")

if __name__ == "__main__":
    main()

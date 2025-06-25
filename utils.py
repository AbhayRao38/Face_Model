import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load data
X = np.load("X_face_rgb_all8classes.npy")
y = np.load("y_face_all8classes.npy")

# Emotion label mapping (index: name)
EMOTION_LABELS = {
    0: 'Anger',
    1: 'Contempt',
    2: 'Disgust',
    3: 'Fear',
    4: 'Happiness',
    5: 'Neutral',
    6: 'Sadness',
    7: 'Surprise'
}

# ✅ Step 1: Unique class counts
unique, counts = np.unique(y, return_counts=True)
print("🔍 Unique Classes and Counts:")
for idx, count in zip(unique, counts):
    label_name = EMOTION_LABELS.get(idx, "Unknown")
    print(f"  {idx}: {label_name:10} -> {count} samples")

# ✅ Step 2: Sanity check shape
print(f"\n📏 X shape: {X.shape} | y shape: {y.shape}")

# ✅ Step 3: Sample visualization per class
def plot_samples_per_class(X, y, EMOTION_LABELS, samples=5):
    plt.figure(figsize=(15, 10))
    for class_idx in range(len(EMOTION_LABELS)):
        class_name = EMOTION_LABELS[class_idx]
        class_indices = np.where(y == class_idx)[0]
        if len(class_indices) == 0:
            print(f"⚠️ No samples found for class {class_name} ({class_idx})")
            continue
        for i in range(samples):
            idx = class_indices[i]
            plt.subplot(len(EMOTION_LABELS), samples, class_idx * samples + i + 1)
            plt.imshow(X[idx])
            plt.axis('off')
            if i == 0:
                plt.ylabel(class_name, fontsize=12)
    plt.suptitle("🖼 Sample Images Per Class", fontsize=16)
    plt.tight_layout()
    plt.show()

plot_samples_per_class(X, y, EMOTION_LABELS)

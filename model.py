# face_test_generator.py
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Load full dataset
X = np.load("X_face_rgb_all8classes.npy")
y = np.load("y_face_all8classes.npy")

# Generate test split
all_indices = np.arange(len(X))
_, test_indices, _, y_test = train_test_split(
    all_indices, y, test_size=0.2, stratify=y, random_state=42
)

# Save test indices and labels
np.save("X_face_test_idx.npy", test_indices)
np.save("y_face_test.npy", y[test_indices])

print("✅ Test index and label files saved!")
print("   🔹 X_face_test_idx.npy shape:", test_indices.shape)
print("   🔹 y_face_test.npy shape     :", y[test_indices].shape)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statistics import mean, stdev

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------- MODEL ----------------------
class ImprovedFaceEmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ImprovedFaceEmotionCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)

# ---------------------- DATASET ----------------------
class AugmentedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.from_numpy(y.astype(np.int64))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        image = (self.X[idx] * 255).astype(np.uint8)
        return self.transform(image), self.y[idx]

# ---------------------- EVALUATE ----------------------
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_softmax = [], [], []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = softmax(output)

            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_softmax.extend(probs.cpu().numpy())

    return accuracy_score(all_targets, all_preds), all_preds, all_targets, all_softmax

def main(runs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("🔍 Loading data...")
    X = np.load("X_face_rgb_all8classes.npy").astype('float32')
    y = np.load("y_face_all8classes.npy").astype(np.int64)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    test_loader = DataLoader(AugmentedDataset(X_test, y_test), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    accs = []

    for i in range(runs):
        logger.info(f"🔁 Run {i+1}/{runs}")
        model = ImprovedFaceEmotionCNN(num_classes=8).to(device)
        model.load_state_dict(torch.load("best_pytorch_improved_model.pth"))
        acc, preds, targets, softmaxes = evaluate(model, test_loader, device)
        accs.append(acc)
        logger.info(f"✅ Accuracy for run {i+1}: {acc*100:.2f}%")

    mean_acc = mean(accs)
    var_acc = np.var(accs)
    std_acc = stdev(accs)

    logger.info("📊 Final Evaluation")
    logger.info(f"📌 Mean Accuracy: {mean_acc*100:.2f}%")
    logger.info(f"📉 Variance: {var_acc:.6f}")
    logger.info(f"📈 Std Deviation: {std_acc:.6f}")

    logger.info("\n🧾 Classification Report (Final Run):\n" +
                classification_report(targets, preds, target_names=[
                    'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'
                ], digits=4))

    conf_mat = confusion_matrix(targets, preds)
    labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Final Run')
    plt.tight_layout()
    plt.savefig("face_confusion_matrix.png")
    logger.info("📸 Saved confusion matrix as 'face_confusion_matrix.png'")
    plt.close()

    # ✅ Save softmax output
    np.save("face_softmax.npy", np.array(softmaxes))
    logger.info("💾 Saved softmax probabilities to 'face_softmax.npy'")

if __name__ == "__main__":
    main(runs=10)

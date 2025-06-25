import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------- MODEL DEFINITION ----------------------
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
    def __init__(self, X, y, is_training=True):
        self.X = X
        self.y = torch.from_numpy(y.astype(np.int64))
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.12, 0.12), scale=(0.88, 1.12), shear=5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        image = (self.X[idx] * 255).astype(np.uint8)
        return self.transform(image), self.y[idx]

# ---------------------- TRAINING ----------------------
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
        pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total

def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Validation')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total, all_preds, all_targets

# ---------------------- MAIN ----------------------
def main():
    CONFIG = {
        'batch_size': 128,
        'num_epochs': 250,
        'learning_rate': 0.0008,
        'weight_decay': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    logger.info("🚀 Starting PyTorch Face Emotion Training (Resume Mode)")
    device = torch.device(CONFIG['device'])

    # Load data
    X = np.load("X_face_rgb_all8classes.npy").astype('float32')
    y = np.load("y_face_all8classes.npy").astype(np.int64)
    class_weights_dict = np.load("class_weights_all8classes.npy", allow_pickle=True).item()

    # Split
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, stratify=y_train_temp, random_state=42)

    # Datasets & Loaders
    train_loader = DataLoader(AugmentedDataset(X_train, y_train, True), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(AugmentedDataset(X_val, y_val, False), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(AugmentedDataset(X_test, y_test, False), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = ImprovedFaceEmotionCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load("best_pytorch_improved_model.pth"))
    logger.info("✅ Loaded model weights from checkpoint")

    # Optimizer / Loss / Scheduler
    class_weights = torch.FloatTensor(list(class_weights_dict.values())).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-5)

    # Resume settings
    start_epoch = 140
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info(f"{'='*60}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device, epoch)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        logger.info(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"📈 Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_pytorch_improved_model.pth")
            logger.info(f"✅ New best val acc: {val_acc:.2f}%")
            if val_acc >= 85.0:
                logger.info(f"🎯 Target reached: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                logger.info("🛑 Early stopping triggered!")
                break

    logger.info("🧪 Final Evaluation...")
    model.load_state_dict(torch.load("best_pytorch_improved_model.pth"))
    _, test_acc, test_preds, test_targets = validate_epoch(model, test_loader, criterion, device, epoch)
    logger.info(f"🎯 Test Accuracy: {test_acc:.2f}% | Best Val Accuracy: {best_val_acc:.2f}%")
    EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    report = classification_report(test_targets, test_preds, target_names=EMOTION_LABELS, digits=4)
    logger.info(f"\n📋 Classification Report:\n{report}")
    logger.info("✅ Training complete!")

if __name__ == "__main__":
    main()

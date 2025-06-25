from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class ImprovedFaceEmotionCNN(nn.Module):
    """Custom deep CNN from your face_model_training.py optimized for 48×48 RGB face images"""
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    face_model = ImprovedFaceEmotionCNN(num_classes=8)
    face_model.load_state_dict(torch.load("best_pytorch_improved_model.pth", map_location=device))
    face_model.to(device)
    face_model.eval()
    logging.info(f"Face model loaded successfully on {device}")
    
except Exception as e:
    logging.error(f"Failed to load face model: {e}")
    face_model = None

def get_face_transforms():
    """Transforms matching your training pipeline for face images"""
    return transforms.Compose([
        transforms.Resize((48, 48)),  # Your model is optimized for 48x48
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': face_model is not None,
        'device': str(device),
        'model_type': 'ImprovedFaceEmotionCNN',
        'input_size': '48x48'
    })

@app.route('/predict/face', methods=['POST'])
def predict_face():
    if face_model is None:
        return jsonify({
            'success': False,
            'error': 'Face model not loaded'
        }), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
            
        image_file = request.files['file']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Load and preprocess image
        image = Image.open(image_file.stream).convert('RGB')
        
        # Apply transforms
        transform = get_face_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = face_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
        
        # Emotion labels from your training
        EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        predicted_emotion = EMOTION_LABELS[predicted_class] if predicted_class < len(EMOTION_LABELS) else 'Unknown'
        
        # Convert to MCI-relevant binary classification
        mci_relevant_emotions = ['Anger', 'Fear', 'Disgust', 'Sadness']
        mci_probability = confidence if predicted_emotion in mci_relevant_emotions else (1 - confidence)
        
        # Create binary probabilities [Non-MCI, MCI]
        binary_probs = [1 - mci_probability, mci_probability]
        
        return jsonify({
            'success': True,
            'probabilities': binary_probs,
            'confidence': confidence,
            'predicted_emotion': predicted_emotion,
            'all_probabilities': probabilities.tolist(),
            'emotion_labels': EMOTION_LABELS
        })
        
    except Exception as e:
        logging.error(f"Error in face prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
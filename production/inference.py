import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import ASTModel
import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
CONFIG = {
    "num_classes": 10,
    "sample_rate": 16000,
    "ast_frames": 1024,
}

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}
AST_MEAN = -4.2677393
AST_STD = 4.5689974

# ==========================================
# MINIMAL ARCHITECTURE (Inference Only)
# ==========================================
class MessyMashupAST_v3(pl.LightningModule):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.config = config
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config["num_classes"])
        )

    def forward(self, x):
        outputs = self.ast(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# ==========================================
# EXPOSED API FUNCTIONS
# ==========================================
def loadAST(checkpoint_path):
    """Initializes architecture and maps the saved weights safely."""
    model = MessyMashupAST_v3(CONFIG)
    
    # weights_only=False required to bypass PyTorch 2.6 security blocks
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract just the weights, ignoring optimizer states
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Clean up any leftover criterion/loss weights from training
    clean_dict = {k: v for k, v in state_dict.items() if 'criterion' not in k}
    
    model.load_state_dict(clean_dict, strict=False)
    model.eval()
    return model

def run_inference(audio_path, model, device="cpu"):
    """
    Handles file conversion, transforms to Mel-Spec, applies Skip-Intro TTA, 
    runs the forward pass, and returns the genre and probabilities.
    """
    model = model.to(device)
    
    # 1. Load and Transform
    y, sr = librosa.load(audio_path, sr=CONFIG["sample_rate"])
    mel_spec = librosa.feature.melspectrogram(y=y, sr=CONFIG["sample_rate"], n_mels=128, n_fft=400, hop_length=160)
    mel_spec_db = (librosa.power_to_db(mel_spec, ref=np.max) - AST_MEAN) / (AST_STD * 2)
    tensor = torch.tensor(mel_spec_db, dtype=torch.float32)

    total_frames = tensor.shape[1]
    frames = CONFIG["ast_frames"]
    frames_30s = 3000

    # 2. Skip-Intro Test-Time Augmentation (TTA)
    if total_frames > frames_30s + frames:
        # >40s: Skip the first 30 seconds entirely
        start_1 = frames_30s
        start_3 = total_frames - frames
        start_2 = start_1 + (start_3 - start_1) // 2
        
        crop1 = tensor[:, start_1 : start_1 + frames]
        crop2 = tensor[:, start_2 : start_2 + frames]
        crop3 = tensor[:, start_3 : start_3 + frames]
        
    elif total_frames > frames:
        # 10s-40s: Skip as much of the start as possible
        start_1 = (total_frames - frames) // 2
        start_3 = total_frames - frames
        start_2 = start_1 + (start_3 - start_1) // 2
        
        crop1 = tensor[:, start_1 : start_1 + frames]
        crop2 = tensor[:, start_2 : start_2 + frames]
        crop3 = tensor[:, start_3 : start_3 + frames]
        
    else:
        # <10s: Pad the audio
        crop = F.pad(tensor, (0, frames - total_frames))
        crop1 = crop2 = crop3 = crop

    # Stack crops: [Batch, Time, Freq]
    batch = torch.stack([crop1.transpose(0, 1), crop2.transpose(0, 1), crop3.transpose(0, 1)]).to(device)

    # 3. Forward Pass
    with torch.no_grad():
        logits = model(batch)
        # Average TTA crop probabilities
        avg_probs = torch.mean(F.softmax(logits, dim=1), dim=0)
        pred_idx = torch.argmax(avg_probs).item()

    return IDX_TO_GENRE[pred_idx], avg_probs.cpu().numpy()
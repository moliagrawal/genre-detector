# %% [code]
import os
import glob
import random
import warnings
import multiprocessing
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import MulticlassF1Score
from transformers import ASTModel, get_cosine_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm

# Update wandb before importing
os.system('uv pip install --upgrade wandb')
import wandb

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
warnings.filterwarnings('ignore')
pl.seed_everything(14, workers=True)

# We are only Training and Inferencing now!
RUN_TRAINING = True
RUN_INFERENCE = True

CONFIG = {
    "project_name": "23f2001635-t12026",
    "experiment_name": "ast_mashup_v3_pseudo_tta",
    "data_root": "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup",
    
    # 1. Your original 10k synthetic dataset
    "synthetic_data_dir": "/kaggle/input/datasets/harshitsahu01/dataset-for-ast-v1/processed_data_ast",
    
    # 2. Your newly generated Pseudo-Labels Dataset
    # ⚠️ IMPORTANT: UPDATE THIS PATH TO YOUR ACTUAL KAGGLE DATASET FOLDER! ⚠️
    "pseudo_labeled_dataset": "/kaggle/input/datasets/harshitsahu01/messymashup-pseudolabeled-audios/pseudo_labels", 
    
    # 3. Your best V1 checkpoint to extract the finely-tuned AST backbone
    "best_checkpoint_v1": "/kaggle/input/models/harshitsahu01/ast-genre-predictor/pytorch/default/1/checkpoints/best-ast-model.ckpt",
    
    # Lowered head_lr to prevent catastrophic forgetting of the backbone
    "head_lr": 1e-5,         
    "backbone_lr": 5e-6,     # Tiny LR to protect the AST backbone (Acts as a soft freeze)
    "weight_decay": 0.03,
    "batch_size": 16,
    
    # Back to 4 since we are using 1 GPU (Effective Batch Size = 64)
    "accumulate_grad": 4,    
    "num_epochs": 30,
    "patience": 5,
    "num_classes": 10,
    "sample_rate": 16000,
    "duration": 30,
    "ast_frames": 1024,
}

GENRES = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}

AST_MEAN = -4.2677393
AST_STD = 4.5689974

if RUN_TRAINING:
    from kaggle_secrets import UserSecretsClient
    try:
        user_secrets = UserSecretsClient()
        wandb.login(key=user_secrets.get_secret("WANDB_API_KEY"))
    except Exception as e:
        print(f"WandB login skipped/failed: {e}")

# ==========================================
# 1. OLD V1 MODEL (Required to extract weights)
# ==========================================
class MessyMashupAST_v1(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.5), nn.Linear(768, config["num_classes"]))
    def forward(self, x):
        outputs = self.ast(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# ==========================================
# 2. DUAL-DIRECTORY DATASET
# ==========================================
class ASTMashupDatasetV3(Dataset):
    def __init__(self, data_dirs, config, is_train=True):
        self.files = []
        for d in data_dirs:
            # Safely grab all .pt files from both datasets
            self.files.extend(glob.glob(f"{d}/*/*.pt"))
            
        self.is_train = is_train
        self.target_frames = config["ast_frames"]
        self.time_masking = T.TimeMasking(time_mask_param=100)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=24)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = GENRE_TO_IDX[Path(file_path).parent.name]
        mel_spec = torch.load(file_path).to(torch.float32) # Ensure precision is correct
        
        total_frames = mel_spec.shape[1]
        if total_frames > self.target_frames:
            if self.is_train:
                start = random.randint(0, total_frames - self.target_frames)
            else:
                start = (total_frames - self.target_frames) // 2
            mel_spec = mel_spec[:, start : start + self.target_frames]
        else:
            mel_spec = F.pad(mel_spec, (0, self.target_frames - total_frames))

        mel_spec = mel_spec.transpose(0, 1)

        # Apply SpecAugment ONLY during training
        if self.is_train:
            mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)
            mel_spec = self.time_masking(mel_spec)
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)

        return mel_spec, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. V3 ARCHITECTURE (FFN Head)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

class MessyMashupAST_v3(pl.LightningModule):
    def __init__(self, config, pretrained_v1_model=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # 1. Extract the finely-tuned AST backbone from V1
        if pretrained_v1_model:
            self.ast = pretrained_v1_model.ast
        else:
            self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            
        # 2. Attach the new robust Feed-Forward Network
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config["num_classes"])
        )

        self.criterion = FocalLoss(gamma=2.0)
        self.train_f1 = MulticlassF1Score(num_classes=config["num_classes"], average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=config["num_classes"], average='macro')

    def forward(self, x):
        outputs = self.ast(x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Mixup Augmentation
        if np.random.rand() < 0.5:
            alpha = 0.4
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(x.size()[0], device=x.device)
            target_a, target_b = y, y[rand_index]
            x = lam * x + (1 - lam) * x[rand_index]
            logits = self(x)
            loss = lam * self.criterion(logits, target_a) + (1 - lam) * self.criterion(logits, target_b)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_f1(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_macro_f1', self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.val_f1(preds, y)
        self.log('val_loss', self.criterion(logits, y), prog_bar=True)
        self.log('val_macro_f1', self.val_f1, prog_bar=True)

    def configure_optimizers(self):
        # Differential Learning Rates act as a soft freeze-thaw
        optimizer = torch.optim.AdamW([
            {'params': self.ast.parameters(), 'lr': self.config["backbone_lr"]},
            {'params': self.classifier.parameters(), 'lr': self.config["head_lr"]}
        ], weight_decay=self.config["weight_decay"])
        
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# ==========================================
# 4. INFERENCE (TEST-TIME AUGMENTATION)
# ==========================================
def run_inference_tta(model, config):
    print("--- RUNNING INFERENCE WITH TTA ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_df = pd.read_csv(f"{config['data_root']}/test.csv")
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        file_id = row['id']
        filename = row.get('filename', f"mashups/{file_id}.wav") 
        audio_path = f"{config['data_root']}/{filename}"
        
        try:
            y, _ = librosa.load(audio_path, sr=config["sample_rate"])
            mel_spec = librosa.feature.melspectrogram(y=y, sr=config["sample_rate"], n_mels=128, n_fft=400, hop_length=160)
            mel_spec_db = (librosa.power_to_db(mel_spec, ref=np.max) - AST_MEAN) / (AST_STD * 2)
            tensor = torch.tensor(mel_spec_db, dtype=torch.float32)
            
            total_frames = tensor.shape[1]
            frames = config["ast_frames"]
            
            if total_frames > frames:
                # TTA: Extract Start, Center, and End crops
                crop1 = tensor[:, 0 : frames]
                crop2 = tensor[:, (total_frames - frames) // 2 : ((total_frames - frames) // 2) + frames]
                crop3 = tensor[:, total_frames - frames : total_frames]
                
                batch = torch.stack([crop1.transpose(0, 1), crop2.transpose(0, 1), crop3.transpose(0, 1)]).to(device)
            else:
                crop = F.pad(tensor, (0, frames - total_frames))
                batch = crop.transpose(0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(batch)
                # Average the probabilities across the 3 TTA crops
                avg_probs = torch.mean(F.softmax(logits, dim=1), dim=0)
                pred_idx = torch.argmax(avg_probs).item()
                predictions.append(IDX_TO_GENRE[pred_idx])
                
        except Exception as e:
            print(f"Failed on {file_id}: {e}")
            predictions.append("pop") 

    test_df['genre'] = predictions
    test_df[['id', 'genre']].to_csv('submission.csv', index=False)
    print("Saved predictions to submission.csv")

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":

    if RUN_TRAINING:
        print("--- STARTING V3 TRAINING ---")
        
        # MIXED TRAINING SET: Synthetic Data + Pseudo Labels
        train_dirs = [f"{CONFIG['synthetic_data_dir']}/train", f"{CONFIG['pseudo_labeled_dataset']}/train"]
        
        # PURE VALIDATION SET: Strictly Ground-Truth Synthetic Data Only (NO pseudo labels!)
        val_dirs = [f"{CONFIG['synthetic_data_dir']}/val"]
        
        train_ds = ASTMashupDatasetV3(train_dirs, CONFIG, is_train=True)
        val_ds = ASTMashupDatasetV3(val_dirs, CONFIG, is_train=False)
        
        # Changed num_workers to 0 to prevent Kaggle DataLoader multiprocessing deadlocks
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
        
        print(f"Total Combined Training Samples: {len(train_ds)}")
        print(f"Total Pure Validation Samples: {len(val_ds)}")
        
        # Initialize V3 with V1's backbone
        print("Extracting weights from V1 checkpoint...")
        v1_model = MessyMashupAST_v1.load_from_checkpoint(CONFIG['best_checkpoint_v1'])
        model = MessyMashupAST_v3(CONFIG, pretrained_v1_model=v1_model)
        
        checkpoint_cb = ModelCheckpoint(
            dirpath="checkpoints_v3", filename="best-ast-v3", monitor="val_macro_f1", mode="max", save_top_k=1
        )
        early_stop_cb = EarlyStopping(monitor="val_macro_f1", patience=CONFIG["patience"], mode="max", verbose=True)
        wandb_logger = WandbLogger(project=CONFIG["project_name"], name=CONFIG["experiment_name"])
        
        trainer = pl.Trainer(
            max_epochs=CONFIG["num_epochs"],
            accelerator="gpu",
            # Single GPU for stability (removes DDP overhead and bugs)
            devices=1,
            logger=wandb_logger,
            callbacks=[checkpoint_cb, early_stop_cb],
            accumulate_grad_batches=CONFIG["accumulate_grad"],
            precision="16-mixed",
            deterministic=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
    if RUN_INFERENCE:
        checkpoint_path = "checkpoints_v3/best-ast-v3.ckpt"
        if os.path.exists(checkpoint_path):
            print(f"Loading best V3 checkpoint: {checkpoint_path}")
            model = MessyMashupAST_v3.load_from_checkpoint(checkpoint_path)
        else:
            print("No V3 checkpoint found! Initializing untrained model.")
            model = MessyMashupAST_v3(CONFIG)
            
        run_inference_tta(model, CONFIG)
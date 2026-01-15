import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.audio.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tqdm import trange
from tqdm import tqdm


import matplotlib.pyplot as plt

from tts.dataset.dataset import TTSDataset, TTSCollator, BatchSampler
from tts.dataset.config import DatasetConfig
from tts.model.tacotron import Tacotron2
from tts.model.config import Tacotron2Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log")])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_config = Tacotron2Config()
dataset_config = DatasetConfig()


# Resume training configuration
RESUME_TRAINING = True  # Set to True to resume from checkpoint
CHECKPOINT_PATH = "checkpoints/model_checkpoint.pt"
CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
EPSILON = 1e-6
START_DECAY_EPOCHS = None
NUM_WORKERS = 0


TRIAIN_DATASET_PATH = "data/train.csv"
VALIDATION_DATASET_PATH = "data/valid.csv"

# dataset loaded
train_dataset = TTSDataset(TRIAIN_DATASET_PATH)
validation_dataset = TTSDataset(VALIDATION_DATASET_PATH)
collator = TTSCollator()
train_sampler = BatchSampler(train_dataset, batch_size=BATCH_SIZE, drop_last=True)

train_loader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=collator, num_workers=NUM_WORKERS)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=NUM_WORKERS)

# model
model = Tacotron2(model_config)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=EPSILON)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters: {total_trainable_params}")

# scheduler
if START_DECAY_EPOCHS is not None:
    logger.info("using scheduler")
    delay_epochs = NUM_EPOCHS - START_DECAY_EPOCHS
    decay_gamma = (MIN_LEARNING_RATE / LEARNING_RATE) ** (1 / delay_epochs)

    def lr_lambda(current_step):
        if current_step < START_DECAY_EPOCHS:
            return 1
        else:
            return decay_gamma ** (current_step - START_DECAY_EPOCHS)

if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    logging.info("Loading from checkpoint: %s", CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_validation_loss = checkpoint['best_validation_loss']

    # Load history
    train_losses = checkpoint['train_losses']
    validation_losses = checkpoint['validation_losses']
    logging.info(f"Resumed training from epoch {start_epoch} with best validation loss: {best_validation_loss:.4f}")
    if use_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch-1)
else:
    logging.info("Starting training from scratch")
    start_epoch = 0
    best_validation_loss = float('inf')
    train_losses = []
    validation_losses = []
    if use_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda)

epoch_range = trange(start_epoch, NUM_EPOCHS, leave=False, desc="Epoch")
for epoch in epoch_range:
    
    model.train()
    losses = []
    loop = tqdm(enumerate(train_loader), leave=False, desc="Batch")
    for batch_idx, (text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask) in loop:
        text_padded = text_padded.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        mel_padded = mel_padded.to(DEVICE)
        gate_padded = gate_padded.to(DEVICE)
        text_mask = text_mask.to(DEVICE)
        mel_mask = mel_mask.to(DEVICE)

        mels_out, mels_out_postnet, stop_outs, _ = model(text_padded, input_lengths, mel_padded, text_mask, mel_mask) 

        mel_loss = F.mse_loss(mels_out, mel_padded)
        refined_mel_loss = F.mse_loss(mels_out_postnet, mel_padded)
        stop_loss = F.binary_cross_entropy_with_logits(stop_outs.reshape(-1, 1), gate_padded.reshape(-1, 1))

        loss = mel_loss + refined_mel_loss + stop_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss = torch.mean(loss).item()
        loop.set_postfix(loss=loss)
        losses.append(loss)
    
    train_loss = sum(losses) / len(losses)
    train_losses.append(train_loss)
    
    model.eval()
    losses = []
    loop = tqdm(enumerate(validation_loader), leave=False, desc="Batch")
    for batch_idx, (text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask) in loop:
        text_padded = text_padded.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        mel_padded = mel_padded.to(DEVICE)
        gate_padded = gate_padded.to(DEVICE)
        text_mask = text_mask.to(DEVICE)
        mel_mask = mel_mask.to(DEVICE)

        with torch.inference_mode():
            mels_out, mels_out_postnet, stop_outs, _ = model(text_padded, input_lengths, mel_padded, text_mask, mel_mask) 

        mel_loss = F.mse_loss(mels_out, mel_padded)
        refined_mel_loss = F.mse_loss(mels_out_postnet, mel_padded)
        stop_loss = F.binary_cross_entropy_with_logits(stop_outs.reshape(-1, 1), gate_padded.reshape(-1, 1))

        loss = mel_loss + refined_mel_loss + stop_loss

        loss = torch.mean(loss).item()
        loop.set_postfix(loss=loss)
        losses.append(loss)
    
    validation_loss = sum(losses) / len(losses)
    validation_losses.append(validation_loss)
    
    if use_scheduler:
        scheduler.step()
        logging.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_validation_loss': best_validation_loss,
            'train_losses': train_losses,
            'validation_losses': validation_losses
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        logging.info(f"Validation loss improved from {best_validation_loss:.4f} to {validation_loss:.4f}. Saved checkpoint to {CHECKPOINT_PATH}")
    
        

    
    
    
    







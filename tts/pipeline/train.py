import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tqdm import trange
from tqdm import tqdm

from scipy.io.wavfile import write

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tts.dataset.dataset import TTSDataset, TTSCollator, BatchSampler
from tts.dataset.utils import denormalize
from tts.dataset.config import TTSDatasetConfig
from tts.model.tacotron import Tacotron2
from tts.model.config import Tacotron2Config
from tts.component.tokenizer import Tokenizer
from tts.dataset.conversion import AudioMelConversions


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log")])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_config = Tacotron2Config()
dataset_config = TTSDatasetConfig()


# Resume training configuration
RESUME_TRAINING = False  # Set to True to resume from checkpoint
CHECKPOINT_PATH = "checkpoints/model_checkpoint_v3.pt"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
EPSILON = 1e-6
START_DECAY_EPOCHS = 5
NUM_WORKERS = 10
PREFETCH_FACTOR = 24
SAVE_AUDIO_GEN = "generated_audio"
INFER_EVERY = 1
SAMPLING_RATE = 22050

os.makedirs(SAVE_AUDIO_GEN, exist_ok=True)

TRAIN_DATASET_PATH = "data/train.csv"
VALIDATION_DATASET_PATH = "data/test.csv"

# dataset loaded
train_dataset = TTSDataset(TRAIN_DATASET_PATH)
validation_dataset = TTSDataset(VALIDATION_DATASET_PATH)
collator = TTSCollator()
train_sampler = BatchSampler(train_dataset, batch_size=BATCH_SIZE, drop_last=True)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collator,
                          num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, collate_fn=collator,
                               num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, pin_memory=True)

# model
model = Tacotron2(model_config)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=EPSILON)
a2m = AudioMelConversions()
tokenizer = Tokenizer()

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters: {total_trainable_params}")


def save_training_inference(true_mel, pred_mel, attention_weights, text, file_path, sample_num):
    '''
    Save true mel and generated audio for training inference
    intputs:
        true_mel (torch.Tensor): true mel spectrogram
        pred_mel (torch.Tensor): predicted mel spectrogram
        attention_weights (torch.Tensor): attention weights generated
        text (torch.tensor): tokenized text
        file_path (str): file path to save the inference
        sample_num (int): sample number
    '''
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    
    # True Mel
    im0 = axes[0].imshow(true_mel, aspect='auto', origin='lower', interpolation='none')
    axes[0].set_title("True Mel")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    # Predicted Mel
    im1 = axes[1].imshow(pred_mel, aspect='auto', origin='lower', interpolation='none')
    axes[1].set_title("Predicted Mel")
    axes[1].set_ylabel("Mel bins")
    fig.colorbar(im1, ax=axes[1])

    # Attention
    im2 = axes[2].imshow(attention_weights, aspect='auto', origin='lower', interpolation='none')
    axes[2].set_title("Alignment")
    axes[2].set_ylabel("Character Index")
    axes[2].set_xlabel("Decoder Mel Timesteps")
    fig.colorbar(im2, ax=axes[2])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f"result_{sample_num}.png"))
    plt.close()

    # save text
    with open(os.path.join(file_path, f"text_{sample_num}.txt"), "w") as f:
        f.write(tokenizer.decode(text))
    
    audio = a2m.mel2audio(pred_mel, do_denorm=True).cpu()
    audio_path = os.path.join(file_path, f"audio_{sample_num}.wav")
    write(audio_path, SAMPLING_RATE, audio.numpy())
    

# scheduler
use_scheduler = False
if START_DECAY_EPOCHS is not None:
    logging.info("using scheduler")
    use_scheduler = True
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
    
    # granular losses
    train_mel_losses = checkpoint.get('train_mel_losses', [])
    train_refined_mel_losses = checkpoint.get('train_refined_mel_losses', [])
    train_stop_losses = checkpoint.get('train_stop_losses', [])

    val_mel_losses = checkpoint.get('val_mel_losses', [])
    val_refined_mel_losses = checkpoint.get('val_refined_mel_losses', [])
    val_stop_losses = checkpoint.get('val_stop_losses', [])

    logging.info(f"Resumed training from epoch {start_epoch} with best validation loss: {best_validation_loss:.4f}")
    if use_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch-1)
else:
    logging.info("Starting training from scratch")
    start_epoch = 0
    best_validation_loss = float('inf')
    train_losses = []
    validation_losses = []
    
    # granular losses
    train_mel_losses = []
    train_refined_mel_losses = []
    train_stop_losses = []

    val_mel_losses = []
    val_refined_mel_losses = []
    val_stop_losses = []

    if use_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda)

try:
    epoch_range = trange(start_epoch, NUM_EPOCHS, leave=False, desc="Epoch")
    for epoch in epoch_range:
    
        model.train()
        losses = []
        batch_mel_losses = []
        batch_refined_mel_losses = []
        batch_stop_losses = []
    
        loop = tqdm(enumerate(train_loader), leave=False, desc="Training Batch", total=len(train_loader))
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

            loss_item = loss.item()
            loop.set_postfix(loss=loss_item)
            losses.append(loss_item)
        
        batch_mel_losses.append(mel_loss.item())
        batch_refined_mel_losses.append(refined_mel_loss.item())
        batch_stop_losses.append(stop_loss.item())
    
        train_loss = sum(losses) / len(losses)
        train_losses.append(train_loss)
    
        train_mel_losses.append(sum(batch_mel_losses) / len(batch_mel_losses))
        train_refined_mel_losses.append(sum(batch_refined_mel_losses) / len(batch_refined_mel_losses))
        train_stop_losses.append(sum(batch_stop_losses) / len(batch_stop_losses))
    
        model.eval()
        losses = []
        batch_mel_losses = []
        batch_refined_mel_losses = []
        batch_stop_losses = []
    
        loop = tqdm(enumerate(validation_loader), leave=False, desc="Validation Batch", total=len(validation_loader))
        for batch_idx, (text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask) in loop:
            text_padded = text_padded.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            mel_padded = mel_padded.to(DEVICE)
            gate_padded = gate_padded.to(DEVICE)
            text_mask = text_mask.to(DEVICE)
            mel_mask = mel_mask.to(DEVICE)

            with torch.inference_mode():
                mels_out, mels_out_postnet, stop_outs, attention_weights = model(text_padded, input_lengths, mel_padded, text_mask, mel_mask) 

            mel_loss = F.mse_loss(mels_out, mel_padded)
            refined_mel_loss = F.mse_loss(mels_out_postnet, mel_padded)
            stop_loss = F.binary_cross_entropy_with_logits(stop_outs.reshape(-1, 1), gate_padded.reshape(-1, 1))

            loss = mel_loss + refined_mel_loss + stop_loss

            loss_item = loss.item()
            loop.set_postfix(loss=loss_item)
            losses.append(loss_item)

            batch_mel_losses.append(mel_loss.item())
            batch_refined_mel_losses.append(refined_mel_loss.item())
            batch_stop_losses.append(stop_loss.item())
            if epoch % INFER_EVERY == 0:
                # inference on the first 10 elements of this batch
                true_mels, pred_mels, attentions, texts = [], [], [], []
                for i in range(10): 
                    pred_mel, attention = model.inference(text_padded[i]) # should be on device during inference
                    true_mel = denormalize(mel_padded[i].T.to("cpu"))
                    pred_mel = denormalize(pred_mel.T.to("cpu"))
                    attention = attention.T.cpu().numpy()
                    true_mels.append(true_mel)
                    pred_mels.append(pred_mel)
                    attentions.append(attention)
                    texts.append(text_padded[i])
                file_dir = f"{SAVE_AUDIO_GEN}/epoch_{epoch}"
                os.makedirs(file_dir, exist_ok=True)
                save_audio_generation(true_mels, pred_mels, attentions, texts, file_dir)
                model_save_file = f"{file_dir}/model.pt"
                torch.save(model.state_dict(), model_save_file)
    
        validation_loss = sum(losses) / len(losses)
        validation_losses.append(validation_loss)

        val_mel_losses.append(sum(batch_mel_losses) / len(batch_mel_losses))
        val_refined_mel_losses.append(sum(batch_refined_mel_losses) / len(batch_refined_mel_losses))
        val_stop_losses.append(sum(batch_stop_losses) / len(batch_stop_losses))
    
        if use_scheduler:
            scheduler.step()
            logging.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
        if validation_loss < best_validation_loss:
            prev_best = best_validation_loss
            best_validation_loss = validation_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_validation_loss': best_validation_loss,
            'train_losses': train_losses,
            'validation_losses': validation_losses,
            'train_mel_losses': train_mel_losses,
            'train_refined_mel_losses': train_refined_mel_losses,
            'train_stop_losses': train_stop_losses,
            'val_mel_losses': val_mel_losses,
            'val_refined_mel_losses': val_refined_mel_losses,
            'val_stop_losses': val_stop_losses
        }
            torch.save(checkpoint, CHECKPOINT_PATH)
            logging.info(f"Validation loss improved from {prev_best:.4f} to {validation_loss:.4f}. During Epoch {epoch} Saved checkpoint to {CHECKPOINT_PATH}")

except KeyboardInterrupt:
    logging.info("Training interrupted by user. Generating plots...")

finally:
    def plot_losses():
        if len(train_losses) == 0:
            logging.info("No training data to plot.")
            return

        epochs = list(range(1, len(train_losses) + 1))
        
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("Total Loss", "Mel Loss", "Refined Mel Loss", "Stop Loss"))

        # Total Loss
        fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train Total'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=validation_losses, mode='lines', name='Val Total'), row=1, col=1)

        # Mel Loss
        fig.add_trace(go.Scatter(x=epochs, y=train_mel_losses, mode='lines', name='Train Mel'), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_mel_losses, mode='lines', name='Val Mel'), row=1, col=2)

        # Refined Mel Loss
        fig.add_trace(go.Scatter(x=epochs, y=train_refined_mel_losses, mode='lines', name='Train Refined'), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_refined_mel_losses, mode='lines', name='Val Refined'), row=2, col=1)
        
        # Stop Loss
        fig.add_trace(go.Scatter(x=epochs, y=train_stop_losses, mode='lines', name='Train Stop'), row=2, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_stop_losses, mode='lines', name='Val Stop'), row=2, col=2)

        fig.update_layout(title_text="Training Metrics", height=800)
        fig.write_html("training_losses.html")
        logging.info("Saved training plots to training_losses.html")

        # --- Matplotlib Plotting ---
        try:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Metrics')

            # Total Loss
            axs[0, 0].plot(epochs, train_losses, label='Train Total')
            axs[0, 0].plot(epochs, validation_losses, label='Val Total')
            axs[0, 0].set_title('Total Loss')
            axs[0, 0].legend()

            # Mel Loss
            axs[0, 1].plot(epochs, train_mel_losses, label='Train Mel')
            axs[0, 1].plot(epochs, val_mel_losses, label='Val Mel')
            axs[0, 1].set_title('Mel Loss')
            axs[0, 1].legend()
            
            # Refined Mel Loss
            axs[1, 0].plot(epochs, train_refined_mel_losses, label='Train Refined')
            axs[1, 0].plot(epochs, val_refined_mel_losses, label='Val Refined')
            axs[1, 0].set_title('Refined Mel Loss')
            axs[1, 0].legend()
            
            # Stop Loss
            axs[1, 1].plot(epochs, train_stop_losses, label='Train Stop')
            axs[1, 1].plot(epochs, val_stop_losses, label='Val Stop')
            axs[1, 1].set_title('Stop Loss')
            axs[1, 1].legend()

            plt.tight_layout()
            plt.savefig("training_losses.png")
            plt.close()
            logging.info("Saved training plots to training_losses.png")
        except Exception as e:
            logging.error(f"Failed to generate matplotlib plot: {e}")

    plot_losses()
    
        

    
    
    
    







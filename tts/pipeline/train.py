import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed
from accelerate import Accelerator
import matplotlib.pyplot as plt

from tts.model.config import Tacotron2Config
from tts.model.model import Tacotron2
from tts.dataset.dataset import TTSDataset,TTSCollator, BatchSampler
from tts.dataset.utils import denormalize
from tts.tokenizer import Tokenizer

NUM_EPOCHS = 50
CONSOLE_OUT_ITERS = 5
CHECKPOINT_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6
ADAM_EPS = 1e-6
MIN_LEARNING_RATE = 1e-5
START_DECAY_EPOCHS = None
save_audio_gen = "audio_gen"
TRAIN_MANIFEST = "train.csv"
VAL_MANIFEST = "test.csv"
RESUME_FROM_CHECKPOINT = False
SEED = 42
EXPERIMENT_NAME = "trailv1"
RUN_NAME = "run1"
WORKING_DIRECTORY = "experiments/"

### Set Seed ###
set_seed(SEED)

### Init Accelerator ###
path_to_experiment = os.path.join(WORKING_DIRECTORY, EXPERIMENT_NAME)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with=None)

### Create Paths for Gen Saves ###
if accelerator.is_main_process:
    os.makedirs(save_audio_gen, exist_ok=True)

### Load Tokenizer ###
tokenizer = Tokenizer()

### Load Model ###
config = Tacotron2Config()

model = Tacotron2(config) 
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.print(f"Total Trainable Parameters: {total_trainable_params}")

### Load Optimizer ###
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LEARNING_RATE, 
                             weight_decay=WEIGHT_DECAY, 
                             eps=ADAM_EPS)

### Load Dataset ###
trainset = TTSDataset(TRAIN_MANIFEST)

testset = TTSDataset(VAL_MANIFEST)

collator = TTSCollator()
train_sampler = BatchSampler(trainset, 
                             batch_size=BATCH_SIZE, 
                             drop_last=accelerator.num_processes > 1)

trainloader = DataLoader(trainset, 
                         batch_sampler=train_sampler, 
                         num_workers=0,
                         collate_fn=collator)

testloader = DataLoader(testset, 
                        batch_size=BATCH_SIZE, 
                        num_workers=0,
                        collate_fn=collator)

### Prepare Everything ###
model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
)

### Create Scheduler ###
using_scheduler = False
if START_DECAY_EPOCHS is not None:
    accelerator.print("Using LR Scheduler!!")
    using_scheduler = True
    init_lr = LEARNING_RATE
    min_lr = MIN_LEARNING_RATE
    decay_epochs = NUM_EPOCHS - START_DECAY_EPOCHS
    decay_gamma = (min_lr / init_lr) ** (1 / decay_epochs)

    def lr_lambda(epoch):
        if epoch < START_DECAY_EPOCHS:
            return 1.0
        else:
            return decay_gamma ** (epoch - START_DECAY_EPOCHS)

### Load Checkpoint ###
if RESUME_FROM_CHECKPOINT:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, RESUME_FROM_CHECKPOINT)

    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_epochs = int(RESUME_FROM_CHECKPOINT.split("_")[-1]) + 1
    completed_steps = completed_epochs * len(trainloader)
    accelerator.print(f"Resuming from Epoch: {completed_epochs}")

    if using_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=completed_epochs-1)

else:
    completed_epochs = 0
    completed_steps = 0

    if using_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

### Train Model ###
for epoch in range(completed_epochs, NUM_EPOCHS):
    
    accelerator.print(f"Epoch: {epoch}")

    model.train()
    for texts, text_lens, mels, stops, encoder_mask, decoder_mask in trainloader:
      
        texts = texts.to(accelerator.device)
        mels = mels.to(accelerator.device)
        stops = stops.to(accelerator.device)
        encoder_mask = encoder_mask.to(accelerator.device)
        decoder_mask = decoder_mask.to(accelerator.device)
        
        ### Generate Mel Spectrogram from Text ###
        mels_out, mels_postnet_out, stop_preds, _ = model(
            texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
        )

        ### Compute Loss ###
        mel_loss = F.mse_loss(mels_out, mels)
        refined_mel_loss = F.mse_loss(mels_postnet_out, mels)
        stop_loss = F.binary_cross_entropy_with_logits(stop_preds.reshape(-1,1), stops.reshape(-1,1))

        loss = mel_loss + refined_mel_loss + stop_loss

        ### Update Model ###
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        ### Grab Metrics from all GPUs for Logging ###
        loss = torch.mean(accelerator.gather_for_metrics(loss)).item()
        mel_loss = torch.mean(accelerator.gather_for_metrics(mel_loss)).item()
        refined_mel_loss = torch.mean(accelerator.gather_for_metrics(refined_mel_loss)).item()
        stop_loss = torch.mean(accelerator.gather_for_metrics(stop_loss)).item()

        if completed_steps % CONSOLE_OUT_ITERS == 0:
            accelerator.print("Completed Steps {}/{} | Loss {:.4f} | Mel Loss {:.4f} | RMel Loss {:.4f} | Stop Loss {:.4f}".format(
                completed_steps, 
                NUM_EPOCHS * len(trainloader), 
                loss, 
                mel_loss, 
                refined_mel_loss, 
                stop_loss
            ))
     
        completed_steps +=1 

    accelerator.wait_for_everyone()

    ### Evaluate Model ###
    model.eval()
    accelerator.print("--VALIDATION--")
    val_mel_loss, val_rmel_loss, val_stop_loss, num_losses = 0, 0, 0, 0
    save_first = True
    for texts, text_lens, mels, stops, encoder_mask, decoder_mask in testloader:
        
        texts = texts.to(accelerator.device)
        mels = mels.to(accelerator.device)
        stops = stops.to(accelerator.device)
        encoder_mask = encoder_mask.to(accelerator.device)
        decoder_mask = decoder_mask.to(accelerator.device)

        ### Generate Mel Spectrogram from Text ###
        with torch.no_grad():
            mels_out, mels_postnet_out, stop_preds, attention_weights = model(
                texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
            )

        ### Compute Loss ###
        mel_loss = F.mse_loss(mels_out, mels)
        refined_mel_loss = F.mse_loss(mels_postnet_out, mels)
        stop_loss = F.binary_cross_entropy_with_logits(stop_preds.reshape(-1,1), stops.reshape(-1,1))

        val_mel_loss += mel_loss
        val_rmel_loss += refined_mel_loss
        val_stop_loss += stop_loss
        num_losses += 1

        if accelerator.is_main_process:
            if save_first:

                # Extract tensors
                true_mel = denormalize(mels[0].T.to("cpu"))
                pred_mel = denormalize(mels_postnet_out[0].T.to("cpu"))
                attention = attention_weights[0].T.to("cpu")

                # Make subplots (3 rows, 1 column)
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
                im2 = axes[2].imshow(attention, aspect='auto', origin='lower', interpolation='none')
                axes[2].set_title("Alignment")
                axes[2].set_ylabel("Character Index")
                axes[2].set_xlabel("Decoder Mel Timesteps")
                fig.colorbar(im2, ax=axes[2])

                # Adjust layout
                plt.tight_layout()

                # Save combined figure
                plt.savefig(os.path.join(save_audio_gen, f"epoch_{epoch}_result.png"))

                plt.close()
        
        save_first = False
    
    val_mel_loss = torch.mean(accelerator.gather_for_metrics(val_mel_loss)).item() / num_losses
    val_rmel_loss = torch.mean(accelerator.gather_for_metrics(val_rmel_loss)).item() / num_losses
    val_stop_loss = torch.mean(accelerator.gather_for_metrics(val_stop_loss)).item() / num_losses
    val_loss = val_mel_loss + val_rmel_loss + val_stop_loss
    
    accelerator.print("Loss {:.4f} | Mel Loss {:.4f} | RMel Loss {:.4f} | Stop Loss {:.4f}".format(
                val_loss, 
                val_mel_loss, 
                val_rmel_loss, 
                val_stop_loss
            ))
    
    
    if completed_epochs % CHECKPOINT_EPOCHS == 0:
        accelerator.print("Saving Checkpoint!")
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_epochs}")
        accelerator.save_state(output_dir=path_to_checkpoint, safe_serialization=False)
    
    completed_epochs += 1

    if using_scheduler:
        scheduler.step(epoch=completed_epochs)
        accelerator.print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

accelerator.save_state(os.path.join(path_to_experiment, "final_checkpoint"), safe_serialization=False)

accelerator.end_training()
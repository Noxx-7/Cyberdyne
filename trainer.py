import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import time
import os
from datetime import datetime

from model import AdvancedLLM
from dataset_loader import MultiDatasetLoader

class LLMTrainer:
    def __init__(self, model_name='cyberdyne-llm', vocab_size=None, emb_size=768,
                 n_layers=12, n_heads=12, ff_size=3072, max_len=512,
                 learning_rate=5e-5, device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if vocab_size is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            vocab_size = len(tokenizer)

        self.model = AdvancedLLM(
            vocab_size=vocab_size,
            emb_size=emb_size,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_size=ff_size,
            max_len=max_len
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        print(f"Model initialized with {self.model.get_num_params():,} parameters")
        print(f"Using device: {self.device}")

    def train(self, dataset_key, num_epochs=3, batch_size=4, max_samples=10000,
              max_len=512, save_dir='models', log_interval=10):
        print(f"\nStarting training on dataset: {dataset_key}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Max samples: {max_samples}")

        os.makedirs(save_dir, exist_ok=True)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        try:
            dataset = MultiDatasetLoader.create_dataset(
                dataset_key,
                tokenizer,
                max_samples=max_samples,
                max_len=max_len,
                streaming=True
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        training_start = time.time()
        training_history = {
            'model_name': self.model_name,
            'dataset': dataset_key,
            'epochs': num_epochs,
            'batch_size': batch_size,
            'losses': [],
            'epoch_losses': []
        }

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                if batch_idx % log_interval == 0:
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    training_history['losses'].append({
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'loss': loss.item()
                    })

            avg_epoch_loss = epoch_loss / batch_count
            training_history['epoch_losses'].append(avg_epoch_loss)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

            checkpoint_path = os.path.join(save_dir, f"{self.model_name}_epoch_{epoch+1}.pt")
            self.model.save_model(checkpoint_path)

        training_duration = time.time() - training_start
        training_history['duration'] = training_duration

        final_path = os.path.join(save_dir, f"{self.model_name}_final.pt")
        self.model.save_model(final_path)

        print(f"\nTraining completed in {training_duration/60:.2f} minutes")
        print(f"Final model saved to: {final_path}")

        return training_history

    def load_checkpoint(self, checkpoint_path):
        self.model = AdvancedLLM.load_model(checkpoint_path, self.device)
        print(f"Checkpoint loaded from: {checkpoint_path}")

    def save_model_with_metadata(self, save_path, tokenizer_name='gpt2', metadata=None):
        config = {
            'model_name': self.model_name,
            'vocab_size': self.model.vocab_size,
            'emb_size': self.model.emb_size,
            'n_layers': self.model.n_layers,
            'n_heads': self.model.n_heads,
            'max_len': self.model.max_len,
            'tokenizer_name': tokenizer_name,
            'state_dict': self.model.state_dict(),
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        torch.save(config, save_path)
        print(f"Model with metadata saved to: {save_path}")

    @staticmethod
    def load_model_with_metadata(load_path, device='cpu'):
        config = torch.load(load_path, map_location=device)
        model = AdvancedLLM(
            vocab_size=config['vocab_size'],
            emb_size=config['emb_size'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_len=config['max_len']
        )
        model.load_state_dict(config['state_dict'])
        model.to(device)
        print(f"Model loaded from: {load_path}")
        return model, config

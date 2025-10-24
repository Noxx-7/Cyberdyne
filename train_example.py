#!/usr/bin/env python3
"""
Example script for training a custom LLM model
"""

from trainer import LLMTrainer
from dataset_loader import MultiDatasetLoader

def main():
    print("=" * 50)
    print("Cyberdyne LLM Training Example")
    print("=" * 50)

    print("\nAvailable datasets:")
    datasets = MultiDatasetLoader.list_available_datasets()
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")

    print("\n" + "=" * 50)
    print("Starting training on 'wikitext' dataset...")
    print("=" * 50 + "\n")

    trainer = LLMTrainer(
        model_name='my-first-llm',
        emb_size=512,
        n_layers=8,
        n_heads=8,
        learning_rate=5e-5
    )

    history = trainer.train(
        dataset_key='wikitext',
        num_epochs=3,
        batch_size=4,
        max_samples=5000,
        save_dir='models'
    )

    if history:
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Model: {history['model_name']}")
        print(f"Dataset: {history['dataset']}")
        print(f"Total epochs: {history['epochs']}")
        print(f"Training time: {history['duration']/60:.2f} minutes")
        print(f"Final loss: {history['epoch_losses'][-1]:.4f}")
        print("\nModel saved to: models/my-first-llm_final.pt")
        print("\nYou can now use this model for inference!")

if __name__ == "__main__":
    main()

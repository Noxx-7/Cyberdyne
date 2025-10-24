#!/usr/bin/env python3
"""
Quick start script - trains a small model and launches the chat interface
"""

import os
from trainer import LLMTrainer
from inference import OfflineInference

def quick_train():
    print("\n" + "=" * 60)
    print("QUICK START: Training a small model for demonstration")
    print("=" * 60 + "\n")

    print("This will train a small model on wikitext dataset...")
    print("Configuration: 6 layers, 512 embedding, 5000 samples, 2 epochs\n")

    trainer = LLMTrainer(
        model_name='cyberdyne-quickstart',
        emb_size=512,
        n_layers=6,
        n_heads=8,
        learning_rate=5e-5
    )

    history = trainer.train(
        dataset_key='wikitext',
        num_epochs=2,
        batch_size=4,
        max_samples=5000,
        save_dir='models'
    )

    return history is not None

def quick_test():
    print("\n" + "=" * 60)
    print("Testing the trained model...")
    print("=" * 60 + "\n")

    model_path = 'models/cyberdyne-quickstart_final.pt'

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False

    try:
        inference = OfflineInference(model_path)
        session_id = inference.create_session()

        test_prompts = [
            "What is artificial intelligence?",
            "Tell me about machine learning.",
            "How does a neural network work?"
        ]

        for prompt in test_prompts:
            print(f"\nUser: {prompt}")
            response, _ = inference.chat(
                prompt,
                session_id=session_id,
                temperature=0.8,
                max_new_tokens=100
            )
            print(f"Assistant: {response}")

        print("\n" + "=" * 60)
        print("Model is working! You can now use it in the Gradio interface.")
        print("=" * 60 + "\n")

        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("🤖 CYBERDYNE LLM - QUICK START")
    print("=" * 60)

    if not os.path.exists('models'):
        os.makedirs('models')

    model_path = 'models/cyberdyne-quickstart_final.pt'

    if os.path.exists(model_path):
        print(f"\nFound existing model at {model_path}")
        response = input("Do you want to use it or train a new one? (use/train): ").lower()

        if response == 'train':
            if quick_train():
                quick_test()
        elif response == 'use':
            quick_test()
        else:
            print("Invalid choice. Exiting.")
            return
    else:
        print("\nNo existing model found. Starting training...")
        if quick_train():
            quick_test()

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Run 'python app.py' to launch the full Gradio interface")
    print("2. Or use 'python inference_example.py' for command-line chat")
    print("3. Train larger models with 'python train_example.py'")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

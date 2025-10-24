#!/usr/bin/env python3
"""
Example script for offline inference with a trained model
"""

from inference import OfflineInference
import sys

def main():
    model_path = 'models/my-first-llm_final.pt'

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print("=" * 50)
    print("Cyberdyne LLM Offline Inference Example")
    print("=" * 50)
    print(f"\nLoading model from: {model_path}\n")

    try:
        inference = OfflineInference(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease train a model first using train_example.py")
        return

    session_id = inference.create_session()

    print("\n" + "=" * 50)
    print("Chat with your model! (type 'exit' to quit)")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            response, _ = inference.chat(
                user_input,
                session_id=session_id,
                temperature=0.8,
                max_new_tokens=150,
                top_k=50,
                top_p=0.95
            )

            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()

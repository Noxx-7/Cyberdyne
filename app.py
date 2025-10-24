import gradio as gr
import os
import json
from datetime import datetime
from inference import OfflineInference, ConversationalInference
from trainer import LLMTrainer
from dataset_loader import MultiDatasetLoader
from transformers import GPT2Tokenizer

class CyberdyneLLMApp:
    def __init__(self):
        self.inference_engine = None
        self.session_id = None
        self.available_models = []
        self.scan_models()

    def scan_models(self):
        models_dir = 'models'
        if os.path.exists(models_dir):
            self.available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        else:
            self.available_models = []

    def load_model(self, model_path):
        if not model_path or not os.path.exists(model_path):
            return "❌ Model path not found. Please train a model first."

        try:
            self.inference_engine = OfflineInference(model_path)
            self.session_id = self.inference_engine.create_session()
            return f"✅ Model loaded successfully from {model_path}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def train_model(self, dataset_name, model_name, num_epochs, batch_size,
                   max_samples, learning_rate, emb_size, n_layers, n_heads):
        try:
            if not dataset_name:
                return "❌ Please select a dataset", None

            trainer = LLMTrainer(
                model_name=model_name,
                emb_size=int(emb_size),
                n_layers=int(n_layers),
                n_heads=int(n_heads),
                learning_rate=float(learning_rate)
            )

            history = trainer.train(
                dataset_key=dataset_name,
                num_epochs=int(num_epochs),
                batch_size=int(batch_size),
                max_samples=int(max_samples),
                save_dir='models'
            )

            if history:
                model_path = f"models/{model_name}_final.pt"
                summary = f"""
✅ Training completed successfully!

📊 Training Summary:
- Model: {model_name}
- Dataset: {dataset_name}
- Epochs: {num_epochs}
- Final Loss: {history['epoch_losses'][-1]:.4f}
- Duration: {history['duration']/60:.2f} minutes
- Model saved to: {model_path}

You can now load this model for inference!
"""
                self.scan_models()
                return summary, model_path
            else:
                return "❌ Training failed. Check the logs.", None

        except Exception as e:
            return f"❌ Training error: {str(e)}", None

    def chat(self, message, history, temperature, max_tokens, top_k, top_p):
        if not self.inference_engine:
            return history + [[message, "⚠️ Please load a model first using the 'Model Management' tab."]]

        if not message.strip():
            return history

        try:
            response, _ = self.inference_engine.chat(
                message,
                session_id=self.session_id,
                temperature=temperature,
                max_new_tokens=int(max_tokens),
                top_k=int(top_k),
                top_p=top_p,
                use_context=True
            )

            history.append([message, response])
            return history

        except Exception as e:
            history.append([message, f"❌ Error: {str(e)}"])
            return history

    def clear_chat(self):
        if self.inference_engine and self.session_id:
            self.inference_engine.clear_session(self.session_id)
        return []

    def get_model_list(self):
        self.scan_models()
        return gr.Dropdown(choices=self.available_models)

def create_interface():
    app = CyberdyneLLMApp()

    with gr.Blocks(title="Cyberdyne LLM", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Cyberdyne LLM
        ### Advanced Language Model Training & Inference System
        Train your own ChatGPT-like model or chat with pre-trained models offline!
        """)

        with gr.Tabs():
            with gr.Tab("💬 Chat Interface"):
                gr.Markdown("### Chat with your trained model")

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=500,
                            label="Conversation",
                            show_label=True
                        )

                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Type your message here...",
                                label="Message",
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        with gr.Row():
                            clear_btn = gr.Button("Clear Chat")

                    with gr.Column(scale=1):
                        gr.Markdown("### Generation Settings")

                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="Temperature"
                        )

                        max_tokens = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=150,
                            step=10,
                            label="Max Tokens"
                        )

                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top K"
                        )

                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            label="Top P"
                        )

                msg.submit(
                    app.chat,
                    inputs=[msg, chatbot, temperature, max_tokens, top_k, top_p],
                    outputs=[chatbot]
                ).then(lambda: "", None, msg)

                send_btn.click(
                    app.chat,
                    inputs=[msg, chatbot, temperature, max_tokens, top_k, top_p],
                    outputs=[chatbot]
                ).then(lambda: "", None, msg)

                clear_btn.click(app.clear_chat, outputs=[chatbot])

            with gr.Tab("🎓 Training"):
                gr.Markdown("### Train a new model on Hugging Face datasets")

                with gr.Row():
                    with gr.Column():
                        dataset_dropdown = gr.Dropdown(
                            choices=MultiDatasetLoader.list_available_datasets(),
                            label="Select Dataset",
                            value="wikitext"
                        )

                        model_name_input = gr.Textbox(
                            label="Model Name",
                            value="cyberdyne-llm",
                            placeholder="my-awesome-model"
                        )

                        with gr.Row():
                            num_epochs = gr.Number(
                                label="Epochs",
                                value=3,
                                minimum=1,
                                maximum=50
                            )

                            batch_size = gr.Number(
                                label="Batch Size",
                                value=4,
                                minimum=1,
                                maximum=32
                            )

                        max_samples = gr.Number(
                            label="Max Training Samples",
                            value=10000,
                            minimum=100,
                            maximum=1000000
                        )

                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=5e-5,
                            minimum=1e-6,
                            maximum=1e-3
                        )

                    with gr.Column():
                        gr.Markdown("### Model Architecture")

                        emb_size = gr.Dropdown(
                            choices=[256, 512, 768, 1024],
                            label="Embedding Size",
                            value=768
                        )

                        n_layers = gr.Slider(
                            minimum=4,
                            maximum=24,
                            value=12,
                            step=2,
                            label="Number of Layers"
                        )

                        n_heads = gr.Slider(
                            minimum=4,
                            maximum=16,
                            value=12,
                            step=2,
                            label="Number of Attention Heads"
                        )

                train_btn = gr.Button("Start Training", variant="primary", size="lg")

                training_output = gr.Textbox(
                    label="Training Status",
                    lines=10,
                    interactive=False
                )

                trained_model_path = gr.Textbox(
                    label="Trained Model Path",
                    interactive=False
                )

                train_btn.click(
                    app.train_model,
                    inputs=[
                        dataset_dropdown, model_name_input, num_epochs, batch_size,
                        max_samples, learning_rate, emb_size, n_layers, n_heads
                    ],
                    outputs=[training_output, trained_model_path]
                )

            with gr.Tab("⚙️ Model Management"):
                gr.Markdown("### Load and manage your models")

                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=app.available_models,
                            label="Available Models",
                            interactive=True
                        )

                        refresh_btn = gr.Button("Refresh Model List")

                        with gr.Row():
                            model_path_input = gr.Textbox(
                                label="Model Path",
                                placeholder="models/cyberdyne-llm_final.pt"
                            )

                        load_btn = gr.Button("Load Model", variant="primary")

                        load_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )

                def update_model_path(model_name):
                    if model_name:
                        return f"models/{model_name}"
                    return ""

                model_dropdown.change(
                    update_model_path,
                    inputs=[model_dropdown],
                    outputs=[model_path_input]
                )

                refresh_btn.click(
                    app.get_model_list,
                    outputs=[model_dropdown]
                )

                load_btn.click(
                    app.load_model,
                    inputs=[model_path_input],
                    outputs=[load_status]
                )

            with gr.Tab("📚 Dataset Info"):
                gr.Markdown("""
                ### Available Datasets

                This system supports training on multiple Hugging Face datasets:

                **General Text Corpora:**
                - **wikitext**: Wikipedia articles (great for general knowledge)
                - **wikipedia**: Full Wikipedia dump
                - **openwebtext**: Web pages from Reddit links
                - **bookcorpus**: Books corpus for long-form text
                - **c4**: Colossal Clean Crawled Corpus
                - **pile**: EleutherAI's diverse dataset

                **Instruction-Following Datasets:**
                - **dolly**: Databricks instruction-following dataset
                - **alpaca**: Stanford Alpaca instruction dataset
                - **squad**: Question-answering dataset

                **Tips:**
                - Start with smaller datasets like `wikitext` for testing
                - Use instruction datasets for chat-like behavior
                - Larger `emb_size` and `n_layers` = more powerful but slower
                - Adjust `max_samples` based on your compute resources
                """)

            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ### Cyberdyne LLM System

                **Features:**
                - 🤖 Train custom language models from scratch
                - 📊 Support for multiple Hugging Face datasets
                - 💾 Offline model inference (no internet required)
                - 💬 ChatGPT-like conversational interface
                - 🔧 Fully configurable model architecture
                - 📈 Training progress tracking

                **How to Use:**
                1. **Training**: Go to the Training tab, select a dataset, configure parameters, and start training
                2. **Loading**: After training, go to Model Management and load your trained model
                3. **Chatting**: Use the Chat Interface to interact with your model

                **Model Architecture:**
                - Decoder-only Transformer (GPT-style)
                - Multi-head self-attention
                - Layer normalization and residual connections
                - Configurable depth and width

                **System Requirements:**
                - GPU recommended for training (CPU supported)
                - Python 3.8+
                - PyTorch, Transformers, Gradio

                Built with ❤️ using PyTorch and Hugging Face
                """)

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

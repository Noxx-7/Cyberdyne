# 🤖 Cyberdyne LLM

A complete system for training and deploying custom language models with ChatGPT-like capabilities. Train on multiple Hugging Face datasets and run models offline!

## ✨ Features

- **🎓 Custom Model Training**: Train your own language models from scratch
- **📊 Multi-Dataset Support**: Choose from 9+ curated Hugging Face datasets
- **💾 Offline Inference**: Run trained models without internet connectivity
- **💬 ChatGPT-Style Interface**: Conversational AI with context management
- **🎨 Beautiful Gradio UI**: Intuitive web interface for training and chatting
- **🔧 Fully Configurable**: Customize model architecture and training parameters
- **📈 Training Tracking**: Monitor loss and progress in real-time
- **🚀 Easy to Use**: Get started with simple example scripts

## 🏗️ Architecture

**Decoder-Only Transformer (GPT-Style)**
- Multi-head self-attention mechanism
- Layer normalization with residual connections
- Configurable depth (4-24 layers)
- Flexible embedding dimensions (256-1024)
- Optimized for conversational tasks

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- GPU recommended (CPU supported)
- 8GB+ RAM (16GB+ for larger models)

## 🚀 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd cyberdyne

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Option 1: Automated Quick Start

```bash
python quick_start.py
```

This will:
1. Train a small model on wikitext dataset (5000 samples, 2 epochs)
2. Test the model with sample prompts
3. Provide instructions for next steps

### Option 2: Launch the Full Interface

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

The interface has 5 tabs:
- **💬 Chat Interface**: Interact with your trained models
- **🎓 Training**: Train new models on various datasets
- **⚙️ Model Management**: Load and manage saved models
- **📚 Dataset Info**: Learn about available datasets
- **ℹ️ About**: System information and usage guide

## 📖 Usage Examples

### Training a Model

**Using Python:**

```python
from trainer import LLMTrainer

trainer = LLMTrainer(
    model_name='my-llm',
    emb_size=768,
    n_layers=12,
    n_heads=12,
    learning_rate=5e-5
)

history = trainer.train(
    dataset_key='wikitext',
    num_epochs=3,
    batch_size=4,
    max_samples=10000,
    save_dir='models'
)
```

**Using the example script:**

```bash
python train_example.py
```

### Offline Inference

**Using Python:**

```python
from inference import OfflineInference

inference = OfflineInference('models/my-llm_final.pt')
session_id = inference.create_session()

response, _ = inference.chat(
    "What is machine learning?",
    session_id=session_id,
    temperature=0.8,
    max_new_tokens=150
)

print(response)
```

**Using the command-line interface:**

```bash
python inference_example.py models/my-llm_final.pt
```

### Advanced Training

```python
from trainer import LLMTrainer

trainer = LLMTrainer(
    model_name='advanced-llm',
    emb_size=1024,
    n_layers=16,
    n_heads=16,
    ff_size=4096,
    learning_rate=3e-5
)

history = trainer.train(
    dataset_key='dolly',
    num_epochs=5,
    batch_size=8,
    max_samples=50000,
    max_len=512,
    save_dir='models'
)
```

## 📚 Available Datasets

### General Text Corpora
- **wikitext**: Wikipedia articles (good for testing)
- **wikipedia**: Full Wikipedia dump
- **openwebtext**: Web pages from Reddit
- **bookcorpus**: Books corpus
- **c4**: Colossal Clean Crawled Corpus
- **pile**: EleutherAI's diverse dataset

### Instruction-Following Datasets
- **dolly**: Databricks instruction dataset
- **alpaca**: Stanford Alpaca dataset
- **squad**: Question-answering dataset

## 🎛️ Configuration Options

### Model Architecture

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `emb_size` | Embedding dimension | 768 | 256-1024 |
| `n_layers` | Number of transformer layers | 12 | 4-24 |
| `n_heads` | Number of attention heads | 12 | 4-16 |
| `ff_size` | Feed-forward layer size | 3072 | 1024-4096 |
| `max_len` | Maximum sequence length | 512 | 128-2048 |

### Training Parameters

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `learning_rate` | Optimizer learning rate | 5e-5 | 1e-5 to 1e-4 |
| `batch_size` | Training batch size | 4 | 1-32 |
| `num_epochs` | Number of training epochs | 3 | 1-50 |
| `max_samples` | Maximum training samples | 10000 | 1000-1000000 |

### Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `temperature` | Sampling randomness | 0.8 | 0.1-2.0 |
| `top_k` | Top-k sampling | 50 | 1-100 |
| `top_p` | Nucleus sampling | 0.95 | 0.1-1.0 |
| `max_new_tokens` | Max generated tokens | 150 | 50-500 |

## 💡 Tips for Best Results

### For Training:
1. **Start Small**: Begin with `wikitext` and 5000 samples to test
2. **GPU Recommended**: Training on CPU is slow (use smaller models)
3. **Memory Management**: Reduce `batch_size` if you run out of memory
4. **Dataset Choice**: Use instruction datasets (dolly, alpaca) for chat-like behavior
5. **Model Size**: Larger `emb_size` and `n_layers` = more capable but slower

### For Inference:
1. **Temperature**: Lower (0.3-0.6) for focused responses, higher (0.8-1.2) for creativity
2. **Context**: Enable context for multi-turn conversations
3. **Token Limit**: Increase `max_new_tokens` for longer responses
4. **Top-p/Top-k**: Adjust for diversity vs coherence tradeoff

## 📁 Project Structure

```
cyberdyne/
├── model.py              # Transformer model architecture
├── dataset_loader.py     # Multi-dataset loading system
├── trainer.py            # Training engine
├── inference.py          # Offline inference engine
├── app.py                # Gradio web interface
├── requirements.txt      # Python dependencies
├── train_example.py      # Training example script
├── inference_example.py  # CLI chat example
├── quick_start.py        # Automated quick start
├── models/               # Saved model checkpoints
└── README.md             # This file
```

## 🔧 Advanced Features

### Custom Dataset Integration

```python
from dataset_loader import HuggingFaceDataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

dataset = HuggingFaceDataset(
    dataset_name='your-custom-dataset',
    tokenizer=tokenizer,
    max_samples=20000,
    max_len=512,
    text_field='text',
    streaming=True
)
```

### Model Checkpointing

Models are automatically saved after each epoch:
```
models/
├── my-llm_epoch_1.pt
├── my-llm_epoch_2.pt
├── my-llm_epoch_3.pt
└── my-llm_final.pt
```

### Conversation Context

The inference engine maintains conversation history:

```python
inference = OfflineInference('models/my-llm_final.pt')
session_id = inference.create_session()

response1, _ = inference.chat("What is AI?", session_id=session_id)
response2, _ = inference.chat("Tell me more", session_id=session_id)

history = inference.get_history(session_id)
```

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` to 1 or 2
- Decrease `emb_size` and `n_layers`
- Use fewer training samples
- Close other applications

### Slow Training
- Use GPU instead of CPU
- Reduce `max_samples`
- Use smaller model architecture
- Enable mixed precision training (advanced)

### Poor Model Quality
- Train for more epochs
- Use more training samples
- Try instruction datasets (dolly, alpaca)
- Increase model size (if resources allow)
- Adjust learning rate (try 3e-5 or 1e-4)

### Dataset Loading Errors
- Check internet connection for first-time download
- Try a different dataset
- Ensure sufficient disk space
- Update `datasets` library: `pip install -U datasets`

## 🎓 Learning Resources

### Understanding the Architecture
- The model uses decoder-only transformers (like GPT)
- Self-attention learns relationships between tokens
- Layer normalization stabilizes training
- Residual connections help gradient flow

### Training Process
1. Model reads text sequences from dataset
2. Predicts next token for each position
3. Compares predictions to actual next tokens
4. Updates weights to minimize prediction error
5. Repeats for multiple epochs

### Generation Process
1. Encode user prompt to tokens
2. Model predicts probability distribution for next token
3. Sample from distribution using temperature/top-k/top-p
4. Append selected token and repeat
5. Stop at EOS token or max length

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 🙏 Acknowledgments

Built with:
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Gradio

---

**Made with ❤️ by the Cyberdyne team**

For questions or support, please open an issue on GitHub.

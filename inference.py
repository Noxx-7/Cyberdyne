import torch
from transformers import GPT2Tokenizer
from typing import List, Optional
import uuid

class ConversationalInference:
    def __init__(self, model, tokenizer_name='gpt2', device=None, max_context_length=5):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.max_context_length = max_context_length
        self.conversations = {}

        print(f"Inference engine initialized on {self.device}")

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        return session_id

    def get_conversation_history(self, session_id):
        return self.conversations.get(session_id, [])

    def clear_session(self, session_id):
        if session_id in self.conversations:
            self.conversations[session_id] = []

    def generate_response(self, prompt, session_id=None, max_new_tokens=150,
                         temperature=0.8, top_k=50, top_p=0.95,
                         use_context=True):
        if session_id is None:
            session_id = self.create_session()

        if session_id not in self.conversations:
            self.conversations[session_id] = []

        if use_context and self.conversations[session_id]:
            context_messages = self.conversations[session_id][-self.max_context_length:]
            context_text = self._build_context(context_messages)
            full_prompt = f"{context_text}\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"

        response = self._generate_text(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        self.conversations[session_id].append({
            'role': 'user',
            'content': prompt
        })
        self.conversations[session_id].append({
            'role': 'assistant',
            'content': response
        })

        return response, session_id

    def _build_context(self, messages):
        context_parts = []
        for msg in messages:
            if msg['role'] == 'user':
                context_parts.append(f"User: {msg['content']}")
            else:
                context_parts.append(f"Assistant: {msg['content']}")
        return "\n".join(context_parts)

    def _generate_text(self, prompt, max_new_tokens=150, temperature=0.8,
                       top_k=50, top_p=0.95):
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        if tokens.size(1) > self.model.max_len - max_new_tokens:
            tokens = tokens[:, -(self.model.max_len - max_new_tokens):]

        generated = tokens

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.size(1) >= self.model.max_len:
                    break

                logits = self.model(generated)
                next_token_logits = logits[:, -1, :]

                next_token_logits = next_token_logits / temperature

                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

                top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), top_k)
                next_token = top_indices[0, torch.multinomial(top_probs[0], 1)]

                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        if "Assistant:" in output_text:
            response = output_text.split("Assistant:")[-1].strip()
        else:
            response = output_text[len(prompt):].strip()

        return response

    def batch_generate(self, prompts, max_new_tokens=150, temperature=0.8,
                      top_k=50, top_p=0.95):
        responses = []
        for prompt in prompts:
            response = self._generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            responses.append(response)
        return responses

    def chat(self, message, session_id=None, **kwargs):
        return self.generate_response(message, session_id=session_id, **kwargs)

class OfflineInference:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        from model import AdvancedLLM
        self.model = AdvancedLLM(
            vocab_size=checkpoint['vocab_size'],
            emb_size=checkpoint['emb_size'],
            n_layers=checkpoint['n_layers'],
            n_heads=checkpoint['n_heads'],
            max_len=checkpoint['max_len']
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

        tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2')
        self.inference_engine = ConversationalInference(
            self.model,
            tokenizer_name=tokenizer_name,
            device=self.device
        )

        print("Offline inference ready!")

    def chat(self, message, session_id=None, **kwargs):
        return self.inference_engine.chat(message, session_id=session_id, **kwargs)

    def create_session(self):
        return self.inference_engine.create_session()

    def clear_session(self, session_id):
        self.inference_engine.clear_session(session_id)

    def get_history(self, session_id):
        return self.inference_engine.get_conversation_history(session_id)

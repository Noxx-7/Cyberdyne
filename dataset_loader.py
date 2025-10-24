import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Optional, Dict, List

class HuggingFaceDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, max_samples=100000,
                 max_len=512, split='train', config=None, streaming=True,
                 text_field='text', instruction_field=None):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_len = max_len
        self.split = split
        self.config = config
        self.streaming = streaming
        self.text_field = text_field
        self.instruction_field = instruction_field

        try:
            if config:
                self.dataset = load_dataset(dataset_name, config, split=split, streaming=streaming)
            else:
                self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            print(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            raise

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if count >= self.max_samples:
                break

            text = self._extract_text(item)
            if not text:
                continue

            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).squeeze(0)

            target = tokens.clone()
            target[:-1] = tokens[1:]
            target[-1] = self.tokenizer.eos_token_id

            yield tokens, target
            count += 1

    def _extract_text(self, item):
        if self.instruction_field and self.instruction_field in item:
            if isinstance(item[self.instruction_field], list):
                instruction = item[self.instruction_field][0] if item[self.instruction_field] else ""
            else:
                instruction = item[self.instruction_field]

            if self.text_field in item:
                if isinstance(item[self.text_field], list):
                    response = item[self.text_field][0] if item[self.text_field] else ""
                else:
                    response = item[self.text_field]
                return f"Instruction: {instruction}\n\nResponse: {response}"
            return instruction

        if self.text_field in item:
            text = item[self.text_field]
            if isinstance(text, list):
                return text[0] if text else ""
            return text

        for key in ['text', 'content', 'document', 'article', 'response', 'output']:
            if key in item:
                value = item[key]
                if isinstance(value, list):
                    return value[0] if value else ""
                return value

        return ""

class MultiDatasetLoader:
    DATASET_CONFIGS = {
        'wikipedia': {
            'name': 'wikipedia',
            'config': '20231101.en',
            'text_field': 'text',
        },
        'openwebtext': {
            'name': 'openwebtext',
            'text_field': 'text',
        },
        'wikitext': {
            'name': 'wikitext',
            'config': 'wikitext-103-v1',
            'text_field': 'text',
        },
        'bookcorpus': {
            'name': 'bookcorpus',
            'text_field': 'text',
        },
        'c4': {
            'name': 'c4',
            'config': 'en',
            'text_field': 'text',
        },
        'dolly': {
            'name': 'databricks/databricks-dolly-15k',
            'text_field': 'response',
            'instruction_field': 'instruction',
        },
        'alpaca': {
            'name': 'tatsu-lab/alpaca',
            'text_field': 'output',
            'instruction_field': 'instruction',
        },
        'squad': {
            'name': 'squad',
            'text_field': 'context',
        },
        'pile': {
            'name': 'EleutherAI/pile',
            'text_field': 'text',
        },
    }

    @classmethod
    def create_dataset(cls, dataset_key, tokenizer, max_samples=100000,
                       max_len=512, split='train', streaming=True):
        if dataset_key not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(cls.DATASET_CONFIGS.keys())}")

        config = cls.DATASET_CONFIGS[dataset_key]
        return HuggingFaceDataset(
            dataset_name=config['name'],
            tokenizer=tokenizer,
            max_samples=max_samples,
            max_len=max_len,
            split=split,
            config=config.get('config'),
            streaming=streaming,
            text_field=config['text_field'],
            instruction_field=config.get('instruction_field')
        )

    @classmethod
    def list_available_datasets(cls):
        return list(cls.DATASET_CONFIGS.keys())

    @classmethod
    def get_dataset_info(cls, dataset_key):
        if dataset_key in cls.DATASET_CONFIGS:
            return cls.DATASET_CONFIGS[dataset_key]
        return None

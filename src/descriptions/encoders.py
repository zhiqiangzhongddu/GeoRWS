import hashlib

import torch
import torch.nn.functional as F


class TextEncoder:
    def encode(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class _TransformerTextEncoder(TextEncoder):
    model_id = ''
    context = 0
    pooling = ''

    def __init__(self, cfg):
        self.batch_size = cfg.text_encoder.batch_size
        self.device = f'cuda:{cfg.device}' if cfg.device >= 0 and torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.identity = {'model': self.model_id, 'pooling': self.pooling,
                         'chunking': f'content-token weighted, context {self.context}', 'normalisation': 'l2'}

    def _load(self):
        raise NotImplementedError

    def _embed_batch(self, ids):
        batch = self.tokenizer.pad({'input_ids': ids}, padding=True, return_attention_mask=True, return_tensors='pt')
        batch = {key: value.to(self.device) for key, value in batch.items()}
        output = self.model(**batch)
        if self.pooling == 'text_embeds':
            return output.text_embeds
        mask = batch['attention_mask'].unsqueeze(-1)
        return (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.dim), dtype=torch.float32)
        if self.model is None:
            self._load()
            self.model.to(self.device).eval()
        unique = list(dict.fromkeys(texts))
        tokenized = self.tokenizer(unique, add_special_tokens=False, truncation=False)['input_ids']
        capacity = self.context - self.tokenizer.num_special_tokens_to_add(pair=False)
        chunks, owners, weights = [], [], []
        for index, tokens in enumerate(tokenized):
            for start in range(0, max(len(tokens), 1), capacity):
                part = tokens[start:start + capacity]
                chunks.append(self.tokenizer.build_inputs_with_special_tokens(part))
                owners.append(index)
                weights.append(max(len(part), 1))
        sums = torch.zeros((len(unique), self.dim), dtype=torch.float32)
        totals = torch.zeros(len(unique), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(chunks), self.batch_size):
                end = start + self.batch_size
                vectors = self._embed_batch(chunks[start:end]).float().cpu()
                owner = torch.tensor(owners[start:end], dtype=torch.long)
                weight = torch.tensor(weights[start:end], dtype=torch.float32)
                sums.index_add_(0, owner, vectors * weight[:, None])
                totals.index_add_(0, owner, weight)
        vectors = F.normalize(sums / totals[:, None], p=2, dim=1)
        lookup = {text: i for i, text in enumerate(unique)}
        return vectors[[lookup[text] for text in texts]]


class ClipTextEncoder(_TransformerTextEncoder):
    model_id, dim, context, pooling = 'openai/clip-vit-base-patch32', 512, 77, 'text_embeds'

    def _load(self):
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.model = CLIPTextModelWithProjection.from_pretrained(self.model_id)


class BertTextEncoder(_TransformerTextEncoder):
    model_id, dim, context, pooling = 'bert-base-uncased', 768, 512, 'valid-token mean'

    def _load(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)


class T5TextEncoder(_TransformerTextEncoder):
    model_id, dim, context, pooling = 't5-base', 768, 512, 'valid-token mean'

    def _load(self):
        from transformers import AutoTokenizer, T5EncoderModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = T5EncoderModel.from_pretrained(self.model_id)


class FakeTextEncoder(TextEncoder):
    def __init__(self, dim=32):
        self.dim = dim
        self.identity = {'model': f'fake-{dim}', 'pooling': 'none', 'chunking': 'none', 'normalisation': 'l2'}

    def encode(self, texts: list[str]) -> torch.Tensor:
        vectors = {}
        for text in dict.fromkeys(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')
            generator = torch.Generator().manual_seed(seed)
            vectors[text] = F.normalize(torch.randn(self.dim, generator=generator), dim=0)
        return torch.stack([vectors[text] for text in texts]) if texts else torch.empty((0, self.dim), dtype=torch.float32)


def make_text_encoder(cfg):
    encoders = {'clip': ClipTextEncoder, 'bert': BertTextEncoder, 't5': T5TextEncoder, 'fake': FakeTextEncoder}
    name = cfg.text_encoder.name
    if name not in encoders:
        raise ValueError(f'Unknown text encoder: {name}')
    return FakeTextEncoder() if name == 'fake' else encoders[name](cfg)

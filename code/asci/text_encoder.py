import torch
from transformers import AutoModel, AutoTokenizer
import clip


class TextEmbedder:
    def __init__(self, model_type="BERT"):
        self.model_type = model_type

        if model_type == "BERT":
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif model_type == "T5":
            self.model = AutoModel.from_pretrained("t5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        elif model_type == "CLIP":
            self.model, _ = clip.load("ViT-B/32", device="cpu")
        else:
            raise ValueError("Unsupported model type. Choose from BERT, T5, or CLIP.")

    def encode(self, text):
        if self.model_type in ["BERT", "T5"]:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
        elif self.model_type == "CLIP":
            text_tokens = clip.tokenize([text]).to("cpu")
            text_features = self.model.encode_text(text_tokens).detach()
            return text_features.mean(dim=1)
        else:
            raise ValueError("Unsupported model type.")


if __name__ == "__main__":
    texts = [
        "Antelopes are herbivorous mammals that inhabit grasslands.",
        "Giraffes are tall mammals with long necks and spotted patterns.",
        "Leopards are solitary predators with spotted fur."
    ]

    for model_type in ["BERT", "T5", "CLIP"]:
        embedder = TextEmbedder(model_type)
        embeddings = torch.stack([embedder.encode(text) for text in texts])
        print(f"{model_type} embeddings shape: {embeddings.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer


class ASCIModel(nn.Module):
    def __init__(self, text_encoder='bert-base-uncased', embedding_dim=768):
        super(ASCIModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.semantic_encoder = nn.Linear(embedding_dim, embedding_dim)
        self.classifier_injection = nn.Linear(embedding_dim, embedding_dim)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, class_descriptions):
        embeddings = torch.stack([self.encode_text(desc) for desc in class_descriptions])
        refined_embeddings = self.semantic_encoder(embeddings)
        classifier_weights = self.classifier_injection(refined_embeddings)
        return classifier_weights


if __name__ == "__main__":
    # Example usage
    model = ASCIModel()
    class_descriptions = [
        "Antelopes are herbivorous mammals that inhabit grasslands.",
        "Giraffes are tall mammals with long necks and spotted patterns.",
        "Leopards are solitary predators with spotted fur."
    ]
    classifier_weights = model(class_descriptions)
    print("Generated classifier weights:", classifier_weights.shape)

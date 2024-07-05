from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    def __init__(self, model_name="all-mpnet-base-v2", device="cpu"):
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def generate_embeddings(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

    @staticmethod
    def cosine_similarity(vector1, vector2):
        dot_product = torch.dot(vector1, vector2)
        norm_vector1 = torch.sqrt(torch.sum(vector1**2))
        norm_vector2 = torch.sqrt(torch.sum(vector2**2))
        return dot_product / (norm_vector1 * norm_vector2)
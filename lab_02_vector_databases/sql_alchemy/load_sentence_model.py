from sentence_transformers import SentenceTransformer


# def load_model():
#     checkpoint = "distiluse-base-multilingual-cased-v2"
#     model = SentenceTransformer(checkpoint)
#     return model
checkpoint = "distiluse-base-multilingual-cased-v2"
model = SentenceTransformer(checkpoint)


def generate_embeddings(text: str) -> list[float]:
    return model.encode(text)

import config
import json_prompt_reader as jpr
from sentence_transformers import SentenceTransformer


class PromptEmbedder:
    def __init__(self, model):
        self.model = model

    def embed(self, prompt):
        # Use the model to generate an embedding for the prompt
        embedding = self.model.encode(prompt)
        return embedding

if __name__ == "__main__":

    # Load the model specified in the config
    model_name = config.EMBEDDING_MODEL
    model = SentenceTransformer(model_name)

    json_path = "prompt.json"

    # Create an instance of PromptEmbedder
    embedder = PromptEmbedder(model)

    #embed prompt from json file
    prompt = jpr.read_json_prompt(json_path)

    # Get the embedding for the prompt
    embedding = embedder.embed(prompt)
    print("Embedding for the prompt:", embedding)

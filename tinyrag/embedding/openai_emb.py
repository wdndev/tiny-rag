from typing import Dict, List, Optional, Tuple, Union

from tinyrag.embedding.base_emb import BaseEmbedding # Assuming this is a custom module for base embeddings

class OpenAIEmbedding(BaseEmbedding):
    """
    A class for generating embeddings using the OpenAI API.
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1", path: str = '', is_api: bool = True) -> None:
        """
        Initializes the OpenAIEmbedding object.

        :param api_key: API key for accessing the OpenAI API.
        :param base_url: Base URL for the OpenAI API.
        :param path: Path to any local resources (not used in this case).
        :param is_api: Flag indicating whether this is an API-based embedding.
        """
        super().__init__(path, is_api=True)  # Call the constructor of the base class

        from openai import OpenAI  # Importing the OpenAI client
        
        self.client = OpenAI()  # Initialize the OpenAI client
        self.client.api_key = api_key  # Set the API key for the client
        self.client.base_url = base_url  # Set the base URL for the client
        self.name = "openai_api"  # Set the name of the embedding source

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the OpenAI API.

        :param text: Text to embed.
        :return: A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")  # Replace newlines with spaces to avoid issues
        # Use the OpenAI API to create an embedding for the input text
        return self.client.embeddings.create(input=[text], model="text-embedding-3-large").data[0].embedding

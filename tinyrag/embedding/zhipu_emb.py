from typing import Dict, List, Optional, Tuple, Union
from tinyrag.embedding.base_emb import BaseEmbedding  # Assuming this is a custom module for base embeddings

class ZhipuEmbedding(BaseEmbedding):
    """
    A class for generating embeddings using the Zhipu AI API.
    """
    def __init__(self, api_key, path: str = '', is_api: bool = True) -> None:
        """
        Initializes the ZhipuEmbedding object.

        :param api_key: API key for accessing the Zhipu AI API.
        :param path: Path to any local resources (not used in this case).
        :param is_api: Flag indicating whether this is an API-based embedding.
        """
        super().__init__(path, is_api)  # Call the constructor of the base class
        
        from zhipuai import ZhipuAI  # Importing the ZhipuAI client
        
        self.client = ZhipuAI(api_key=api_key)  # Initialize the ZhipuAI client
        self.name = "zhipu_api"  # Set the name of the embedding source

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the Zhipu AI API.

        :param text: Text to embed.
        :return: A list of floats representing the embedding.
        """
        # Use the Zhipu AI API to create an embedding for the input text
        response = self.client.embeddings.create(model="embedding-2", input=text)
        return response.data[0].embedding


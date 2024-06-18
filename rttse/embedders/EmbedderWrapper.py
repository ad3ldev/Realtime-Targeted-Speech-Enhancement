from torch import Tensor, zeros

class EmbedderWrapper:
    def embed(self, file_path: str) -> Tensor:
        pass

    def embed_batch(self, file_paths: list) -> Tensor:
        embeddings = zeros((len(file_paths), 192))
        for i, file_path in enumerate(file_paths):
            embeddings[i] = self.embed(file_path)
        return embeddings
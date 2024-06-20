from torch import Tensor, zeros, nn

class EmbedderWrapper(nn.Module):
    def __init__(self):
        super(EmbedderWrapper, self).__init__()

    def embed(self, file_path: str) -> Tensor:
        pass

    def embed_batch(self, file_paths: list) -> Tensor:
        embeddings = zeros((len(file_paths), 192))
        for i, file_path in enumerate(file_paths):
            embeddings[i] = self.embed(file_path)
        return embeddings
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion for a given prompt."""
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for given text."""
        pass

import google.generativeai as genai

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)


class GeminiLLM(CustomLLM):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        super().__init__()

        # ✅ configure globally (allowed)
        genai.configure(api_key=api_key)

        # ✅ store safely outside pydantic fields
        object.__setattr__(self, "_model_name", model)
        object.__setattr__(
            self,
            "_model",
            genai.GenerativeModel(model)
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self._model_name,
            context_window=100000,
            num_output=2048,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self._model.generate_content(prompt)
        text = getattr(response, "text", str(response))
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs):
        response = self._model.generate_content(prompt)
        text = getattr(response, "text", str(response))
        yield CompletionResponse(text=text)

    def __call__(self, prompt: str, **kwargs):
        return self.complete(prompt, **kwargs).text
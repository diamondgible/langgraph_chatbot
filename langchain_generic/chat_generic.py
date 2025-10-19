import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Any, List, Optional, Mapping, AsyncIterator, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# Function to make a request to the API and generate a response
def generate_response_with_chat_completion(
    messages, 
    model, 
    temperature=0.7, 
    max_tokens=4096, 
    top_p = 1.0, 
    frequency_penalty = 0.0, 
    presence_penalty = 0.0, 
    logprobs = None, 
    seed = None,
    stream=False):

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "logprobs": logprobs,
        "seed": seed,
        "stream": stream
    }

    clean_kwargs = {k: v for k, v in request_kwargs.items() if v is not None}

    if stream:
        return client.chat.completions.create(**clean_kwargs)
    else:
        completion = client.chat.completions.create(**clean_kwargs)
        # Extract the bot's message content from the response
        bot_response = completion.choices[0].message.content
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        return bot_response, input_tokens, output_tokens 

class ChatGeneric(BaseChatModel):
    """
    Drop-in LangChain Chat model for Generic API.
    Behaves like ChatOpenAI or ChatOllama.
    Internally uses generate_response_with_chat_completion().
    """

    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: Optional[int] = None
    seed: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "chat_generic"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def _format_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """
        Convert LangChain BaseMessage list to generic API message format.
        Assumes OpenAI-style roles: 'user', 'assistant', 'system'.
        Adjust if your API requires something else.
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "system"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """
        Generate a response from the model. Passed in messages are converted to the generic API message format.
        """
        formatted_messages = self._format_messages(messages)

        bot_response, input_tokens, output_tokens = generate_response_with_chat_completion(
            messages=formatted_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=self.logprobs,
            seed=self.seed
        )

        ai_message = AIMessage(content=bot_response)
        generation = ChatGeneration(message=ai_message, generation_info={'input_tokens': input_tokens, 'output_tokens': output_tokens})
        return ChatResult(generations=[generation])

    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream a response from the model.
        """
        formatted_messages = self._format_messages(messages)
        
        stream = generate_response_with_chat_completion(
            messages=formatted_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=self.logprobs,
            seed=self.seed,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunk_message = AIMessageChunk(content=chunk.choices[0].delta.content)
                chunk_generation = ChatGenerationChunk(message=chunk_message)
                yield chunk_generation

    async def _astream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async stream a response from the model.
        """
        formatted_messages = self._format_messages(messages)
        
        stream = generate_response_with_chat_completion(
            messages=formatted_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=self.logprobs,
            seed=self.seed,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunk_message = AIMessageChunk(content=chunk.choices[0].delta.content)
                chunk_generation = ChatGenerationChunk(message=chunk_message)
                yield chunk_generation

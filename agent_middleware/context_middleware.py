"""
Context Management Middleware

Middleware components for managing context window and tool responses:
1. TokenLimitCheckMiddleware - Monitors and manages token limits
2. tool_response_summarizer - Summarizes large tool responses
"""

import json
import logging
from typing import Any, List, Optional
from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


class TokenLimitCheckMiddleware(AgentMiddleware):
    """
    Custom middleware that checks token count before model calls.
    If input tokens exceed threshold (100k), triggers summarization.
    """

    def __init__(
        self,
        model: ChatBedrockConverse,
        summarization_llm: ChatBedrockConverse,
        token_threshold: int = 100000,
        summary_prompt: Optional[str] = None,
    ):
        """
        Initialize token limit checking middleware.

        Args:
            model: The main model to check token counts for
            summarization_llm: LLM to use for summarization
            token_threshold: Maximum input tokens before summarization (default: 100k)
            summary_prompt: Optional custom summarization prompt
        """
        super().__init__()
        self.model = model
        self.summarization_llm = summarization_llm
        self.token_threshold = token_threshold
        self.summary_prompt = summary_prompt
        print(f"Initialized TokenLimitCheckMiddleware with threshold={token_threshold}")

    def _summarize_messages(
        self,
        messages: List[BaseMessage],
    ) -> List[BaseMessage]:
        """
        Summarize messages when token limit is exceeded.

        Args:
            messages: List of messages to summarize

        Returns:
            Summarized messages list
        """
        print(f"Token limit exceeded ({self.token_threshold}). Summarizing conversation...")

        try:
            # Convert messages to text for summarization
            conversation_text = "\n\n".join([
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in messages
            ])

            # Create summarization prompt
            if self.summary_prompt:
                summary_request = f"{self.summary_prompt}\n\nConversation:\n{conversation_text}"
            else:
                summary_request = (
                    "Please provide a concise summary of the following conversation, "
                    "preserving key information, context, and important details:\n\n"
                    f"{conversation_text}"
                )

            # Get summary from LLM
            summary_response = self.summarization_llm.invoke([
                SystemMessage(content="You are a helpful assistant that creates summaries."),
                HumanMessage(content=summary_request)
            ])

            summary_text = summary_response.content
            print(f"Successfully summarized conversation. Summary length: {len(summary_text)} chars")

            # Return summarized conversation as a single system message
            return [
                SystemMessage(content=f"Previous conversation summary:\n{summary_text}")
            ]

        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)
            # If summarization fails, return original messages
            logger.warning("Summarization failed, returning original messages")
            return messages

    def wrap_model_call(
        self,
        request: Any,
        handler: Any,
    ) -> Any:
        """
        Intercept model call to check token count and summarize if needed.

        Args:
            request: ModelRequest containing state and runtime
            handler: Callback to execute the model

        Returns:
            ModelResponse from handler
        """
        # Access messages from request state
        state = request.state
        messages = state.get("messages", [])

        if not messages:
            print("No messages to check")
            return handler(request)

        try:
            # Count input tokens
            token_count = self.model.get_num_tokens_from_messages(messages)
            print(f"[TokenLimitCheckMiddleware] Current input token count: {token_count}")

            # Check if we exceed threshold
            if token_count > self.token_threshold:
                print(f"[TokenLimitCheckMiddleware] Token count ({token_count}) exceeds threshold ({self.token_threshold})")
                # Summarize messages
                summarized_messages = self._summarize_messages(messages)

                # Update state with summarized messages
                state["messages"] = summarized_messages

                # Verify new token count
                new_token_count = self.model.get_num_tokens_from_messages(summarized_messages)
                print(f"[TokenLimitCheckMiddleware] Reduced token count from {token_count} to {new_token_count}")
            else:
                print(f"[TokenLimitCheckMiddleware] Token count within threshold. Proceeding normally.")

        except Exception as e:
            print(f"[TokenLimitCheckMiddleware] Could not check token count: {e}")
            logger.error(f"Token counting error: {e}", exc_info=True)
            # If token counting fails, proceed without modification

        # Call the model with potentially modified state
        return handler(request)


# Global storage for full tool responses (could be replaced with S3/DynamoDB)
_TOOL_RESPONSE_STORAGE = {}


def create_tool_response_summarizer(
    summarization_llm: ChatBedrockConverse,
    token_threshold: int = 100000,
    store_full_responses: bool = True,
    summary_prompt: Optional[str] = None,
    summarization_model_max_tokens: int = 150000,
):
    """
    Create a tool response summarization middleware using wrap_tool_call.

    Uses the same token threshold and summarization approach as TokenLimitCheckMiddleware
    to ensure consistent context management across the agent.

    Args:
        summarization_llm: LLM to use for summarization
        token_threshold: Maximum allowed response tokens before summarization (default: 100k)
        store_full_responses: Whether to store full responses
        summary_prompt: Optional custom summarization prompt (same as TokenLimitCheckMiddleware)
        summarization_model_max_tokens: Max tokens the summarization model can handle (default: 150k)

    Returns:
        Middleware that summarizes large tool responses
    """
    logger.info(
        f"Creating tool_response_summarizer with "
        f"token_threshold={token_threshold}, "
        f"store_full_responses={store_full_responses}, "
        f"summarization_model_max_tokens={summarization_model_max_tokens}"
    )

    def _chunk_content(content: str, max_chunk_tokens: int) -> List[str]:
        """
        Split content into chunks that fit within the summarization model's token limit.
        Uses actual token counting to ensure chunks don't exceed the limit.

        Args:
            content: The content to chunk
            max_chunk_tokens: Maximum tokens per chunk

        Returns:
            List of content chunks
        """
        chunks = []

        # Start with character-based estimate for initial chunk size
        # We'll refine this as we go
        chars_per_token = 4  # Initial estimate

        current_pos = 0
        content_length = len(content)

        while current_pos < content_length:
            # Estimate chunk size based on current chars_per_token ratio
            estimated_chunk_size = int(max_chunk_tokens * chars_per_token)
            end_pos = min(current_pos + estimated_chunk_size, content_length)

            chunk = content[current_pos:end_pos]

            # Check actual token count
            chunk_tokens = summarization_llm.get_num_tokens(chunk)

            # If chunk is too large, reduce it
            while chunk_tokens > max_chunk_tokens and len(chunk) > 100:
                # Reduce chunk size by 10%
                end_pos = current_pos + int(len(chunk) * 0.9)
                chunk = content[current_pos:end_pos]
                chunk_tokens = summarization_llm.get_num_tokens(chunk)

            # Update our estimate of chars per token for next iteration
            if chunk_tokens > 0:
                chars_per_token = len(chunk) / chunk_tokens

            chunks.append(chunk)
            current_pos = end_pos
            logger.info(f"Created chunk {len(chunks)} with {chunk_tokens} tokens ({len(chunk)} chars)")

        logger.info(f"Split content into {len(chunks)} chunks (max {max_chunk_tokens} tokens each)")
        return chunks

    def _summarize_tool_content(tool_name: str, content: str) -> str:
        """
        Summarize tool response content using LLM.
        Uses the same summarization prompt as TokenLimitCheckMiddleware for consistency.
        If content is too large for the summarization model, chunks it and summarizes each chunk.
        """
        logger.info(f"Summarizing response from tool '{tool_name}' (length: {len(content)} chars)")

        try:
            # Check actual token count of the content
            content_tokens = summarization_llm.get_num_tokens(content)
            logger.info(f"Content has {content_tokens} tokens")

            # If content is too large for summarization model, chunk it first
            # Leave room for prompt overhead (~2000 tokens)
            if content_tokens > (summarization_model_max_tokens - 2000):
                logger.warning(
                    f"Content too large for summarization model ({content_tokens} tokens > "
                    f"{summarization_model_max_tokens - 2000} tokens). Using chunked summarization."
                )

                # Chunk the content - reserve tokens for prompt overhead
                max_chunk_tokens = summarization_model_max_tokens - 2000
                chunks = _chunk_content(content, max_chunk_tokens)

                # Summarize each chunk
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Summarizing chunk {i+1}/{len(chunks)} for tool '{tool_name}'")

                    # Create summarization prompt for this chunk
                    if summary_prompt:
                        summary_request = (
                            f"{summary_prompt}\n\n"
                            f"Tool Name: {tool_name}\n"
                            f"Chunk {i+1} of {len(chunks)}\n\n"
                            f"Tool Response:\n{chunk}"
                        )
                    else:
                        summary_request = (
                            f"Please provide a concise summary of the following chunk (part {i+1} of {len(chunks)}) "
                            f"from tool '{tool_name}', preserving key information and important details. "
                            f"If it's structured data (JSON, lists), maintain the structure and include counts/statistics.\n\n"
                            f"Tool Response:\n{chunk}"
                        )

                    # Get summary for this chunk
                    response = summarization_llm.invoke([
                        SystemMessage(content="You are a helpful assistant that creates concise summaries."),
                        HumanMessage(content=summary_request)
                    ])

                    chunk_summaries.append(f"[Chunk {i+1}/{len(chunks)}]\n{response.content}")

                # Combine all chunk summaries
                combined_summary = "\n\n".join(chunk_summaries)
                logger.info(
                    f"Successfully summarized {len(chunks)} chunks. "
                    f"Combined summary length: {len(combined_summary)} chars"
                )

                # Add metadata
                summary_with_metadata = (
                    f"[CHUNKED SUMMARIZED TOOL RESPONSE FROM: {tool_name}]\n"
                    f"[Original: {content_tokens} tokens, Split into {len(chunks)} chunks]\n"
                    f"[Summarized to: {len(combined_summary)} chars]\n\n"
                    f"{combined_summary}\n\n"
                    f"[Note: Full response has been stored and is available if needed]"
                )
                print(f"Returning the chunked tool call summary")
                return summary_with_metadata

            else:
                # Content is small enough - summarize directly
                # Create summarization prompt - use custom prompt if provided, otherwise use default
                if summary_prompt:
                    summary_request = (
                        f"{summary_prompt}\n\n"
                        f"Tool Name: {tool_name}\n\n"
                        f"Tool Response:\n{content}"
                    )
                else:
                    summary_request = (
                        f"Please provide a concise summary of the following tool response from '{tool_name}', "
                        f"preserving key information, context, and important details. "
                        f"If it's structured data (JSON, lists), maintain the structure and include counts/statistics.\n\n"
                        f"Tool Response:\n{content}"
                    )

                # Get summary from LLM using the same approach as TokenLimitCheckMiddleware
                response = summarization_llm.invoke([
                    SystemMessage(content="You are a helpful assistant that creates concise summaries."),
                    HumanMessage(content=summary_request)
                ])

                summary = response.content
                logger.info(f"Successfully summarized tool response. Summary length: {len(summary)} chars")

                # Add metadata to summary
                summary_with_metadata = (
                    f"[SUMMARIZED TOOL RESPONSE FROM: {tool_name}]\n"
                    f"[Original: {content_tokens} tokens, Summarized to: {len(summary)} chars]\n\n"
                    f"{summary}\n\n"
                    f"[Note: Full response has been stored and is available if needed]"
                )
                print(f"Returning the tool call summary \n: {summary_with_metadata}")
                return summary_with_metadata

        except Exception as e:
            logger.error(f"Error summarizing tool response: {e}", exc_info=True)
            # If summarization fails, truncate the response to ~500 chars
            truncated = content[:500] + "\n...[truncated due to summarization failure]"
            return f"[TRUNCATED TOOL RESPONSE FROM: {tool_name}]\n{truncated}"

    @wrap_tool_call
    def tool_response_summarizer(request, handler):
        """
        Middleware to intercept and summarize large tool responses.

        This middleware:
        1. Executes the tool via handler
        2. Checks if response exceeds token threshold
        3. Summarizes large responses using an LLM
        4. Stores full responses if enabled
        5. Returns summarized response
        """
        # Execute the tool
        result = handler(request)

        # Check if result is a ToolMessage
        if not isinstance(result, ToolMessage):
            return result

        content = result.content
        tool_name = getattr(result, 'name', 'unknown_tool')

        # Count tokens in the tool response using the same approach as TokenLimitCheckMiddleware
        try:
            # Use the summarization LLM to count tokens
            token_count = summarization_llm.get_num_tokens(content)
            logger.info(f"Tool '{tool_name}' response token count: {token_count}")
        except Exception as e:
            logger.warning(f"Could not count tokens for tool response, using char length estimate: {e}")
            # Fallback: estimate tokens as chars/4 (rough approximation)
            token_count = len(content) // 4

        # Check if token count exceeds threshold (same as TokenLimitCheckMiddleware)
        if token_count > token_threshold:
            logger.info(
                f"Tool '{tool_name}' response exceeds threshold "
                f"({token_count} tokens > {token_threshold} tokens). Summarizing..."
            )

            # Store full response if enabled
            if store_full_responses:
                storage_key = f"{tool_name}_{result.id}"
                _TOOL_RESPONSE_STORAGE[storage_key] = content
                logger.info(f"Stored full response with key: {storage_key}")
            # Summarize the response
            summarized_content = _summarize_tool_content(tool_name, content)
            # Create new ToolMessage with summarized content
            return ToolMessage(
                content=summarized_content,
                tool_call_id=result.tool_call_id,
                name=tool_name,
                id=result.id,
            )
        else:
            # Return original if under threshold
            logger.debug(f"Tool '{tool_name}' response within threshold ({token_count} tokens)")
            return result

    return tool_response_summarizer


def get_full_tool_response(storage_key: str) -> Optional[str]:
    """
    Retrieve a full tool response from storage.

    Args:
        storage_key: The storage key for the response

    Returns:
        Full response if found, None otherwise
    """
    return _TOOL_RESPONSE_STORAGE.get(storage_key)


def clear_tool_response_storage() -> None:
    """Clear all stored full tool responses."""
    _TOOL_RESPONSE_STORAGE.clear()
    logger.info("Cleared tool response storage")


# Export the function that creates the middleware
tool_response_summarizer = create_tool_response_summarizer

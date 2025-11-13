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
    If input tokens exceed threshold (100k), summarizes only the oldest messages
    up to a certain percentage, keeping the rest as-is.
    """

    def __init__(
        self,
        model: ChatBedrockConverse,
        summarization_llm: ChatBedrockConverse,
        token_threshold: int = 100000,
        summarization_percentage: float = 0.40,  # Summarize first 40% of messages
        summary_prompt: Optional[str] = None,
        keep_recent_messages: int = 5,
    ):
        """
        Initialize token limit checking middleware.

        Args:
            model: The main model to check token counts for
            summarization_llm: LLM to use for summarization
            token_threshold: Maximum input tokens before summarization (default: 100k)
            summarization_percentage: Percentage of old messages to summarize (default: 0.40 = 40%)
            summary_prompt: Optional custom summarization prompt
            keep_recent_messages: Minimum number of most recent messages to keep unsummarized (default: 5)
        """
        super().__init__()
        self.model = model
        self.summarization_llm = summarization_llm
        self.token_threshold = token_threshold
        self.summarization_percentage = summarization_percentage
        self.summary_prompt = summary_prompt
        self.keep_recent_messages = keep_recent_messages
        
        print(f"Initialized TokenLimitCheckMiddleware with threshold={token_threshold}, "
              f"summarization_percentage={summarization_percentage * 100}%, "
              f"keep_recent={keep_recent_messages}")

    def _find_summarization_boundary(
        self,
        messages: List[BaseMessage],
        target_percentage: float,
    ) -> tuple[int, int]:
        """
        Find the index where we should stop summarizing based on percentage of tokens.
        
        Args:
            messages: List of messages
            target_percentage: Percentage of messages to summarize (0.0 - 1.0)
            
        Returns:
            Tuple of (boundary_index, tokens_in_chunk)
            - boundary_index: Index of the last message to include in summarization (exclusive upper bound)
            - tokens_in_chunk: Total tokens in the chunk to be summarized
        """
        # First, calculate total tokens in all messages
        total_tokens = 0
        message_tokens = []
        
        for msg in messages:
            try:
                msg_tokens = self.summarization_llm.get_num_tokens_from_messages([msg])
            except Exception as e:
                # Fallback: estimate tokens
                logger.warning(f"Could not count tokens for message, using estimate: {e}")
                msg_tokens = len(str(msg.content)) // 4
            
            message_tokens.append(msg_tokens)
            total_tokens += msg_tokens
        
        # Calculate target tokens (percentage of total)
        target_tokens = int(total_tokens * target_percentage)
        
        logger.info(
            f"Total tokens: {total_tokens}, "
            f"Target to summarize ({target_percentage * 100}%): {target_tokens} tokens"
        )
        
        # Find the boundary where we've accumulated ~target_tokens
        cumulative_tokens = 0
        boundary_index = 0
        
        for i, msg_tokens in enumerate(message_tokens):
            cumulative_tokens += msg_tokens
            
            # Stop when we've accumulated enough tokens
            if cumulative_tokens >= target_tokens:
                # Return index + 1 to include this message
                boundary_index = i + 1
                break
        
        # If we never reached target_tokens, return all messages
        if boundary_index == 0:
            boundary_index = len(messages)
            cumulative_tokens = total_tokens
        
        logger.info(f"Boundary set at message index {boundary_index} ({cumulative_tokens} tokens)")
        
        return boundary_index, cumulative_tokens

    def _summarize_first_chunk(
        self,
        messages: List[BaseMessage],
        chunk_tokens: int,
    ) -> str:
        """
        Summarize the first chunk of messages.
        
        Args:
            messages: List of messages to summarize
            chunk_tokens: Number of tokens in this chunk
            
        Returns:
            Summary text
        """
        try:
            # Convert messages to text for summarization
            conversation_text = "\n\n".join([
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in messages
            ])

            # Create summarization prompt
            if self.summary_prompt:
                summary_request = (
                    f"{self.summary_prompt}\n\n"
                    f"[Summarizing first {len(messages)} messages (~{chunk_tokens} tokens) from conversation]\n\n"
                    f"{conversation_text}"
                )
            else:
                summary_request = (
                    f"Please provide a concise summary of the following conversation "
                    f"(first {len(messages)} messages, approximately {chunk_tokens} tokens), "
                    f"preserving key information, context, decisions made, and important details:\n\n"
                    f"{conversation_text}"
                )

            # Get summary from LLM
            summary_response = self.summarization_llm.invoke([
                SystemMessage(content="You are a helpful assistant that creates concise summaries."),
                HumanMessage(content=summary_request)
            ])

            summary_text = summary_response.content
            logger.info(
                f"Successfully summarized first {len(messages)} messages "
                f"(~{chunk_tokens} tokens). Summary length: {len(summary_text)} chars"
            )
            
            return summary_text

        except Exception as e:
            logger.error(f"Error summarizing first chunk: {e}", exc_info=True)
            # If summarization fails, create a simple truncated version
            truncated = (
                f"[Summary of first {len(messages)} messages (~{chunk_tokens} tokens) - summarization failed]\n"
            )
            truncated += "\n".join([
                f"- {msg.__class__.__name__}: {str(msg.content)[:100]}..."
                for msg in messages[:5]  # Just show first 5 messages
            ])
            return truncated

    def _summarize_messages(
        self,
        messages: List[BaseMessage],
    ) -> List[BaseMessage]:
        """
        Summarize messages when token limit is exceeded.
        
        Strategy:
        1. Keep the most recent N messages unsummarized (for context continuity)
        2. From the remaining old messages, summarize only the FIRST X% (default 40%)
        3. Keep all other middle messages (60%) as-is
        
        Result: [Summary of first 40%] + [Middle 60% as-is] + [Recent N messages as-is]
        
        Args:
            messages: List of messages to summarize

        Returns:
            Messages list with first X% summarized, rest preserved
        """
        print(
            f"Token limit exceeded ({self.token_threshold}). "
            f"Summarizing first {self.summarization_percentage * 100}% of old messages..."
        )

        try:
            # Step 1: Separate recent messages to keep
            if len(messages) <= self.keep_recent_messages:
                # If we have very few messages, we still need to summarize
                # This is an edge case - shouldn't normally happen
                logger.warning(
                    f"Only {len(messages)} messages but exceeding token limit. "
                    f"This suggests very long messages. Summarizing based on percentage."
                )
                messages_to_process = messages
                recent_messages = []
            else:
                messages_to_process = messages[:-self.keep_recent_messages]
                recent_messages = messages[-self.keep_recent_messages:]
            
            logger.info(
                f"Processing {len(messages_to_process)} old messages, "
                f"keeping {len(recent_messages)} recent messages as-is"
            )

            # Step 2: Find boundary for first X% to summarize
            summarize_boundary, chunk_tokens = self._find_summarization_boundary(
                messages_to_process,
                self.summarization_percentage
            )
            
            messages_to_summarize = messages_to_process[:summarize_boundary]
            middle_messages = messages_to_process[summarize_boundary:]
            
            logger.info(
                f"Summarizing first {len(messages_to_summarize)} messages (~{chunk_tokens} tokens), "
                f"keeping {len(middle_messages)} middle messages as-is"
            )

            # Step 3: Summarize only the first chunk
            summary_text = self._summarize_first_chunk(messages_to_summarize, chunk_tokens)
            
            # Create summary message
            summary_message = SystemMessage(
                content=(
                    f"Previous conversation summary "
                    f"(from first {len(messages_to_summarize)} messages, ~{chunk_tokens} tokens):\n\n"
                    f"{summary_text}"
                )
            )

            # Step 4: Combine: [Summary] + [Middle messages] + [Recent messages]
            result_messages = [summary_message] + middle_messages + recent_messages
            
            logger.info(
                f"Final message structure: 1 summary + {len(middle_messages)} middle + "
                f"{len(recent_messages)} recent = {len(result_messages)} total messages"
            )
            
            # Verify the new token count
            try:
                new_token_count = self.model.get_num_tokens_from_messages(result_messages)
                original_token_count = self.model.get_num_tokens_from_messages(messages)
                reduction = original_token_count - new_token_count
                reduction_percent = (reduction / original_token_count * 100) if original_token_count > 0 else 0
                
                logger.info(
                    f"Token count: {original_token_count} → {new_token_count} "
                    f"(reduced by {reduction} tokens, {reduction_percent:.1f}%)"
                )
                
                print(
                    f"[TokenLimitCheckMiddleware] Summarization complete:\n"
                    f"  Original: {original_token_count:,} tokens\n"
                    f"  New: {new_token_count:,} tokens\n"
                    f"  Saved: {reduction:,} tokens ({reduction_percent:.1f}%)\n"
                    f"  Structure: 1 summary + {len(middle_messages)} middle + {len(recent_messages)} recent"
                )
                
            except Exception as e:
                logger.warning(f"Could not verify new token count: {e}")
            
            return result_messages

        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)
            
            # Emergency fallback: if summarization completely fails, 
            # just keep the most recent messages
            logger.warning(
                f"Summarization failed completely. Keeping only the "
                f"{self.keep_recent_messages} most recent messages."
            )
            
            if len(messages) > self.keep_recent_messages:
                fallback_messages = [
                    SystemMessage(
                        content=(
                            f"[EMERGENCY CONTEXT TRUNCATION] "
                            f"Previous {len(messages) - self.keep_recent_messages} messages "
                            f"were truncated due to context window limitations. "
                            f"Only the {self.keep_recent_messages} most recent messages are shown below."
                        )
                    )
                ] + messages[-self.keep_recent_messages:]
                return fallback_messages
            else:
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
            try:
                token_count = self.model.get_num_tokens_from_messages(messages)
            except Exception as e:
                logger.warning(f"Could not count tokens precisely, using estimate: {e}")
                # Fallback estimation
                token_count = sum(len(str(msg.content)) for msg in messages) // 4
            
            print(f"[TokenLimitCheckMiddleware] Current input token count: {token_count}")

            # Check if we exceed threshold
            if token_count > self.token_threshold:
                print(
                    f"[TokenLimitCheckMiddleware] Token count ({token_count}) "
                    f"exceeds threshold ({self.token_threshold})"
                )
                
                # Summarize only the first X% of old messages
                summarized_messages = self._summarize_messages(messages)

                # Update state with modified messages
                state["messages"] = summarized_messages

                # Verify new token count
                try:
                    new_token_count = self.model.get_num_tokens_from_messages(summarized_messages)
                    print(
                        f"[TokenLimitCheckMiddleware] Reduced token count "
                        f"from {token_count} to {new_token_count}"
                    )
                except Exception as e:
                    logger.warning(f"Could not verify new token count: {e}")
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

class PruneToolCallMiddleware(AgentMiddleware):
    """
    Custom middleware that prunes verbose tool responses to essential fields only.
    
    This middleware reduces token usage by filtering out unnecessary metadata from
    tool responses, especially useful for LangSmith tools that return verbose data.
    """
    
    def __init__(
        self,
        tools_to_prune: Optional[List[str]] = None,
        max_error_length: int = 500,
        max_input_length: int = 200,
    ):
        """
        Initialize tool response pruning middleware.
        
        Args:
            tools_to_prune: List of tool names to prune. If None, prunes all tools.
                           Example: ["get_runs_from_session", "list_session_runs_summary"]
            max_error_length: Maximum length for error messages (default: 500 chars)
            max_input_length: Maximum length for input content (default: 200 chars)
        """
        super().__init__()
        self.tools_to_prune = tools_to_prune
        self.max_error_length = max_error_length
        self.max_input_length = max_input_length
        
        logger.info(
            f"Initialized PruneToolCallMiddleware with "
            f"tools_to_prune={tools_to_prune}, "
            f"max_error_length={max_error_length}, "
            f"max_input_length={max_input_length}"
        )
    
    def _extract_essential_fields(self, data: Any, tool_name: str) -> Any:
        """
        Keep only essential fields for tool responses:
        - id, name, error, status, content
        - For outputs: only keep final insights, not all intermediate messages
        - Remove all metadata, nested messages, and verbose fields

        This dramatically reduces context window usage while preserving
        the information needed for the agent to work.
        """
        if isinstance(data, dict):
            # At the top level of a run/trace object, keep only these fields
            if 'id' in data and 'name' in data:  # This looks like a run/trace object
                essential_data = {}

                # Always keep these if present
                for key in ['id', 'name', 'error', 'status', 'start_time', 'end_time', 'run_count', 'latency_p50', 'latency_p99', 'error_rate']:
                    if key in data:
                        essential_data[key] = data[key]

                # For outputs, extract only the final insights/content
                if 'outputs' in data:
                    outputs = data['outputs']
                    if isinstance(outputs, dict):
                        # Extract just the output value, not all the messages
                        if 'output' in outputs:
                            essential_data['content'] = outputs['output']
                        else:
                            # Fallback: look for meaningful data
                            extracted_output = {}
                            for out_key in ['output', 'result', 'answer', 'response']:
                                if out_key in outputs:
                                    extracted_output[out_key] = outputs[out_key]
                            if extracted_output:
                                essential_data['content'] = extracted_output

                # For inputs, extract only the user question
                if 'inputs' in data:
                    inputs = data['inputs']
                    if isinstance(inputs, dict):
                        user_question = self._extract_input_query(inputs)
                        if user_question:
                            essential_data['input_query'] = user_question[:200]  # Truncate to 200 chars

                return essential_data

            # For nested dicts that aren't run objects, recursively process
            filtered_dict = {}
            for key, value in data.items():
                # Skip verbose metadata fields entirely
                if key in {
                    "ResponseMetadata", "HTTPHeaders", "HTTPStatusCode",
                    "RetryAttempts", "RequestId", "x-amzn-requestid",
                    "FunctionArn", "Role", "Handler", "CodeSha256",
                    "RevisionId", "TracingConfig", "RuntimeVersionConfig",
                    "LoggingConfig", "SnapStart", "Architectures",
                    "EphemeralStorage", "PackageType", "State",
                    "LastUpdateStatus", "Version", "LastModified",
                    "inputs_s3_urls", "outputs_s3_urls", "s3_urls",
                    "manifest_id", "manifest_s3_id", "inputs_preview", "outputs_preview",
                    "app_path", "last_queued_at", "in_dataset", "share_token",
                    "ttl_seconds", "trace_upgrade", "trace_first_received_at",
                    "trace_min_start_time", "trace_max_start_time",
                    "events", "tags", "execution_order", "serialized",
                    "run_type", "thread_id", "test_run_number",
                    "additional_kwargs", "invalid_tool_calls", "usage_metadata",
                    "tool_calls", "lc", "kwargs",
                }:
                    continue

                # Recursively process the value
                filtered_dict[key] = self._extract_essential_fields(value, tool_name)

            return filtered_dict

        elif isinstance(data, list):
            # For lists, process each item
            return [self._extract_essential_fields(item, tool_name) for item in data]

        else:
            # Primitive type, return as-is
            return data
    
    def _extract_input_query(self, inputs: Any) -> Optional[str]:
        """Extract user input query from nested input structure.

        Looks for HumanMessage in the messages array, not SystemMessage.
        This ensures we get the actual user question, not the system prompt.
        """
        input_query = None

        # Handle different input structures
        if isinstance(inputs, dict):
            # Check for messages array
            if "messages" in inputs:
                messages = inputs["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    # Find the HumanMessage (user question), not SystemMessage
                    for msg in messages:
                        # Handle nested list structure
                        if isinstance(msg, list) and len(msg) > 0:
                            msg = msg[0]

                        if isinstance(msg, dict):
                            # Check if this is a HumanMessage
                            msg_type = msg.get('type', '')
                            msg_id = msg.get('id', '')

                            # Look for HumanMessage indicators
                            if msg_type == 'human' or 'HumanMessage' in msg_id:
                                # LangChain message format
                                if "kwargs" in msg:
                                    input_query = msg["kwargs"].get("content", "")
                                elif "content" in msg:
                                    input_query = msg["content"]

                                if input_query:
                                    break  # Found the user question

            # If no messages, check for direct input field
            if not input_query and "input" in inputs:
                input_query = str(inputs["input"])

        elif isinstance(inputs, str):
            input_query = inputs

        return input_query
    
    def _should_prune_tool(self, tool_name: str) -> bool:
        """Determine if this tool should be pruned."""
        if self.tools_to_prune is None:
            # Prune all tools by default
            return True
        return tool_name in self.tools_to_prune
    
    def wrap_tool_call(
        self,
        request: Any,
        handler,
    ) -> Any:
        """
        Intercept tool calls to prune verbose responses.
        
        This method:
        1. Executes the tool via handler
        2. Checks if the tool should be pruned
        3. Extracts only essential fields from the response
        4. Logs the token savings
        5. Returns the pruned response
        
        Args:
            request: ToolCallRequest containing tool call information
            handler: Callback to execute the tool
            
        Returns:
            ToolMessage with pruned content or original response
        """
        # Execute the tool
        result = handler(request)
        
        # Check if result is a ToolMessage
        if not isinstance(result, ToolMessage):
            logger.debug(f"Result is not a ToolMessage, returning as-is")
            return result
        
        tool_name = getattr(result, 'name', 'unknown_tool')
        
        # Check if this tool should be pruned
        if not self._should_prune_tool(tool_name):
            logger.debug(f"Tool '{tool_name}' not in prune list, returning original response")
            return result
        
        content = result.content
        original_length = len(content)
        
        try:
            # Try to parse as JSON
            try:
                data = json.loads(content)
                is_json = True
            except (json.JSONDecodeError, TypeError):
                # Not JSON, treat as plain text
                data = content
                is_json = False
            
            if is_json:
                # Extract essential fields
                pruned_data = self._extract_essential_fields(data, tool_name)
                
                # Convert back to JSON string
                pruned_content = json.dumps(pruned_data, indent=2)
                pruned_length = len(pruned_content)
                
                # Calculate savings
                reduction_percent = (
                    ((original_length - pruned_length) / original_length * 100) 
                    if original_length > 0 else 0
                )
                
                logger.info(
                    f"Pruned tool '{tool_name}' response: "
                    f"{original_length} chars → {pruned_length} chars "
                    f"({reduction_percent:.1f}% reduction)"
                )
                
                print(
                    f"[PRUNE_TOOL_CALL] Tool: {tool_name}\n"
                    f"  Original: {original_length:,} chars\n"
                    f"  Pruned: {pruned_length:,} chars\n"
                    f"  Reduction: {reduction_percent:.1f}%\n"
                    f"  Estimated token savings: ~{(original_length - pruned_length) // 4:,} tokens"
                )
                
                # Create new ToolMessage with pruned content
                return ToolMessage(
                    content=pruned_content,
                    tool_call_id=result.tool_call_id,
                    name=tool_name,
                    id=result.id,
                )
            else:
                # For non-JSON content, apply simple truncation
                if original_length > 1000:
                    pruned_content = content
                    logger.info(
                        f"Truncated non-JSON tool '{tool_name}' response: "
                        f"{original_length} chars → {len(pruned_content)} chars"
                    )
                    
                    print(
                        f"[PRUNE_TOOL_CALL] Tool: {tool_name}\n"
                        f"  Original: {original_length:,} chars\n"
                        f"  Truncated to: {len(pruned_content):,} chars"
                    )
                    
                    return ToolMessage(
                        content=pruned_content,
                        tool_call_id=result.tool_call_id,
                        name=tool_name,
                        id=result.id,
                    )
                else:
                    # Content is short enough, return as-is
                    logger.debug(f"Tool '{tool_name}' response is short enough, no pruning needed")
                    return result
        
        except Exception as e:
            logger.error(f"Error pruning tool response from '{tool_name}': {e}", exc_info=True)
            # If pruning fails, return original response
            logger.warning(f"Pruning failed for tool '{tool_name}', returning original response")
            return result


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


# Memory middleware classes removed - now using direct hooks in agent.py


# Export the function that creates the middleware
tool_response_summarizer = create_tool_response_summarizer

"""
Custom Memory Prompts for AgentCore Memory

Custom extraction and consolidation prompts for the error analysis strategy.
"""

# Custom extraction prompt for error analysis and insights
ERROR_INSIGHTS_EXTRACTION_PROMPT = """
Extract and structure ANY of the following information from the conversation (extract even if only some categories are present):

1. **User Questions and Context**:
   - What the user is asking about
   - Session IDs or project identifiers mentioned
   - Specific runs, traces, or agent executions discussed

2. **Errors and Issues** (if present):
   - Error messages and stack traces
   - Error patterns and frequencies
   - Root causes identified
   - Note: Exclude user-initiated errors like keyboard interrupts

3. **Agent Insights and Analysis**:
   - Analysis and findings about agent behavior
   - Performance observations
   - Code quality issues
   - Patterns identified in traces

4. **Solutions and Recommendations** (if present):
   - Solutions implemented or suggested
   - Code changes made or recommended
   - Best practices suggested
   - Fixes that resolved issues

5. **Research and External Resources** (if present):
   - External resources consulted
   - Documentation references
   - Similar issues and solutions found

6. **Key Facts and Details**:
   - Session metadata (session IDs, run counts, timestamps)
   - Tool responses and their summaries
   - Important data points from LangSmith traces
   - Any other factual information discussed

**Important**: Extract information even if the conversation doesn't contain errors. Focus on ANY technical details, user questions, agent insights, or factual information that would be useful for future reference.
"""

# Custom consolidation prompt for error analysis
ERROR_INSIGHTS_CONSOLIDATION_PROMPT = """
Consolidate information from multiple conversations into a coherent knowledge base:

1. **Group Related Information**:
   - Group related errors and their solutions
   - Connect related questions and insights across sessions
   - Link similar agent traces and patterns

2. **Identify Patterns**:
   - Identify recurring error patterns
   - Find common user questions or workflows
   - Track trends in agent performance or behavior

3. **Organize by Context**:
   - Connect insights to specific sessions, runs, or code areas
   - Preserve contextual information (session IDs, timestamps)
   - Link solutions to the problems they solved

4. **Track Outcomes**:
   - Track which issues were resolved vs. still pending
   - Note solution effectiveness
   - Record best practices and recommendations

Create a structured knowledge base that enables quick lookup of:
- Known errors and their fixes
- Frequently asked questions and their answers
- Agent behavior patterns and insights
- Session-specific information
- Code improvement recommendations
- Unresolved issues requiring attention
"""

from lightrag.utils import TokenTracker

# Create TokenTracker instance
token_tracker = TokenTracker()

# Method 1: Using context manager (Recommended)
# Suitable for scenarios requiring automatic token usage tracking
# with token_tracker:
#     result1 = await llm_model_func("your question 1")
#     result2 = await llm_model_func("your question 2")

# Method 2: Manually adding token usage records
# Suitable for scenarios requiring more granular control over token statistics
# token_tracker.reset()

# rag.insert()

# rag.query("your question 1", param=QueryParam(mode="naive"))
# rag.query("your question 2", param=QueryParam(mode="mix"))

# Display total token usage (including insert and query operations)
# print("Token usage:", token_tracker.get_usage())

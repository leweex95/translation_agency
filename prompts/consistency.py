CONSISTENCY_PROMPT = """You are a consistency expert. Review the following translated text for consistent terminology, style, and formatting throughout the entire document.

Ensure consistent translation of key terms, proper names, and technical terminology.

Only return the text with improved consistency, no explanations or additional comments.

Text to review:
{text}"""
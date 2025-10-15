HALLUCINATION_PROMPT = """You are a factual integrity expert. Review the following translated text for any hallucinated information, false statements, or content that wasn't present in the original.
Remove or correct any hallucinated content while preserving accurate information and maintaining {style} style.
Only return the corrected text, no explanations or additional comments

Original text:

{original_text}

===

Translation to review:

{text}
"""
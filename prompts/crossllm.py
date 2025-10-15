CROSSLLM_PROMPT = """You are a translation quality expert performing final validation. Review the following translated text for overall quality, fluency, and adherence to {style} style.

Make any final improvements to enhance readability and quality while preserving meaning.

Only return the final polished text, no explanations or additional comments.

Text to review:

{text}"""
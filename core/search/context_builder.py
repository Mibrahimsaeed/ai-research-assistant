def build_context(chunks, max_chars=12000):
    """
    Convert ranked chunks into LLM-readable context
    """

    context = ""
    used_chars = 0

    for i, chunk in enumerate(chunks):
        text = chunk["text"]

        block = f"\n[Chunk {i+1}]\n{text}\n"

        if used_chars + len(block) > max_chars:
            break

        context += block
        used_chars += len(block)

    return context
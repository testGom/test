async def pipe(self, user_message, model_id, messages, body, __event_emitter__=None, *__args, **__kwargs):
    query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=1, response_mode="refine")
    response = query_engine.query(user_message)

    # Emit citations
    for node in response.source_nodes:
        await __event_emitter__({
            "type": "citation",
            "data": {
                "document": [node.text],
                "metadata": [{"source": node.metadata.get("file_name", "unknown")}],
                "source": {
                    "name": node.metadata.get("file_name", "unknown"),
                    "url": node.metadata.get("file_path", "#"),
                },
            }
        })

    # This is a *sync* generator
    in_thinking = False
    buffer = ""

    for chunk in response.response_gen:  # ✅ this works
        buffer += chunk
        while True:
            if in_thinking:
                end_idx = buffer.find("</think>")
                if end_idx == -1:
                    if buffer.strip():
                        await __event_emitter__({
                            "type": "message",
                            "data": {"content": buffer, "role": "assistant-thinking"}
                        })
                    buffer = ""
                    break
                thinking_text = buffer[:end_idx]
                if thinking_text.strip():
                    await __event_emitter__({
                        "type": "message",
                        "data": {"content": thinking_text, "role": "assistant-thinking"}
                    })
                buffer = buffer[end_idx + len("</think>"):]
                in_thinking = False
            else:
                start_idx = buffer.find("<think>")
                if start_idx == -1:
                    if buffer.strip():
                        await __event_emitter__({
                            "type": "message",
                            "data": {"content": buffer, "role": "assistant"}
                        })
                    buffer = ""
                    break
                if start_idx > 0:
                    await __event_emitter__({
                        "type": "message",
                        "data": {"content": buffer[:start_idx], "role": "assistant"}
                    })
                buffer = buffer[start_idx + len("<think>"):]
                in_thinking = True

    # Flush remaining buffer
    if buffer.strip():
        await __event_emitter__({
            "type": "message",
            "data": {"content": buffer, "role": "assistant-thinking" if in_thinking else "assistant"}
        })

    return ""

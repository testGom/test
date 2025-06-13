import re

async def pipe(self, user_message, model_id, messages, body, __event_emitter__=None, *__args, **__kwargs):
    query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=1, response_mode="refine")
    response = query_engine.query(user_message)

    # Emit citations if needed
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

    # Streaming reasoning then final answer
    in_thinking = False
    buffer = ""

    async for chunk in response.response_gen:
        buffer += chunk

        # Start thinking block
        if not in_thinking and "<think>" in buffer:
            in_thinking = True
            parts = buffer.split("<think>", 1)
            if parts[0].strip():
                await __event_emitter__({
                    "type": "message",
                    "data": {"content": parts[0], "role": "assistant"}
                })
            buffer = parts[1]
            continue

        # End thinking block
        if in_thinking and "</think>" in buffer:
            parts = buffer.split("</think>", 1)
            if parts[0].strip():
                await __event_emitter__({
                    "type": "message",
                    "data": {"content": parts[0], "role": "assistant-thinking"}
                })
            buffer = parts[1]
            in_thinking = False
            if buffer.strip():
                await __event_emitter__({
                    "type": "message",
                    "data": {"content": buffer, "role": "assistant"}
                })
            buffer = ""
            continue

        # Stream ongoing content
        if not in_thinking and buffer.strip():
            await __event_emitter__({
                "type": "message",
                "data": {"content": buffer, "role": "assistant"}
            })
            buffer = ""
        elif in_thinking and buffer.strip():
            await __event_emitter__({
                "type": "message",
                "data": {"content": buffer, "role": "assistant-thinking"}
            })
            buffer = ""

    # Flush leftover
    if buffer.strip():
        await __event_emitter__({
            "type": "message",
            "data": {
                "content": buffer,
                "role": "assistant-thinking" if in_thinking else "assistant"
            }
        })

    return ""

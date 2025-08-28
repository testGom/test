def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
    # optional status event
    yield {
        "event": {
            "type": "status",
            "data": {"description": "Recherche dans les documents...", "done": False},
        }
    }

    query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=3)
    response = query_engine.query(user_message)

    # Stream OpenAI-style deltas so OpenWebUI can build the reasoning block
    in_think = False
    buf = ""

    # Tag sets you might see from “thinking” models
    start_tags = ["<think>", "<|begin_of_thought|>", "◁think▷", "<|begin_of_solution|>"]
    end_tags   = ["</think>", "<|end_of_thought|>", "◁/think▷", "<|end_of_solution|>"]

    def _find_any(s, tags):
        for t in tags:
            i = s.find(t)
            if i != -1:
                return i, t
        return -1, ""

    for tok in response.response_gen:
        buf += tok

        while True:
            if not in_think:
                i, t = _find_any(buf, start_tags)
                if i != -1:
                    # flush any pre-think as normal content
                    pre = buf[:i]
                    if pre:
                        yield {"choices": [{"index": 0, "delta": {"content": pre}}]}
                    buf = buf[i + len(t):]
                    in_think = True
                    continue
                # no start tag; flush what we have as normal content
                if buf:
                    yield {"choices": [{"index": 0, "delta": {"content": buf}}]}
                    buf = ""
                break
            else:
                j, t2 = _find_any(buf, end_tags)
                if j != -1:
                    # stream the reasoning chunk up to the end tag
                    chunk = buf[:j]
                    if chunk:
                        yield {"choices": [{"index": 0, "delta": {"reasoning_content": chunk}}]}
                    buf = buf[j + len(t2):]
                    in_think = False
                    continue
                # still inside thinking; stream whatever we have
                if buf:
                    yield {"choices": [{"index": 0, "delta": {"reasoning_content": buf}}]}
                    buf = ""
                break

    # flush any leftover
    if buf:
        if in_think:
            yield {"choices": [{"index": 0, "delta": {"reasoning_content": buf}}]}
        else:
            yield {"choices": [{"index": 0, "delta": {"content": buf}}]}

    # Send citations as events (this is fine to keep as events)
    for node in response.source_nodes:
        yield {
            "event": {
                "type": "citation",
                "data": {
                    "document": [node.text],
                    "metadata": [{"source": node.metadata.get("file_name", "unknown")}],
                    "source": {
                        "name": node.metadata.get("file_name", "unknown"),
                        "url": node.metadata.get("file_path", "#"),
                    },
                    "distances": [node.score],
                },
            }
        }

    yield {"event": {"type": "status", "data": {"description": "", "done": True}}}

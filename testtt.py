
def stream_thinking_as_collapsible(text_stream):
    buffer = ""
    open_tag = "<thinking>"
    close_tag = "</thinking>"

    for chunk in text_stream:
        buffer += chunk
        while True:
            start = buffer.find(open_tag)
            end = buffer.find(close_tag, start + len(open_tag))

            # Case 1: Found full block
            if start != -1 and end != -1:
                before = buffer[:start]
                thinking_content = buffer[start + len(open_tag):end]
                buffer = buffer[end + len(close_tag):]

                if before:
                    yield {"event": {"type": "message", "data": {"content": before}}}

                details = f"\n<details><summary>Thinking</summary>\n\n{thinking_content.strip()}\n\n</details>\n"
                yield {"event": {"type": "message", "data": {"content": details}}}

            # Case 2: Found start but not full block — wait for more
            elif start != -1:
                break

            # Case 3: No <thinking> tag — stream all, keep last few chars in buffer
            else:
                # Don't flush too much in case tag is split
                flush_len = max(0, len(buffer) - len(open_tag))
                if flush_len > 0:
                    yield {"event": {"type": "message", "data": {"content": buffer[:flush_len]}}}
                    buffer = buffer[flush_len:]
                break

    # Flush any leftovers
    if buffer.strip():
        yield {"event": {"type": "message", "data": {"content": buffer}}}

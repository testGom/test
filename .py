    def _emit_collapsible_html(self, inner_text: str) -> str:
        # Keep it simple; OWUI must allow <details>/<summary> tags
        return (
            "<details>\n"
            "<summary>Click to expand thoughts</summary>\n"
            f"{inner_text}\n"
            "</details>"
        )

    def _process_stream_chunk(self, chunk: str) -> List[dict]:
        """
        Consume a chunk, return a list of 'message' events to emit immediately.
        Thought content is buffered until closing tag.
        """
        events: List[dict] = []
        self._buf += chunk

        # Build tag patterns (case-sensitive by default; change re.IGNORECASE if needed)
        open_tag = f"<{self.valves.thought_tag}>"
        close_tag = f"</{self.valves.thought_tag}>"

        # We’ll greedily process any complete segments we can see in the buffer
        while True:
            if self._in_thought:
                # Look for the closing tag
                close_idx = self._buf.find(close_tag)
                if close_idx == -1:
                    # No close yet; stash everything and keep waiting
                    self._thought_acc.append(self._buf)
                    self._buf = ""
                    break
                # capture thought until close tag
                self._thought_acc.append(self._buf[:close_idx])
                # drop the consumed part + closing tag
                self._buf = self._buf[close_idx + len(close_tag):]
                self._in_thought = False

                # Emit collapsible (or silently drop if you prefer to hide without disclosure)
                if self.valves.use_collapsible:
                    html = self._emit_collapsible_html("".join(self._thought_acc))
                    events.append({"event": {"type": "message", "data": {"content": html}}})
                # reset accumulator
                self._thought_acc = []
                # continue loop in case more plaintext follows
                continue

            # not in thought: look for an opening tag
            open_idx = self._buf.find(open_tag)
            if open_idx == -1:
                # No open tag in buffer → emit as normal output, but keep a small tail
                if self._buf:
                    # To avoid breaking a tag across token boundaries, keep a small tail
                    tail_keep = max(len(open_tag), len(close_tag)) - 1
                    emit_upto = max(0, len(self._buf) - tail_keep)
                    if emit_upto > 0:
                        text = self._buf[:emit_upto]
                        if text:
                            events.append({"event": {"type": "message", "data": {"content": text}}})
                        self._buf = self._buf[emit_upto:]
                break
            else:
                # Emit anything before the tag as normal content
                if open_idx > 0:
                    pre = self._buf[:open_idx]
                    if pre:
                        events.append({"event": {"type": "message", "data": {"content": pre}}})
                # Enter thought mode and drop the opening tag
                self._buf = self._buf[open_idx + len(open_tag):]
                self._in_thought = True
                # loop to look for a close tag in same buffer iteration
                continue

        return events


                "data": {"description": "Recherche dans les documents...", "done": False},
            }
        }

        # Build the query engine (streaming)
        query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=3)
        response = query_engine.query(user_message)

        # Stream tokens, but filter/convert <think> ... </think> into <details>
        for token in response.response_gen:
            for ev in self._process_stream_chunk(token):
                yield ev

        # After the generator ends, flush any leftover text (non-thought)
        if not self._in_thought and self._buf:
            yield {"event": {"type": "message", "data": {"content": self._buf}}}
            self._buf = ""

        # If the stream ended while still inside a thought, you can decide to:
        # - emit nothing (hide incomplete thought), or
        # - emit a collapsible anyway. Here we’ll just drop it silently.
        self._thought_acc = []
        self._in_thought = False

        # Citations
        for node in response.source_nodes:

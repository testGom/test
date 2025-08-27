"""
title: Show Source as Artifact
author: me
version: 1.0.0
required_open_webui_version: 0.5.0
"""

from typing import Optional
from pydantic import BaseModel

HTML_SNIPPET = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Source Preview</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 1.25rem; line-height: 1.5; }
      .src { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }
      h2 { margin: 0 0 .5rem 0; }
      small { color: #666; }
    </style>
  </head>
  <body>
    <div class="src">
      <h2>Source Preview</h2>
      <small>Rendered as an Artifact</small>
      <p>Hello</p>
      <p>Test.</p>
    </div>
  </body>
</html>"""

class Action:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        await __event_call__({
            "type": "message",
            "data": {
                "content": HTML_SNIPPET
            },
        })

        # OPTION 2 (alternative): Replace the current message entirely with the artifact HTML
        # await __event_call__({
        #     "type": "replace",
        #     "data": {
        #         "content": HTML_SNIPPET
        #     },
        # })

        return {"ok": True}

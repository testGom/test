import json
import markdown
from pathlib import Path

INPUT_FILE = "responses.jsonl"
OUTPUT_FILE = "responses.html"

def jsonl_to_html(jsonl_path):
    md = markdown.Markdown()
    html_entries = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("prompt", "").strip()
            response = item.get("response", "").strip()
            rendered_response = md.convert(response)

            html_entries.append(f"""
                <div class="entry">
                    <h3>Prompt</h3>
                    <pre>{prompt}</pre>
                    <h3>Response</h3>
                    <div class="response">{rendered_response}</div>
                    <hr>
                </div>
            """)
            md.reset()  # reset between documents

    return html_entries

def build_html(entries):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM Responses</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 1em; }}
        pre {{ background: #f4f4f4; padding: 0.5em; white-space: pre-wrap; word-wrap: break-word; }}
        .response {{ background: #fafafa; padding: 0.5em; border-left: 4px solid #ccc; }}
        hr {{ margin: 2em 0; }}
    </style>
</head>
<body>
    <h1>Model Responses</h1>
    {''.join(entries)}
</body>
</html>
"""

def main():
    entries = jsonl_to_html(INPUT_FILE)
    html = build_html(entries)
    Path(OUTPUT_FILE).write_text(html, encoding="utf-8")
    print(f"✅ HTML written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

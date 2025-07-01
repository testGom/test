from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b-it-qat"

@app.route("/ask", methods=["POST"])
def ask():
    user_prompt = request.json.get("prompt", "")
    response_text = ""

    with requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": user_prompt},
        stream=True
    ) as r:
        for line in r.iter_lines():
            if line:
                try:
                    data = line.decode("utf-8")
                    response_text += data
                except:
                    pass

    # You can parse and extract just the "response" field if needed
    return jsonify({"full_output": response_text})

if __name__ == "__main__":
    app.run(port=5005)



Function CallOllamaProxy(prompt As String) As String
    Dim http As Object
    Dim url As String
    Dim requestBody As String
    Dim responseText As String

    On Error GoTo ErrHandler

    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")

    url = "http://localhost:5005/ask"
    requestBody = "{""prompt"":""" & EscapeJson(prompt) & """}"

    http.Open "POST", url, False
    http.SetRequestHeader "Content-Type", "application/json"
    http.Send requestBody

    responseText = http.ResponseText
    Debug.Print "Raw proxy response: " & responseText

    CallOllamaProxy = responseText  ' You can parse JSON here if needed
    Exit Function

ErrHandler:
    CallOllamaProxy = "Error: " & Err.Description
End Function

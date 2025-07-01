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


                                
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get("prompt", "")
        if not prompt:
            return Response("Error: No prompt provided", status=400)

        # Collect full streamed response from Ollama
        result = ""
        with requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt},
            stream=True
        ) as r:
            for line in r.iter_lines():
                if line:
                    line_data = json.loads(line.decode("utf-8"))
                    result += line_data.get("response", "")

        # Return only the response text (no JSON)
        return Response(result.strip(), mimetype="text/plain")

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)

Function CallIncident(anomaliesRange As Range, incidentRange As Range, headerRange As Range) As String
    Dim anomalyText As String
    Dim prompt As String
    Dim row As Range
    Dim cell As Range
    Dim sep As String: sep = " | "

    ' Step 1: Build anomaly context
    anomalyText = ""
    For Each row In anomaliesRange.Rows
        Dim rowText As String: rowText = ""
        For Each cell In row.Cells
            rowText = rowText & cell.Text & sep
        Next cell
        anomalyText = anomalyText & Left(rowText, Len(rowText) - Len(sep)) & vbCrLf
    Next row

    ' Step 2: Build structured incident context using explicit header range
    Dim headers() As Variant
    Dim values() As Variant
    Dim i As Long
    Dim incidentText As String: incidentText = ""
    
    headers = headerRange.Value
    values = incidentRange.Value

    For i = 1 To incidentRange.Columns.Count
        incidentText = incidentText & headers(1, i) & ": " & values(1, i) & vbCrLf
    Next i

    ' Step 3: Build final prompt
    prompt = "You are a system that maps IT incidents to known anomalies for faster root-cause identification." & vbCrLf & _
             "Below is a list of known anomalies. Each line represents one anomaly and includes relevant context." & vbCrLf & _
             "Then, a full incident record is provided with structured fields." & vbCrLf & _
             "Return the most relevant anomaly IDs, along with short justifications." & vbCrLf & vbCrLf & _
             "### Anomalies ###" & vbCrLf & anomalyText & vbCrLf & _
             "### Incident ###" & vbCrLf & incidentText & vbCrLf & _
             "### Output ###"

    ' Step 4: Send to LLM proxy
    CallIncident = CallOllamaProxy(prompt)
End Function

End Function


                                

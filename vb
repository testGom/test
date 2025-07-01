Function CallOllama(prompt As String) As String
    Dim http As Object
    Dim url As String
    Dim requestBody As String
    Dim responseText As String

    On Error GoTo ErrHandler

    ' Initialize HTTP object
    Set http = CreateObject("MSXML2.XMLHTTP")

    ' Endpoint for Ollama
    url = "http://localhost:11434/api/generate"

    ' JSON body with prompt and model name (change "llama3" if needed)
    requestBody = "{""model"":""llama3"",""prompt"":""" & EscapeJson(prompt) & """}"

    ' Send POST request
    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "application/json"
    http.Send requestBody

    ' Parse response (assuming JSON with field "response")
    responseText = http.responseText
    CallOllama = ParseOllamaResponse(responseText)
    Exit Function

ErrHandler:
    CallOllama = "Error: " & Err.Description
End Function

' Helper to escape quotes and backslashes in JSON
Function EscapeJson(text As String) As String
    text = Replace(text, "\", "\\")
    text = Replace(text, """", "\""")
    EscapeJson = text
End Function

' Basic parser to extract the response field (simple but works)
Function ParseOllamaResponse(json As String) As String
    Dim startPos As Long, endPos As Long
    startPos = InStr(json, """response"":""")
    If startPos = 0 Then
        ParseOllamaResponse = "No response"
        Exit Function
    End If
    startPos = startPos + Len("""response"":""")
    endPos = InStr(startPos, json, """}")
    ParseOllamaResponse = Mid(json, startPos, endPos - startPos)
End Function

Function CallOllama(prompt As String) As String
    Dim http As Object
    Dim url As String
    Dim requestBody As String
    Dim responseText As String
    Dim parsedResponse As String

    On Error GoTo ErrHandler

    Debug.Print "Starting CallOllama with prompt: " & prompt

    ' Initialize HTTP object
    Set http = CreateObject("MSXML2.XMLHTTP")

    url = "http://localhost:11434/api/generate"
    requestBody = "{""model"":""llama3"",""prompt"":""" & EscapeJson(prompt) & """}"

    Debug.Print "Sending request to: " & url
    Debug.Print "Request body: " & requestBody

    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "application/json"
    http.Send requestBody

    responseText = http.responseText
    Debug.Print "Raw response: " & responseText

    parsedResponse = ParseOllamaResponse(responseText)
    Debug.Print "Parsed response: " & parsedResponse

    CallOllama = parsedResponse
    Exit Function

ErrHandler:
    Debug.Print "Error occurred: " & Err.Description
    CallOllama = "Error: " & Err.Description
End Function

Function EscapeJson(text As String) As String
    text = Replace(text, "\", "\\")
    text = Replace(text, """", "\""")
    EscapeJson = text
End Function

Function ParseOllamaResponse(json As String) As String
    Dim startPos As Long, endPos As Long

    On Error GoTo ParseError

    startPos = InStr(json, """response"":""")
    If startPos = 0 Then
        Debug.Print "Could not find 'response' field in JSON"
        ParseOllamaResponse = "Invalid response format"
        Exit Function
    End If

    startPos = startPos + Len("""response"":""")
    endPos = InStr(startPos, json, """}")
    ParseOllamaResponse = Mid(json, startPos, endPos - startPos)
    Exit Function

ParseError:
    Debug.Print "Error parsing JSON: " & Err.Description
    ParseOllamaResponse = "Parse error: " & Err.Description
End Function

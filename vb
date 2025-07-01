Attribute VB_Name = "Module1"

Public Function CallOllama(ByVal prompt As String, Optional ByVal model As String = "llama2", Optional ByVal ollamaUrl As String = "http://localhost:11434/api/generate") As String

    Dim winHttpReq As Object
    Dim jsonRequest As String
    Dim jsonResponse As String
    Dim responseText As String
    Dim startPos As Long
    Dim endPos As Long
    Dim i As Long
    Dim parsedJson As Object

    If Trim(prompt) = "" Then
        CallOllama = "Error: Prompt cannot be empty."
        Exit Function
    End If

    On Error GoTo ErrorHandler
    Set winHttpReq = CreateObject("WinHttp.WinHttpRequest.5.1")

    jsonRequest = "{""model"": """ & model & """, ""prompt"": """ & Replace(prompt, """", "\""") & """, ""stream"": false}"

    Debug.Print "Ollama URL: " & ollamaUrl
    Debug.Print "JSON Request: " & jsonRequest

    With winHttpReq
        .Open "POST", ollamaUrl, False
        .SetRequestHeader "Content-Type", "application/json"
        .Send jsonRequest
        .WaitForResponse
    End With

    If winHttpReq.Status = 200 Then
        jsonResponse = winHttpReq.ResponseText
        Debug.Print "Raw JSON Response: " & jsonResponse

        startPos = InStr(jsonResponse, """response"":""")
        If startPos > 0 Then
            startPos = startPos + Len("""response"":""")

            endPos = InStr(startPos, jsonResponse, """")
            If endPos > 0 Then
                responseText = Mid(jsonResponse, startPos, endPos - startPos)
                responseText = Replace(responseText, "\""", """")
                CallOllama = responseText
            Else
                CallOllama = "Error: Could not parse 'response' field from Ollama JSON (missing closing quote)."
            End If
        Else
            CallOllama = "Error: 'response' field not found in Ollama JSON. Raw response: " & jsonResponse
        End If

    Else
        CallOllama = "Error: Ollama API call failed with status " & winHttpReq.Status & ". Response: " & winHttpReq.ResponseText
    End If

    GoTo CleanUp

ErrorHandler:
    CallOllama = "Error: An unexpected error occurred during Ollama API call. " & Err.Description
    Debug.Print "Runtime Error: " & Err.Number & " - " & Err.Description

CleanUp:
    Set winHttpReq = Nothing
    Set parsedJson = Nothing

End Function

Sub TestCallOllama()
    Dim result As String
    Dim testPrompt As String

    testPrompt = "What is the capital of France?"
    result = CallOllama(testPrompt, "llama2")

    MsgBox "Prompt: " & testPrompt & vbCrLf & vbCrLf & "Ollama Response: " & result, vbInformation, "Ollama Test"

    testPrompt = "Tell me a short story about a brave knight."
    result = CallOllama(testPrompt, "llama2")
    MsgBox "Prompt: " & testPrompt & vbCrLf & vbCrLf & "Ollama Response: " & result, vbInformation, "Ollama Test"

End Sub

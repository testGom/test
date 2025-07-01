Function CallOllama(prompt As String) As String
    Dim http As Object
    Dim url As String
    Dim requestBody As String
    Dim responseText As String
    Dim startTime As Double

    On Error GoTo ErrHandler

    Debug.Print "== Start CallOllama =="
    startTime = Timer

    ' Use WinHttp for better timeout control
    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")
    
    url = "http://127.0.0.1:11434/api/generate"
    requestBody = "{""model"":""gemma3:1b-it-qat"",""prompt"":""" & EscapeJson(prompt) & """, ""stream"": false}"

    Debug.Print "[Request Body]: " & requestBody
    Debug.Print "[Sending request to]: " & url

    ' Set a reasonable timeout (in seconds)
    http.SetTimeouts 5000, 5000, 30000, 30000 ' DNS, connect, send, receive
    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "application/json"

    Debug.Print "[Sending at]: " & Format(Now, "hh:nn:ss")
    http.Send requestBody

    Debug.Print "[Response Status]: " & http.Status
    Debug.Print "[Raw Response]: " & http.ResponseText
    Debug.Print "[Duration]: " & Format(Timer - startTime, "0.00") & " seconds"

    If http.Status = 200 Then
        CallOllama = ParseOllamaResponse(http.ResponseText)
    Else
        CallOllama = "HTTP Error: " & http.Status
    End If
    Exit Function

ErrHandler:
    Debug.Print "[ERROR]: " & Err.Description
    CallOllama = "Error: " & Err.Description
End Function

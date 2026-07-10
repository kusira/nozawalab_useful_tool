Option Explicit

Dim fso, shell, dir, mainPy, errNum

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
mainPy = fso.BuildPath(dir, "main.py")

If Not fso.FileExists(mainPy) Then
    MsgBox "main.py が見つかりません。" & vbCrLf & mainPy, vbCritical, "image_viewer"
    WScript.Quit 1
End If

shell.CurrentDirectory = dir

On Error Resume Next
shell.Run "pyw -3 """ & mainPy & """", 0, False
errNum = Err.Number
Err.Clear
If errNum = 0 Then WScript.Quit 0

shell.Run "pythonw """ & mainPy & """", 0, False
errNum = Err.Number
Err.Clear
If errNum = 0 Then WScript.Quit 0

shell.Run "py -3 """ & mainPy & """", 0, False
errNum = Err.Number
Err.Clear
If errNum = 0 Then WScript.Quit 0

shell.Run "python """ & mainPy & """", 0, False
errNum = Err.Number
Err.Clear
If errNum = 0 Then WScript.Quit 0

On Error GoTo 0

MsgBox "Python が見つかりません。" & vbCrLf & vbCrLf & "Python 3.10 以上をインストールし、PATH に追加してください。", vbCritical, "image_viewer"
WScript.Quit 1
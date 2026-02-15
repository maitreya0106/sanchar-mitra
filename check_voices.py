import subprocess
# Check both Desktop (SAPI5) and OneCore voices
result = subprocess.run(
    ['powershell', '-Command', '''
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
Write-Output "=== SAPI5 Desktop Voices ==="
foreach ($v in $s.GetInstalledVoices()) {
    Write-Output ("  " + $v.VoiceInfo.Name + " | " + $v.VoiceInfo.Gender + " | " + $v.VoiceInfo.Culture)
}
Write-Output ""
Write-Output "=== OneCore Voices (from Settings) ==="
$keys = Get-ChildItem "HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens" -ErrorAction SilentlyContinue
if ($keys) {
    foreach ($k in $keys) {
        $name = (Get-ItemProperty $k.PSPath).'(default)'
        if ($name) { Write-Output ("  " + $name) }
        else { Write-Output ("  " + $k.PSChildName) }
    }
} else {
    Write-Output "  (none found)"
}
'''],
    capture_output=True, text=True, timeout=10
)
print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)

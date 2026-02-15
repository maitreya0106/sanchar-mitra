param([string]$text = "Hello, this is a test")

$voice = New-Object -ComObject SAPI.SpVoice
$cat = New-Object -ComObject SAPI.SpObjectTokenCategory
$cat.SetId("HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices", $false)

$found = $false
foreach ($t in $cat.EnumerateTokens()) {
    $desc = $t.GetDescription()
    if ($desc -match "Heera") {
        $voice.Voice = $t
        $voice.Rate = -2
        $voice.Speak($text)
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Error "Heera voice not found in OneCore voices"
    exit 1
}

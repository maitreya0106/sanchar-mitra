# Run this script as Administrator to make OneCore voices available to SAPI5
# OneCore voices (like Heera) are installed via Settings but live in a different registry location.
# This copies them to the SAPI5 location so System.Speech.Synthesis can use them.

$source = "HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens"
$dest   = "HKLM:\SOFTWARE\Microsoft\Speech\Voices\Tokens"

$copied = 0
Get-ChildItem $source | ForEach-Object {
    $tokenDest = "$dest\$($_.PSChildName)"
    if (-not (Test-Path $tokenDest)) {
        Copy-Item -Path $_.PSPath -Destination $tokenDest -Recurse
        Write-Host "Copied: $($_.PSChildName)" -ForegroundColor Green
        $copied++
    } else {
        Write-Host "Already exists: $($_.PSChildName)" -ForegroundColor Yellow
    }
}

if ($copied -eq 0) {
    Write-Host "`nNo new voices to copy — all already registered." -ForegroundColor Cyan
} else {
    Write-Host "`nDone! Copied $copied voice(s). Heera should now work in SAPI5." -ForegroundColor Green
}

Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

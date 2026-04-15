# RAG Client - Invia un prompt al sistema RAG su localhost:8001

param(
    [Parameter(Mandatory = $false)]
    [string]$Prompt
)

# Se il prompt non è passato come parametro, chiedi input interattivo
if (-not $Prompt) {
    $Prompt = Read-Host "Inserisci il prompt"
}

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    Write-Error "Il prompt non può essere vuoto."
    exit 1
}

# Costruisci il body JSON
$Body = @{
    prompt = $Prompt
} | ConvertTo-Json -Depth 2

$Uri = "http://localhost:8001/process"

Write-Host "`nInvio richiesta a $Uri ..." -ForegroundColor Cyan
Write-Host "Payload: $Body" -ForegroundColor DarkGray

try {
    $Response = Invoke-RestMethod `
        -Uri $Uri `
        -Method Post `
        -ContentType "application/json" `
        -Body $Body `
        -ErrorAction Stop

    Write-Host "`nRisposta del sistema RAG:" -ForegroundColor Green
    $Response | ConvertTo-Json -Depth 10
}
catch {
    $StatusCode = $_.Exception.Response.StatusCode.value__
    $ErrorMessage = $_.Exception.Message

    Write-Host "`nErrore nella chiamata al sistema RAG!" -ForegroundColor Red
    Write-Host "Status code : $StatusCode" -ForegroundColor Yellow
    Write-Host "Dettaglio   : $ErrorMessage" -ForegroundColor Yellow
}
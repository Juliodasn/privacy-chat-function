param(
  [Parameter(Mandatory = $true)] [string] $ResourceGroup,
  [Parameter(Mandatory = $true)] [string] $FunctionAppName,
  [Parameter(Mandatory = $true)] [string] $AzureOpenAIEndpoint,
  [Parameter(Mandatory = $true)] [string] $AzureOpenAIApiKey,
  [string] $AzureOpenAIApiVersion = "2024-06-01",
  [Parameter(Mandatory = $true)] [string] $ChatDeployment,
  [Parameter(Mandatory = $true)] [string] $EmbeddingDeployment
)

$settings = @(
  "RAG_ENABLED=true",
  "RAG_ROLLOUT_STAGE=shadow",
  "RAG_TOP_K=4",
  "RAG_EVAL_MODEL=$ChatDeployment",
  "RAG_FALLBACK_TO_OLD_LLM=true",
  "RAG_ONLY_FILL_MISSING=true",
  "AZURE_OPENAI_ENDPOINT=$AzureOpenAIEndpoint",
  "AZURE_OPENAI_API_KEY=$AzureOpenAIApiKey",
  "AZURE_OPENAI_API_VERSION=$AzureOpenAIApiVersion",
  "AZURE_OPENAI_CHAT_DEPLOYMENT=$ChatDeployment",
  "AZURE_OPENAI_EMBEDDING_DEPLOYMENT=$EmbeddingDeployment"
)

Write-Host "Aplicando Application Settings em $FunctionAppName ..."
az functionapp config appsettings set `
  --resource-group $ResourceGroup `
  --name $FunctionAppName `
  --settings $settings

if ($LASTEXITCODE -ne 0) {
  Write-Error "Falha ao aplicar app settings."
  exit 1
}

Write-Host "Settings aplicadas. Reinicie a Function App se necessario."

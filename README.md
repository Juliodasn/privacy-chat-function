# PrivacyPoint Site - API do Chat

## Configuração de Ambiente

### Pré-requisitos

- Python 3.10 ou superior
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes e ambientes
  virtuais)
- [Azure Functions Core Tools](https://docs.microsoft.com/pt-br/azure/azure-functions/functions-run-local?tabs=v4%2Cwindows%2Ccsharp%2Cportal%2Cbash)
- Uma conta na OpenAI com acesso à API

### Configuração do Ambiente com `uv`

1. Clone o repositório:
   ```
   git clone https://github.com/hellmrf/privacy-chat-function
   cd privacy-chat-function
   ```

2. Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis (ou
   configure as variáveis de ambiente):
   ```
   OPENAI_API_KEY=sua_chave_api_da_openai
   OPENAI_ASSISTANT_ID=id_do_seu_assistente_openai
   THREAD_COOKIE_NAME=pp_thread_id
   # Origem do front (ajuste conforme ambiente)
   # Homolog: https://homologa.privacypoint.com.br
   CORS_ALLOW_ORIGIN=https://homologa.privacypoint.com.br
   CORS_ALLOW_CREDENTIALS=true
   ```

3. Execute a função localmente para testes:
   ```
   uv run func start
   ```

## Executando Localmente

Para executar a função localmente:

1. Execute o comando:
   ```
   uv run func start
   ```
   - Para testes locais com cookies, defina `CORS_ALLOW_ORIGIN=http://localhost:3000` no `.env`.

## Deploy (Azure Functions)

- Em cada Function App (homolog e produ��o), em **Configuration > Application settings**, configure:
  - `OPENAI_API_KEY`
  - `OPENAI_ASSISTANT_ID`
  - `THREAD_COOKIE_NAME=pp_thread_id`
  - `CORS_ALLOW_ORIGIN=https://homologa.privacypoint.com.br` (ajuste para o dom��nio do front em produ��o)
  - `CORS_ALLOW_CREDENTIALS=true`
- Em **CORS** no portal Azure, adicione a mesma origem (`https://homologa.privacypoint.com.br`) e remova `*`.
- Salve e reinicie a Function App se necess��rio.

## Estrutura do Projeto

- `function_app.py`: Contém as definições das rotas de API.
- `src/api.py`: Implementa a lógica principal de interação com a API da OpenAI.
- `src/defs/`: Contém definições de tipos, erros e respostas.

## Desenvolvimento

Para desenvolvimento, recomenda-se usar o Visual Studio Code com as extensões
Azure Functions e Python.

## RAG Rollout (Avaliacao Data Mapping)

Feature flags principais:
- `RAG_ENABLED` (default: `false`)
- `RAG_ONLY_FILL_MISSING` (default: `true`)
- `RAG_FALLBACK_TO_OLD_LLM` (default: `true`)
- `RAG_ROLLOUT_STAGE` (opcional: `manual|legacy|shadow|hybrid|rag_primary`)
- `RAG_AUTO_CUTOVER_DATE` (opcional, formato `YYYY-MM-DD`; desliga fallback ao atingir a data)
- `RAG_TOP_K` (default: `4`)
- `RAG_QUESTION_BATCH_SIZE` (default: `6`, faixa recomendada `2-8`)
- `RAG_CONTEXT_MAX_CHARS` (default: `12000`)
- `RAG_CONTEXT_MAX_TOKENS` (default: `3000`)
- `RAG_EVAL_MODEL` (default: `OPENAI_MODEL`)

Rollout sugerido:
1. `RAG_ENABLED=true` e `RAG_FALLBACK_TO_OLD_LLM=true` para operar com fallback.
2. Validar divergencias em amostra real usando o script abaixo.
3. Quando estabilizar, manter `RAG_ENABLED=true` e trocar para `RAG_FALLBACK_TO_OLD_LLM=false`.

Comparacao old vs RAG (10 arquivos):
```bash
python Backend/scripts/compare_old_vs_rag.py "C:\\caminho\\amostra" --limit 10 --top-k 4
```

Teste automatizado RAG (local/CI):
```bash
python -m unittest discover -s Backend/tests -p "test_*.py" -v
```

Validacao de config/deployments (chat + embedding):
```bash
python Backend/scripts/validate_rag_config.py --local-settings Backend/local.settings.json
```

Aplicar settings RAG na Function App (Azure):
```powershell
.\Backend\scripts\set_azure_rag_appsettings.ps1 `
  -ResourceGroup "<rg>" `
  -FunctionAppName "<function-app>" `
  -AzureOpenAIEndpoint "https://<recurso>.openai.azure.com/" `
  -AzureOpenAIApiKey "<key>" `
  -ChatDeployment "<deployment-chat>" `
  -EmbeddingDeployment "<deployment-embedding>"
```

Check rapido de erros comuns:
- `401/403`: chave invalida ou sem permissao no recurso Azure OpenAI.
- `404`: deployment de chat/embedding incorreto.
- `400`: `AZURE_OPENAI_API_VERSION` incompativel com o endpoint/deployment.

## Respondibilidade (QA Blocks + Hybrid Retrieval)

Ordem de decisao por pergunta:
1. `qa_block` (parser deterministico por ancoras)
2. `rag` (embedding + lexical rerank)
3. `col_llm` (regras por coluna/LLM legado)

Campos novos no `debug` da resposta:
- `qa_blocks.found`
- `qa_blocks.anchors_found`
- `retrieval.<secao>.<question_code>.before_rerank`
- `retrieval.<secao>.<question_code>.after_rerank`

Campos por pergunta (`debug.per_question`):
- `answerable`
- `answerable_source` (`qa_block|rag|col_llm`)
- `evidence`
- `used_chunk_ids`
- `reason` (quando `answerable=0`)

Hardening de respondibilidade (zero falso-positivo):
- `answerable=1` so quando existe evidencia explicita + rastreavel.
- Evidencia que e pergunta/header/label ou texto generico e descartada.
- Em `STRICT_MODE=true`, RAG nao pode auto-atribuir chunk quando o modelo nao enviar `used_chunk_ids`.
- Em RAG, evidencia precisa existir literalmente no texto do chunk usado.

Variaveis de ambiente uteis para rerank:
- `RAG_RETRIEVAL_EMBED_TOP_K` (default: `12`)
- `RAG_RETRIEVAL_ALPHA` (default: `0.65`)

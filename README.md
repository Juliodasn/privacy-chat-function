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

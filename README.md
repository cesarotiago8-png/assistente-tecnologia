# 🤖 Assistente de Tecnologia (GenAI + RAG)

Um assistente de IA que responde perguntas **com base em documentos**, cita a fonte e evita inventar respostas. Construído com uma API de LLM (Google Gemini) e a técnica de **RAG (Retrieval-Augmented Generation)**.

**▶️ App no ar:** https://assistente-tecnologia-mjy6vunyrjyike6bz4rgwa.streamlit.app/

---

## O que ele faz

Diferente de um chatbot genérico, que responde "de cabeça" e às vezes inventa (alucina), este assistente responde **grudado em documentos**: ele busca os trechos mais relevantes no material fornecido, responde com base neles e mostra **de onde tirou a resposta**. Se a pergunta não está nos documentos, ele diz honestamente *"não encontrei isso nos documentos"* em vez de inventar.

Analogia: um chatbot comum é uma pessoa respondendo de memória; este assistente é a mesma pessoa, mas **com o livro certo aberto na frente**, respondendo pela página e apontando o parágrafo.

Ele já vem com documentos de exemplo sobre tecnologia, e qualquer pessoa pode **subir o próprio PDF** para perguntar sobre ele.

## Como usar

1. Abra o [app](https://assistente-tecnologia-mjy6vunyrjyike6bz4rgwa.streamlit.app/).
2. Faça uma pergunta no chat (ex.: *"O que é RAG?"*, *"Quais cargos de IA estão em alta?"*).
3. (Opcional) Suba um PDF seu e pergunte sobre o conteúdo dele.

> A primeira abertura pode levar alguns segundos (o app conecta na API e indexa os exemplos), e o serviço gratuito "hiberna" quando fica sem uso.

## Como funciona (RAG) — passo a passo

1. **Documentos** → o material que o assistente conhece (exemplos embutidos ou um PDF enviado).
2. **Pedaços + embeddings** → o texto é quebrado em trechos, e cada trecho vira um vetor de números que representa seu significado (indexação, feita uma vez).
3. **Pergunta** → a pergunta do usuário também vira um vetor.
4. **Busca** → o sistema acha os trechos mais parecidos com a pergunta (similaridade entre vetores).
5. **Resposta com fonte** → o LLM recebe os trechos encontrados + a pergunta, com a instrução de responder **somente** com base neles, e cita a fonte.

## Stack

| Camada | Ferramenta |
| --- | --- |
| Linguagem | Python |
| LLM | Google Gemini (API, `google-genai`) |
| Embeddings | API de embeddings do Gemini |
| Leitura de PDF | pypdf |
| Busca por similaridade | NumPy |
| Interface web | Streamlit |
| Deploy | Streamlit Community Cloud |

## Destaques de engenharia

- **Grounding (anti-alucinação):** o modelo é instruído a responder apenas com os trechos recuperados, reduzindo respostas inventadas.
- **Citação de fonte:** cada resposta mostra o trecho de origem, o que aumenta a confiança.
- **Robustez com a API:** *retry* automático quando a API está ocupada e *fallback* entre modelos (se um modelo estiver indisponível/sobrecarregado, tenta outro).
- **Segurança:** a chave da API fica em *secrets*, nunca no código.

## Como rodar localmente

```bash
git clone https://github.com/cesarotiago8-png/assistente-tecnologia.git
cd assistente-tecnologia
pip install -r requirements.txt
```

Crie o arquivo `.streamlit/secrets.toml` com sua chave do Gemini:

```toml
GEMINI_API_KEY = "sua_chave_aqui"
```

E rode:

```bash
streamlit run streamlit_app.py
```

> Pegue uma chave gratuita em https://aistudio.google.com/apikey (sem cartão de crédito).

## Limitações e próximos passos

Sendo honesto sobre o escopo atual:

- Os documentos de exemplo são curtos e os PDFs enviados existem **só durante a sessão** (não há armazenamento permanente).
- A busca usa similaridade simples; para bases grandes, um banco vetorial (ex.: FAISS) escalaria melhor.
- Suporta PDF e texto; outros formatos ficam de fora por enquanto.

**Próximos passos:** destacar o trecho exato usado na resposta, suportar vários documentos ao mesmo tempo, e melhorar a divisão em pedaços (chunking) respeitando parágrafos.

## Autor

**Tiago Cesaro** — estudante de Inteligência Artificial (FIAP) e estagiário de IA, com foco em Machine Learning e GenAI/LLM.

- LinkedIn: https://linkedin.com/in/cesaro-tiago
- GitHub: https://github.com/cesarotiago8-png

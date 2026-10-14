# 🤖 Assistente de Tecnologia (GenAI + RAG)

Um assistente de IA focado em tecnologia, construído com uma API de LLM (Google Gemini). Ele tem **dois modos**: responde sobre tecnologia em geral (opcionalmente pesquisando na web, com fontes) e — quando você sobe um PDF — responde **com base nesse documento**, citando o trecho de origem e evitando inventar (técnica de **RAG, Retrieval-Augmented Generation**). Também **lembra do contexto da conversa** para responder perguntas de acompanhamento.

**▶️ App no ar:** https://assistente-tecnologia-mjy6vunyrjyike6bz4rgwa.streamlit.app/

---

## O que ele faz

O assistente tem dois modos, e cada um tem um trabalho diferente:

🌐 **Modo web** (quando nenhum documento está carregado) — responde sobre tecnologia, IA, programação e mercado de trabalho em tech. Por padrão responde de forma rápida, usando o conhecimento do próprio modelo. Se você marcar **"🔎 Pesquisar na web"** na barra lateral, ele busca na internet para trazer informação mais atual e mostra as **fontes** que usou. Esse modo *deve* responder amplamente, inclusive sobre coisas que não estão em nenhum documento — é para isso que ele existe.

📄 **Modo documento (RAG)** (quando você sobe um PDF) — responde **grudado no documento**: busca os trechos mais relevantes do material, responde com base neles e mostra **de onde tirou a resposta**. Se a pergunta não está no documento, ele diz honestamente *"não encontrei isso no documento"* em vez de inventar.

Analogia: no modo documento, é como uma pessoa respondendo **com o livro certo aberto na frente**, respondendo pela página e apontando o parágrafo — em vez de responder de memória e correr o risco de errar.

Os dois comportamentos são propositais: o modo web responde de cabeça (e, se você pedir, pesquisando), o modo documento se segura ao material. A ideia é demonstrar *quando* uma IA deve responder livremente e *quando* ela deve se restringir à fonte.

🧠 **Memória de conversa** — em qualquer um dos modos, o assistente leva em conta as últimas mensagens da conversa. Assim você pode fazer perguntas de acompanhamento ("e sobre isso?", "me explica melhor a segunda parte") sem repetir o contexto toda vez.

## Como usar

1. Abra o [app](https://assistente-tecnologia-mjy6vunyrjyike6bz4rgwa.streamlit.app/).
2. Faça uma pergunta no chat sobre tecnologia (ex.: *"Qual a diferença entre Machine Learning e Deep Learning?"*). → **modo web** (resposta rápida, do conhecimento do modelo).
3. (Opcional) Marque **"🔎 Pesquisar na web"** na barra lateral se quiser que ele busque informação atual na internet e mostre as fontes. É mais lento, mas traz novidades e links.
4. (Opcional) Suba um PDF seu na barra lateral e pergunte sobre o conteúdo dele. → **modo documento (RAG).**

> A primeira abertura pode levar alguns segundos (o app conecta na API), e o serviço gratuito "hiberna" quando fica sem uso.

## Como funciona (RAG) — passo a passo

Esse é o fluxo do **modo documento**, que é a parte mais interessante tecnicamente:

1. **Documento** → o PDF que você sobe.
2. **Pedaços + embeddings** → o texto é quebrado em trechos, e cada trecho vira um vetor de números que representa seu significado (indexação, feita uma vez e guardada em cache).
3. **Pergunta** → a pergunta do usuário também vira um vetor.
4. **Busca** → o sistema acha os trechos mais parecidos com a pergunta (similaridade entre vetores).
5. **Resposta com fonte** → o LLM recebe os trechos encontrados + a pergunta, com a instrução de responder **somente** com base neles, e mostra o trecho de origem.

No **modo web**, o fluxo é mais direto: a pergunta vai para o modelo e, se a busca estiver ligada, com o **Google Search ativado** (grounding) — então a resposta vem acompanhada dos links das fontes consultadas. Nos dois modos, as últimas mensagens da conversa são enviadas junto como contexto.

## Avaliação

Para não ficar só no "confie em mim", medi o comportamento do **modo documento (RAG)** com um pequeno conjunto de testes. A ideia foi verificar duas coisas:

- quando a resposta **está** no material, ele responde; e
- quando a resposta **não está** no material (mesmo sendo algo que qualquer LLM "sabe de cabeça", como a capital da França), ele **se segura** e admite que não encontrou — em vez de alucinar.

Resultado: **6 de 6 casos corretos (100%)** — 4 perguntas dentro do material (respondidas) e 2 fora do material (recusadas corretamente com *"não encontrei isso no documento"*).

É um teste pequeno e proposital no escopo: ele mede o **comportamento de grounding do modo documento** (responder vs. recusar), não a qualidade fina da redação das respostas e não o modo web (cuja resposta é aberta e não tem um "certo" único). O valor dele é dar evidência concreta de que o anti-alucinação do RAG funciona.

## Stack

| Camada | Ferramenta |
| --- | --- |
| Linguagem | Python |
| LLM | Google Gemini (API, `google-genai`) |
| Busca na web | Grounding com Google Search (nativo do Gemini, opcional) |
| Embeddings | API de embeddings do Gemini |
| Leitura de PDF | pypdf |
| Busca por similaridade | NumPy |
| Interface web | Streamlit |
| Deploy | Streamlit Community Cloud |

## Destaques de engenharia

- **Dois modos no mesmo app:** modo web (geral, com busca na internet opcional + fontes) e RAG sobre documento (respostas grudadas no material), cada um para um tipo de pergunta.
- **Busca na web opcional:** por padrão o assistente responde rápido com o conhecimento do modelo; a busca na web (mais lenta, mas atual) fica atrás de um toggle, para equilibrar velocidade e atualidade.
- **Memória de conversa:** as últimas mensagens são enviadas como contexto, permitindo perguntas de acompanhamento sem repetir o assunto.
- **Grounding (anti-alucinação):** no modo documento, o modelo é instruído a responder apenas com os trechos recuperados, e isso foi **medido** (ver seção Avaliação).
- **Citação de fonte:** cada resposta mostra de onde veio — o trecho do documento (modo RAG) ou os links da web (modo web com busca ligada).
- **Persona:** o assistente se apresenta como o assistente de tecnologia do Tiago, não como um modelo genérico, graças a uma *system instruction*.
- **Robustez com a API:** *retry* automático com backoff quando a API está ocupada, *fallback* entre modelos (se um modelo estiver indisponível/sobrecarregado, tenta outro) e tratamento específico para o limite diário gratuito de embeddings — quando ele estoura, o app avisa com clareza e continua funcionando no modo web, em vez de quebrar.
- **Desempenho:** a indexação do PDF é guardada em cache (feita uma vez por documento) e a conexão com a API foi enxugada para reduzir o tempo da primeira resposta.
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

- Os PDFs enviados existem **só durante a sessão** (não há armazenamento permanente).
- A busca do modo documento usa similaridade simples; para bases grandes, um banco vetorial (ex.: FAISS) escalaria melhor.
- A avaliação cobre **o modo documento** e é pequena (6 casos) — mede comportamento, não qualidade fina da resposta.
- No plano gratuito do Gemini, a busca na web e os embeddings têm cotas apertadas (por minuto e por dia), o que pode deixar o app mais lento ou pedir para esperar.
- Suporta PDF; outros formatos ficam de fora por enquanto.

**Próximos passos:** expandir o conjunto de testes e avaliar também o modo web; destacar o trecho exato usado na resposta; suportar vários documentos ao mesmo tempo; melhorar a divisão em pedaços (chunking) respeitando parágrafos; e evoluir para um banco vetorial + API própria (FastAPI).

## Autor

**Tiago Cesaro** — estudante de Inteligência Artificial (FIAP) e estagiário de IA, com foco em Machine Learning e GenAI/LLM.

- LinkedIn: https://linkedin.com/in/cesaro-tiago
- GitHub: https://github.com/cesarotiago8-png

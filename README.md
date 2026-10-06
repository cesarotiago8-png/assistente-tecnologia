# 🤖 Assistente de Tecnologia (GenAI + RAG)

Um assistente de IA focado em tecnologia, construído com uma API de LLM (Google Gemini). Ele tem **dois modos**: responde sobre tecnologia em geral pesquisando na web (com fontes), e — quando você sobe um PDF — responde **com base nesse documento**, citando o trecho de origem e evitando inventar (técnica de **RAG, Retrieval-Augmented Generation**).

**▶️ App no ar:** https://assistente-tecnologia-mjy6vunyrjyike6bz4rgwa.streamlit.app/

---

## O que ele faz

O assistente tem dois modos, e cada um tem um trabalho diferente:

🌐 **Modo web** (quando nenhum documento está carregado) — responde sobre tecnologia, IA, programação e mercado de trabalho em tech, **pesquisando na internet** para trazer informação atual, e mostrando as **fontes** que usou. Esse modo *deve* responder amplamente, inclusive sobre coisas que não estão em nenhum documento — é para isso que ele existe.

📄 **Modo documento (RAG)** (quando você sobe um PDF) — responde **grudado no documento**: busca os trechos mais relevantes do material, responde com base neles e mostra **de onde tirou a resposta**. Se a pergunta não está no documento, ele diz honestamente *"não encontrei isso no documento"* em vez de inventar.

Analogia: no modo documento, é como uma pessoa respondendo **com o livro certo aberto na frente**, respondendo pela página e apontando o parágrafo — em vez de responder de memória e correr o risco de errar.

Os dois comportamentos são propositais: o modo web responde de cabeça (e pesquisando), o modo documento se segura ao material. A ideia é demonstrar *quando* uma IA deve responder livremente e *quando* ela deve se restringir à fonte.

## Como usar

1. Abra o [app](https://assistente-tecnologia-mjy6vunyrjyike6bz4rgwa.streamlit.app/).
2. Faça uma pergunta no chat sobre tecnologia (ex.: *"Qual a diferença entre Machine Learning e Deep Learning?"*, *"O que há de novo em IA recentemente?"*). → **modo web, com busca na internet.**
3. (Opcional) Suba um PDF seu na barra lateral e pergunte sobre o conteúdo dele. → **modo documento (RAG).**

> A primeira abertura pode levar alguns segundos (o app conecta na API), e o serviço gratuito "hiberna" quando fica sem uso.

## Como funciona (RAG) — passo a passo

Esse é o fluxo do **modo documento**, que é a parte mais interessante tecnicamente:

1. **Documento** → o PDF que você sobe.
2. **Pedaços + embeddings** → o texto é quebrado em trechos, e cada trecho vira um vetor de números que representa seu significado (indexação, feita uma vez).
3. **Pergunta** → a pergunta do usuário também vira um vetor.
4. **Busca** → o sistema acha os trechos mais parecidos com a pergunta (similaridade entre vetores).
5. **Resposta com fonte** → o LLM recebe os trechos encontrados + a pergunta, com a instrução de responder **somente** com base neles, e mostra o trecho de origem.

No **modo web**, o fluxo é mais direto: a pergunta vai para o modelo com a **busca do Google ligada** (grounding), e a resposta vem acompanhada dos links das fontes consultadas.

## Avaliação

Para não ficar só no "confie em mim", medi o comportamento do **modo documento (RAG)** com um pequeno conjunto de testes. A ideia foi verificar duas coisas:

- quando a resposta **está** no material, ele responde; e
- quando a resposta **não está** no material (mesmo sendo algo que qualquer LLM "sabe de cabeça", como a capital da França), ele **se segura** e admite que não encontrou — em vez de alucinar.

Resultado: **6 de 6 casos corretos (100%)** — 4 perguntas dentro do material (respondidas) e 2 fora do material (recusadas corretamente com *"não encontrei isso nos documentos"*).

É um teste pequeno e proposital no escopo: ele mede o **comportamento de grounding do modo documento** (responder vs. recusar), não a qualidade fina da redação das respostas e não o modo web (cuja resposta é aberta e não tem um "certo" único). O valor dele é dar evidência concreta de que o anti-alucinação do RAG funciona.

## Stack

| Camada | Ferramenta |
| --- | --- |
| Linguagem | Python |
| LLM | Google Gemini (API, `google-genai`) |
| Busca na web | Grounding com Google Search (nativo do Gemini) |
| Embeddings | API de embeddings do Gemini |
| Leitura de PDF | pypdf |
| Busca por similaridade | NumPy |
| Interface web | Streamlit |
| Deploy | Streamlit Community Cloud |

## Destaques de engenharia

- **Dois modos no mesmo app:** busca na web (informação atual + fontes) e RAG sobre documento (respostas grudadas no material), cada um para um tipo de pergunta.
- **Grounding (anti-alucinação):** no modo documento, o modelo é instruído a responder apenas com os trechos recuperados, e isso foi **medido** (ver seção Avaliação).
- **Citação de fonte:** cada resposta mostra de onde veio — o trecho do documento (modo RAG) ou os links da web (modo web).
- **Persona:** o assistente se apresenta como o assistente de tecnologia do Tiago, não como um modelo genérico, graças a uma *system instruction*.
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

- Os PDFs enviados existem **só durante a sessão** (não há armazenamento permanente).
- A busca do modo documento usa similaridade simples; para bases grandes, um banco vetorial (ex.: FAISS) escalaria melhor.
- A avaliação cobre **o modo documento** e é pequena (6 casos) — mede comportamento, não qualidade fina da resposta.
- Suporta PDF; outros formatos ficam de fora por enquanto.

**Próximos passos:** expandir o conjunto de testes e avaliar também o modo web; destacar o trecho exato usado na resposta; suportar vários documentos ao mesmo tempo; e melhorar a divisão em pedaços (chunking) respeitando parágrafos.

## Autor

**Tiago Cesaro** — estudante de Inteligência Artificial (FIAP) e estagiário de IA, com foco em Machine Learning e GenAI/LLM.

- LinkedIn: https://linkedin.com/in/cesaro-tiago
- GitHub: https://github.com/cesarotiago8-png

import time
from io import BytesIO

import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader

st.set_page_config(page_title="Assistente de Tecnologia", page_icon="\U0001F916")

# Documentos de exemplo: ja vem prontos pra qualquer um testar na hora
DOCUMENTOS_EXEMPLO = [
    "A IA generativa, como os modelos de linguagem (LLMs), gera texto novo a partir de um pedido (prompt). Ela aprende padroes de muito texto, mas pode errar ou inventar informacoes, o que se chama alucinacao.",
    "RAG (Retrieval-Augmented Generation) conecta um LLM a documentos externos. Antes de responder, o sistema busca os trechos mais relevantes e os envia ao modelo, que responde com base neles. Isso reduz alucinacoes e permite citar a fonte.",
    "O mercado de tecnologia valoriza cada vez mais profissionais de IA. Cargos como Engenheiro de Machine Learning e Engenheiro de GenAI/LLM estao em alta, e muitas vagas pedem Python, uso de APIs de modelos e deploy de aplicacoes.",
    "Python e a linguagem mais usada em ciencia de dados e IA, pela simplicidade e pelas bibliotecas como pandas, scikit-learn e as de LLMs. Costuma ser o primeiro requisito em vagas da area.",
    "Colocar um modelo em producao (deploy) e tao importante quanto treina-lo. O Streamlit permite criar uma interface web rapidamente, e nuvens gratuitas permitem publicar o projeto com um link publico.",
    "Embeddings sao vetores de numeros que representam o significado de um texto. Textos com significado parecido ficam com vetores proximos, o que permite buscar trechos relevantes comparando a similaridade.",
]


@st.cache_resource
def conectar():
    """Conecta no Gemini e escolhe um modelo de chat e um de embeddings que funcionem."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    modelo = None
    for nome in ["gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]:
        try:
            client.models.generate_content(model=nome, contents="oi")
            modelo = nome
            break
        except Exception:
            continue

    emb = None
    for nome in ["gemini-embedding-001", "gemini-embedding-2"]:
        try:
            client.models.embed_content(model=nome, contents=["teste"])
            emb = nome
            break
        except Exception:
            continue

    return client, modelo or "gemini-flash-lite-latest", emb or "gemini-embedding-001"


client, MODELO, EMB_MODELO = conectar()


def embed(textos):
    for _ in range(5):
        try:
            r = client.models.embed_content(model=EMB_MODELO, contents=textos)
            return [e.values for e in r.embeddings]
        except Exception:
            time.sleep(4)
    raise RuntimeError("API de embeddings ocupada, tente de novo.")


def similaridade(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def quebrar_em_pedacos(texto, tamanho=500):
    palavras = texto.split()
    pedacos, atual = [], ""
    for p in palavras:
        if len(atual) + len(p) > tamanho:
            pedacos.append(atual.strip())
            atual = ""
        atual += " " + p
    if atual.strip():
        pedacos.append(atual.strip())
    return pedacos


def responder(pergunta, documentos, vetores, k=2):
    emb_pergunta = embed([pergunta])[0]
    notas = [similaridade(emb_pergunta, v) for v in vetores]
    melhores = np.argsort(notas)[::-1][:k]
    trechos = [documentos[i] for i in melhores]
    contexto = "\n\n".join(trechos)

    prompt = f"""Responda a pergunta usando SOMENTE os trechos abaixo.
Se a resposta nao estiver neles, responda: "Nao encontrei isso nos documentos."

Trechos:
{contexto}

Pergunta: {pergunta}"""

    for _ in range(5):
        try:
            resposta = client.models.generate_content(model=MODELO, contents=prompt).text
            return resposta, trechos
        except Exception:
            time.sleep(4)
    return "A API esta ocupada agora, tente de novo em instantes.", trechos


@st.cache_data
def indexar_exemplos():
    return DOCUMENTOS_EXEMPLO, embed(DOCUMENTOS_EXEMPLO)


@st.cache_data
def indexar_pdf(conteudo_bytes):
    leitor = PdfReader(BytesIO(conteudo_bytes))
    texto = " ".join((pagina.extract_text() or "") for pagina in leitor.pages)
    pedacos = quebrar_em_pedacos(texto)
    return pedacos, embed(pedacos)


# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Sobre")
    st.write(
        "Assistente de IA que responde **com base em documentos** (técnica RAG), "
        "cita a fonte e evita inventar."
    )
    st.markdown(
        "**Experimente perguntar:**\n"
        "- O que é RAG?\n"
        "- Quais cargos de IA estão em alta?\n"
        "- O que são embeddings?"
    )

    # Mostra os documentos que o assistente conhece de fábrica
    with st.expander("📚 Ver documentos de exemplo"):
        for i, doc in enumerate(DOCUMENTOS_EXEMPLO, 1):
            st.markdown(f"**{i}.** {doc}")

    st.divider()
    st.subheader("Seu documento")
    pdf = st.file_uploader("Suba um PDF para perguntar sobre ele (opcional):", type="pdf")
    st.divider()
    if st.button("🧹 Limpar conversa"):
        st.session_state.historico = []
    st.caption("Feito por Tiago Cesaro")


# ---------------- Area principal ----------------
st.title("🤖 Assistente de Tecnologia")
st.caption("Pergunte sobre tecnologia — respostas baseadas em documentos, com a fonte citada.")

# Escolhe a base: PDF enviado ou os exemplos
if pdf is not None:
    documentos, vetores = indexar_pdf(pdf.getvalue())
    st.info(f"📄 Usando o seu PDF ({len(documentos)} trechos). Pergunte sobre ele!")
else:
    documentos, vetores = indexar_exemplos()
    st.info("📚 Usando os documentos de exemplo sobre tecnologia. (Suba um PDF na barra lateral para usar o seu.)")

# Historico da conversa
if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("fonte"):
            with st.expander("📄 Ver fonte usada"):
                st.write(msg["fonte"])

pergunta = st.chat_input("Faça sua pergunta...")
if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resposta, trechos = responder(pergunta, documentos, vetores)
        st.write(resposta)
        with st.expander("📄 Ver fonte usada"):
            st.write(trechos[0])

    st.session_state.historico.append({"role": "assistant", "content": resposta, "fonte": trechos[0]})

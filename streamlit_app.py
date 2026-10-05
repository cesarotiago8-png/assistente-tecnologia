import time
from io import BytesIO

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="Assistente de Tecnologia", page_icon="\U0001F916")

CANDIDATOS = [
    "gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-latest",
    "gemini-2.5-flash", "gemini-3.8-flash", "gemini-flash-lite-latest",
]


@st.cache_resource
def conectar():
    """Conecta no Gemini e escolhe um modelo que suporte busca na web, e um de embeddings."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    ferramenta_busca = types.Tool(google_search=types.GoogleSearch())

    # tenta achar um modelo que aceite a busca no Google (grounding)
    modelo = None
    for nome in CANDIDATOS:
        try:
            client.models.generate_content(
                model=nome, contents="oi",
                config=types.GenerateContentConfig(tools=[ferramenta_busca]),
            )
            modelo = nome
            break
        except Exception:
            continue
    # se nenhum aceitar busca, pega qualquer um que gere texto
    if modelo is None:
        for nome in CANDIDATOS:
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

    return client, modelo or "gemini-3.5-flash", emb or "gemini-embedding-001"


client, MODELO, EMB_MODELO = conectar()


# ---------- Modo WEB: tecnologia em geral, com busca ao vivo ----------
def responder_web(pergunta):
    """Responde usando busca no Google (informacao atual + fontes)."""
    config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    # tenta com busca na web
    for _ in range(3):
        try:
            return client.models.generate_content(model=MODELO, contents=pergunta, config=config)
        except Exception:
            time.sleep(3)
    # fallback: sem busca (conhecimento geral do modelo)
    for _ in range(3):
        try:
            return client.models.generate_content(model=MODELO, contents=pergunta)
        except Exception:
            time.sleep(3)
    return None


def extrair_fontes(resp):
    """Pega os links que o Gemini usou na busca (se houver)."""
    fontes = []
    try:
        chunks = resp.candidates[0].grounding_metadata.grounding_chunks
        for c in chunks:
            web = getattr(c, "web", None)
            if web:
                fontes.append((web.title or web.uri, web.uri))
    except Exception:
        pass
    return fontes


# ---------- Modo DOCUMENTO (RAG): responde com base num PDF enviado ----------
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


@st.cache_data
def indexar_pdf(conteudo_bytes):
    leitor = PdfReader(BytesIO(conteudo_bytes))
    texto = " ".join((pagina.extract_text() or "") for pagina in leitor.pages)
    pedacos = quebrar_em_pedacos(texto)
    return pedacos, embed(pedacos)


def responder_doc(pergunta, documentos, vetores, k=2):
    emb_pergunta = embed([pergunta])[0]
    notas = [similaridade(emb_pergunta, v) for v in vetores]
    melhores = np.argsort(notas)[::-1][:k]
    trechos = [documentos[i] for i in melhores]
    contexto = "\n\n".join(trechos)

    prompt = f"""Responda a pergunta usando SOMENTE os trechos abaixo.
Se a resposta nao estiver neles, responda: "Nao encontrei isso no documento."

Trechos:
{contexto}

Pergunta: {pergunta}"""

    for _ in range(5):
        try:
            resposta = client.models.generate_content(model=MODELO, contents=prompt).text
            return resposta, trechos[0]
        except Exception:
            time.sleep(4)
    return "A API esta ocupada agora, tente de novo em instantes.", trechos[0]


# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Sobre")
    st.write(
        "Assistente de tecnologia com dois modos:\n\n"
        "🌐 **Web** — responde sobre tecnologia em geral e novidades, pesquisando na internet (com fontes).\n\n"
        "📄 **Documento (RAG)** — se você subir um PDF, ele responde com base nesse documento."
    )
    st.markdown(
        "**Experimente perguntar:**\n"
        "- Qual a diferença entre Machine Learning e Deep Learning?\n"
        "- O que há de novo em IA recentemente?\n"
        "- O que é um banco vetorial?"
    )
    st.divider()
    st.subheader("Seu documento")
    pdf = st.file_uploader("Suba um PDF para o modo documento (opcional):", type="pdf")
    st.divider()
    if st.button("🧹 Limpar conversa"):
        st.session_state.historico = []
    st.caption("Feito por Tiago Cesaro")


# ---------------- Area principal ----------------
st.title("🤖 Assistente de Tecnologia")
st.caption("Pergunte sobre tecnologia — com busca na web ao vivo ou com base no seu documento.")

if pdf is not None:
    documentos, vetores = indexar_pdf(pdf.getvalue())
    st.info(f"📄 Modo documento: respondendo com base no seu PDF ({len(documentos)} trechos).")
else:
    documentos = None
    st.info("🌐 Modo web: respondendo sobre tecnologia em geral, com busca na internet. (Suba um PDF para o modo documento.)")

if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tipo") == "doc" and msg.get("fonte"):
            with st.expander("📄 Ver trecho usado"):
                st.write(msg["fonte"])
        elif msg.get("tipo") == "web" and msg.get("fonte"):
            with st.expander("🌐 Fontes da web"):
                for titulo, url in msg["fonte"]:
                    st.markdown(f"- [{titulo}]({url})")

pergunta = st.chat_input("Faça sua pergunta...")
if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if documentos is not None:
                resposta, trecho = responder_doc(pergunta, documentos, vetores)
                tipo, fonte = "doc", trecho
            else:
                resp = responder_web(pergunta)
                resposta = resp.text if resp else "A API esta ocupada agora, tente de novo."
                tipo, fonte = "web", (extrair_fontes(resp) if resp else [])

        st.write(resposta)
        if tipo == "doc":
            with st.expander("📄 Ver trecho usado"):
                st.write(fonte)
        elif fonte:
            with st.expander("🌐 Fontes da web"):
                for titulo, url in fonte:
                    st.markdown(f"- [{titulo}]({url})")

    st.session_state.historico.append(
        {"role": "assistant", "content": resposta, "tipo": tipo, "fonte": fonte}
    )

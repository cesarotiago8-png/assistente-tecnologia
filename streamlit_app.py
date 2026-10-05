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

    # fallback caso tudo esteja ocupado no momento da conexao
    return client, modelo or "gemini-flash-lite-latest", emb or "gemini-embedding-001"


client, MODELO, EMB_MODELO = conectar()


def embed(textos):
    """Transforma uma lista de textos em vetores (com retry se a API estiver ocupada)."""
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
    """Quebra um texto grande em pedacos de ~tamanho caracteres."""
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
    """O RAG: acha os trechos mais parecidos e pede ao modelo uma resposta baseada neles."""
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


# ---------------- Interface ----------------
st.title("\U0001F916 Assistente de Tecnologia")
st.write("Pergunte sobre tecnologia. Eu respondo com base em documentos e cito a fonte.")

# Por padrao usa os documentos de exemplo
documentos, vetores = indexar_exemplos()

# Upload opcional do proprio PDF
pdf = st.file_uploader("Quer usar seu proprio documento? Suba um PDF (opcional):", type="pdf")
if pdf is not None:
    documentos, vetores = indexar_pdf(pdf.getvalue())
    st.success(f"PDF carregado: {len(documentos)} pedacos indexados. Agora pergunte sobre ele.")

# Historico da conversa
if "historico" not in st.session_state:
    st.session_state.historico = []

for papel, msg in st.session_state.historico:
    st.chat_message(papel).write(msg)

pergunta = st.chat_input("Faca sua pergunta...")
if pergunta:
    st.chat_message("user").write(pergunta)
    st.session_state.historico.append(("user", pergunta))

    with st.spinner("Pensando..."):
        resposta, trechos = responder(pergunta, documentos, vetores)

    texto_final = resposta + "\n\n---\n*Fonte usada:* " + trechos[0][:150] + "..."
    st.chat_message("assistant").write(texto_final)
    st.session_state.historico.append(("assistant", texto_final))

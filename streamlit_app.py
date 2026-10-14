
# imports e configuracoes

import time         
from io import BytesIO

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(page_title="Assistente de Tecnologia", page_icon="🤖")     # define o titulo e icone da pagina

# modelos e persona ------------

CANDIDATOS = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash"]  # modelos do Gemini que podem ser usados para gerar respostas

PERSONA = (                                                                                # define a persona do assistente, que é o estilo e tom de voz que ele vai usar para responder
    "Voce e o Assistente de Tecnologia do Tiago Cesaro, especializado em tecnologia, "
    "inteligencia artificial, programacao e mercado de trabalho em tech. "
    "Ao se apresentar, diga que e o assistente do Tiago voltado para tecnologia e IA — "
    "nunca se apresente como um modelo generico do Google. "
    "Responda sempre em portugues, de forma clara e amigavel, explicando de um jeito simples "
    "e usando analogias quando ajudar a entender. "
    "Prefira respostas objetivas; so se aprofunde se a pessoa pedir. "
    "Foque em temas de tecnologia, IA, programacao e carreira em tech; se perguntarem algo "
    "muito fora disso, responda de forma breve e gentilmente puxe de volta para a area de tecnologia. "
    "Se nao tiver certeza da resposta, admita em vez de inventar. "
    "Quando pesquisar na web, cite as fontes."
)

# conexao com a API do Gemini ---------

@st.cache_resource
def conectar():                 # conecta com a API do Gemini e retorna o cliente, modelo de resposta e modelo de embedding
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    modelo = None     # tenta encontrar um modelo de resposta que funcione, testando os candidatos
    for nome in CANDIDATOS:
        try:
            client.models.generate_content(model=nome, contents="oi")
            modelo = nome
            break
        except Exception:
            continue

    emb = None       # tenta encontrar um modelo de embedding que funcione, testando os candidatos
    for nome in ["gemini-embedding-001", "gemini-embedding-2"]:
        try:
            client.models.embed_content(model=nome, contents="teste")
            emb = nome
            break
        except Exception:
            continue

    return client, modelo or "gemini-flash-lite-latest", emb or "gemini-embedding-001"

client, MODELO, EMB_MODELO = conectar() # conecta com a API do Gemini e guarda o cliente, modelo de resposta e modelo de embedding

# memoria da conversa ---------

def historico_recente(n=6):
    msgs = st.session_state.get("historico", [])[:-1][-n:]
    linhas = []
    for m in msgs:
        quem = "Usuario" if m["role"] == "user" else "Assistente"
        linhas.append(f"{quem}: {m['content']}")
    return "\n".join(linhas)

# modo web ---------

def responder_web(pergunta, usar_busca=False):    # responde a pergunta usando o modelo do Gemini, com ou sem busca na web
    contexto = historico_recente()
    entrada = pergunta
    if contexto:   
        entrada = f"Conversa ate agora:\n{contexto}\n\nNova pergunta: {pergunta}"

    config = types.GenerateContentConfig(system_instruction=PERSONA)   # define a persona do assistente, que é o estilo e tom de voz que ele vai usar para responder
    if usar_busca:
        config = types.GenerateContentConfig(
            system_instruction=PERSONA,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

    for _ in range(3):   # tenta gerar a resposta 3 vezes, esperando 2 segundos entre cada tentativa se houver erro
        try:
            return client.models.generate_content(model=MODELO, contents=entrada, config=config)
        except Exception:
            time.sleep(2)
    return None


def extrair_fontes(resp):      # extrai as fontes da resposta do modelo, se houver, e retorna uma lista de tuplas (titulo, url)
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


# modo documento (RAG) ---------

class CotaEsgotada(Exception): # excecao personalizada para quando a cota de embeddings do Gemini for atingida
    pass

def embed(textos): # gera embeddings para uma lista de textos usando o modelo de embedding do Gemini, com tratamento de erros e tentativas
    vetores = []
    for texto in textos:
        vetor = None
        espera = 4
        ultimo_erro = None
        for _ in range(3):
            try:
                r = client.models.embed_content(model=EMB_MODELO, contents=texto)
                vetor = r.embeddings[0].values
                break
            except Exception as e:
                ultimo_erro = e
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    raise CotaEsgotada(
                        "O limite diario gratuito de embeddings do Gemini "
                        "(1000 por dia) foi atingido. Ele reseta sozinho em "
                        "algumas horas — o modo web continua funcionando."
                    )
                time.sleep(espera)
                espera = min(espera * 2, 20)
        if vetor is None:
            raise RuntimeError(f"Falha ao gerar embedding. Erro real da API: {ultimo_erro}")
        vetores.append(vetor)
    return vetores


def similaridade(a, b):      # calcula a similaridade do cosseno entre dois vetores, retornando um valor entre -1 e 1
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def quebrar_em_pedacos(texto, tamanho=1500):   # quebra um texto em pedaços de tamanho máximo definido, retornando uma lista de pedaços
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
def indexar_pdf(conteudo_bytes):  # indexa um PDF, extraindo o texto e gerando embeddings para cada pedaço, retornando uma lista de pedaços e uma lista de vetores
    leitor = PdfReader(BytesIO(conteudo_bytes))
    texto = " ".join((pagina.extract_text() or "") for pagina in leitor.pages)
    pedacos = quebrar_em_pedacos(texto)
    return pedacos, embed(pedacos)


def responder_doc(pergunta, documentos, vetores, k=2):   # responde a pergunta usando o modo documento (RAG), buscando os k trechos mais relevantes do PDF e gerando uma resposta com base neles
    emb_pergunta = embed([pergunta])[0]
    notas = [similaridade(emb_pergunta, v) for v in vetores]
    melhores = np.argsort(notas)[::-1][:k]
    trechos = [documentos[i] for i in melhores]
    contexto = "\n\n".join(trechos)
    conversa = historico_recente()

    prompt = f"""{PERSONA}

Historico recente da conversa (apenas para contexto):
{conversa}

Responda a pergunta usando SOMENTE os trechos do documento abaixo.
Se a resposta nao estiver neles, responda: "Nao encontrei isso no documento."

Trechos:
{contexto}

Pergunta: {pergunta}"""  

    for _ in range(5):  # tenta gerar a resposta 5 vezes, esperando 4 segundos entre cada tentativa se houver erro
        try:
            resposta = client.models.generate_content(model=MODELO, contents=prompt).text
            return resposta, trechos[0]
        except Exception:
            time.sleep(4)
    return "A API esta ocupada agora, tente de novo em instantes.", trechos[0]

# Barra lateral e interface do Streamlit ---------

with st.sidebar:   # cria a barra lateral do Streamlit
    st.header("Sobre")
    st.write(
        "Assistente de tecnologia do Tiago, com dois modos:\n\n"
        "🌐 **Web** - responde sobre tecnologia em geral e novidades, pesquisando na internet (com fontes).\n\n"
        "📄 **Documento (RAG)** - se voce subir um PDF, ele responde com base nesse documento."
    )
    st.markdown(     # adiciona um markdown com exemplos de perguntas
        "**Experimente perguntar:**\n"
        "- Quem é você?\n"
        "- Qual a diferença entre Machine Learning e Deep Learning?\n"
        "- O que há de novo em IA recentemente?"
    )
    st.divider()     # adiciona uma linha divisória na barra lateral
    st.subheader("Seu documento")
    pdf = st.file_uploader("Suba um PDF para o modo documento (opcional):", type="pdf")
    buscar_web = st.checkbox("🔎 Pesquisar na web (mais lento)", value=False)
    st.divider()     
    if st.button("🧹 Limpar conversa"):
        st.session_state.historico = []
    st.caption("Feito por Tiago Thomaz Cesaro")


# interface principal do Streamlit ---------

st.title("🤖 Assistente de Tecnologia")  
st.caption("Pergunte sobre tecnologia — com busca na web ao vivo ou com base no seu documento.")

documentos = None   # se o usuário subir um PDF, tenta indexá-lo e gerar embeddings para cada trecho, armazenando os trechos e vetores em variáveis
if pdf is not None:
    try:
        documentos, vetores = indexar_pdf(pdf.getvalue())
        st.info(f"📄 Modo documento: respondendo com base no seu PDF ({len(documentos)} trechos).")
    except CotaEsgotada as e:
        st.warning(f"⏳ {e} Continuando no modo web por enquanto.")
        documentos = None
    except Exception as e:
        st.warning(f"Não consegui indexar o PDF agora ({e}). Continuando no modo web.")
        documentos = None

if documentos is None:   # se não houver PDF ou se houve erro ao indexá-lo, informa que o assistente está no modo web
    st.info("🌐 Modo web: respondendo sobre tecnologia em geral. Marque '🔎 Pesquisar na web' na barra lateral para buscar fontes atualizadas.")

if "historico" not in st.session_state:  # inicializa o histórico da conversa na sessão do Streamlit, se ainda não existir
    st.session_state.historico = []

for msg in st.session_state.historico:    # exibe o histórico da conversa no Streamlit, mostrando as mensagens do usuário e do assistente, e adicionando expansores para mostrar trechos do documento ou fontes da web, se houver
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tipo") == "doc" and msg.get("fonte"):
            with st.expander("📄 Ver trecho usado"):
                st.write(msg["fonte"])
        elif msg.get("tipo") == "web" and msg.get("fonte"):
            with st.expander("🌐 Fontes da web"):
                for titulo, url in msg["fonte"]:
                    st.markdown(f"- [{titulo}]({url})")

pergunta = st.chat_input("Faça sua pergunta...") # cria um campo de entrada de texto para o usuário digitar sua pergunta, e armazena a pergunta na variável 'pergunta'
if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):  # cria uma mensagem de chat para o assistente, e exibe um spinner enquanto o assistente pensa na resposta
        with st.spinner("Pensando..."):
            if documentos is not None:
                try:
                    resposta, trecho = responder_doc(pergunta, documentos, vetores)
                    tipo, fonte = "doc", trecho
                except CotaEsgotada as e:
                    resposta, tipo, fonte = str(e), "web", []
            else:
                resp = responder_web(pergunta, buscar_web)
                resposta = resp.text if resp else "A API esta ocupada agora, tente de novo."
                tipo, fonte = "web", (extrair_fontes(resp) if (resp and buscar_web) else [])

        st.write(resposta)    # exibe a resposta do assistente no Streamlit, e adiciona expansores para mostrar trechos do documento ou fontes da web, se houver
        if tipo == "doc":
            with st.expander("📄 Ver trecho usado"):
                st.write(fonte)
        elif fonte:
            with st.expander("🌐 Fontes da web"):
                for titulo, url in fonte:
                    st.markdown(f"- [{titulo}]({url})")

    st.session_state.historico.append(  # adiciona a resposta do assistente ao histórico da conversa na sessão do Streamlit, incluindo o tipo de resposta (documento ou web) e as fontes usadas, se houver
        {"role": "assistant", "content": resposta, "tipo": tipo, "fonte": fonte}
    )
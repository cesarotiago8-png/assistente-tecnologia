import time                    #fazer o programa esperar alguns segundos
from io import BytesIO         #le o pdf que o usuario sobe direto na memoria

import numpy as np        # biblioteca de cauculo com numeros/vetores
import streamlit as st    # biblioteca que transforma o codigo Python em uma pagina web com chat
from google import genai          #kit para falar com gemini (o cliente que faz as chamadas)
from google.genai import types    #kit para falar com gemini (traz uns moldes de configuracao que a gente usa para ligar a busca web
from pypdf import PdfReader       #ferramenta que le o texto de dentro de um PDF

st.set_page_config(page_title="Assistente de Tecnologia", page_icon="🤖") # definindo o icone e o titulo que aparece na pagina web


# Persona e lista de modelos

CANDIDATOS = [
    "gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-latest",
    "gemini-2.5-flash", "gemini-3.8-flash", "gemini-flash-lite-latest",
]

PERSONA = (
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


# Conexao com o Gemini

@st.cache_resource                                                                                        # Embrulha a funcao e muda o comportamento dela, diz pro Streamlit: Roda conectar(),
def conectar(): # cria a funcao                                                                           # uma vez e guarda o resultado, conecta uma vez e reaproveita
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"]) # seguranca, guarda a chave no cofre do Streamlit                           
    ferramenta_busca = types.Tool(google_search=types.GoogleSearch()) # acessorio de busca do google                   

    modelo = None
    for nome in CANDIDATOS:     # percorre a lista de candidatos, para cada nome tem uma chamadinha de teste "oi" com a busca ligada, o primeiro que funcionar vira modelo, e o break para o loop
        try:
            client.models.generate_content(
                model=nome, contents="oi",
                config=types.GenarateContentConfig(tools=[ferramenta_busca]),             # try/except: tenta fazer isso, se der erro, nao quebra o programa, faz outra coisa,
            )                                                                             # no nosso caso tenta usar o modelo se der errado, parte pro proximo    
            modelo = nome
            break
        except Exception:
            continue
    if modelo is None:               # Plano B , se nenhum modelo aceitou a busca, ele tenta denovo, so que sem a busca, assim pelo menos o chat funciona
        for nome in CANDIDATOS:
            try:
                client.models.genarate_content(model=nome, contents="oi")
                modelo = nome
                break                                                                     # return client, modelo, emb devolve 3 coisas de uma vez, e client, MODELO, EMB_MODELO = conectar(),
            except Exception:                                                             # e guarda em 3 variaveis separadas, na ordem.
                continue

    emb = None
    for nome in ["gemini-embedding-001", "gemini-embedding-2"]:      # mesma logica, mais procurando um modelo de embedding que transforma texto em numeros, pro RAG
        try:
            client.models.embed_content(model=nome, contents=["teste"])
            emb = nome
            break
        except Exception:
            continue

    return client, modelo or "gemini-3.5-flash", emb or "gemini-embedding-001"       # se por algum motivo modelo ainda estiver vazio, ele usa o nome padrao depois do or, use o modelo que achei OU se nao achei nada, use esse aqui.

client, MODELO, EMB_MODELO = conectar()          



# MODO WEB (BUSCA + FONTES)

def responder_web(pergunta):                                                     # faz a pergunta com busca na web, config e o pacote de instrucoes que vai junto com a pergunta, contem a persona,
    config = types.GenerateContentConfig(                                        # para ele responder como seu assistente e liga a ferramenta de busca na internet.  
        system_instruction=PERSONA,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    for _ in range(3):                                                                                   
        try:
            return client.models.generate_content(model=MODELO, contents=pergunta, config=config)
        except Exception:
            time.sleep(3)                                                                                   # tenta 3 vezes, se a resposta vier de primeira, o return ja devolve e encerra,
    config2 = types.GenerateContentConfig(system_instruction=PERSONA)                                       # se a API estiver ocupada e der erro, ele espera 3 segundos e tenta denovo
    for _ in range(3):
        try:
            return client.models.generate_content(model=MODELO, contents=pergunta, config=config2)   
        except Exception:
            time.sleep(3)     
    return None


def extrair_fontes(resp):                                                                                     # pega os links que utilizou, quando o Gemini pesquisa na web, ele devolve a resposta e tambem,
    fontes = []                                                                                               # escondidinho, a lista de sites que consultou.   
    try:
        chunks = resp.candidates[0].grounding_metadata.grounding_chunks
        for c in chunks:
            web = getattr(c, "web", None)
            if web:
                fontes.append((web.title or web.uri, web.uri))
    except Exception:
        pass
    return fontes         


# RAG PARTE 1: FERRAMENTAS

class CotaEsgotada(Exception):                                 # uma etiqueta de erro, (acabou a cota do dia)
    pass

def embed(textos):                                             # embed transforma textos em numeros, ela faz um texto de cada vez
    vetores = []
    for texto in textos:
        vetor = None
        espera = 4
        ultimo_erro = None
        for _ in range(3):                                                                # Para cada texto, tenta ate 3 vezes. Se a API estiver ocupada, ele espera e tenta de novo,
            try:                                                                          # com uma espera que cresce (4s, 8s depois 16s/ backoff)
                r = client.models.embed_content(model=EMB_MODELO, contents=texto)
                vetor = r.embeddings[0].values
                break
            except Exception as e:
                ultimo_erro = e
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:                             # se o erro for de cota esgotada, so volta daqui a horas, levanta na etiqueta, para mostrar uma mensagem na web
                    raise CotaEsgotada(
                        "O limite diario gratuito de embeddings do Gemini "
                        "(1000 por dia) foi atingido. Ele reseta sozinho em "
                        "algumas horas - o modo web continua funcionando."
                    )
                time.sleep(espera)
                espera - min(espera *2, 20)
        if vetor is None:
            raise RuntimeError(f"Falha ao gerar embedding. Erro real da API: {ultimo_erro}")   
        vetores.append(vetor)
    return vetores

def similaridade(a, b):                                                      # mede o quao parecidos dois textos sao, recebe dois vetores e devolve uma nota de semelhanca, quanto maior a nota,
    a, b = np.array(a), np.array(b)                                          # mais parecidos sao no significado, linha com np e so a formula matematica
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def quebrar_em_pedacos(texto, tamanho=1500):                                  # pega um texto grande e corta em pedacos de 1500 caracteres, sem cortar a palavra no meio,
    palavras = texto.split()                                                  # ela vai juntando palavra por palavra ate encher o pedaco.
    pedacos, atual = [], ""                                                   # pedacos maiores = menos pedacos = menos chamadas de embedding = menos cota gasta. Devolve a lista de pedacos   
    for p in palavras:
        if len(atual) + len(p) > tamanho:
            pedacos.append(atual.strip())
            atual = ""
        atual += " " + p
    if atual.strip():
        pedacos.append(atual.strip())
    return pedacos        




# RAG PARTE 2: LER O PDF E RESPONDER

@st.cache_data # faca isso uma vez por PDF e guarde o resultado.
def indexar_pdf(conteudo_bytes):                                                      # prepara o documento. Le o pdf e extrai o texto de todas as paginas, juntando todo num texto grandao;
    leitor = PdfReader (BytesIO(conteudo_bytes))                                      # quebra o texto em pedacos, gera os embeddings de cada pedaco, devolve os pedacos e os vetores
    texto = " ".join((pagina.extract_text() or "") for pagina in leitor.pages)
    pedacos = quebrar_em_pedacos(texto)                                              
    return pedacos, embed(pedacos)


def responder_doc(pergunta, documentos, vetores, k=2):                                 # responde com base no PDF (o RAG acontecendo)
    emb_pergunta = embed([pergunta])[0]                                                # vira numeros
    notas = [similaridade(emb_pergunta, v) for v in vetores]                           # compara a pergunta com cada pedaco do documento, gerando uma nota pra cada
    melhores = np.argsort(notas)[::-1][:k]                                             # ele acha os 2 pedacos mais parecidos com a pergunta, k2 = os 2 melhores, 
    trechos = [documentos[i] for i in melhores]
    contexto = "\n\n".join(trechos)
                                                                                       # junta esses trechos no contexto e monta o prompt, damos a persona, os trechos encontrados e a ordem clara: "responda com base nesses trechos, se nao estiver neles diga nao encontrei isso no documento"
    prompt = f"""{PERSONA}
                                                                            
Responda a pergunta usando SOMENTE os trechos do documento abaixo.                         
Se a resposta nao estiver neles, responda: "Nao encontrei isso no documento."

Trechos:
{contexto}

Pergunta: {pergunta}"""

    for _ in range(5):
        try:
            resposta = client.models.generate_content(model=MODELO, contents=prompt).text
            return resposta, trechos [0]
        except Exception:
            time.sleep(4)
    return "A API esta ocupada agora, tente de novo em instantes.", trechos[0]   


# INTERFACE PARTE 1: BARRA LATERAL

with st.sidebar:                                                                         # tudo que estiver na barra lateral aparece aqui embaixo,
    st.header("Sobre")                                                                   # jeitos de mostrar texto, do maior (header) ao menor (caption)
    st.write(
        "Assistente de tecnologia do Tiago, com dois modos:\n\n"                                                      # st.write e markdown entendem aquele negrito e os topicos com -
        "🌐 **Web** - responde sobre tecnologia em geral e novidades, pesquisando na internet (com fontes).\n\n"
        "📄 **Documento (RAG)** - se voce subir um PDF, ele responde com base nesse documento."
    )
    st.markdown(
        "**Experimente perguntar:**\n"
        "- Quem é você?\n"
        "- Qual a diferença entre Machine Learning e Deep Learning?\n"
        "- O que há de novo em IA recentemente?"
    )

    st.divider() # linha divisoria para separar as secoes
    st.subheader("Seu documento")
    pdf= st.file_uploader("Suba um PDF para o modo documento (opcional):", type="pdf")          # cria o botao de subir o pdf
    st.divider()
    if st.button("🧹 Limpar conversa"):                                 # zera o historico da conversa
        st.session_state.historico = []
    st.caption("Feito por Tiago Thomaz Cesaro")    


# INTERFACE PARTE 2: AREA PRINCIPAL + CHAT

st.title("🤖 Assistente de Tecnologia")                                                                    # titulo grande e a frase menor embaixo
st.caption("Pergunte sobre tecnologia - com busca na web ao vivo ou com base no seu documento.")

documentos = None
if pdf is not None:                                                                                          # interruptor, se a pessoa subir um PDF, tenta indexar o RAG, Se der certo,
    try:                                                                                                     # entra no modo documento e mostra o aviso com o numero de trechos.
        documentos, vetores = indexar_pdf(pdf.getvalue())                                                    # Se bater a cota, mostra um aviso amigavel e cai pro modo web. Se der outro erro, e se nao tem pdf, mostra o aviso do modo web                                        
        st.info(f"📄 Modo documento: respondendo com base no seu PDF ({len(documentos)} trechos).")
    except CotaEsgotada as e:
        st.warning(f"⏳ {e} Continuando no modo web por enquanto.")
        documentos = None
    except Exception as e:
        st.warning(f"Nao consegui indexar o PDF agora {e}. Continuando no modo web.")
        documentos = None

if documentos is None:
    st.info("🌐 Modo web: respondendo sobre tecnologia em geral, com busca na internet. (Suba um PDF para o modo documento.)")


if "historico" not in st.session_state:                            # uma caixinha onde guardamos o historico, so cria essa caixinha vazia na primeira vez.
    st.session_state.historico = []

for msg in st.session_state.historico:                             # redesenha a conversa toda a cada rerun, para cada msg guardada, desenha um balao e, se ela tinha fonte,
    with st.chat_message(msg["role"]):                             # mostra aquele expander, e isso que faz as msg antigas na tela
        st.write(msg["content"])
        if msg.get("tipo") == "doc" and msg.get("fonte"):
            with st.expander("📄 Ver trecho usado"):
                st.write(msg["fonte"])
        elif msg.get("tipo") == "web" and msg.get("fonte"):
            with st.expander("🌐 Fontes da web"):
                for titulo, url in msg["fonte"]:
                    st.markdown(f"- [{titulo}]({url})")

pergunta = st.chat_input("Faca sua pergunta...")                            # caixa de digitar la embaixo.
if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):                             # abre o balao do assistente com um pensando
            if documentos is not None:
                try:
                    resposta, trecho = responder_doc(pergunta, documentos, vetores)         # decide qual cerebro usar: se tem documento = responder_doc, senao = responder web
                    tipo, fonte = "doc", trecho
                except CotaEsgotada as e:
                    resposta, tipo, fonte = str(e), "web", []
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
                    st.markdown(f"- [{titulo}] ({url})") 

    st.session_state.historico.append(
        {"role": "assistant", "content": resposta, "tipo": tipo, "fonte": fonte}
    )                       


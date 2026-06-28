import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

st.set_page_config(page_title="Banco Atlas Inteligente", page_icon="🏦", layout="wide")
st.title("🏦 Sistema Inteligente - Banco Atlas Financiero")
st.caption("Powered by Gemini + RAG + AI Agent")

# ── FIX PRINCIPAL: obtener la API key de forma segura ──────────────────────
# Primero intenta desde st.secrets (Streamlit Cloud), luego desde variable de entorno
api_key = None

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
elif os.environ.get("GOOGLE_API_KEY"):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ No se encontró GOOGLE_API_KEY. Agrégala en los Secrets de Streamlit Cloud.")
    st.info("Ve a tu app en Streamlit Cloud → Settings → Secrets y añade:\n```\nGOOGLE_API_KEY = \"tu_clave_aqui\"\n```")
    st.stop()

# Establecer la variable de entorno para que langchain la use
os.environ["GOOGLE_API_KEY"] = api_key

# ── Ruta de la base de datos Chroma ────────────────────────────────────────
# En Streamlit Cloud no existe /content/drive, usar ruta relativa al repo
CHROMA_PATH = os.environ.get("CHROMA_DB_PATH", "chroma_db_atlas")

@st.cache_resource
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key  # pasar explícitamente para evitar problemas de entorno
    )
    return db, llm

try:
    db, llm = cargar_sistema()
except Exception as e:
    st.error(f"❌ Error al cargar el sistema: {e}")
    st.stop()

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800, lang="es")
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def buscar_en_manual(consulta: str) -> str:
    """Busca políticas internas del Banco Atlas en el manual oficial."""
    docs = db.similarity_search(consulta, k=3)
    return "\n\n".join([d.page_content for d in docs])

@tool
def buscar_en_web(consulta: str) -> str:
    """Busca información pública y actual en Wikipedia."""
    return wikipedia_tool.invoke(consulta)

herramientas = [buscar_en_manual, buscar_en_web]

prompt_agente = ChatPromptTemplate.from_messages([
    ("system",
     "Eres asistente del Banco Atlas Financiero. "
     "Usa buscar_en_manual para consultar políticas y procedimientos internos "
     "y buscar_en_web para obtener información pública actualizada. "
     "Siempre cita la fuente de donde obtuviste la información."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agente = create_tool_calling_agent(llm, herramientas, prompt_agente)
ejecutor = AgentExecutor(agent=agente, tools=herramientas, verbose=False)

# ── Modo RAG puro (muestra fuentes) ────────────────────────────────────────
st.subheader("📚 Modo RAG — Consulta con fuentes")
pregunta_rag = st.text_input("Escribe tu pregunta al manual:", key="rag")

if st.button("Consultar RAG"):
    if pregunta_rag:
        with st.spinner("Buscando en el manual..."):
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs_recuperados = retriever.invoke(pregunta_rag)

            prompt = PromptTemplate.from_template(
                "Responde esta pregunta: {question}\nUsando este contexto: {context}\nRespuesta:"
            )
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            respuesta = chain.invoke(pregunta_rag)

            st.markdown("### 🤖 Respuesta")
            st.write(respuesta)

            st.markdown("### 📄 Fuentes recuperadas")
            for i, doc in enumerate(docs_recuperados):
                with st.expander(f"Fragmento {i+1}"):
                    st.write(doc.page_content)
    else:
        st.warning("Escribe una pregunta primero.")

st.divider()

# ── Modo Agente ─────────────────────────────────────────────────────────────
st.subheader("🤖 Modo Agente — Con herramientas")
pregunta_agente = st.text_input("Pregunta al agente:", key="agente")

if st.button("Consultar Agente"):
    if pregunta_agente:
        with st.spinner("El agente está pensando..."):
            try:
                resultado = ejecutor.invoke({"input": pregunta_agente})
                st.markdown("### ✅ Respuesta del Agente")
                st.write(resultado["output"])
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Escribe una pregunta primero.")

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuración de página
st.set_page_config(page_title="Sistema Inteligente Banco Atlas", page_icon="🏦")
st.title("🏦 Sistema Inteligente - Banco Atlas")

# Configuración de API Key (Asegúrate de tener la variable de entorno o configurarla aquí)
# En Colab/Streamlit, puedes usar st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY") 

# Carga de la base de datos (debe estar en una ruta accesible)
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    # Asegúrate de ajustar esta ruta a donde realmente están tus archivos en Drive/Colab
    ruta_chroma = '/content/drive/MyDrive/u/cd2/chroma_db_atlas'
    return Chroma(persist_directory=ruta_chroma, embedding_function=embeddings)

base_vectorial = load_db()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# Interfaz de usuario
uploaded_file = st.sidebar.file_uploader("Subir documento PDF", type="pdf")
pregunta = st.text_input("¿Qué deseas consultar sobre el Banco Atlas?")

if st.button("Consultar"):
    if pregunta:
        with st.spinner("Consultando manuales..."):
            # Lógica RAG
            recuperador = base_vectorial.as_retriever(search_kwargs={"k": 3})
            plantilla = """Eres un experto del Banco Atlas Financiero. Responde basándote en el contexto:
            {context}
            Pregunta: {question}"""
            prompt = PromptTemplate.from_template(plantilla)
            
            cadena_rag = (
                {"context": recuperador, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            respuesta = cadena_rag.invoke(pregunta)
            st.write("### Respuesta:")
            st.write(respuesta)
            
            # Mostrar fuentes (Rúbrica: Ver fuentes recuperadas)
            fuentes = base_vectorial.similarity_search(pregunta, k=3)
            st.write("### Fuentes recuperadas:")
            for i, doc in enumerate(fuentes):
                st.write(f"Fuente {i+1}: {doc.page_content[:150]}...")
    else:
        st.warning("Por favor, ingresa una consulta.")
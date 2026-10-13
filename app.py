
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Configuração da API
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("Configure a chave GROQ_API_KEY no arquivo .env")
    st.stop()

# 2. Interface
st.title("Chatbot Trabalho Garantido")
st.write("Faça perguntas sobre o edital do programa.")

# 3. Preparação do RAG
@st.cache_resource
def iniciar_rag():

    # Carregar o edital
    documentos = PyPDFLoader(
        "EDITAL-Trabalho-Garantido-26-2.pdf"
    ).load()

    # Dividir o documento em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documentos)

    # Gerar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Criar o banco vetorial
    banco = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Configurar a recuperação de informações
    retriever = banco.as_retriever(
        search_kwargs={"k": 5}
    )

    # Configurar o modelo de linguagem
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return retriever, llm


# 4. Inicializar o chatbot
try:
    with st.spinner("Preparando o chatbot..."):
        retriever, llm = iniciar_rag()
except Exception as erro:
    st.error(f"Erro ao iniciar: {erro}")
    st.stop()


# 5. Histórico de mensagens
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])


# 6. Receber a pergunta do usuário
if pergunta := st.chat_input("Digite sua pergunta..."):

    with st.chat_message("user"):
        st.write(pergunta)

    st.session_state.mensagens.append({
        "role": "user",
        "content": pergunta
    })

    # 7. Recuperar informações e gerar a resposta
    with st.chat_message("assistant"):
        with st.spinner("Consultando o edital..."):
            try:
                documentos = retriever.invoke(pergunta)

                contexto = "\n\n".join(
                    doc.page_content for doc in documentos
                )

                prompt = f"""
                Você é um assistente do Programa Trabalho Garantido.

                Responda utilizando apenas as informações
                do edital fornecidas abaixo.

                Se não encontrar a resposta, informe que
                a informação não foi localizada no edital.

                Contexto:
                {contexto}

                Pergunta:
                {pergunta}
                """

                resposta = llm.invoke(prompt).content

                st.markdown(resposta)

                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": resposta
                })

            except Exception as erro:
                st.error(f"Erro ao responder: {erro}")

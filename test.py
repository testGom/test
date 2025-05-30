import os
import streamlit as st
from datetime import datetime
import logging
import uuid
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient

# Arize Phoenix
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor


os.environ["OPENAI_API_KEY"] = ""


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Phoenix instrumentation
tracer_provider = register()
LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

# Phoenix Session
if "phoenix_session" not in st.session_state:
    st.session_state.phoenix_session = px.launch_app()
    logger.info(f"Phoenix app launched at: {st.session_state.phoenix_session.url}")

def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

chroma_client = PersistentClient(path="./chroma_db")

st.title("RAG V0: Arize Phoenix / OpenAI API")

prompt_template = PromptTemplate.from_template("""
    You are an assistant helping with questions about a document. Answer the question using only the context provided.
    If the answer is not in the context provided respond with "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
""")

# Sidebar: Upload
with st.sidebar:
    st.title("Upload Document")
    uploaded_file = st.file_uploader("Uploadez votre PDF", type=["pdf"])

    if uploaded_file and "qa_chain" not in st.session_state:
        file_path = "C:/Users/giaco/Desktop/testRAG/uploaded.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ PDF uploadé")
        logger.info(f"PDF uploadé et sauvé: {file_path}")

        with st.spinner("Processing document..."):
            # Load and split PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            logger.info(f"{len(pages)} pages chargées")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
            all_splits = text_splitter.split_documents(pages)
            logger.info(f"Document divisé en {len(all_splits)} chunks")
            st.success(f"Document divisé en {len(all_splits)} chunks")

            # Embeddings & Vector DB
            embedding_model = OpenAIEmbeddings()
            vector_store = Chroma(
                client=chroma_client,
                collection_name="RagV0",
                embedding_function=embedding_model
            )
            for batch in batch_documents(all_splits, 100):
                vector_store.add_documents(documents=batch)

            # Phoenix visu embeddings
            logger.info("Génération des embeddings pour Phoenix...")
            embeddings = embedding_model.embed_documents([doc.page_content for doc in all_splits])
            df = pd.DataFrame({
                "chunk_id": [str(uuid.uuid4()) for _ in all_splits],
                "page_number": [doc.metadata.get("page", -1) for doc in all_splits],
                "text": [doc.page_content for doc in all_splits],
                "embedding": embeddings,
                "timestamp": [datetime.now().isoformat()] * len(all_splits),
            })

            schema = px.Schema(
                timestamp_column_name="timestamp",
                embedding_feature_column_names={
                    "text_embedding": px.EmbeddingColumnNames(
                        vector_column_name="embedding"
                    )
                },
            )

            chunk_ds = px.Inferences(dataframe=df, schema=schema, name="document_chunks")
            chunk_session = px.launch_app(primary=chunk_ds)
            st.session_state.phoenix_chunk_session = chunk_session
            logger.info(f"Phoenix chunk visualization launched at: {chunk_session.url}")

            # LLM & RAG setup
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )

            # Store in session
            st.session_state.qa_chain = qa_chain
            st.session_state.pages = pages
            st.session_state.messages = []

    if "pages" in st.session_state:
        with st.expander("Aperçu du document"):
            for i, page in enumerate(st.session_state.pages):
                st.markdown(f"**Page {i + 1}:**")
                st.write(page.page_content)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez une question sur votre document:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        logger.info(f"Question utilisateur: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):

                response = st.session_state.qa_chain.invoke({"query": prompt})
                answer = response["result"]
                source_docs = response["source_documents"]

                context = "\n\n".join([doc.page_content for doc in source_docs])
                constructed_prompt = prompt_template.format(context=context, question=prompt)
                logger.info(f"Prompt complet envoyé au modèle: {constructed_prompt}")
                logger.info(f"Réponse du modèle: {answer}")

            st.markdown(answer)

            seen = set()
            unique_sources = []
            for doc in source_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_sources.append(doc)

            with st.expander("📚 Sources utilisées"):
                for i, doc in enumerate(unique_sources):
                    st.markdown(f"**Source {i + 1}:**")
                    st.write(doc.page_content)

            with st.expander("Prompt envoyé"):
                st.code(constructed_prompt, language=None)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("📄 Veuillez uploader un document dans le menu à gauche pour démarrer.")

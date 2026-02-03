import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory   
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()   


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF upload and Chat History")
st.write("Upload a PDF file and ask questions about it")

api_key = st.text_input("Enter Groq API Key", type="password")

if api_key:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key.strip())
    session_id = st.text_input("Session ID",value='default_session')

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf",accept_multiple_files=False)

    if uploaded_files:
        documents = []
        temp_pdf = "./temp.pdf"

        # ✅ get name BEFORE reading
        file_name = uploaded_files.name
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_files.read())

        
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        
        contextualize_q_system_prompt= '''
        Given a chat history and latest question, 
        1. Contextualize the question based on the chat history
        2. Return the contextualized question
        '''
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        #answer prompt

        system_prompt = '''
        You are a helpful assistant that can answer questions about a given document.
        If you dont know the answer say no. Make the answer concise and to the point and not more than 3 sentences.
        {context}
        '''
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(llm,qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key='chat_history',
        output_messages_key='answer')

        user_input = st.text_input("Ask a question about the document:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable":{'session_id':session_id}}
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write('Chat History:',session_history.messages)
else:
    st.warning("Please enter a valid Groq API key")


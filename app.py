import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile


load_dotenv()


groq_api_key = os.getenv('GROQ_API_KEY')


st.markdown("<h2 style='text-align: center;'>PDF Insights: Interactive Q&A with Llama3 & Groq API</h2>", unsafe_allow_html=True)


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):


    if "vector_store" not in st.session_state:

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:

            temp_file.write(pdf_file.read())

            pdf_file_path = temp_file.name

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        
        st.session_state.loader = PyPDFLoader(pdf_file_path)

        st.session_state.text_document_from_pdf = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)

        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)


pdf_input_from_user = st.file_uploader("Upload the PDF file", type=['pdf'])


if pdf_input_from_user is not None:

    if st.button("Create the Vector DB from the uploaded PDF file"):
        
        if pdf_input_from_user is not None:
            
            create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
            
            st.success("Vector Store DB for this PDF file Is Ready")
        
        else:
            
            st.write("Please upload a PDF file first")



if "vector_store" in st.session_state:

    user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")

    if st.button('Submit Prompt'):

        if user_prompt:
            
            if "vector_store" in st.session_state:

                document_chain = create_stuff_documents_chain(llm, prompt)

                retriever = st.session_state.vector_store.as_retriever()

                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                response = retrieval_chain.invoke({'input': user_prompt})

                st.write(response['answer'])

            else:

                st.write("Please embed the document first by uploading a PDF file.")

        else:

            st.error('Please write your prompt')

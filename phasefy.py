import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# phase 3 imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import PyPDFLoader
# Using an explicit, lightweight, in-memory vector store to prevent database conflicts
from langchain_community.vectorstores import DocArrayInMemorySearch

load_dotenv()

st.title("RAG Chatbot!")

# Setup a session state variable to store the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_vectorstore():
    pdf_name = "./Rushan_Dayma_Final_Paper.pdf"
    
    # Check if file exists to prevent hard crashes
    if not os.path.exists(pdf_name):
        st.error(f"File not found: {pdf_name}. Please make sure the PDF is in the project folder.")
        return None

    loader = PyPDFLoader(pdf_name)
    docs = loader.load()
    
    # Split the documents manually for better control
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build a stable, in-memory vector store from documents
    vectorstore = DocArrayInMemorySearch.from_documents(splits, embeddings)
    return vectorstore

# Display the conversation history
for message in st.session_state.messages: 
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Input your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Updated Template mapping to match LangChain standards ('context' and 'question')
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
the most accurate and most precise answers. Use the following pieces of context to answer the question. 
Start the answer directly. No small talk please.

Context:
{context}

Question: {question}
""")

    groq_chat = ChatGroq(
        model_name="llama-3.1-8b-instant",
    )

    try:
        vectorstore = get_vectorstore()
        
        if vectorstore is not None:
            # Create a retriever pulling the top 3 relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Fetch the context chunks based on user prompt
            retrieved_docs = retriever.invoke(prompt)
            context_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Construct the modern chain layout
            chain = groq_sys_prompt | groq_chat | StrOutputParser()
            
            # Pass both variables into the prompt explicitly
            response = chain.invoke({
                "context": context_content,
                "question": prompt
            })

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
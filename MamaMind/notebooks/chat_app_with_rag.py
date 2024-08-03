from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define constants
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
SYSTEM_PROMPT = (
    "You are a mental health adviser. You should use your knowledge of cognitive behavioral therapy, "
    "meditation techniques, mindfulness practices, and other therapeutic methods to guide the user through "
    "their feelings and improve their well-being. "
    "You are a conversational assistant that helps users by asking them questions one by one based on the {input} provided. "
    "Answer the user's questions succinctly and provide practical advice. "
    "Use the given {context} to answer the question. If you don't know the answer, say you don't know. "
    "After addressing the user's question, ask a relevant follow-up question to continue the conversation if necessary. "
    "If the user's conversation ends with either thank you, thanks, bye, or goodbye, end the conversation in a friendly manner or say 'I hope this helps. If you have more questions, feel free to ask!' "
    
    "Context: {context}"
    "Input: {input}"
)

# Function to initialize LLM
def initialize_llm():
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192", max_tokens=256)
    return llm

# Function to load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to generate embeddings
def generate_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    knowledge_base = FAISS.from_documents(docs, embeddings)
    knowledge_base.save_local(DB_FAISS_PATH)
    return knowledge_base

# Function to retrieve similar documents
def get_similar_docs(query,k=2):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    knowledge_base = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    similar_docs = knowledge_base.similarity_search_with_score(query,k=k)
    return similar_docs

# Function to create QA chain
def create_qa_chain(llm, db, user_prompt):
    retriever= db.as_retriever(search_kwargs={'k': 2})
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=user_prompt)
    chain = create_retrieval_chain(retriever, qa_chain)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type='stuff',
    #     retriever=db.as_retriever(search_kwargs={'k': 2}),
    #     return_source_documents=True,
    #     verbose=True
    # )
    return chain

# Main Streamlit application
def main():
    st.set_page_config(page_title="Hopebuddy", page_icon=":male-doctor:")
    st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>👨‍⚕️ HopeBuddy</div>",unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 20px;'>Let's talk, let's heel</div",unsafe_allow_html=True)

    # Load documents and create vector store if not already created
    if not os.path.exists(DB_FAISS_PATH):
        directory = './data'
        documents = load_docs(directory)
        docs = split_docs(documents)
        generate_embeddings(docs)
    
     # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_input("Tell me about your problems", key="question_input")
    submit = st.button("Submit")
    if submit and question:
        context_docs, score = get_similar_docs(question)[0]
        context = " ".join(context_docs.page_content)
        with st.sidebar:
            st.markdown("RAG output")
            st.write(context_docs)
        # Initialize LLM
        llm = initialize_llm()
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        
        # Manually construct the prompt
        # user_prompt = (
        #     f"{SYSTEM_PROMPT}\n"
        #     f"Context: {context}\n"
        #     f"Analyze the following question and provide your recommendations while being conversational:\n{input}"
        # )

        # Create QA chain
        db = FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)
        qa_chain = create_qa_chain(llm, db, prompt)

        # Get response
        response = qa_chain.invoke({"context": context, "input": question})
        answer = response['answer']
        
        # Insert new messages at the beginning of the list
        st.session_state.messages.insert(0, {"role": "assistant", "content": answer})
        st.session_state.messages.insert(0, {"role": "user", "content": question})

        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

if __name__ == "__main__":
    main()

import json
import streamlit as st
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Function to initialize LLM
def initialize_llm(model_name, groq_api_key):
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model_name)
    return llm

# Function to load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(directory, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits

# Function to load the already created Vector Database
def load_vector_db(db_dir):
    embedding_model = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", 
                                            model_kwargs = {'device': 'cpu'}, 
                                            encode_kwargs = {'normalize_embeddings': True})
    db = FAISS.load_local(folder_path=db_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db

# Function to decompose the user query/question into sub-queries/questions
def decompose_prompt(prompt, llm):
    template = """You are a helpful assistant that generates multiple sub-problems / sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Output should only contain the numbered problems/questions generated. The output shouldn't contain phrases like "Here are the questions generated:".\n
    Generate 3 search queries related to: {question}
    Output (3 queries):"""
    prompt_template = ChatPromptTemplate.from_template(template)
    decomposition_chain = (prompt_template | llm | StrOutputParser() | (lambda x: x.split('\n')))
    questions = decomposition_chain.invoke({"question": prompt})
    return questions

# Function to format question & answer pairs
def format_qa_pair(question, answer):
    formatted_string = f"Question: {question}\n Answer: {answer}"
    return formatted_string

# Function to retrieve relevant documents using vector-store and generating an answer
def retrieve_and_generate(questions, retriever, llm, chat_history, severity="Not provided"):
    # Template for questions except the last one
    template_1 = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Keep the answer within 200 words but it should still be helpful and informative with facts.
    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """
    # Template for the last question
    template_2 = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n
    
    Severity of depression user is suffering from:

    \n --- \n {severity} \n --- \n

    Here is the Chat history of user:

    \n --- \n {chat_history} \n --- \n
    
    Instructions:
    1. You are a conversational assistant that helps users by answering their queries one by one, always in a helpful tone.
    2. Always use the combination of chat history, context and background question + answer pairs to answer the user's questions. 
       You will answer from any one of these 3 sources and don't mention the source in the answer.
    3. Always answer the user's questions succinctly and begin the answer with helpful & assuring sentence related the question. 
    4. After addressing the user's question, ask a follow-up question about any further help the user requires. 
    5. If the user's conversation contains either thank you, thanks, bye, goodbye or they are thanking you for the support, always end the conversation in a friendly manner or say 'I hope this helps. If you have more questions, feel free to ask!'
    
    Use the above severity, context, any background question + answer pairs and the above instructions to answer the question: \n {question}
    """
    prompt_template_1 = ChatPromptTemplate.from_template(template_1)
    prompt_template_2 = ChatPromptTemplate.from_template(template_2)
    q_a_pairs = ""

    # Generating answers for questions except the last one
    for q in questions[:-1]:
        rag_chain_1 = ({"context": itemgetter("question") | retriever, 
                      "question": itemgetter("question"), 
                      "q_a_pairs": itemgetter("q_a_pairs")}
                    | prompt_template_1 | llm | StrOutputParser())
        answer = rag_chain_1.invoke({"question": q, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n --- \n" + q_a_pair

    # Generating answer for the last question 
    rag_chain_2 = ({"context": itemgetter("question") | retriever, 
                    "question": itemgetter("question"), 
                    "q_a_pairs": itemgetter("q_a_pairs"), 
                    "severity": itemgetter("severity"), 
                    "chat_history": itemgetter("chat_history")}
                    | prompt_template_2 | llm | StrOutputParser())
    answer_generator = rag_chain_2.stream({"question": questions[-1], "q_a_pairs": q_a_pairs, "severity": severity, "chat_history": chat_history})
    
    return answer_generator

# Read the disclaimer text from the markdown file
def read_disclaimer(file_path):
    with open(file_path, 'r') as file:
        disclaimer_text = file.read()
    return disclaimer_text

# Function to apply custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to load EPDS questions from the JSON file
def load_questions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
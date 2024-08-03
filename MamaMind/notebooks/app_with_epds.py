import os
import time
import json
from dotenv import load_dotenv
import streamlit as st
from utils import *
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="MamaMind", page_icon=":male-doctor:")

# Define constants
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
SYSTEM_PROMPT = (
    "You are a mental health adviser. You should use your knowledge of cognitive behavioral therapy, "
    "meditation techniques, mindfulness practices, and other therapeutic methods to guide the user through "
    "their feelings and improve their well-being. "
    "You are a conversational assistant that helps users by asking them questions one by one based on the {input} or the {input} and {severity} provided. "
    "Answer the user's questions succinctly and provide practical advice. "
    "Use the given {context} to answer the question. If you don't know the answer, say you don't know. "
    "After addressing the user's question, ask a relevant follow-up question to continue the conversation if necessary. "
    "If the user's conversation ends with either thank you, thanks, bye, or goodbye, end the conversation in a friendly manner or say 'I hope this helps. If you have more questions, feel free to ask!' "
    
    "Context: {context}"
    "Input: {input},{severity}"
)

# Function to load questions from the JSON file
def load_questions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to show/hide sidebar based on the current tab
def toggle_sidebar(visible):
    if visible:
        st.markdown('<style>.css-1lcbmhc.e1fqkh3o4 {display: block;}</style>', unsafe_allow_html=True)
    else:
        st.markdown('<style>.css-1lcbmhc.e1fqkh3o4 {display: none;}</style>', unsafe_allow_html=True)

def display_welcome_message(llm):
    prompt = """
            You are an AI assistant designed to welcome users to the "MamaMind-Gentle Guidance for new beginnings" app.
            Your task is to greet the user with an appealing welcome message and then prompt them to provide their query or concern. Follow these steps:

            1. Start with a warm and friendly welcome message.
            2. Briefly explain what the "MamaMind" app does. 
            3. Invite the user to answer the Edinurgh depression questionnaire first so as to assess the depression level.

            The app provides the expecting and new mothers with mental health recommendations based on cognitive behavioral therapy, 
            assesses the severity of perinatal depression using Edinburgh Depression Scale and provides the response accordingly.
            
            Here is an example format you can follow:
            ---
            **Welcome to MamaMind!**

            We are here to support you through your journey with perinatal depression. Whether you are expecting a baby or navigating the postpartum period, MamaMind offers compassionate advice, helpful resources, and a listening ear.

            **How can we assist you today?**

            Please share your questions or concerns, and let us provide the support you need.
            ---
            """
    response = llm.invoke(prompt)
    message_text = response.content
    lines = message_text.split(" ")
    for line in lines:
        yield line + " "
        time.sleep(0.02)
            

local_css("custom.css")

# File path to the disclaimer markdown file
disclaimer_file_path = 'disclaimer.md'
disclaimer_text = read_disclaimer(disclaimer_file_path)

# Main Streamlit application
def main():

    # Initialize session state to manage sidebar visibility
    if 'sidebar_visible' not in st.session_state:
        st.session_state.sidebar_visible = False

    tab1, tab2 = st.tabs(["About","Chat with MamaMind"])
    
    with tab1:
        toggle_sidebar(False)
        st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>👨‍⚕️ MamaMind</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-size: 24px;'>Gentle Guidance for New Beginnings</div", unsafe_allow_html=True)
        st.divider()
        
        st.header("About the application")
        st.markdown("""
            This application helps the people with perinatal depression.
        """)

        # Display disclaimer
        st.subheader("Disclaimer")
        st.markdown(disclaimer_text)

    
    with tab2:
        toggle_sidebar(True)
        st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>👨‍⚕️ MamaMind</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-size: 24px;'>Gentle Guidance for New Beginnings</div", unsafe_allow_html=True)
        
        # Get user's Groq api key
        with st.sidebar:
            groq_api_key = st.text_input(label = "**Groq API key**", placeholder="Ex gsk-2twmA8tfCb8un4...",
            key ="groq_api_key_input", help = "How to get a Groq api key: Visit https://console.groq.com/login")

            # Initialize session state for the model if it doesn't already exist
            if 'selected_model' not in st.session_state:
                st.session_state['selected_model'] = ""
            # Container for markdown text
            with st.container():
                st.markdown("""Make sure you have entered your API key.
                            Don't have an API key yet?
                            Visit https://console.groq.com/login and Get your API key""")
                st.session_state['selected_model'] = st.selectbox("Choose the model",("","llama3-70b-8192","gemma2-9b-it"),key="tab2_sidebar_selectbox")
                model_chosen = st.session_state['selected_model']
            if model_chosen:
                st.write(f"You have selected the model: {model_chosen}")
            else:
                st.write("Please select a model.")
                    
        if groq_api_key and model_chosen:
            
            llm = initialize_llm(model_chosen, groq_api_key)

            # Initialize session state to keep track of question index and responses
            if 'started' not in st.session_state:
                st.session_state.started = False
            if 'question_index' not in st.session_state:
                st.session_state.question_index = 0  # or however you want to initialize it
            if 'responses' not in st.session_state:
                st.session_state.responses = []
            if 'scores' not in st.session_state:
                st.session_state.scores = []

            # Load the EPDS questions
            epds_questions = load_questions('epds_questions.json')
            if "welcome_message_displayed" not in st.session_state:
                #Generate Welcome Message with User Input
                with st.chat_message("assistant"):
                    llm = llm
                    st.write_stream(display_welcome_message(llm))
                    st.session_state["welcome_message_displayed"] = True

            # Function to move to the next question
            def next_question(response, score):
                st.session_state.responses.append(response)
                st.session_state.scores.append(score)
                if st.session_state.question_index < len(epds_questions) - 1:
                    st.session_state.question_index += 1
                else:
                    st.session_state.question_index = 'completed'


            # Ask if the user wants to start the questionnaire
            if not st.session_state.started:
                with st.chat_message("assistant"):
                    # st.write("To assess the level of severity of the depression, we would like you to answer the Ediburgh Depression questionnaire.")
                    start = st.radio("Would you like to answer the EPDS questionnaire?", ("Yes", "No"), index=None, horizontal=True)
                if start == "Yes":
                    st.session_state.started = True
                elif start == "No":
                    st.write("Ask any questions you have.")
            else:
                # Display the current question
                if st.session_state.question_index == 'completed':
                    with st.chat_message():
                        st.write("You have completed the EPDS questionnaire.")
                    # st.write("Your responses:", st.session_state.responses)
                    # st.write("Your scores:", st.session_state.scores)
                    # st.write("Total score:", sum(st.session_state.scores))
                    epds_score = sum(st.session_state.scores)
                    st.write("Your total score:", epds_score)
                else:
                    current_question = epds_questions[st.session_state.question_index]
                    with st.chat_message("assistant"):
                        st.write(current_question["question"])
                        response = st.radio("Select your response:", current_question["options"],horizontal=True)
                    if st.button("**Next**"):
                        if response:  # Ensure a response is selected
                            score = current_question["scores"][current_question["options"].index(response)]
                            next_question(response, score)
                        else:
                            st.warning("Please select a response before proceeding.")

        #     with st.chat_message("assistant"):
        #         st.write("How can I help you?")

        #     query = st.text_input("Tell me about your problems")

        #     if query:
        #         if os.path.exists(DB_FAISS_PATH):
        #             context_docs = get_similar_docs(question)
        #             context = " ".join(context_docs.page_content)
        #             with st.sidebar:
        #                 st.markdown("RAG output")
        #                 st.write(context_docs)
        #         else:
        #             directory = './data'
        #             documents = load_docs(directory)
        #             docs = split_docs(documents)
        #             generate_embeddings(docs)


        #         # Initialize LLM
        #         llm = llm
                
        #         prompt = ChatPromptTemplate.from_messages(
        #             [
        #                 ("system", SYSTEM_PROMPT),
        #                 ("human", "{input}"),
        #             ]
        #         )
                
        #         # Create QA chain
        #         db = FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)
        #         qa_chain = create_qa_chain(llm, db, prompt)

        #         # Get response
        #         response = qa_chain.invoke({"context": context, "input": query})
        #         answer = response['answer']
            
        #         # Insert new messages at the beginning of the list
        #         st.session_state.messages.insert(0, {"role": "assistant", "content": answer})
        #         st.session_state.messages.insert(0, {"role": "user", "content": query})

        #         # Display messages
        #         for message in st.session_state.messages:
        #             if message["role"] == "user":
        #                 with st.chat_message("user"):
        #                     st.write(message["content"])
        #             else:
        #                 with st.chat_message("assistant"):
        #                     st.write(message["content"])
        # else:
        #     st.error("Please enter your Groq API key")

        # Apply the CSS class to hide the sidebar

if __name__ == "__main__":
    main()

import os
import streamlit as st
from textwrap import dedent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from utils import local_css, load_questions, load_vector_db, decompose_prompt, retrieve_and_generate, initialize_llm

# Load environment variables
load_dotenv()

# Streamlit app settings
st.set_page_config(page_title="MamaMind", page_icon=":female-doctor:")
# col1, col2 = st.columns([3, 1])
with st.sidebar:
    st.image("static/image1-removebg-preview.png")
# Apply custom CSS
local_css("styles.css")

# Folder path
FAISS_DB_PATH = './vectorstore'

# Function to     
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
    return llm.stream(prompt)

# Function to return Severity of user's depression
def interpret_epds_score(score):
    if score <= 9:
        return "None to minimal depression"
    elif 10 <= score <= 12:
        return "Mild depression"
    elif 13 <= score <= 14:
        return "Moderate depression"
    else:
        return "Severe depression"

def begin_chat(llm, retriever, severity):
    with st.chat_message("assistant"):
        st.write('How can I help you?')

    # Display chat_history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    query = st.chat_input("Tell me about your problems")

    if query is not None and query != "":
        # Display the query
        with st.chat_message('user') :
            st.markdown(query)
            
        # Decompose the question
        decomposed_questions = decompose_prompt(query, llm)

        # Get response
        with st.chat_message('assistant'):
            answer = st.write_stream(retrieve_and_generate(decomposed_questions, retriever, llm, chat_history=st.session_state.chat_history, severity=severity))

        # Append to chat history
        st.session_state.chat_history.append(HumanMessage(query))
        st.session_state.chat_history.append(AIMessage(answer))

def main():    
    
    # Get user's Groq api key
    with st.sidebar:
        groq_api_key = st.text_input(label = "**Groq API key**", placeholder="Ex gsk-2twmA8tfCb8un4...",
        key ="groq_api_key_input", help = "How to get a Groq api key: Visit https://console.groq.com/login", type="password")

        # Initialize session state for the model if it doesn't already exist
        if 'selected_model' not in st.session_state:
            st.session_state['selected_model'] = ""
        # Container for markdown text
        with st.container():
            st.markdown("Make sure you have entered your API key")
            model_chosen = st.selectbox("Choose the model", ("llama3-70b-8192","mixtral-8x7b-32768"), key="tab2_sidebar_selectbox", index=None)
            st.session_state['selected_model'] = model_chosen
    
    st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>👩‍⚕️ MamaMind</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 24px;'>Gentle Guidance for New Beginnings</div", unsafe_allow_html=True)
    st.divider() 

    if groq_api_key and model_chosen:

        llm = initialize_llm(model_chosen, groq_api_key)

        # Load the EPDS questionnaire
        epds_questions = load_questions('static/epds_questions.json')

        # Load Vector store DB & Retriever
        vectorstore = load_vector_db(FAISS_DB_PATH)
        retriever = vectorstore.as_retriever(search_kwargs={'k':3})

        # Initialize session state for messages
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "query" not in st.session_state:
            st.session_state.query = ""

        # Initialize session state to keep track of question index and responses
        if 'started' not in st.session_state:
            st.session_state.started = False
        if 'question_index' not in st.session_state:
            st.session_state.question_index = 0  # to start from the first question in the list
        if 'responses' not in st.session_state:
            st.session_state.responses = []
        if 'scores' not in st.session_state:
            st.session_state.scores = []

        #Generate Welcome Message with User Input
        if "welcome_message_displayed" not in st.session_state:
            with st.chat_message("assistant"):
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
                start = st.radio("Would you like to answer the EPDS questionnaire?", ("Yes", "No"), index=None, horizontal=True, key='start_radio')
            if start == "Yes":                    
                st.session_state.started = True
                st.session_state.question_index = 0
                st.session_state.responses = []
                st.session_state.scores = []
                st.rerun()
            elif start == "No":
                st.session_state.started = False
                severity = "Not provided"
                with st.chat_message('assistant'):
                    st.write("No problem. Please ask any questions you have to MamaMind 🙂.")
                begin_chat(llm, retriever, severity)
        else:
            if st.session_state.question_index == 'completed':
                with st.chat_message('assistant'):
                    st.write("You have completed the EPDS questionnaire.")
                epds_score = sum(st.session_state.scores)
                severity = interpret_epds_score(epds_score)
                begin_chat(llm, retriever, severity)
            else:
                current_question = epds_questions[st.session_state.question_index]
                st.write("Click on the options that comes closest to how you have felt IN THE PAST 7 DAYS—not just how you feel today.\
                        Complete all 10 questions.\n\
                        Kindly Note: This is a screening test; not a medical diagnosis.")
                with st.chat_message("assistant"):
                    st.write(current_question["question"])
                    response = st.radio("Select your response:", current_question["options"], index=None, horizontal=False)
                if st.button("**Next**"):
                    if response:  # Ensure a response is selected
                        score = current_question["scores"][current_question["options"].index(response)]
                        next_question(response, score)
                        st.rerun()
                    else:
                        st.warning("Please select a response before proceeding.")

    # with col2:
    #     st.write("Example Prompts")
    #     with st.expander("View Example Prompts"):
    #         st.markdown("""
    #         - **How can I manage my anxiety during pregnancy?**
    #         - **What are some coping strategies for postpartum depression?**
    #         - **Can you suggest activities to improve my mental health?**
    #         - **What resources are available for new mothers?**
    #         - **How do I recognize the signs of perinatal depression?**
    #         """)
        
if __name__ == "__main__":
    main()

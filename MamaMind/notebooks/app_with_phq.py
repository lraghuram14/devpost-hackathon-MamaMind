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
from utils import *

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

# Introductory statement
intro_statement = "Over the last 2 weeks, how often have you been bothered by the following problems?"

# PHQ-2 Questions
phq2_questions = [
    "1. Little interest or pleasure in doing things?",
    "2. Feeling down, depressed, or hopeless?"
]

# PHQ-9 Questions
phq9_questions = [
    "1. Little interest or pleasure in doing things?",
    "2. Feeling down, depressed, or hopeless?",
    "3. Trouble falling or staying asleep, or sleeping too much?",
    "4. Feeling tired or having little energy?",
    "5. Poor appetite or overeating?",
    "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "7. Trouble concentrating on things, such as reading the newspaper or watching television?",
    "8. Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
    "9. Thoughts that you would be better off dead, or thoughts of hurting yourself in some way?"
]

# File path to the disclaimer markdown file
disclaimer_file_path = 'disclaimer.md'
disclaimer_text = read_disclaimer(disclaimer_file_path)

# Main Streamlit application
def main():
    st.set_page_config(page_title="MamaMind", page_icon=":male-doctor:")
    st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>👨‍⚕️ MamaMind</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 24px;'>Gentle Guidance for New Beginnings</div", unsafe_allow_html=True)
    st.divider()
    st.sidebar.markdown("Please read the following disclaimer before using this application:")

    
    # Display disclaimer
    with st.sidebar.popover("Disclaimer"):
        st.markdown(disclaimer_text)
    
    # Get user's GooglePalm key
    with st.sidebar:
        groq_api_key = st.text_input(label = "**Groq API key**", placeholder="Ex gsk-2twmA8tfCb8un4...",
        key ="groq_api_key_input", help = "How to get a Groq api key: Visit https://console.groq.com/login")

        # Container for markdown text

        with st.container():
            st.markdown("""Make sure you have entered your API key.
                        Don't have an API key yet?
                        Read this: Visit https://console.groq.com/login and Get your API key""")
            model_chosen = st.radio("Choose the model",("llama3-70b-8192","gemma2-9b-it"))
            if model_chosen == "llama3-70b-8192":
                model = "llama3-70b-8192"
            else:
                model = "gemma2-9b-it"
    
    if groq_api_key:
        st.markdown("<div style='text-align: center: font-size: 20px;'>Please use the sidebar to complete the PHQ-2 and PHQ-9 screenings.</div>", unsafe_allow_html=True)
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model)
        # Sidebar with PHQ-2
        st.sidebar.header("PHQ-2 Screening")
        st.sidebar.write(intro_statement)
        phq2_scores = []
        for i, question in enumerate(phq2_questions):
            response = st.sidebar.radio(question, ("Not at all", "Several days", "More than half the days", "Nearly every day"), key=f"phq2_{i}")
            phq2_scores.append(response)
        
        # PHQ-2 Scoring
        phq2_score_mapping = {
            "Not at all": 0,
            "Several days": 1,
            "More than half the days": 2,
            "Nearly every day": 3
        }
        phq2_total_score = sum(phq2_score_mapping[response] for response in phq2_scores)
        st.sidebar.write(f"PHQ-2 Total Score: {phq2_total_score}")

        # Display PHQ-9 only if PHQ-2 score is >= 3
        if phq2_total_score >= 3:
            severity = "potential depression"
            st.sidebar.info("PHQ-2 indicates potential depression. Further assessment with PHQ-9 is recommended.")
            st.sidebar.header("PHQ-9 Screening")
            st.sidebar.write(intro_statement)
            phq9_scores = []
            for i, question in enumerate(phq9_questions):
                response = st.sidebar.radio(question, ("Not at all", "Several days", "More than half the days", "Nearly every day"), key=f"phq9_{i}")
                phq9_scores.append(response)

            # PHQ-9 Scoring
            phq9_total_score = sum(phq2_score_mapping[response] for response in phq9_scores)
            st.sidebar.write(f"PHQ-9 Total Score: {phq9_total_score}")

            # Interpretation of Scores
            if phq9_total_score >= 10:
                st.sidebar.info("PHQ-9 indicates moderate to severe depression. Professional evaluation is recommended.")
                severity = "moderate to severe depression"
            elif 5 <= phq9_total_score < 10:
                st.sidebar.info("PHQ-9 indicates mild depression. Monitoring and support are recommended.")
                severity = "mild depression"
            else:
                st.sidebar.info("PHQ-9 score is below the threshold for depression.")
                severity = "below the threshold for depression"
        else:
            st.sidebar.info("PHQ-2 score is below the threshold for depression.")
            severity = "below the threshold for depression"

        submitted = st.sidebar.button("**Submit**",key="sidebar_button")

        # Initialize session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "question" not in st.session_state:
            st.session_state.question = ""
            
        if submitted:
            st.info(f"Detected Severity: {severity}")

            with st.chat_message("assistant"):
                st.write("How can I help you?")

            query = st.text_input("Tell me about your problems")

            if query:
                if os.path.exists(DB_FAISS_PATH):
                    context_docs = get_similar_docs(question)
                    context = " ".join(context_docs.page_content)
                    with st.sidebar:
                        st.markdown("RAG output")
                        st.write(context_docs)
                else:
                    directory = './data'
                    documents = load_docs(directory)
                    docs = split_docs(documents)
                    generate_embeddings(docs)


                # Initialize LLM
                llm = llm
                
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", SYSTEM_PROMPT),
                        ("human", "{input}"),
                    ]
                )
                
                # Create QA chain
                db = FAISS.load_local(DB_FAISS_PATH, HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'}), allow_dangerous_deserialization=True)
                qa_chain = create_qa_chain(llm, db, prompt)

                # Get response
                response = qa_chain.invoke({"context": context, "input": query})
                answer = response['answer']
            
                # Insert new messages at the beginning of the list
                st.session_state.messages.insert(0, {"role": "assistant", "content": answer})
                st.session_state.messages.insert(0, {"role": "user", "content": query})

                # Display messages
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
    else:
        st.error("Please enter your Groq API key")

if __name__ == "__main__":
    main()

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# ---------------------------
# Load .env and API Key
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Please set it as an environment variable.")
    st.stop() # Stop the app if the key is missing

# ---------------------------
# Streamlit App Configuration
# ---------------------------
st.set_page_config(page_title="Context-Aware Chatbot", page_icon="🤖")
st.title("🤖 Context-Aware Chatbot")

# ---------------------------
# Asynchronous Setup Helper
# ---------------------------
async def _async_model_setup():
    """
    An async function to set up the models.
    This is run inside the cached function to ensure it has an event loop.
    """
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        model_kwargs={"device": "cpu"}
    )

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    return embeddings, llm

# ---------------------------
# Cached Resource Setup
# This function will run only once and cache the entire QA chain.
# ---------------------------
@st.cache_resource
def initialize_qa_chain():
    """
    Initializes the ConversationalRetrievalChain.
    This function is cached to prevent re-running on every Streamlit update.
    """
    # The 'no current event loop' error happens because the model
    # initialization is an async operation. We explicitly run it
    # within an event loop here to guarantee it works.
    embeddings, llm = asyncio.run(_async_model_setup())

    # Load your text data from the file
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        st.error("❌ 'data.txt' not found. Please make sure it's in the same directory as app.py.")
        return None

    documents = [Document(page_content=text)]

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create a vector store from the documents
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Initialize conversational memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return qa_chain

# ---------------------------
# Session State & Main Logic
# ---------------------------
# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get the cached QA chain
st.session_state.qa_chain = initialize_qa_chain()

# Check if the chain was initialized successfully before proceeding
if st.session_state.qa_chain is None:
    st.stop()

user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking... ⏳"):
        # Get the response from the chain
        response = st.session_state.qa_chain({"question": user_input})
        answer = response["answer"]

    # Append the user's question and the bot's answer to the chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

    # Display the bot's response
    st.success(answer)

# ---------------------------
# Display Chat History
# ---------------------------
st.subheader("📜 Chat History")
if st.session_state.chat_history:
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 {role}:** {message}")
        else:
            st.markdown(f"**🤖 {role}:** {message}")


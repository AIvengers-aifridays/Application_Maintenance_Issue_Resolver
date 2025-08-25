import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import httpx
import tiktoken
import requests
from dotenv import load_dotenv

# ---- Load environment variables ----
load_dotenv()

# ---- Disable SSL verification for internal endpoints ----
requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = False
requests.get = session.get

# Cache dir for tokens
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
client = httpx.Client(verify=False)

# ---- LLM Setup ----
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",   # internal deployed model
    api_key=os.getenv("API_KEY"),               # replace with valid key
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("API_KEY"),               # replace with valid key
    http_client=client
)

# ---- Streamlit UI ----
st.set_page_config(page_title="Application Maintenance Issue Resolver")
st.title("🛠️ AI-Powered Application Maintenance Issue Resolver")

# ---- Step 1: Load VectorDB ----
try:
    vectorstore = Chroma(
        persist_directory="vectordb",  # points to your vectordb.sqlite3
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"❌ Failed to load VectorDB: {e}")
    st.stop()

# ---- Step 2: Build Retrieval QA Chain ----
system_prompt = """
You are an AI Maintenance Assistant.
You can ONLY help with application maintenance issues using the provided logs and troubleshooting manuals. 
If the user asks anything unrelated (like jokes, personal questions, general knowledge, etc.), 
reply with: "⚠️ I am designed only to help with application maintenance issues. Please describe your issue."
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": None},
)

# ---- Step 3: Query UI ----
st.subheader("💬 Ask about your issue")

query = st.text_input("⚡ Describe the maintenance issue you are facing:")
submit = st.button("🔍 Submit Query")

if query and (submit or query):  # Enter key OR Submit button
    if any(word in query.lower() for word in ["joke", "weather", "who", "what", "when", "where", "game"]):
        st.subheader("⚠️ Notice:")
        st.write("I am designed only to help with application maintenance issues. Please describe your issue.")
    else:
        with st.spinner("🔎 Searching knowledge base..."):
            answer = qa_chain.run(query)
        st.subheader("✅ Suggested Resolution Steps:")
        st.write(answer)

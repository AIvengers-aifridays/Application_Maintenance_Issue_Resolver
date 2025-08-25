import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key=os.getenv("API_KEY"),   # replace with valid key
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("API_KEY"),   # replace with valid key
    http_client=client
)

# ---- Streamlit UI ----
st.set_page_config(page_title="Application Maintenance Issue Resolver")
st.title("🛠️ AI-Powered Application Maintenance Issue Resolver")

# ---- Step 1: Load CSV Data ----
logs_path = "data/logs.csv"
manuals_path = "data/manuals.csv"

try:
    logs_df = pd.read_csv(logs_path)
    manuals_df = pd.read_csv(manuals_path)
except Exception as e:
    st.error(f"❌ Failed to load CSV files: {e}")
    st.stop()

# Debug: Show available columns
st.sidebar.write("📂 Logs CSV Columns:", logs_df.columns.tolist())
st.sidebar.write("📂 Manuals CSV Columns:", manuals_df.columns.tolist())

# ---- Step 2: Convert into documents ----
docs = []

# Logs CSV
if {"issue", "log"}.issubset(logs_df.columns):
    for _, row in logs_df.iterrows():
        docs.append(f"Issue: {row['issue']}\nLog: {row['log']}")
else:
    for _, row in logs_df.iterrows():
        docs.append(f"Issue: {row.iloc[0]}\nLog: {row.iloc[1]}")

# Manuals CSV
if {"title", "steps"}.issubset(manuals_df.columns):
    for _, row in manuals_df.iterrows():
        docs.append(f"Manual Title: {row['title']}\nResolution Steps: {row['steps']}")
else:
    for _, row in manuals_df.iterrows():
        docs.append(f"Manual Title: {row.iloc[0]}\nResolution Steps: {row.iloc[1]}")

# ---- Step 3: Split text ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for d in docs:
    chunks.extend(splitter.split_text(d))

# ---- Step 4: Create Vector Store ----
vectorstore = Chroma.from_texts(chunks, embedding_model)
retriever = vectorstore.as_retriever()

# ---- Step 5: RetrievalQA with strict prompt ----
system_prompt = """
You are an AI Maintenance Assistant. a
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

# ---- Step 6: Query UI ----
query = st.text_input("⚡ Describe the maintenance issue you are facing:")

if query:
    if any(word in query.lower() for word in ["joke", "weather", "who", "what", "when", "where"]):
        st.subheader("⚠️ Notice:")
        st.write("I am designed only to help with application maintenance issues. Please describe your issue.")
    else:
        answer = qa_chain.run(query)
        st.subheader("✅ Suggested Resolution Steps:")
        st.write(answer)

# convert_to_vectordb.py

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import httpx
import requests
from dotenv import load_dotenv

# ---- Load environment variables ----
load_dotenv()

# ---- Disable SSL verification for internal endpoints ----
requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = False
requests.get = session.get
client = httpx.Client(verify=False)

# ---- Paths ----
logs_path = "data/logs.csv"
manuals_path = "data/manuals.csv"
persist_directory = "vectordb"   # Folder where vector DB will be saved

# ---- Embedding Model ----
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("API_KEY"),   # replace with valid key
    http_client=client
)

# ---- Step 1: Load CSV Data ----
logs_df = pd.read_csv(logs_path)
manuals_df = pd.read_csv(manuals_path)

docs = []

# Logs CSV -> Convert rows to documents
if {"issue", "log"}.issubset(logs_df.columns):
    for _, row in logs_df.iterrows():
        docs.append(f"Issue: {row['issue']}\nLog: {row['log']}")
else:
    for _, row in logs_df.iterrows():
        docs.append(f"Issue: {row.iloc[0]}\nLog: {row.iloc[1]}")

# Manuals CSV -> Convert rows to documents
if {"title", "steps"}.issubset(manuals_df.columns):
    for _, row in manuals_df.iterrows():
        docs.append(f"Manual Title: {row['title']}\nResolution Steps: {row['steps']}")
else:
    for _, row in manuals_df.iterrows():
        docs.append(f"Manual Title: {row.iloc[0]}\nResolution Steps: {row.iloc[1]}")

# ---- Step 2: Split into chunks ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for d in docs:
    chunks.extend(splitter.split_text(d))

# ---- Step 3: Create and Persist VectorDB ----
vectorstore = Chroma.from_texts(chunks, embedding_model, persist_directory=persist_directory)
vectorstore.persist()

print(f"✅ VectorDB created successfully at: {persist_directory}")

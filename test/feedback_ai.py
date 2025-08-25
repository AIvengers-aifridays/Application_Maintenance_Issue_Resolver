import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import httpx
import tiktoken
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    api_key=os.getenv("API_KEY"),  # replace with valid key
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("API_KEY"),  # replace with valid key
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
    if any(word in query.lower() for word in ["joke", "weather", "who", "what", "when", "where","games"]):
        st.subheader("⚠️ Notice:")
        st.write("I am designed only to help with application maintenance issues. Please describe your issue.")
    else:
        with st.spinner("🔎 Searching knowledge base..."):
            answer = qa_chain.run(query)

        # ---- Step 4: Handle Weak/No Context ----
        if not answer or "I don't know" in answer or len(answer.strip()) < 30:
            st.warning("Currently I don’t have any context regarding this prompt right now.")

            if st.button("💡 Submit Feedback"):
                feedback = st.text_area("✍️ Provide the correct resolution/solution for this issue:")
                if st.button("📩 Submit Feedback Now"):
                    # Save feedback into JSON for persistence
                    feedback_entry = {"query": query, "solution": feedback}
                    feedback_file = "feedback.json"
                    if os.path.exists(feedback_file):
                        with open(feedback_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        data = []
                    data.append(feedback_entry)
                    with open(feedback_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)

                    # ---- 🔄 Retrain VectorDB immediately with feedback ----
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_text(f"User Query: {query}\nSuggested Solution: {feedback}")

                    vectorstore.add_texts(chunks)
                    vectorstore.persist()

                    st.success("✅ Thank you! Your feedback has been submitted and the system has been updated. It will be used to improve future responses.")
        else:
            st.subheader("✅ Suggested Resolution Steps:")
            st.write(answer)

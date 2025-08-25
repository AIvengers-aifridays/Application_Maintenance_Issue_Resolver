import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import httpx
import requests
import json
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
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",  # chosen for efficiency + reasoning
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

# ---- Dark Mode Toggle ----
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121212;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown, .stTextInput label, .stSelectbox label {
            color: white !important;
        }
        .stButton>button {
            background-color: #333333;
            color: white !important;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            background-color: #1e1e1e;
            color: white !important;
        }
        .stAlert, .stWarning, .stSuccess, .stInfo {
            background-color: #2a2a2a !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: white;
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown, .stTextInput label, .stSelectbox label {
            color: black !important;
        }
        .stButton>button {
            background-color: #f0f0f0;
            color: black !important;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            background-color: white;
            color: black !important;
        }
        .stAlert, .stWarning, .stSuccess, .stInfo {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        """,
        unsafe_allow_html=True
    )


st.title("🛠️ AI-Powered Application Maintenance Issue Resolver")

# ---- Step 1: Load VectorDB ----
try:
    vectorstore = Chroma(
        persist_directory="vectordb",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"❌ Failed to load VectorDB: {e}")
    st.stop()

# ---- Step 2: Build Retrieval QA Chain ----
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)

# ---- Feedback File ----
feedback_file = "feedback.json"

# ---- Step 3: Query UI ----
st.subheader("💬 Ask about your issue")
query = st.text_input("⚡ Describe the maintenance issue you are facing:")
submit = st.button("🔍 Submit Query")

if query and (submit or query):  # Enter key OR button
    # ---- Step 3A: Check irrelevant prompts ----
    irrelevant_keywords = [
        "joke", "weather", "movie", "song", "friend", "personal", 
        "love", "life", "story", "general", "funny", "poem","who", "games"
    ]
    if any(word in query.lower() for word in ["joke", "weather", "who", "what", "when", "where", "game"]):
        st.subheader("⚠️ Notice:")
        st.write("I am designed only to help with application maintenance issues. Please describe your issue.")
        st.stop()

    # ---- Step 3B: Check if feedback has validated solution ----
    validated_answer = None
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
        for entry in feedback_data:
            if entry["query"].lower() == query.lower() and entry.get("validated"):
                validated_answer = entry["solution"]
                break

    if validated_answer:
        st.subheader("✅ Best Known Resolution (from feedback):")
        st.write(validated_answer)
        st.info("This solution has been validated by users and prioritized for this issue.")
    else:
        # ---- Step 3C: Query AI model ----
        with st.spinner("🔎 Searching knowledge base..."):
            answer = qa_chain.run(query)

        if not answer or "I don't know" in answer or len(answer.strip()) < 30:
            st.warning("Currently I don’t have any context regarding this prompt right now.")
            
            if st.button("💡 Submit Feedback (Provide Resolution)"):
                feedback = st.text_area("✍️ Provide the correct resolution/solution for this issue:")
                if st.button("📩 Submit Feedback Now"):
                    feedback_entry = {"query": query, "solution": feedback, "validated": True}
                    if os.path.exists(feedback_file):
                        with open(feedback_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        data = []
                    data.append(feedback_entry)
                    with open(feedback_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                    st.success("✅ Thank you! Your feedback has been submitted. It will be used to improve future responses.")
        else:
            st.subheader("✅ Suggested Resolution Steps:")
            st.write(answer)

            # ---- Step 4: Post-Answer Feedback ----
            st.subheader("🤔 Was this solution helpful?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Yes, it worked"):
                    st.session_state['solution_validated'] = True
                    st.session_state['solution_failed'] = False

            with col2:
                if st.button("👎 No, it didn’t work"):
                    st.session_state['solution_failed'] = True
                    st.session_state['solution_validated'] = False


            # --- Outside the columns, render results full-width ---
            # ✅ Case: User clicked Yes
            if st.session_state.get('solution_validated', False):
                feedback_entry = {"query": query, "solution": answer, "validated": True}
                if os.path.exists(feedback_file):
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = []
                data.append(feedback_entry)
                with open(feedback_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

                st.success("✅ Great! This solution is now marked as the best for this issue and will be prioritized.")


            # ❌ Case: User clicked No
            if st.session_state.get('solution_failed', False):
                st.warning("⚠️ Generating an alternative resolution...")

                new_prompt = f"The previous solution did not work. Please suggest an alternative resolution for: {query}"
                new_answer = qa_chain.run(new_prompt)

                st.subheader("🔄 Alternative Resolution Steps:")
                st.write(new_answer)

                feedback_entry = {"query": query, "solution": new_answer, "validated": False}
                if os.path.exists(feedback_file):
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = []
                data.append(feedback_entry)
                with open(feedback_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

                st.info("📝 This alternative has been logged for review and improvement.")

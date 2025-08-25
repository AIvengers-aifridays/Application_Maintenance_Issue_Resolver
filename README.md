---

# 🛠️ Application Maintenance Issue Resolver

An **AI-powered assistant** designed to help maintenance teams resolve repetitive application issues faster.
It uses **LangChain, OpenAI embeddings, ChromaDB, and Streamlit** to provide **step-by-step resolutions** for known problems while continuously learning from user feedback.

---

## 🚀 Features

* **AI-Powered Query Resolution**

  * Uses LLMs (DeepSeek-V3, Azure AI, etc.) to provide solutions for application issues.
* **Dark/Light Mode Toggle** 🌙☀️

  * Seamless theme switching with custom styles (labels, text, inputs, buttons).
* **Knowledge Base with VectorDB** 📚

  * Issues and resolutions stored in **ChromaDB** using embeddings.
* **Feedback System** 🔄

  * Users can validate solutions (`✅ Helpful`) or mark them as incorrect (`❌ Not helpful`).
  * Validated solutions are **prioritized** for future queries.
* **Alternative Resolution Suggestions**

  * If a solution fails, an alternative resolution is generated.
* **Irrelevant Query Filtering** ⚠️

  * Non-maintenance prompts (jokes, weather, songs, etc.) are automatically rejected.
* **Secure API Key Handling** 🔑

  * API keys are stored in `.env` (not pushed to GitHub).

---

## 🏗️ Architecture

```plaintext
┌─────────────────────────────────────────┐
│           Streamlit Frontend             │
│   - Dark/Light Mode UI                   │
│   - Query input + feedback system        │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        Retrieval + Reasoning Layer       │
│   - LangChain RetrievalQA                │
│   - OpenAI/DeepSeek models               │
│   - Embedding Model (text-embedding-3)   │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        Vector Database (ChromaDB)        │
│   - Stores issue logs + resolutions      │
│   - Provides context to LLM              │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        Feedback Management (JSON)        │
│   - Stores validated/alternative answers │
│   - Improves response quality over time  │
└─────────────────────────────────────────┘
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vishal10kesharwani/application-maintenance-resolver.git
cd application-maintenance-resolver
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup `.env` File

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

*(The app will automatically load the key from `.env`.)*

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Launch the app (`streamlit run app.py`).
2. Enter your **application maintenance issue** in the query box.
3. View the **suggested resolution** from the knowledge base or AI model.
4. Provide **feedback**:

   * ✅ Yes → Solution is validated & stored.
   * ❌ No → Alternative resolution is generated.
5. Contribute missing solutions via the feedback submission form.

---

## 📂 Project Structure

```
app-maintenance-resolver/
│── app.py                # Main Streamlit app
│── vectordb/             # ChromaDB persistent storage
│── feedback.json         # Stores user feedback/solutions
│── requirements.txt      # Python dependencies
│── .env                  # API key (ignored in Git)
│── README.md             # Project documentation
```

---

## 📌 Future Enhancements

* ✅ Admin dashboard to review submitted feedback
* ✅ Export feedback logs to CSV/Excel for audit
* ✅ Integration with ITSM tools (ServiceNow, Jira)
* ✅ Role-based access (Admin/User)
* ✅ Multi-language support

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`feature-new`)
3. Commit changes and push
4. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.

---



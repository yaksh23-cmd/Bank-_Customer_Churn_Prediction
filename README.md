# 📰 TruthLens: AI-Powered Fake News Analyzer

## 🚀 Overview
**TruthLens** is a misinformation detection tool designed to analyze news articles and determine their credibility. In an era where market manipulation and fake financial news can crash stocks (or mislead customers), this tool uses **LangChain** and **LLMs (Large Language Models)** to cross-reference claims against logical patterns of misinformation.

## 🎯 Problem Statement
Manual verification of news is slow and prone to bias. This project automates the "fact-checking" process by analyzing the **linguistic patterns** and **source credibility** of a given text.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Framework:** LangChain (for orchestration)
* **AI Model:** ]HuggingFace Hub
* **Frontend:** Streamlit (for a simple web UI)
* **Vector Store:** FAISS (for storing reference contexts - *optional*)
* **Libraries:** Pandas, NumPy, Scikit-learn
<img width="1012" height="541" alt="image" src="https://github.com/user-attachments/assets/1350d969-4029-4101-a2b1-7692280e1898" />

## ⚙️ How It Works (System Architecture)
The system follows a 3-step pipeline:

1.  **Input Ingestion:** The user pastes a news URL or raw text into the Streamlit UI.
2.  **Processing (LangChain):**
    * The text is cleaned (removing HTML tags/special characters).
    * **Prompt Engineering:** The model is prompted to analyze the text for *sensationalism*, *lack of sources*, and *emotional manipulation*.
3.  **Verdict Generation:** The system outputs a **"Credibility Score" (0-100%)** and highlights suspicious sentences.

## 🖥️ Installation & Setup
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/truthlens.git](https://github.com/yourusername/truthlens.git)
    cd truthlens
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys**
    Create a `.env` file and add your OpenAI key:
    ```
    HUGGINGFACEHUB_API_TOKEN= xx    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📊 Future Improvements
* **Web Search Integration:** Connect to Google Search API to cross-check facts in real-time.
* **Financial Domain Fine-Tuning:** Specialize the model to detect "Pump and Dump" stock schemes specifically for banking use cases.

---
*Created by [Your Name] - 2025*

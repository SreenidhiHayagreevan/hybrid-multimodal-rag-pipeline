# 🚀 Hybrid Multi-Modal RAG Pipeline

### Solution for the Aparavi RAG Coding Challenge

This project is a complete, end-to-end solution for the Aparavi Hybrid Multi-Modal Retrieval-Augmented Generation (RAG) Coding Challenge. It implements a sophisticated RAG pipeline designed to process and answer questions from a diverse set of documents, including financial reports, scanned PDFs, invoices, and PowerPoints, which contain both German and English text.

The system is built with a modular, **agentic architecture** and demonstrates advanced RAG optimization techniques, aiming for high accuracy, precision, recall, and F1 score as evaluated by an LLM-as-a-Judge.

---

## 🏗️ Architecture Overview

The pipeline is orchestrated by LangGraph and follows a robust, multi-stage process to ensure high-quality answer generation.

![Architecture Diagram](diagrams/architecture.png)

**The data flows through the following stages:**

1.  **Data Ingestion & Agentic OCR:** Documents are initially processed by an intelligent agent (part of `ocr_agent.py`). This agent dynamically determines the optimal extraction method:
    * It first attempts a fast, standard text extraction with **PyMuPDF**.
    * It analyzes the output; if the text is sparse or garbled (indicating a scanned page or image-based content), it flags the page for a more advanced OCR process using **Google Cloud Vision AI**.

2.  **ETL Pipeline:** The raw extracted text, along with structured data from tables, is systematically cleaned, normalized, and segmented into smaller, semantically coherent chunks using an advanced `RecursiveCharacterTextSplitter`. Crucial metadata, such as page numbers, is preserved for accurate source citation. This stage also includes the **Knowledge Graph Builder**, which creates a knowledge graph capturing relationships between documents and entities.

3.  **Indexing:** Each processed text chunk is converted into a vector embedding using a **fine-tuned `all-MiniLM-L6-v2`** model (generated from over 4,800 synthetic query triplets). These dense embeddings are stored in a **PostgreSQL database with the `pgvector` extension** for efficient similarity search. Sparse embeddings (BM25) and document metadata are also indexed.

4.  **Agentic RAG Pipeline (Orchestrated by LangGraph):** This is the core query engine, coordinated by LangGraph to act as a modular and observable agentic workflow:
    * **a) Retrieve:** When a user asks a question, the pipeline first performs a **hybrid search** (combining dense vector search and sparse BM25 keyword search) to fetch an initial set of candidate documents from the PostgreSQL database and retrieves related information from the **Knowledge Graph**.
    * **b) Re-rank:** These candidates are then passed to a more powerful **Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`)**, which re-ranks them based on true relevance to the question. This two-stage retrieval significantly improves the quality of the context provided to the LLM.
    * **c) Generate:** The top 5 re-ranked documents, along with the original question, are passed to an **OpenAI LLM (gpt-3.5-turbo)**. The LLM synthesizes a final, accurate, and human-readable answer, which always includes **source citations** for traceability (managed by `answer_generator.py`).

5.  **Evaluation & Observability:** The entire pipeline's execution is traced using **Arize Phoenix** for end-to-end observability and debugging. The final quality of the generated answers is measured against a ground-truth dataset using an advanced **"LLM-as-a-Judge"** methodology to calculate Accuracy, Precision, Recall, and F1 Score.

---

## ✨ Key Technologies & Features

| Category                  | Technology / Feature                                    | Purpose                                                                                             |
| :------------------------ | :------------------------------------------------------ | :-------------------------------------------------------------------------------------------------- |
| **Orchestration** | LangGraph                                               | To build a robust, modular, and observable agentic workflow for the RAG pipeline.                   |
| **Databases** | PostgreSQL + `pgvector`                                 | For efficient and scalable storage of dense embeddings and structured data, supporting hybrid search. |
| **OCR & Data Extraction** | PyMuPDF, Google Cloud Vision AI                         | Intelligent extraction of text and tables from diverse, multi-modal documents.                      |
| **Embedding Model** | Fine-tuned `all-MiniLM-L6-v2`                           | Specialized for document domain, generated from 4,800+ synthetic query triplets.                  |
| **Advanced Retrieval** | Hybrid Search (Dense + Sparse), Knowledge Graph, Cross-Encoder Re-ranker (`ms-marco-MiniLM-L-6-v2`) | To ensure highly relevant document retrieval by combining multiple strategies and re-ranking.        |
| **LLM** | OpenAI GPT-3.5-Turbo                                    | For answer synthesis and powering the sophisticated "LLM-as-a-Judge" evaluation.                  |
| **Evaluation & Tracing** | Arize Phoenix                                           | For end-to-end tracing, observability, and debugging of the LangGraph pipeline.                     |
| **Environment** | Docker, Python 3.10+                                    | For reproducible and isolated database deployment and consistent development environment.         |

---

## 📊 Evaluation & Results

The RAG pipeline was rigorously evaluated against the 163 questions in the provided Q&A dataset that had ground-truth answers. To overcome the limitations of simple string matching, a sophisticated **LLM-as-a-Judge** methodology was employed to score the correctness of the generated answers, providing a gold standard assessment.

### Final Metrics (as of latest evaluation with Re-ranker):

* **Total Questions Evaluated:** 163
* **Accuracy (Correct Answers):** `27.61%` (Target: $\ge0.8$ or 80%)
* **Precision:** `1.00` (Target: $\ge0.8$)
* **Recall:** `0.28` (Target: $\ge0.8$)
* **F1 Score:** `0.43` (Target: 0.85)

*Note: While our Precision met the target, ongoing optimization efforts are focused on significantly improving Recall, Accuracy, and F1 Score to fully meet the challenge's stringent requirements. This project demonstrated a clear, data-driven optimization process, with measurable improvements achieved through advanced features like the Cross-Encoder re-ranker, which boosted accuracy from an initial baseline.*

---

## ⚙️ Setup & Installation

To set up and run this project locally, ensure you have the prerequisites and then follow these steps:

### Prerequisites
* Git
* Python 3.10+
* Docker and Docker Compose

### 1. Clone the Repository
```bash
git clone https://github.com/SreenidhiHayagreevan/hybrid-multimodal-rag-pipeline/
cd hybrid-multimodal-rag-pipeline
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup (PostgreSQL with `pgvector` via Docker)
* Ensure Docker Desktop (or equivalent) is installed and running on your system.
* Navigate to the root directory of your project (where `docker-compose.yml` should be located).
* Run the following command to spin up the PostgreSQL container:
    ```bash
    docker-compose up -d
    ```
* This will deploy a production-grade PostgreSQL instance with the `pgvector` extension, crucial for vector embeddings.

### 5. Configure Google Cloud Vision AI Credentials
* Ensure you have a Google Cloud project set up with the Vision AI API enabled.
* Create a service account and download its JSON key file.
* Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the **absolute path** of this JSON file in your `.env` file (see next step).

### 6. Configure Environment Variables
* Create a `.env` file in the root directory of the project.
* Populate it with your API keys and database credentials. **Do not commit this file to Git.**
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-service-account.json" # e.g., /Users/username/Desktop/my-gcp-key.json
    DB_HOST="localhost"
    DB_PORT="5432"
    DB_NAME="rag_db"
    DB_USER="rag_user"
    DB_PASSWORD="rag_password"
    # Add any other necessary environment variables used by your scripts
    ```

### 7. Download Pre-trained Models
* The first time the embedding and cross-encoder models are used (e.g., during ingestion or retrieval), they will typically be downloaded automatically. Ensure you have an active internet connection.

---

## 🚀 Usage

### 1. Data Ingestion & ETL

Run the ingestion script to process your raw documents (PDFs, invoices, etc.), apply agentic OCR, perform ETL, and populate the databases:

```bash
python src/ingestion/run_ingestion.py --input_dir data/raw_documents
```
* This script activates the **Agentic OCR Framework**, processes documents using PyMuPDF and Google Cloud Vision AI as needed, cleans and structures the data, extracts metadata, and builds the knowledge graph.

### 2. Run the RAG Pipeline (Querying)

Once data is ingested and indexed, you can query the RAG pipeline:

```bash
python src/rag/run_rag_pipeline.py --query "What is the key takeaway from the latest financial report regarding market growth in Germany?"
```
* The system will perform hybrid retrieval, re-ranking, and use the LLM to generate an answer with source citations.

### 3. Run Evaluation

To run the full evaluation and generate performance metrics:

```bash
python src/evaluation/run_evaluation.py --qa_dataset_path data/qa_evaluation_dataset.xlsx
```
* This script processes the provided Q&A dataset, uses the "LLM-as-a-Judge" methodology, and calculates the final Accuracy, Precision, Recall, and F1 Score.

---

## 📦 Submission Deliverables

This submission package includes the following as per the challenge requirements:

1.  **Public GitHub Repository:** [Link to your GitHub Repo]
2.  **Well-Designed Architecture Diagram:** Located at `diagrams/architecture.png`.
3.  **Comprehensive README.md file:** This document.
4.  **5-10 minute Demo Video:** [Link to your Demo Video] showcasing the end-to-end solution working and answering at least 2-3 questions from the Q&A Excel dataset.

---

## 💡 Future Enhancements

* **Advanced Table & Image Understanding:** Deepen the integration of VLMs and structured parsing for more nuanced querying of tables and visual content within documents.
* **Dynamic Knowledge Graph Updates:** Implement mechanisms for incremental updates and evolution of the knowledge graph as new documents are ingested.
* **Multi-Agent Coordination for Complex Queries:** Explore more sophisticated multi-agent setups (e.g., using AutoGen alongside LangGraph) for handling highly complex, multi-step reasoning queries.
* **User Feedback Loop:** Incorporate a mechanism for capturing user feedback to continuously improve retrieval relevance and answer quality.
* **Scalability & Deployment:** Optimize for cloud deployment and horizontal scalability using technologies like Kubernetes.

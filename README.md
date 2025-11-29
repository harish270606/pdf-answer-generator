PDF Answer Generator (Local QA + Streamlit)

The PDF Answer Generator is a lightweight application that allows users to upload a PDF and receive precise answers based on natural language questions. It performs local text extraction, semantic search, and transformer-based question answering without requiring internet access or API keys. All processing is done locally, making it suitable for academic, research, legal, and private document use.

Features

Fully offline execution (no API keys required)

Uses a local transformer QA model for answer extraction

Streamlit-based user interface for easy interaction

TF-IDF and cosine similarity-based text retrieval

Shows final answer with confidence score, page number, and source context

How It Works

The system extracts text from the uploaded PDF.

The text is divided into overlapping chunks to preserve context.

TF-IDF similarity identifies the most relevant text segments based on the user query.

A transformer QA model processes the selected text and generates an answer.

The result is displayed with confidence scores and supporting document context.

Installation

Clone the repository:

git clone https://github.com/your-repo-name/pdf-answer-generator.git
cd pdf-answer-generator


Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux


Install required dependencies:

pip install -r requirements.txt

Run the Application
streamlit run pdf_answer_streamlit_qa.py


Once launched, the Streamlit interface will open in a browser. Upload a PDF, type your question, and the model will generate the answer with evidence.

File Structure
📁 PDF-Answer-Generator
│
├── pdf_answer_streamlit_qa.py
├── requirements.txt
├── README.md
│
├── sample_pdfs/
│   └── example.pdf
│
├── assets/
│   └── preview.png
│
└── modules/
    └── utils.py

Requirements
streamlit
PyPDF2
numpy
scikit-learn
transformers
torch

Future Enhancements

OCR support for scanned documents

Support for larger and faster local LLMs

Vector database integration (FAISS or Chroma)

Web deployment or desktop packaging

License

This project is open-source. You may modify or use it for personal, academic, or research purposes. For commercial use, consider reviewing licensing terms of included third-party libraries.

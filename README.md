# Factual-Chatbot-with-Data-Ingestion


## Overview

This project implements a complete pipeline that ingests a research document (PDF, Word, or plain text), extracts meaningful facts, generates natural question-answer pairs, augments them for paraphrase robustness, and builds a semantic retrieval backend so that a chatbot can answer user queries even when they are paraphrased or reworded. It is designed to run end-to-end in Google Colab but can be adapted for production/chatbot backends.

### Key Capabilities
- Document normalization and cleaning (headers/footers removal, hyphenation fixing, quote normalization)
- Sentence-level decomposition
- Hybrid fact extraction and question generation
- Paraphrase augmentation for question diversity
- Semantic embedding of questions
- Fast retrieval (FAISS) with cross-encoder reranking for precision
- Interactive querying demonstrating robustness to paraphrased user questions

## Components / Pipeline Description

### 1. Document Ingestion & Cleaning
- Supports `.pdf`, `.docx`, and `.txt`.
- Loads the document using LangChain's loaders.
- Cleans the text by:
  - Fixing hyphenations.
  - Normalizing whitespace and quotes.
  - Detecting and stripping repeated header/footer lines.

### 2. Sentence-Level Breakdown
- Uses spaCy to split cleaned text into individual sentences.
- Assigns each sentence a unique ID for traceability.

### 3. Hybrid Fact & Question Generation
- Chunks nearby sentences to create local context windows.
- Summarizes each chunk into a concise “fact” using a summarization model.
- Generates a natural question for each fact using a question-generation model (`valhalla/t5-base-qg-hl`), overwriting the original placeholder question.

### 4. Paraphrase Augmentation 
- For each generated question, produces paraphrased variants using a paraphrasing model (e.g., `ramsrigouthamg/t5_paraphraser`).
- These paraphrases are treated as alternative question surface forms pointing to the same answer, improving robustness to varied user phrasing.

### 5. Semantic Store Construction
- Embeds all canonical questions (and optionally their paraphrases) using a sentence embedding model (`sentence-transformers/all-MiniLM-L6-v2`).
- Builds a FAISS index over these embeddings to allow fast semantic similarity search (cosine similarity via normalized inner product).
- Persists both the index and the QA store for reuse.

### 6. Retrieval with Reranking
- Incoming user query is embedded into the same semantic space.
- FAISS retrieves top candidate stored questions.
- A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring refines the best match by comparing the user query to candidate questions.
- The top match’s answer (and optional metadata) is returned, even if the user question is paraphrased.

### 7. Interactive Demo / Query Loop
- Provides a prompt-based loop for testing: type arbitrary questions and see which canonical question is matched, the answer, context snippet, and confidence scores.

## Installation & Setup (Colab-ready)

This is meant to be run in **Google Colab**, but can be adapted elsewhere with an appropriate GPU and Python environment.

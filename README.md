# BFSI Call Center AI Assistant

## Overview
A lightweight, compliant AI assistant for BFSI call centers. The system prioritizes safe, curated responses and uses retrieval-based grounding for complex financial queries.

## Architecture
User → Dataset Similarity → LLM Fallback → RAG (if required) → Final Response

## Features
- Dataset-first responses for compliance
- LLM fallback (Groq API for prototype; swappable with local SLM)
- RAG for policy/EMI/penalty explanations
- Guardrails to prevent hallucinations and unsafe content
- Versioned datasets and documents for maintainability

## Tech Stack
Python, sentence-transformers, ChromaDB, Groq API, LangChain loaders

## How to Run
1. Install:
   ```bash
   pip install -r requirements.txt
2. Set API key:
   ```bash
setx GROQ_API_KEY "your_key"

3.Run:
   ```bash
python src/bfsi_assistant_groq.py

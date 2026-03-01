# RAG System from Scratch

## Overview
A fully local Retrieval-Augmented Generation (RAG) pipeline built from scratch
using numpy, sentence-transformers, and flan-t5-base. The knowledge base uses
fictional company data (NovaMind Technologies) to ensure all correct answers
come from RAG, not model pretraining.

## Pipeline
Documents → Chunk → Embed (LLM1) → numpy Vector Store
Query → Embed (LLM1) → Cosine Similarity Search → Build Prompt → Generate Answer (LLM2)

## Models Used
- LLM1 (Embedding): sentence-transformers/all-MiniLM-L6-v2
- LLM2 (Generation): google/flan-t5-base

## Experiments
- Experiment 1 — Chunking Paradox: compared chunk sizes 50, 200, and 500 characters
- Experiment 2 — Top-K Retrieval: compared k = 1, 3, and 10
- Experiment 3 — Prompting Techniques: zero-shot vs few-shot prompting

## Key Findings
- chunk_size=200 was the sweet spot for this knowledge base
- top_k=3 retrieved the right content but hit flan-t5-base's 512-token limit
- Few-shot prompting produced better structured output than zero-shot
- LLM1 and LLM2 are kept separate because they serve different objectives

## How to Run
pip install sentence-transformers transformers langchain-text-splitters numpy
jupyter notebook assignment2.ipynb



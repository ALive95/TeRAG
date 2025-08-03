# TeRAG: A RAG System for Medical Students (and Beyond)

"The professor suggested a 1000-pages tome for the exam preparation, but in the end they wanted to hear what they said during lectures"

"I should have studied on my notes"

"Why are the professor's slides different than what was said at lecture?"

"I can't wait to burn all of my notes"

Do you relate to some or all of the above questions?
Then chances are you are studying at medical school! 

Studying shouldn't be this hard. Often, information for the exam is scattered through different sources. Some might even be contradictory. Yet, somehow the student is expected to summarize and synthesize everything.
TeRAG is my (early) attempt at providing a solution: a Retrieval-Augmented Generation (RAG) system designed specifically for  students to query their course materials using natural language.

(Note: TeRAG will NOT help you to burn your notes)

## What is RAG?

RAG (Retrieval-Augmented Generation) is an AI technique that combines information retrieval with text generation. Instead of relying solely on pre-trained knowledge, RAG systems:

1. **Retrieve** relevant information from your specific documents
2. **Augment** the AI's response with this retrieved context
3. **Generate** accurate answers based on your actual course materials

Think of it as having an AI assistant that has read all your textbooks, lecture notes, and PDFs, and can instantly find and explain relevant information when you ask questions.

## Why This System?

Medical education often involves highly specialized, professor-specific content that differs from standard textbooks. This RAG system helps you:

- **Study from YOUR materials**: Get answers based on your actual course content, not generic medical information
- **Save time**: Instantly find information across multiple documents instead of manually searching
- **Understand better**: Get explanations that match your professor's teaching style and emphasis
- **Exam preparation**: Query specific topics as they were taught in your courses

## How It Works

```
Your Question → [Search Documents] → [Find Relevant Chunks] → [Generate Answer] → Response with Sources
```

1. You ask a question in natural language
2. The system searches through your processed documents
3. It finds the most relevant 100-word chunks (typically 5 chunks)
4. An AI model generates a comprehensive answer using these chunks
5. You get an answer with source citations

## File Structure

```
medical-rag-system/
├── Archive/                    # Your course materials
│   ├── lecture_notes.pdf
│   ├── textbook_chapter.docx
│   └── professor_slides.pdf
├── embeddings/                 # Generated index files (auto-created)
│   ├── faiss_index.bin
│   ├── metadata.jsonl
│   └── config.json
├── chunk_extractor.py          # Extracts text from PDFs/DOCX
├── embedding_generator.py      # Creates searchable index
└── query_system.py            # Main interface (run this)
```

## Quick Start

### 1. Setup Your Materials
Place your course materials (PDFs, DOCX files) in the `Archive/` folder.

### 2. Process Your Documents
```bash
# Extract text from documents
python chunk_extractor.py

# Create searchable index
python embedding_generator.py
```

### 3. Start Querying
```bash
python query_system.py
```

### 4. Ask Questions
```
🔍 Your question: What are the symptoms of hypertension?
🔍 Your question: How is diabetes diagnosed according to our course materials?
🔍 Your question: Explain the pathophysiology of heart failure
```

## Example Interaction

```
🔍 Your question: What are the contraindications for ACE inhibitors?

💬 Answer:
Based on your course materials, ACE inhibitors have several important contraindications:
- Pregnancy (teratogenic effects)
- Bilateral renal artery stenosis
- Hyperkalemia (K+ > 5.5 mEq/L)
- Previous angioedema with ACE inhibitors
...

📖 Sources (3 chunks found):
   1. cardiology_notes.pdf (similarity: 0.892)
   2. pharmacology_lecture.pdf (similarity: 0.847)
   3. clinical_guidelines.docx (similarity: 0.803)

⏱️ Processing time: 2.34 seconds
```

## Current Specifications

- **Chunk size**: 100 words each
- **Context retrieval**: Up to 5 most relevant chunks per query
- **Supported formats**: PDF, DOCX
- **Similarity method**: Cosine similarity with embeddings
- **AI model**: GPT-4o-mini for fast, cost-effective responses

## Future Enhancements

🔬 **Planned improvements:**
- **GUI**: You should not have to know python to use TeRAG. I am implementing an online GUI to use the system anytime, anywhere
- **Image retrieval**: Extract and query diagrams, charts, and medical images from your documents
- **Advanced similarity**: Implement attention mechanisms and sophisticated ML architectures beyond simple scalar products
- **Multi-modal support**: Handle text + image queries simultaneously
- **Enhanced chunking**: Smart boundary detection for better context preservation

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages: `faiss-cpu`, `openai`, `numpy`, `python-docx`, `PyPDF2`

## Installation

```bash
pip install faiss-cpu openai numpy python-docx PyPDF2
```

Add your OpenAI API key to `query_system.py` before running.

---

**Perfect for**: Medical students who want to study more efficiently using their actual course materials rather than generic resources. Especially useful when professors emphasize specific interpretations or details that differ from standard textbooks.

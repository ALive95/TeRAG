#!/usr/bin/env python3
"""
RAG Query System
Complete RAG pipeline for querying medical documents with semantic search and answer generation.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

import numpy as np
import faiss
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGQuerySystem:
    def __init__(self,
                 openai_api_key: str = None,
                 embeddings_dir: str = "embeddings",
                 embedding_model: str = "text-embedding-3-small",
                 chat_model: str = "gpt-4o-mini",
                 max_context_chunks: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Initialize the RAG query system.

        Args:
            openai_api_key: OpenAI API key
            embeddings_dir: Directory containing FAISS index and metadata
            embedding_model: Model used for query embeddings (must match stored embeddings)
            chat_model: Model for generating answers
            max_context_chunks: Maximum number of chunks to use as context
            similarity_threshold: Minimum similarity score for chunk relevance
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.embeddings_dir = Path(embeddings_dir)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.max_context_chunks = max_context_chunks
        self.similarity_threshold = similarity_threshold

        # Will be loaded from disk
        self.index = None
        self.metadata = []
        self.config = {}

        # Load the vector database
        self.load_vector_db()

    def load_vector_db(self) -> None:
        """Load FAISS index, metadata, and configuration from disk."""
        try:
            # Load FAISS index
            index_path = self.embeddings_dir / "faiss_index.bin"
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found at {index_path}")

            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

            # Load metadata
            metadata_path = self.embeddings_dir / "metadata.jsonl"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found at {metadata_path}")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = [json.loads(line.strip()) for line in f]
            logger.info(f"Loaded metadata for {len(self.metadata)} chunks")

            # Load configuration
            config_path = self.embeddings_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration: model {self.config.get('model', 'unknown')}")

            # Verify data consistency
            if len(self.metadata) != self.index.ntotal:
                logger.warning(
                    f"Metadata count ({len(self.metadata)}) doesn't match index count ({self.index.ntotal})")

        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query text."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Normalize for cosine similarity (same as during indexing)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def search_similar_chunks(self, query: str, k: int = None) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks using semantic similarity.

        Args:
            query: The search query
            k: Number of results to return (defaults to max_context_chunks)

        Returns:
            List of (chunk_metadata, similarity_score) tuples
        """
        if k is None:
            k = self.max_context_chunks

        # Generate query embedding
        query_embedding = self.get_query_embedding(query)

        # Search FAISS index
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1),
            min(k, self.index.ntotal)
        )

        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= 0 and similarity >= self.similarity_threshold:
                chunk_metadata = self.metadata[idx].copy()
                chunk_metadata['similarity_score'] = float(similarity)
                chunk_metadata['rank'] = i + 1
                results.append((chunk_metadata, float(similarity)))

        logger.info(f"Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results

    def format_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """Format retrieved chunks into context for the LLM."""
        if not relevant_chunks:
            return "No relevant information found in the knowledge base."

        context_parts = []
        context_parts.append("=== RELEVANT CONTEXT FROM MEDICAL DOCUMENTS ===\n")

        for i, (chunk, similarity) in enumerate(relevant_chunks, 1):
            source_info = f"Source: {chunk['source_file']} | Similarity: {similarity:.3f}"
            context_parts.append(f"[Context {i}] {source_info}")
            context_parts.append(f"{chunk['text']}\n")

        return "\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using the LLM with retrieved context."""
        system_prompt = """You are a medical AI assistant helping with medical education and research.

Your task is to answer questions based on the context provided from medical documents.

Guidelines:
- Base your response primarily on the provided context
- If the context doesn't contain sufficient information, clearly indicate what is available and what is missing
- Use clear, professional medical language appropriate for medical students and healthcare professionals
- Include relevant details from the context such as symptoms, diagnostic criteria, treatments, etc.
- If you mention specific information, indicate which source it comes from
- If the question cannot be answered from the context, state this clearly

Remember: This is for educational purposes. Always recommend consulting current medical literature and healthcare professionals for clinical decisions."""

        user_prompt = f"""Question: {query}

{context}

Provide a comprehensive answer based on the context above."""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I'm sorry, I encountered an error generating the response: {e}"

    def query(self, question: str, return_sources: bool = True, verbose: bool = False) -> Dict[str, Any]:
        """
        Complete RAG query: search + generate answer.

        Args:
            question: The user's question
            return_sources: Whether to include source information in response
            verbose: Whether to include detailed similarity scores and chunks

        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Semantic search
            relevant_chunks = self.search_similar_chunks(question)

            if not relevant_chunks:
                return {
                    'question': question,
                    'answer': "I couldn't find relevant information in the knowledge base to answer your question. Try rephrasing your question or asking about a different topic covered in the medical documents.",
                    'sources': [],
                    'chunks_found': 0,
                    'query_time': time.time() - start_time
                }

            # Step 2: Format context
            context = self.format_context(relevant_chunks)

            # Step 3: Generate answer
            answer = self.generate_answer(question, context)

            # Step 4: Prepare response
            response = {
                'question': question,
                'answer': answer,
                'chunks_found': len(relevant_chunks),
                'query_time': time.time() - start_time
            }

            if return_sources:
                sources = []
                for chunk, similarity in relevant_chunks:
                    source_info = {
                        'source_file': chunk['source_file'],
                        'similarity_score': similarity,
                        'word_count': chunk['word_count']
                    }
                    if verbose:
                        source_info['text_preview'] = chunk['text'][:200] + "..."
                    sources.append(source_info)

                response['sources'] = sources

            return response

        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return {
                'question': question,
                'answer': f"An error occurred while processing your query: {e}",
                'sources': [],
                'chunks_found': 0,
                'query_time': time.time() - start_time,
                'error': str(e)
            }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded knowledge base."""
        if not self.metadata:
            return {"error": "No database loaded"}

        source_files = {}
        total_words = 0

        for chunk in self.metadata:
            source = chunk.get('source_file', 'unknown')
            if source not in source_files:
                source_files[source] = {'chunks': 0, 'words': 0}
            source_files[source]['chunks'] += 1
            source_files[source]['words'] += chunk.get('word_count', 0)
            total_words += chunk.get('word_count', 0)

        return {
            'total_chunks': len(self.metadata),
            'total_words': total_words,
            'unique_sources': len(source_files),
            'vector_dimension': self.config.get('dimension', 'unknown'),
            'embedding_model': self.config.get('model', 'unknown'),
            'source_breakdown': source_files
        }


def interactive_mode(rag_system: RAGQuerySystem):
    """Run the system in interactive mode for testing."""
    print("\n" + "=" * 60)
    print("🏥 MEDICAL RAG SYSTEM - INTERACTIVE MODE")
    print("=" * 60)

    # Show database stats
    stats = rag_system.get_database_stats()
    print(f"\n📚 Knowledge Base Statistics:")
    print(f"   • Total chunks: {stats['total_chunks']}")
    print(f"   • Total words: {stats['total_words']:,}")
    print(f"   • Source files: {stats['unique_sources']}")
    print(f"   • Model: {stats['embedding_model']}")

    print(f"\n💡 Example questions you can ask:")
    print(f"   • 'What are the symptoms of hypertension?'")
    print(f"   • 'How is diabetes diagnosed?'")
    print(f"   • 'What are the treatment options for asthma?'")
    print(f"   • 'Explain the pathophysiology of heart failure'")

    print(f"\n📝 Commands:")
    print(f"   • Type your question and press Enter")
    print(f"   • Type 'stats' to see database statistics")
    print(f"   • Type 'quit' or 'exit' to exit")
    print("-" * 60)

    while True:
        try:
            question = input("\n🔍 Your question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if question.lower() == 'stats':
                stats = rag_system.get_database_stats()
                print(f"\n📊 Database Statistics:")
                for key, value in stats.items():
                    if key != 'source_breakdown':
                        print(f"   • {key}: {value}")
                continue

            print(f"\n🔎 Searching...")
            result = rag_system.query(question, return_sources=True, verbose=False)

            print(f"\n💬 Answer:")
            print(f"{result['answer']}")

            if result['sources']:
                print(f"\n📖 Sources ({result['chunks_found']} chunks found):")
                for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                    print(f"   {i}. {source['source_file']} (similarity: {source['similarity_score']:.3f})")

            print(f"\n⏱️  Processing time: {result['query_time']:.2f} seconds")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main execution function."""
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  • Windows: set OPENAI_API_KEY=your_key_here")
        print("  • Mac/Linux: export OPENAI_API_KEY=your_key_here")
        print("  • Or create a .env file with: OPENAI_API_KEY=your_key_here")
        return

    try:
        # Initialize RAG system
        print("🚀 Initializing RAG System...")
        rag_system = RAGQuerySystem(
            openai_api_key=api_key,
            embeddings_dir="embeddings",
            embedding_model="text-embedding-3-small",  # Must match embedding generation
            chat_model="gpt-4o-mini",  # Fast and cost-effective
            max_context_chunks=5,
            similarity_threshold=0.3
        )

        print("✅ RAG System initialized successfully!")

        # Run interactive mode
        interactive_mode(rag_system)

    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        print(f"❌ Initialization failed: {e}")


if __name__ == "__main__":
    main()
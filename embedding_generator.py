#!/usr/bin/env python3
"""
RAG Embedding Generator
Reads chunks from all_chunks.jsonl, generates OpenAI embeddings, and stores in FAISS vector DB.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

import numpy as np
import faiss
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self,
                 openai_api_key: str = None,
                 model: str = "text-embedding-3-small",
                 batch_size: int = 100,
                 rate_limit_delay: float = 0.1,
                 output_dir: str = "embeddings"):
        """
        Initialize the embedding generator.

        Args:
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in each batch
            rate_limit_delay: Delay between API calls to avoid rate limits
            output_dir: Directory to save FAISS index and metadata
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Will be set after first embedding to determine vector dimension
        self.dimension = None
        self.index = None
        self.metadata = []  # Store chunk metadata alongside vectors

    def load_chunks(self, chunks_file: str = "all_chunks.jsonl") -> List[Dict[str, Any]]:
        """Load chunks from JSONL file."""
        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        chunks = []
        logger.info(f"Loading chunks from {chunks_file}...")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts from OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def initialize_faiss_index(self, dimension: int):
        """Initialize FAISS index with the correct dimension."""
        self.dimension = dimension
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        # Alternative: IndexFlatL2 for L2 distance
        self.index = faiss.IndexFlatIP(dimension)
        logger.info(f"Initialized FAISS index with dimension {dimension}")

    def normalize_vector(self, vector: List[float]) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        vec_array = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec_array)
        return vec_array / norm if norm > 0 else vec_array

    def process_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Process all chunks: generate embeddings and add to FAISS index."""
        logger.info(f"Processing {len(chunks)} chunks...")

        # Process in batches
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Processing batches"):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk['content'] for chunk in batch_chunks]

            # Get embeddings for this batch
            try:
                batch_embeddings = self.get_embeddings_batch(batch_texts)

                # Initialize FAISS index on first batch
                if self.index is None:
                    self.initialize_faiss_index(len(batch_embeddings[0]))

                # Process each embedding in the batch
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    # Normalize vector for cosine similarity
                    normalized_vector = self.normalize_vector(embedding)

                    # Add to FAISS index
                    self.index.add(normalized_vector.reshape(1, -1))

                    # Store metadata
                    metadata_entry = {
                        'text': chunk['content'],  # Store as 'text' for consistency
                        'source_file': chunk.get('source_file', 'unknown'),
                        'file_type': chunk.get('file_type', 'unknown'),
                        'chunk_id': chunk.get('chunk_id', len(self.metadata)),
                        'word_count': chunk.get('word_count', 0)
                    }
                    self.metadata.append(metadata_entry)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error processing batch {i // self.batch_size + 1}: {e}")
                continue

        logger.info(f"Successfully processed {len(self.metadata)} chunks")

    def save_vector_db(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.error("No index to save. Process chunks first.")
            return

        # Save FAISS index
        index_path = self.output_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata
        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for entry in self.metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved metadata to {metadata_path}")

        # Save configuration
        config_path = self.output_dir / "config.json"
        config = {
            'model': self.model,
            'dimension': self.dimension,
            'total_vectors': len(self.metadata),
            'index_type': 'IndexFlatIP'
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        if not self.metadata:
            return {"error": "No data processed yet"}

        total_chunks = len(self.metadata)
        total_words = sum(chunk.get('word_count', 0) for chunk in self.metadata)
        source_files = set(chunk.get('source_file', 'unknown') for chunk in self.metadata)

        return {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'unique_source_files': len(source_files),
            'vector_dimension': self.dimension,
            'model_used': self.model
        }


def main():
    """Main execution function."""
    # Hardcoded API key for testing (replace with your actual key)

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    try:
        # Initialize embedding generator
        generator = EmbeddingGenerator(
            openai_api_key=api_key,
            model="text-embedding-3-small",  # Fast and cost-effective
            batch_size=100,  # Adjust based on your rate limits
            rate_limit_delay=0.1,  # Adjust to avoid rate limiting
            output_dir="embeddings"
        )

        # Load chunks
        chunks = generator.load_chunks("all_chunks.jsonl")

        if not chunks:
            logger.error("No chunks found to process!")
            return

        # Process chunks and generate embeddings
        logger.info("Starting embedding generation...")
        start_time = time.time()

        generator.process_chunks(chunks)

        # Save vector database
        generator.save_vector_db()

        # Print statistics
        stats = generator.get_stats()
        elapsed_time = time.time() - start_time

        logger.info("=== EMBEDDING GENERATION COMPLETE ===")
        logger.info(f"Total chunks processed: {stats['total_chunks']}")
        logger.info(f"Total words: {stats['total_words']:,}")
        logger.info(f"Source files: {stats['unique_source_files']}")
        logger.info(f"Vector dimension: {stats['vector_dimension']}")
        logger.info(f"Model used: {stats['model_used']}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per chunk: {elapsed_time / stats['total_chunks']:.3f} seconds")

        # Show sample of first few chunks
        logger.info("\n=== SAMPLE METADATA ===")
        for i, chunk in enumerate(generator.metadata[:3]):
            logger.info(f"Chunk {i + 1}:")
            logger.info(f"  Source: {chunk['source_file']}")
            logger.info(f"  Words: {chunk['word_count']}")
            logger.info(f"  Text preview: {chunk['text'][:100]}...")
            logger.info("")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
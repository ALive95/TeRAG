import os
import pymupdf  # PyMuPDF - much faster than PyPDF2
from docx import Document
import re
from multiprocessing import Pool, cpu_count
import json


# To add
# Time estimate
# Ask whether to write over existing chunks or adding new ones


def read_pdf_streaming(file_path):
    """Extract text from PDF page by page (memory efficient)."""
    try:
        doc = pymupdf.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only yield non-empty pages
                yield text
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")


def read_word_streaming(file_path):
    """Extract text from Word document paragraph by paragraph."""
    try:
        doc = Document(file_path)
        current_text = ""

        for paragraph in doc.paragraphs:
            current_text += paragraph.text + "\n"

            # Yield chunks of text periodically to avoid memory buildup
            if len(current_text) > 10000:  # ~10KB chunks for processing
                yield current_text
                current_text = ""

        # Yield remaining text
        if current_text.strip():
            yield current_text

    except Exception as e:
        print(f"Error reading Word document {file_path}: {e}")


def clean_text(text):
    """Clean and normalize text efficiently."""
    # Single regex pass for efficiency
    text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace to single spaces
    return text.strip()


def create_chunks_streaming(text_generator, chunk_size=100):
    """Create chunks from text generator without loading everything into memory."""
    word_buffer = []

    for text_block in text_generator:
        if not text_block.strip():
            continue

        cleaned_text = clean_text(text_block)
        words = cleaned_text.split()
        word_buffer.extend(words)

        # Yield complete chunks as they're ready
        while len(word_buffer) >= chunk_size:
            chunk_words = word_buffer[:chunk_size]
            word_buffer = word_buffer[chunk_size:]
            yield ' '.join(chunk_words)

    # Yield remaining words as final chunk (if any)
    if word_buffer:
        yield ' '.join(word_buffer)


def process_single_file(file_info):
    """Process a single file and return chunk information."""
    file_path, filename = file_info
    chunks_created = 0

    try:
        # Determine file type and get text generator
        if filename.lower().endswith('.pdf'):
            text_generator = read_pdf_streaming(file_path)
            file_type = "PDF"
        elif filename.lower().endswith(('.docx', '.doc')):
            text_generator = read_word_streaming(file_path)
            file_type = "Word"
        else:
            return filename, 0, "Unsupported file type"

        # Process chunks and save immediately to avoid memory buildup
        output_file = f"chunks_{filename.replace('.', '_')}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk_id, chunk_content in enumerate(create_chunks_streaming(text_generator), 1):
                chunk_data = {
                    'source_file': filename,
                    'file_type': file_type,
                    'chunk_id': chunk_id,
                    'content': chunk_content,
                    'word_count': len(chunk_content.split())
                }

                # Write each chunk as a JSON line (JSONL format)
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                chunks_created += 1

        return filename, chunks_created, "Success"

    except Exception as e:
        return filename, 0, f"Error: {str(e)}"


def process_documents_parallel(folder_path, max_workers=None):
    """Process all documents using parallel processing."""

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []

    # Get all supported files
    files = os.listdir(folder_path)
    supported_files = []

    for filename in files:
        if filename.lower().endswith(('.pdf', '.docx', '.doc')):
            file_path = os.path.join(folder_path, filename)
            supported_files.append((file_path, filename))

    if not supported_files:
        print("No supported files found in the folder.")
        return []

    print(f"Found {len(supported_files)} supported files")

    # Use parallel processing (but limit workers to avoid memory issues)
    if max_workers is None:
        max_workers = min(cpu_count(), 4)  # Limit to 4 to control memory usage

    results = []

    if len(supported_files) > 1:
        print(f"Processing files in parallel using {max_workers} workers...")
        with Pool(max_workers) as pool:
            results = pool.map(process_single_file, supported_files)
    else:
        print("Processing single file...")
        results = [process_single_file(supported_files[0])]

    return results


def combine_chunk_files(output_file='all_chunks.jsonl'):
    """Combine all individual chunk files into one master file."""
    chunk_files = [f for f in os.listdir('.') if f.startswith('chunks_') and f.endswith('.jsonl')]

    if not chunk_files:
        print("No chunk files found to combine.")
        return 0

    total_chunks = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    total_chunks += 1

            # Clean up individual files
            os.remove(chunk_file)

    print(f"Combined {len(chunk_files)} chunk files into {output_file}")
    return total_chunks


def get_processed_files():
    """Get list of files that have already been processed from existing chunks."""
    processed_files = set()

    if os.path.exists('all_chunks.jsonl'):
        try:
            with open('all_chunks.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_data = json.loads(line.strip())
                    processed_files.add(chunk_data['source_file'])
        except Exception as e:
            print(f"Error reading existing chunks: {e}")

    return processed_files


def get_archive_files(folder_path):
    """Get list of supported files in the Archive folder."""
    if not os.path.exists(folder_path):
        return set()

    files = os.listdir(folder_path)
    supported_files = set()

    for filename in files:
        if filename.lower().endswith(('.pdf', '.docx', '.doc')):
            supported_files.add(filename)

    return supported_files


def ask_user_choice(existing_files, new_files, all_files):
    """Ask user what they want to do with existing and new files."""
    print(f"\nFound {len(existing_files)} files already processed:")
    for f in sorted(existing_files):
        print(f"  ✓ {f}")

    if new_files:
        print(f"\nFound {len(new_files)} new files:")
        for f in sorted(new_files):
            print(f"  + {f}")

    print(f"\nTotal files in Archive: {len(all_files)}")

    print("\nWhat would you like to do?")
    print("1. Process only NEW files and add to existing chunks")
    print("2. Process ALL files again (overwrites existing chunks)")
    print("3. Exit without processing")

    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Please enter 1, 2, or 3")


def process_incremental(folder_path, files_to_process):
    """Process only specific files and append to existing chunks."""
    if not files_to_process:
        print("No files to process.")
        return []

    # Create file info for processing
    file_info_list = []
    for filename in files_to_process:
        file_path = os.path.join(folder_path, filename)
        file_info_list.append((file_path, filename))

    print(f"Processing {len(files_to_process)} files...")

    # Process files
    results = []
    max_workers = min(cpu_count(), 4)

    if len(file_info_list) > 1:
        print(f"Using {max_workers} workers for parallel processing...")
        with Pool(max_workers) as pool:
            results = pool.map(process_single_file, file_info_list)
    else:
        results = [process_single_file(file_info_list[0])]

    return results


def append_new_chunks():
    """Append newly created chunk files to the existing all_chunks.jsonl."""
    chunk_files = [f for f in os.listdir('.') if f.startswith('chunks_') and f.endswith('.jsonl')]

    if not chunk_files:
        print("No new chunk files found to append.")
        return 0

    new_chunks = 0

    with open('all_chunks.jsonl', 'a', encoding='utf-8') as outfile:
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    new_chunks += 1

            # Clean up individual files
            os.remove(chunk_file)

    print(f"Appended {new_chunks} new chunks to all_chunks.jsonl")
    return new_chunks


def main():
    archive_folder = "Archive"

    print("Starting optimized document processing...")
    print(f"Available CPU cores: {cpu_count()}")

    # Check what files exist and what's already processed
    processed_files = get_processed_files()
    archive_files = get_archive_files(archive_folder)

    if not archive_files:
        print(f"No supported files found in '{archive_folder}' folder.")
        return

    new_files = archive_files - processed_files
    existing_files = archive_files & processed_files

    # Decide what to process
    files_to_process = archive_files
    incremental_mode = False

    if processed_files and os.path.exists('all_chunks.jsonl'):
        choice = ask_user_choice(existing_files, new_files, archive_files)

        if choice == '1':  # Process only new files
            if not new_files:
                print("No new files to process. All files are already processed.")
                return
            files_to_process = new_files
            incremental_mode = True
        elif choice == '2':  # Process all files
            files_to_process = archive_files
            incremental_mode = False
            # Remove existing chunks file to start fresh
            if os.path.exists('all_chunks.jsonl'):
                os.remove('all_chunks.jsonl')
                print("Removed existing chunks file. Starting fresh.")
        else:  # Exit
            print("Exiting without processing.")
            return

    # Process the selected files
    if incremental_mode:
        results = process_incremental(archive_folder, files_to_process)
    else:
        file_info_list = [(os.path.join(archive_folder, f), f) for f in files_to_process]
        if len(file_info_list) > 1:
            max_workers = min(cpu_count(), 4)
            print(f"Processing files in parallel using {max_workers} workers...")
            with Pool(max_workers) as pool:
                results = pool.map(process_single_file, file_info_list)
        else:
            print("Processing single file...")
            results = [process_single_file(file_info_list[0])]

    if not results:
        print("No files were processed.")
        return

    # Display results
    total_chunks = 0
    successful_files = 0

    print("\nProcessing Results:")
    print("-" * 50)

    for filename, chunk_count, status in results:
        if status == "Success":
            print(f"✓ {filename}: {chunk_count} chunks")
            total_chunks += chunk_count
            successful_files += 1
        else:
            print(f"✗ {filename}: {status}")

    if successful_files > 0:
        # Combine or append chunk files
        if incremental_mode:
            combined_total = append_new_chunks()
        else:
            combined_total = combine_chunk_files()

        print(f"\nSummary:")
        print(f"Files processed successfully: {successful_files}/{len(results)}")
        print(f"New chunks created: {combined_total}")
        print(f"Chunks saved to: all_chunks.jsonl")

        # Show total chunks in file
        try:
            total_chunks_in_file = 0
            with open('all_chunks.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    total_chunks_in_file += 1
            print(f"Total chunks in file: {total_chunks_in_file}")

            # Show sample of first chunk
            with open('all_chunks.jsonl', 'r', encoding='utf-8') as f:
                first_chunk = json.loads(f.readline())
                print(f"\nSample chunk:")
                print(f"Source: {first_chunk['source_file']}")
                print(f"Words: {first_chunk['word_count']}")
                print(f"Preview: {first_chunk['content'][:100]}...")
        except:
            pass

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
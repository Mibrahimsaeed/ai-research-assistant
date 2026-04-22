import sys
import os
import json
import re
import tiktoken
from sentence_transformers import SentenceTransformer
from state_manager import set_status, STATUS

# -----------------------------
# CONFIG
# -----------------------------
METADATA_FILE = "data/metadata.json"
CLEAN_PAPERS_FILE = "data/clean_papers.json"  # Where cleaning script saves clean_text
CHUNKS_FILE = "data/chunks.json"

CHUNK_SIZE = 800
OVERLAP_SIZE = 120
MAX_PARAGRAPH_TOKENS = 1024

enc = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# LOAD / SAVE HELPERS
# -----------------------------
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_metadata():
    return load_json(METADATA_FILE)


def save_metadata(metadata):
    save_json(METADATA_FILE, metadata)


def save_chunks(chunks):
    save_json(CHUNKS_FILE, chunks)


# -----------------------------
# TEXT UTILITIES
# -----------------------------
def split_paragraphs(text):
    return [p.strip() for p in text.split("\n") if p.strip()]


def count_tokens(text):
    return len(enc.encode(text))


def detect_sections(text):
    sections = {}
    patterns = {
        "abstract": r"(?i)abstract(.*?)(introduction|1\.)",
        "introduction": r"(?i)introduction(.*?)(method|methods|2\.)",
        "method": r"(?i)(methodology|methods)(.*?)(results|3\.)",
        "results": r"(?i)results(.*?)(conclusion|discussion|4\.)"
    }

    for k, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match and len(match.group(1).strip()) > 50:
            sections[k] = match.group(1).strip()

    if not sections or len(" ".join(sections.values())) < 100:
        sections = {"full": text}

    return sections


# -----------------------------
# CHUNKING LOGIC
# -----------------------------
def chunk_paper(text, paper_id, metadata_entry=None):
    """Chunk a paper's clean_text into overlapping segments."""
    chunks = []
    sections = detect_sections(text)
    
    for section_name, section_text in sections.items():
        paragraphs = split_paragraphs(section_text)
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = count_tokens(para)
            
            if para_tokens > MAX_PARAGRAPH_TOKENS:
                # Split oversized paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = count_tokens(sent)
                    if current_tokens + sent_tokens > CHUNK_SIZE and current_chunk:
                        chunks.append({
                            "chunk_id": f"{paper_id}_{len(chunks)}",
                            "paper_id": paper_id,
                            "section": section_name,
                            "text": current_chunk.strip(),
                            "tokens": current_tokens,
                            "title": metadata_entry.get("title", "") if metadata_entry else "",
                            "authors": metadata_entry.get("authors", "") if metadata_entry else "",
                        })
                        # Keep overlap
                        words = current_chunk.split()
                        overlap_text = " ".join(words[-OVERLAP_SIZE:]) if len(words) > OVERLAP_SIZE else current_chunk
                        current_chunk = overlap_text + " " + sent
                        current_tokens = count_tokens(current_chunk)
                    else:
                        current_chunk += " " + sent
                        current_tokens += sent_tokens
            else:
                if current_tokens + para_tokens > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        "chunk_id": f"{paper_id}_{len(chunks)}",
                        "paper_id": paper_id,
                        "section": section_name,
                        "text": current_chunk.strip(),
                        "tokens": current_tokens,
                        "title": metadata_entry.get("title", "") if metadata_entry else "",
                        "authors": metadata_entry.get("authors", "") if metadata_entry else "",
                    })
                    # Keep overlap
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-OVERLAP_SIZE:]) if len(words) > OVERLAP_SIZE else current_chunk
                    current_chunk = overlap_text + " " + para
                    current_tokens = count_tokens(current_chunk)
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens
        
        # Don't forget the last chunk in this section
        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"{paper_id}_{len(chunks)}",
                "paper_id": paper_id,
                "section": section_name,
                "text": current_chunk.strip(),
                "tokens": current_tokens,
                "title": metadata_entry.get("title", "") if metadata_entry else "",
                "authors": metadata_entry.get("authors", "") if metadata_entry else "",
            })
    
    return chunks


# -----------------------------
# MERGE CLEAN DATA INTO METADATA
# -----------------------------
def merge_clean_text_into_metadata():
    """Merge clean_text from clean_papers.json into metadata.json."""
    metadata = load_metadata()
    clean_papers = load_json(CLEAN_PAPERS_FILE)
    
    # Build lookup by paper_id
    clean_lookup = {p["paper_id"]: p for p in clean_papers}
    
    merged_count = 0
    for paper in metadata:
        paper_id = paper.get("paper_id")
        if paper_id in clean_lookup and not paper.get("clean_text"):
            paper["clean_text"] = clean_lookup[paper_id].get("clean_text", "")
            paper["sections"] = clean_lookup[paper_id].get("sections", {})
            merged_count += 1
    
    if merged_count > 0:
        save_metadata(metadata)
        print(f"✅ Merged clean_text into {merged_count} metadata entries.")
    
    return metadata


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_chunking_pipeline():
    # Step 1: Ensure metadata has clean_text from clean_papers.json
    metadata = merge_clean_text_into_metadata()
    
    all_chunks = []
    processed_count = 0
    skipped_count = 0

    for paper in metadata:
        paper_id = paper.get("paper_id", "unknown")
        
        # Check for clean_text
        text = paper.get("clean_text", "")
        if not text:
            print(f"⚠️ Skipping {paper_id}: No 'clean_text' found.")
            skipped_count += 1
            continue

        # Skip already chunked
        if paper.get("status") == STATUS["CHUNKED"]:
            print(f"ℹ️ Skipping {paper_id}: Already chunked.")
            continue

        chunks = chunk_paper(text, paper_id, metadata_entry=paper)
        all_chunks.extend(chunks)
        processed_count += 1
        
        # Update status
        metadata = set_status(metadata, paper_id, STATUS["CHUNKED"])

    # Save results
    if all_chunks:
        save_metadata(metadata)
        save_chunks(all_chunks)
        print(f"\n✅ Chunking complete!")
        print(f"   Papers processed: {processed_count}")
        print(f"   Papers skipped: {skipped_count}")
        print(f"   Total chunks created: {len(all_chunks)}")
    else:
        print(f"\n❌ No chunks created.")
        print(f"   Papers skipped (no clean_text): {skipped_count}")
        print("   Check that clean_papers.json exists and contains clean_text.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_chunking_pipeline()
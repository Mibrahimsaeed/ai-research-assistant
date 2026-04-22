import fitz
import os
import json
import re
from state_manager import set_status, STATUS
PDF_DIR = "data/papers"
METADATA_FILE = "data/metadata.json"
OUTPUT_FILE = "data/clean_papers.json"


# -----------------------------
# BASIC TEXT CLEANING
# -----------------------------
def clean_text(text):
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # remove table-like junk (lines with lots of numbers)
    text = re.sub(r'(\d+\s+){3,}', '', text)

    # remove figure/table labels
    text = re.sub(r'Figure\s*\d+.*?(?=\.)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Table\s*\d+.*?(?=\.)', '', text, flags=re.IGNORECASE)

    return text.strip()


# -----------------------------
# SECTION EXTRACTION (simple heuristic)
# -----------------------------
def extract_sections(text):
    sections = {
        "abstract": "",
        "introduction": "",
        "method": "",
        "results": ""
    }

    patterns = {
        "abstract": r"(?i)abstract(.*?)(introduction|1\.)",
        "introduction": r"(?i)introduction(.*?)(method|methods|2\.)",
        "method": r"(?i)(methodology|methods)(.*?)(results|3\.)",
        "results": r"(?i)results(.*?)(discussion|conclusion|4\.)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return sections


# -----------------------------
# EXTRACT TEXT FROM PDF
# -----------------------------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_papers():
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    processed = []
    

    for paper in metadata:
        if paper.get("status") == STATUS["CLEANED"]:
            continue
            
        pdf_path = paper["pdf_path"]
        paper_id = paper["paper_id"]

        if not os.path.exists(pdf_path):
            continue

        print(f"Processing: {paper['title']}")

        raw_text = extract_pdf_text(pdf_path)
        cleaned = clean_text(raw_text)
        sections = extract_sections(cleaned)
        processed.append({
            "paper_id": paper_id,
            "title": paper.get("title"),
            "clean_text": cleaned,
            "sections": sections,
            "source": paper.get("source", "unknown"),
            "pdf_path": pdf_path,
      

        })

        # Update status in metadata
        metadata = set_status(metadata, paper_id, STATUS["CLEANED"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(processed, f, indent=2)

    # Save updated metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Cleaning complete!")

if __name__ == "__main__":
    process_papers()
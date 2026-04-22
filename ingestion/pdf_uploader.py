import os
import json
import uuid
import shutil
from datetime import datetime
from state_manager import set_status, STATUS
SAVE_DIR = "data/papers"
METADATA_FILE = "data/metadata.json"

os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# LOAD EXISTING METADATA
# -----------------------------
def load_existing_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return []


# -----------------------------
# SAVE METADATA
# -----------------------------
def save_metadata(data):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


# -----------------------------
# MAIN UPLOAD FUNCTION
# -----------------------------
def upload_pdf(file_path):
    try:
        # 1. Validate file
        if not os.path.exists(file_path):
            raise Exception("File does not exist")

        if not file_path.lower().endswith(".pdf"):
            raise Exception("Only PDF files are allowed")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 50:
            raise Exception("File exceeds 50MB limit")

        print(f"Uploading: {file_path}")

        # 2. Generate unique ID + filename
        upload_id = str(uuid.uuid4())
        new_filename = f"{upload_id}.pdf"
        new_path = os.path.join(SAVE_DIR, new_filename)

        # 3. Copy file into storage
        shutil.copy(file_path, new_path)

        # 4. Create minimal metadata entry
        metadata_entry = {
            "paper_id": upload_id,
            "title": os.path.basename(file_path),   # temporary title
            "authors": [],
            "summary": "",
            "pdf_path": new_path,
            "published": str(datetime.now().date()),
            "source": "user_upload",
            "upload_id": upload_id,
            "status": STATUS["UPLOADED"] 
        }

        # 5. Append to metadata.json
        existing_data = load_existing_metadata()
        existing_data.append(metadata_entry)
        save_metadata(existing_data)

        print("✅ Upload successful!")
        print(f"Stored at: {new_path}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    # Replace with your test file
    file_path = input("Enter PDF file path: ")
    upload_pdf(file_path)
import arxiv
import os
import json
import time

SAVE_DIR = "data/papers"
METADATA_FILE = "data/metadata.json"

os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_papers(query="(ti:transformer OR ti:attention OR ti:\"large language model\")",
                 max_results=20):

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers_metadata = []

    results = search.results()

    for result in results:

        try:
            print(f"Downloading: {result.title}")

            pdf_path = result.download_pdf(dirpath=SAVE_DIR)

            paper_data = {
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "summary": result.summary,
                "pdf_path": pdf_path,
                "published": str(result.published),
                "arxiv_id": result.get_short_id()
            }

            papers_metadata.append(paper_data)

            time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")

    # append instead of overwrite
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            old_data = json.load(f)
        papers_metadata = old_data + papers_metadata

    with open(METADATA_FILE, "w") as f:
        json.dump(papers_metadata, f, indent=4)

    print("✅ Done!")

if __name__ == "__main__":
    fetch_papers(max_results=30)
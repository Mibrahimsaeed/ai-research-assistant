
STATUS = {
    "UPLOADED": "uploaded",
    "INGESTED": "ingested",
    "CLEANED": "cleaned",
    "CHUNKED": "chunked",
    "EMBEDDED": "embedded",
    "DONE": "done",
    "FAILED": "failed"
}


def set_status(metadata, paper_id, status):
    for paper in metadata:
        if paper["paper_id"] == paper_id:
            paper["status"] = status
            return metadata

    metadata.append({
        "paper_id": paper_id,
        "status": status
    })

    return metadata


def should_process(metadata, paper_id):
    for paper in metadata:
        if paper["paper_id"] == paper_id:
            return paper["status"] != STATUS["DONE"]
    return True
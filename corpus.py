import os
import re
import ijson
from huggingface_hub import HfFileSystem
REPO_PATH= "datasets/Exploration-Lab/CS-781-Capstone/Phase_1_Pretraining"
sources={"acts":"train_acts.json","supreme_court": "train_supreme_court.json","high_courts": "train_high_courts.json","tribunals": "train_tribunals.json"}

def open_source(name, local_dir=None):
    """Return a binary file handle for one training source.
    local_dir -> read local file (laptop tests); else stream from HF (no full download)."""
    filename = sources[name]
    if local_dir is not None:
        return open(os.path.join(local_dir, filename), "rb")
    fs = HfFileSystem()  # uses HF_TOKEN from env
    return fs.open(f"{REPO_PATH}/Train/{filename}", "rb", block_size=8 * 1024 * 1024)


def clean(text):
    """Basic document cleaning. Returns None to drop the doc.
    TODO(you): extend after inspecting real docs (boilerplate, OCR junk, etc.)."""
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)        # collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)       # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)     # max one blank line
    return text.strip() or None


def iter_docs(name, local_dir=None, min_chars=200, stride=1, max_docs=None):
    """Stream cleaned document texts from one source, one at a time.
    stride=k keeps every k-th doc (unbiased sampling of big files);
    max_docs stops early (for quick tests)."""
    yielded = 0
    with open_source(name, local_dir) as f:
        for i, rec in enumerate(ijson.items(f, "item")):
            if stride > 1 and i % stride:
                continue
            text = clean(rec.get("text") if isinstance(rec, dict) else None)
            if text is None or len(text) < min_chars:
                continue
            yield text
            yielded += 1
            if max_docs is not None and yielded >= max_docs:
                return

"""Recognize canonical refusals independently of reference-list rendering."""
import re

def classify_refusal(answer, templates):
    body=re.split(r"\n{2,}---\s*\n{2,}### 参考来源",str(answer or ''),maxsplit=1)[0].strip()
    normalize=lambda value:value.strip().rstrip("。.! \t\r\n")
    canonical=[normalize(t) for t in templates]
    if normalize(body) in canonical:
        return True
    if any(t and t in body for t in canonical):
        return None  # Mixed substantive answer + refusal: requires review, not keyword guessing.
    return False

# src/data/mcq.py — nguồn chân lý DUY NHẤT cho detect MCQ
import re

_MCQ_OPT_RE = re.compile(r"(?:^|\s)([A-E])[.)]\s+\S")

def is_mcq(rec: dict) -> bool:
    """MCQ nếu có >=3 lựa chọn A/B/C... trong question+response."""
    text = f"{rec.get('question','')}\n{rec.get('response','')}"
    letters = {m.group(1) for m in _MCQ_OPT_RE.finditer(text)}
    return len(letters) >= 3
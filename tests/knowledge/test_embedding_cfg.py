import sys, io, ast
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 1) syntax OK?
src = open("src/knowledge/embedding.py", encoding="utf-8").read()
ast.parse(src)
print("[ok] syntax embedding.py")

# 2) config đọc kb đúng? (không cần Qdrant/GPU)
from src.knowledge.embedding import config_from_yaml, load_chunks
c = config_from_yaml()
print("kb_collection :", c.kb_collection)
print("kb_chunk_paths:", c.kb_chunk_paths)
print("qa collections:", list(c.collections.keys()))
print("model         :", c.model_name, "| device:", c.device)

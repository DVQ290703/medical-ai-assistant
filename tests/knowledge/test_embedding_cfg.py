"""Test config embedding đọc từ rag.yaml (không cần GPU/Qdrant).

Lưu ý: import src.knowledge.embedding sẽ thử import torch+FlagEmbedding (top-level),
nhưng module bọc try/except nên import được kể cả khi thiếu — chỉ BGEM3FlagModel=None.
"""
from src.knowledge.embedding import config_from_yaml


def test_config_doc_tu_rag_yaml():
    c = config_from_yaml("configs/rag.yaml")
    assert c.model_name == "BAAI/bge-m3"
    assert c.dense_dim == 1024
    assert c.kb_collection == "vinmec_kb"
    # 2 collection Q&A
    assert set(c.collections.keys()) == {"vinmec_q", "vinmec_qa"}


def test_kb_chunk_paths_tro_dung_file_chunk():
    c = config_from_yaml("configs/rag.yaml")
    # knowledge_base có chunk=true -> path đổi .jsonl thành _chunks.jsonl
    assert any("vn_healthcare_chunks.jsonl" in p for p in c.kb_chunk_paths)
    assert any("byt_kcb_chunks.jsonl" in p for p in c.kb_chunk_paths)


def test_config_thieu_file_dung_default():
    c = config_from_yaml("configs/__khong_ton_tai__.yaml")
    assert c.model_name == "BAAI/bge-m3"   # fallback dataclass default

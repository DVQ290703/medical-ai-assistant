"""Dev: kiểm trạng thái Qdrant + embed state. Chạy: python scripts/dev/check_qdrant.py"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from src.knowledge.embedding import config_from_yaml
from qdrant_client import QdrantClient

cfg = config_from_yaml()
client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
print("=== Qdrant collections ===")
for c in client.get_collections().collections:
    info = client.get_collection(c.name)
    print(f"  {c.name}: {info.points_count:,} points")

print("\n=== embed_state.json ===")
if os.path.exists(cfg.state_path):
    print(json.dumps(json.load(open(cfg.state_path, encoding="utf-8")), ensure_ascii=False, indent=2))
else:
    print("  (chưa có state)")

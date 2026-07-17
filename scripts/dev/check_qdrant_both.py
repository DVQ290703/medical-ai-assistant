"""Dev script: đối chiếu Qdrant LOCAL vs CLOUD -> biết data migrate thật sự nằm đâu.

Khi app báo 'Collection doesn't exist' mà migrate từng báo thành công: chạy cái này để
so cả 2 nơi cùng lúc. Nếu LOCAL có data còn CLOUD trống -> migrate đã vào nhầm local.

Chạy:  python scripts/dev/check_qdrant_both.py
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from qdrant_client import QdrantClient


def show(label, client):
    print(f"\n=== {label} ===")
    try:
        cols = client.get_collections().collections
        if not cols:
            print("  (không có collection nào)")
        for col in cols:
            n = client.get_collection(col.name).points_count
            print(f"  {col.name}: {n:,} điểm")
    except Exception as e:
        print(f"  [lỗi kết nối: {type(e).__name__}: {e}]")


# LOCAL — luôn localhost:6333
show("QDRANT LOCAL (localhost:6333)", QdrantClient(host="localhost", port=6333, timeout=15))

# CLOUD — từ .env
url = os.environ.get("QDRANT_URL", "").strip()
key = os.environ.get("QDRANT_API_KEY", "").strip()
if url:
    print(f"\n(CLOUD url: {url[:50]}...)")
    show("QDRANT CLOUD (từ .env)", QdrantClient(url=url, api_key=key or None, timeout=30))
else:
    print("\n=== QDRANT CLOUD ===\n  (.env không có QDRANT_URL)")

"""Dev script: xem app (đọc .env) đang connect Qdrant nào + có collection gì.

Nạp .env giống hệt app rồi liệt kê collection + số điểm. Dùng để chẩn đoán khi app báo
'Collection doesn't exist' — biết cluster .env trỏ tới có data thật không.

Chạy:  python scripts/dev/check_qdrant_env.py
"""
import os
import sys

# Cho phép import 'src' khi chạy trực tiếp từ scripts/dev/ (thêm thư mục gốc vào path).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Nạp .env đúng như app (dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # fallback: tự parse .env
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

url = os.environ.get("QDRANT_URL", "")
print(f"QDRANT_URL app dùng: {url[:55]}{'...' if len(url) > 55 else ''}")
print(f"QDRANT_API_KEY     : {'(có)' if os.environ.get('QDRANT_API_KEY') else '(RỖNG!)'}")
print("-" * 50)

from src.knowledge.vectorstore import connect

try:
    c = connect()
    cols = c.get_collections().collections
    if not cols:
        print("⚠️  Cluster này KHÔNG có collection nào (rỗng)!")
        print("    -> Dữ liệu migrate KHÔNG ở cluster mà .env đang trỏ.")
    else:
        print("Collections trên cluster .env trỏ tới:")
        for col in cols:
            n = c.get_collection(col.name).points_count
            print(f"  - {col.name}: {n:,} điểm")
except SystemExit as e:
    print(f"❌ Không kết nối được: {e}")
    sys.exit(1)

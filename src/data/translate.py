"""Phase 1 (VN) — Dịch seed EN -> tiếng Việt + QUALITY GATE 3 chiều.

Thiết kế:
  * Backend dịch PLUGGABLE (đổi qua config): NLLB (offline) hoặc LLM API (medical-aware).
  * QualityGate chấm mỗi mẫu theo 3 chiều rồi TỰ LỌC:
      1. Terminology  — giữ tên thuốc + liều/số (an toàn, quan trọng nhất)
      2. Naturalness  — thật sự là tiếng Việt, không phải bản dịch cứng / bỏ dịch
      3. CoT structure— chuỗi suy luận không bị co cụt / cắt cụt (nghiêm nhất ở đây)
  * Chấp nhận VỨT bớt mẫu để giữ chất lượng: 40k sạch > 90k nửa vời.
  * Log tỷ lệ pass/fail + ví dụ lỗi -> reports/translation_quality.md.

CẢNH BÁO: bản dịch tự động có thể sai liều/thuốc. Mẫu safety-critical (có liều) mà gate
nghi ngờ -> đưa vào hàng review tay, KHÔNG tự động nhận.

Usage:
    python -m src.data.translate --config configs/data.yaml \
        --in data/raw/seed_en.jsonl --out data/raw/seed_vi.jsonl
    python -m src.data.translate --demo     # chạy thử quality-gate bằng dữ liệu giả (offline)
"""
from __future__ import annotations

import argparse
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---- ký tự riêng tiếng Việt (proxy nhận diện đã dịch sang VI) ----
_VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
                "ĂÂĐÊÔƠƯ")
# ---- liều/số + đơn vị (giữ nguyên khi dịch) ----
_DOSE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg/kg|mcg|mg|g|kg|ml|l|iu|mmol|mmhg|%|units?)\b",
    re.IGNORECASE,
)
# ---- token thuốc: chữ hoa Latin không đứng đầu câu (Warfarin, Metformin...) ----
_DRUG_RE = re.compile(r"(?<![.\n]\s)(?<!^)\b[A-Z][a-z]{3,}(?:in|ol|ine|ide|one|am|il|ate)\b")


@dataclass
class GateResult:
    passed: bool
    scores: dict            # {"terminology":.., "naturalness":.., "cot":..}
    reasons: list = field(default_factory=list)
    needs_human: bool = False   # safety-critical + nghi ngờ -> review tay


class QualityGate:
    """Chấm chất lượng bản dịch mà KHÔNG cần model (thuần đối chiếu src/tgt).

    Nhờ đó test được offline. Các ngưỡng đọc từ config, có default hợp lý.
    """

    def __init__(self, cfg: dict | None = None):
        g = (cfg or {}).get("quality_gate_thresholds", {})
        self.min_terminology = g.get("terminology", 1.0)   # phải giữ HẾT liều/thuốc
        self.min_naturalness = g.get("naturalness", 0.6)
        self.min_cot = g.get("cot", 0.6)

    # ---- chiều 1: thuật ngữ (an toàn) ----
    @staticmethod
    def _norm_dose(s: str) -> str:
        return s.lower().replace(" ", "").replace(",", ".")

    def check_terminology(self, src: str, tgt: str) -> tuple[float, list[str]]:
        reasons = []
        src_doses = {self._norm_dose(x) for x in _DOSE_RE.findall(src)}
        # findall với group -> lấy lại full match
        src_doses = {self._norm_dose(m.group()) for m in _DOSE_RE.finditer(src)}
        tgt_doses = {self._norm_dose(m.group()) for m in _DOSE_RE.finditer(tgt)}
        src_drugs = set(_DRUG_RE.findall(src))
        tgt_text = tgt

        missing_dose = src_doses - tgt_doses
        missing_drug = {d for d in src_drugs if d not in tgt_text}

        total = len(src_doses) + len(src_drugs)
        if total == 0:
            return 1.0, reasons  # không có thuật ngữ cần giữ
        kept = (len(src_doses) - len(missing_dose)) + (len(src_drugs) - len(missing_drug))
        score = kept / total
        if missing_dose:
            reasons.append(f"MẤT liều: {sorted(missing_dose)}")
        if missing_drug:
            reasons.append(f"MẤT tên thuốc: {sorted(missing_drug)}")
        return score, reasons

    # ---- chiều 2: độ tự nhiên / thật sự đã dịch ----
    def check_naturalness(self, src: str, tgt: str) -> tuple[float, list[str]]:
        reasons = []
        if not tgt.strip():
            return 0.0, ["bản dịch rỗng"]
        vi_ratio = sum(c in _VI_CHARS for c in tgt) / max(len(tgt), 1)
        # gần trùng nguyên văn tiếng Anh -> chưa dịch
        same = tgt.strip().lower() == src.strip().lower()
        len_ratio = len(tgt) / max(len(src), 1)
        score = 1.0
        if same:
            score = 0.0
            reasons.append("gần như GIỮ NGUYÊN tiếng Anh (chưa dịch)")
        elif vi_ratio < 0.02:
            score = 0.3
            reasons.append(f"rất ít ký tự tiếng Việt (vi_ratio={vi_ratio:.3f})")
        if not (0.5 <= len_ratio <= 2.2):
            score = min(score, 0.4)
            reasons.append(f"độ dài lệch bất thường (len_ratio={len_ratio:.2f})")
        return score, reasons

    # ---- chiều 3: cấu trúc CoT (nghiêm nhất) ----
    @staticmethod
    def _n_sent(s: str) -> int:
        return len([x for x in re.split(r"[.!?…]\s", s) if x.strip()])

    def check_cot(self, src_cot: str, tgt_cot: str) -> tuple[float, list[str]]:
        reasons = []
        if not src_cot.strip():
            return 1.0, reasons  # mẫu không có CoT
        if not tgt_cot.strip():
            return 0.0, ["CoT bị mất hoàn toàn"]
        ns, nt = self._n_sent(src_cot), self._n_sent(tgt_cot)
        ratio = nt / max(ns, 1)
        # cắt cụt: kết thúc giữa chừng
        truncated = not tgt_cot.rstrip().endswith((".", "!", "?", "…", ":", ")"))
        score = 1.0
        if ratio < 0.6:
            score = 0.4
            reasons.append(f"CoT co cụt ({nt}/{ns} câu)")
        if truncated:
            score = min(score, 0.5)
            reasons.append("CoT có vẻ bị CẮT CỤT")
        return score, reasons

    # ---- tổng hợp ----
    def evaluate(self, src: dict, tgt: dict) -> GateResult:
        term_s, term_r = self.check_terminology(
            f"{src['question']} {src['cot']} {src['response']}",
            f"{tgt['question']} {tgt['cot']} {tgt['response']}",
        )
        nat_s, nat_r = self.check_naturalness(src["response"], tgt["response"])
        cot_s, cot_r = self.check_cot(src["cot"], tgt["cot"])

        passed = (
            term_s >= self.min_terminology
            and nat_s >= self.min_naturalness
            and cot_s >= self.min_cot
        )
        has_dose = bool(_DOSE_RE.search(f"{src['cot']} {src['response']}"))
        needs_human = has_dose and term_s < 1.0  # có liều + nghi ngờ thuật ngữ
        return GateResult(
            passed=passed,
            scores={"terminology": round(term_s, 3),
                    "naturalness": round(nat_s, 3),
                    "cot": round(cot_s, 3)},
            reasons=term_r + nat_r + cot_r,
            needs_human=needs_human,
        )


# ============ Backend dịch (pluggable) ============
class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str: ...

    def translate_record(self, rec: dict) -> dict:
        return {k: (self.translate(v) if v else "") for k, v in rec.items()}


class NLLBTranslator(Translator):
    """Offline, chạy được trên Kaggle free. Dịch thuần (không hiểu ngữ cảnh y khoa)."""

    def __init__(self, model="facebook/nllb-200-distilled-600M",
                 src_lang="eng_Latn", tgt_lang="vie_Latn"):
        from transformers import pipeline  # lazy: chỉ cần khi thật sự dịch
        self.pipe = pipeline("translation", model=model,
                             src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)

    def translate(self, text: str) -> str:
        # NLLB giới hạn độ dài -> nên chia câu; ở đây rút gọn cho scaffold.
        return self.pipe(text)[0]["translation_text"]


class LLMTranslator(Translator):
    """Medical-aware qua LLM API (GPT-5.5/Gemini...). Chất lượng cao hơn, TỐN PHÍ, cần mạng.

    Ưu điểm: prompt được để GIỮ tên thuốc, dịch thuật ngữ theo chuẩn VN, giữ format CoT —
    thứ NLLB không làm được.
    """

    SYSTEM = (
        "Bạn là chuyên gia dịch y khoa Anh->Việt. Quy tắc BẮT BUỘC:\n"
        "1. GIỮ NGUYÊN tên thuốc gốc Latin (Warfarin, Metformin...).\n"
        "2. GIỮ NGUYÊN mọi liều/số + đơn vị (5mg, 2.5 mg/kg...).\n"
        "3. Dịch thuật ngữ theo chuẩn y khoa Việt Nam, văn phong tự nhiên như bác sĩ Việt.\n"
        "4. GIỮ NGUYÊN cấu trúc/thứ tự các bước suy luận.\n"
        "Chỉ trả về bản dịch, không giải thích."
    )

    def __init__(self, model="gpt-5.5", api_key_env="OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env
        # TODO: khởi tạo client thật (openai/google). Để lười để scaffold không cần key.

    def translate(self, text: str) -> str:
        # TODO: gọi API thật với self.SYSTEM + text. Nhớ retry + rate limit.
        raise NotImplementedError(
            "Điền API call ở đây. Cần key + mạng (không chạy offline trên Kaggle free)."
        )


def get_translator(cfg: dict) -> Translator:
    engine = (cfg.get("translation", {}).get("engine") or "nllb").lower()
    if engine == "nllb":
        return NLLBTranslator()
    if engine in ("llm", "gpt", "gemini", "api"):
        return LLMTranslator(model=cfg["translation"].get("model", "gpt-5.5"))
    raise ValueError(f"engine không hỗ trợ: {engine}")


def translate_dataset(config_path: str, in_path: str, out_path: str,
                      report_path: str = "reports/translation_quality.md") -> dict:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    translator = get_translator(cfg)
    gate = QualityGate(cfg)

    src_recs = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    kept, rejected, human = [], [], []
    for rec in src_recs:
        tgt = translator.translate_record(rec)
        res = gate.evaluate(rec, tgt)
        row = {**tgt, "_gate": res.scores, "_reasons": res.reasons}
        if res.needs_human:
            human.append(row)
        elif res.passed:
            kept.append(row)
        else:
            rejected.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"total": len(src_recs), "kept": len(kept),
             "rejected": len(rejected), "needs_human": len(human)}
    _write_report(report_path, stats, rejected[:5], human[:5])
    print(f"[translate] {stats}")
    return stats


def _write_report(path, stats, rej_ex, hum_ex):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Translation Quality Report\n",
             f"- total: {stats['total']}",
             f"- kept: {stats['kept']}",
             f"- rejected: {stats['rejected']}",
             f"- needs_human (có liều + nghi ngờ): {stats['needs_human']}\n",
             "## Ví dụ bị loại"]
    for r in rej_ex:
        lines.append(f"- scores={r['_gate']} | reasons={r['_reasons']}")
    lines.append("\n## Ví dụ cần review tay")
    for r in hum_ex:
        lines.append(f"- scores={r['_gate']} | reasons={r['_reasons']}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ============ DEMO offline (không cần HF/model) ============
def _demo() -> None:
    """Chứng minh quality-gate bắt lỗi đúng bằng dữ liệu giả."""
    gate = QualityGate()
    cases = [
        ("✔ dịch tốt", {
            "question": "What is the max daily dose of Paracetamol for adults?",
            "cot": "Paracetamol max is 4000mg per day. Above this risks hepatotoxicity. So the answer is 4000mg.",
            "response": "The maximum daily dose of Paracetamol for adults is 4000mg."},
         {"question": "Liều Paracetamol tối đa mỗi ngày cho người lớn là bao nhiêu?",
          "cot": "Paracetamol tối đa là 4000mg mỗi ngày. Vượt mức này nguy cơ độc gan. Vậy đáp án là 4000mg.",
          "response": "Liều Paracetamol tối đa mỗi ngày cho người lớn là 4000mg."}),

        ("MẤT liều thuốc (nguy hiểm)", {
            "question": "Warfarin starting dose?",
            "cot": "Typical Warfarin starting dose is 5mg daily, then adjust by INR.",
            "response": "Start Warfarin at 5mg daily."},
         {"question": "Liều khởi đầu Warfarin?",
          "cot": "Liều khởi đầu Warfarin thường dùng hằng ngày, rồi chỉnh theo INR.",   # rớt 5mg
          "response": "Khởi đầu Warfarin hằng ngày."}),

        ("✗ CHƯA dịch (giữ nguyên tiếng Anh)", {
            "question": "What is COPD?", "cot": "COPD is a chronic lung disease.",
            "response": "COPD is a chronic obstructive pulmonary disease."},
         {"question": "What is COPD?", "cot": "COPD is a chronic lung disease.",
          "response": "COPD is a chronic obstructive pulmonary disease."}),

        ("✗ CoT bị co cụt / cắt cụt", {
            "question": "How does insulin lower glucose?",
            "cot": "Insulin binds its receptor. This triggers GLUT4 translocation. Glucose enters cells. Blood glucose falls. Therefore insulin lowers glucose.",
            "response": "Insulin lowers blood glucose by promoting cellular uptake."},
         {"question": "Insulin hạ đường huyết thế nào?",
          "cot": "Insulin gắn thụ thể và",   # cụt + co cụt
          "response": "Insulin hạ đường huyết bằng cách tăng hấp thu vào tế bào."}),
    ]
    print("=" * 72)
    for label, src, tgt in cases:
        r = gate.evaluate(src, tgt)
        verdict = "PASS" if r.passed else ("REVIEW" if r.needs_human else "REJECT")
        print(f"\n[{label}] -> {verdict}")
        print(f"   scores    : {r.scores}")
        print(f"   needs_human: {r.needs_human}")
        if r.reasons:
            print(f"   reasons   : {r.reasons}")
    print("\n" + "=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--in", dest="in_path", default="data/raw/seed_en.jsonl")
    ap.add_argument("--out", default="data/raw/seed_vi.jsonl")
    ap.add_argument("--demo", action="store_true", help="chạy demo quality-gate offline")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        translate_dataset(args.config, args.in_path, args.out)


if __name__ == "__main__":
    main()
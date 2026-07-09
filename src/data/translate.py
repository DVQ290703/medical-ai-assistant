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
# ---- tên thuốc: WHITELIST thuốc THẬT (so khớp không phân biệt hoa/thường) ----
# Bài học từ probe 500: regex hậu tố (-in/-ol/-ate...) bắt nhầm HÀNG LOẠT từ tiếng Anh thường
# (Borderline, Control, Protein, Program...) -> 177 mẫu oan. Whitelist chỉ cờ thuốc thật.
# Chưa đủ? Bổ sung dần / thay bằng Dược thư-RxNorm. Thiếu 1 thuốc = bỏ sót 1 (chấp nhận được),
# còn hơn loại oan hàng trăm mẫu tốt.
_DRUG_WHITELIST = {
    "warfarin", "heparin", "aspirin", "metformin", "insulin", "lisinopril", "losartan",
    "bosentan", "tolvaptan", "dapsone", "eszopiclone", "zolpidem", "diazepam", "lorazepam",
    "phenytoin", "carbamazepine", "valproate", "lithium", "haloperidol", "clozapine",
    "risperidone", "fluoxetine", "sertraline", "amoxicillin", "penicillin", "ceftriaxone",
    "azithromycin", "ciprofloxacin", "vancomycin", "gentamicin", "rifampicin", "isoniazid",
    "ethambutol", "pyrazinamide", "prednisone", "prednisolone", "dexamethasone",
    "hydrocortisone", "furosemide", "spironolactone", "amlodipine", "atenolol", "propranolol",
    "digoxin", "atorvastatin", "simvastatin", "omeprazole", "ranitidine", "levothyroxine",
    "methotrexate", "cyclophosphamide", "cisplatin", "doxorubicin", "morphine", "fentanyl",
    "naloxone", "adrenaline", "epinephrine", "atropine", "salbutamol", "paracetamol",
    "ibuprofen", "acetaminophen", "clopidogrel", "enoxaparin", "nitroglycerin",
}


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

    def check_terminology(self, src: str, tgt: str) -> tuple[float, set, list[str]]:
        """Trả (dose_score, missing_drugs, reasons).

        - dose_score: tỷ lệ liều/số được giữ (LOẠI CỨNG nếu <1 — rớt liều là nguy hiểm).
        - missing_drugs: thuốc trong WHITELIST có ở nguồn nhưng mất ở bản dịch (không phân
          biệt hoa/thường) -> chỉ REVIEW, không loại.
        """
        reasons = []
        # --- liều/số (an toàn, loại cứng) ---
        src_doses = {self._norm_dose(m.group()) for m in _DOSE_RE.finditer(src)}
        tgt_doses = {self._norm_dose(m.group()) for m in _DOSE_RE.finditer(tgt)}
        missing_dose = src_doses - tgt_doses
        dose_score = 1.0 if not src_doses else (len(src_doses) - len(missing_dose)) / len(src_doses)
        if missing_dose:
            reasons.append(f"MẤT liều/số (loại): {sorted(missing_dose)}")

        # --- tên thuốc: chỉ thuốc THẬT trong whitelist, so khớp không phân biệt hoa/thường ---
        src_low, tgt_low = src.lower(), tgt.lower()
        missing_drugs = {d for d in _DRUG_WHITELIST if d in src_low and d not in tgt_low}
        if missing_drugs:
            reasons.append(f"Nghi mất tên thuốc (review): {sorted(missing_drugs)}")
        return dose_score, missing_drugs, reasons

    # ---- chiều 2: độ tự nhiên / thật sự đã dịch ----
    def check_naturalness(self, src: str, tgt: str) -> tuple[float, list[str]]:
        reasons = []
        if not tgt.strip():
            return 0.0, ["bản dịch rỗng"]
        same = tgt.strip().lower() == src.strip().lower()
        len_ratio = len(tgt) / max(len(src), 1)
        score = 1.0
        # "chưa dịch" chỉ tính khi response ĐỦ DÀI (đáp án MCQ ngắn như 'D. Mononucleosis'
        # trùng EN/VI là bình thường -> để khâu MCQ ở validate xử lý, không loại ở đây)
        if same and len(tgt) >= 40:
            return 0.0, ["gần như GIỮ NGUYÊN tiếng Anh (chưa dịch)"]
        if len(tgt) >= 40:
            vi_ratio = sum(c in _VI_CHARS for c in tgt) / max(len(tgt), 1)
            if vi_ratio < 0.03:
                score = 0.3
                reasons.append(f"response dài nhưng rất ít tiếng Việt (vi_ratio={vi_ratio:.3f})")
        if not (0.4 <= len_ratio <= 2.5):
            score = min(score, 0.5)
            reasons.append(f"độ dài lệch bất thường (len_ratio={len_ratio:.2f})")
        return score, reasons

    # ---- chiều 3: cấu trúc CoT (dùng ĐỘ DÀI, KHÔNG đếm câu) ----
    def check_cot(self, src_cot: str, tgt_cot: str) -> tuple[float, list[str]]:
        """Bài học probe 500: đếm câu Anh/Việt SAI (hai ngôn ngữ ngắt câu khác nhau -> 336
        mẫu 'co cụt' oan). Dùng TỶ LỆ ĐỘ DÀI: cụt thật thì ngắn hẳn. Cắt-cụt-cuối chỉ nhắc nhẹ.
        """
        reasons = []
        if not src_cot.strip():
            return 1.0, reasons
        if not tgt_cot.strip():
            return 0.0, ["CoT bị mất hoàn toàn"]
        len_ratio = len(tgt_cot) / max(len(src_cot), 1)
        score = 1.0
        if len_ratio < 0.4:                 # ngắn bất thường -> dịch thiếu/cụt (loại)
            score = 0.3
            reasons.append(f"CoT ngắn bất thường (len_ratio={len_ratio:.2f})")
        # kết thúc giữa chừng -> nhắc nhẹ, KHÔNG tự loại (0.7 vẫn qua ngưỡng 0.6)
        if not tgt_cot.rstrip().endswith((".", "!", "?", "…", ":", ")", "%", "'", '"', "”")):
            score = min(score, 0.7)
            reasons.append("CoT có thể bị cắt cuối (nhắc)")
        return score, reasons

    # ---- tổng hợp ----
    def evaluate(self, src: dict, tgt: dict) -> GateResult:
        dose_score, missing_drugs, term_r = self.check_terminology(
            f"{src['question']} {src['cot']} {src['response']}",
            f"{tgt['question']} {tgt['cot']} {tgt['response']}",
        )
        nat_s, nat_r = self.check_naturalness(src["response"], tgt["response"])
        cot_s, cot_r = self.check_cot(src["cot"], tgt["cot"])

        # base_pass: liều giữ đủ + tự nhiên + CoT ok. (thuốc nghi KHÔNG chặn ở đây)
        base_pass = (
            dose_score >= self.min_terminology
            and nat_s >= self.min_naturalness
            and cot_s >= self.min_cot
        )
        # thuốc nghi mất + mọi thứ khác ổn -> đưa REVIEW (có thể do viết tắt hợp lệ)
        needs_human = base_pass and bool(missing_drugs)
        passed = base_pass and not missing_drugs   # kept chỉ khi hoàn toàn sạch
        return GateResult(
            passed=passed,
            scores={"terminology": round(dose_score, 3),   # = mức giữ liều/số
                    "naturalness": round(nat_s, 3),
                    "cot": round(cot_s, 3)},
            reasons=term_r + nat_r + cot_r,
            needs_human=needs_human,
        )


# ============ Backend dịch (pluggable) ============
class ContentFiltered(Exception):
    """API từ chối vì content filter (invalid_prompt). Retry vô ích -> bỏ qua mẫu, không sập."""


class Translator(ABC):
    last_usage: dict = {"in": 0, "out": 0}

    def reset_usage(self) -> None:
        self.last_usage = {"in": 0, "out": 0}

    @abstractmethod
    def translate(self, text: str) -> str: ...

    def translate_record(self, rec: dict) -> dict:
        return {k: (self.translate(v) if v else "") for k, v in rec.items()}


class NLLBTranslator(Translator):
    """Offline, chạy được trên Kaggle free. Dịch thuần (không hiểu ngữ cảnh y khoa).

    Dùng AutoModelForSeq2SeqLM TRỰC TIẾP (không dùng pipeline('translation') vì task string
    kén version transformers -> lỗi 'Invalid translation task'). Chia câu để CoT dài không
    bị cắt cụt (nếu không quality-gate sẽ REJECT nhầm vì tưởng dịch tệ).
    """

    def __init__(self, model="facebook/nllb-200-distilled-600M",
                 src_lang="eng_Latn", tgt_lang="vie_Latn", device=None, max_length=512):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.tok = AutoTokenizer.from_pretrained(model, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # id token ngôn ngữ đích — API khác nhau giữa các version transformers
        bos = self.tok.convert_tokens_to_ids(tgt_lang)
        if bos is None or bos == self.tok.unk_token_id:
            bos = getattr(self.tok, "lang_code_to_id", {}).get(tgt_lang)
        if bos is None:
            raise ValueError(f"Không lấy được token id cho ngôn ngữ đích {tgt_lang}")
        self.forced_bos = bos
        self.reset_usage()   # offline -> luôn 0, không đụng budget

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?…])\s+", text.strip())
        return [p for p in parts if p]

    def _translate_one(self, text: str) -> str:
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_length).to(self.device)
        out = self.model.generate(**enc, forced_bos_token_id=self.forced_bos,
                                  max_length=self.max_length)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        # dịch từng câu rồi ghép -> đoạn dài (CoT) không bị cắt cụt
        return " ".join(self._translate_one(s) for s in self._split_sentences(text))


class LLMTranslator(Translator):
    """Medical-aware qua LLM API (OpenAI GPT-5.5). Chất lượng cao, TỐN PHÍ, cần mạng.

    Ưu điểm so với NLLB: hiểu ngữ cảnh y khoa -> GIỮ tên thuốc, dịch thuật ngữ theo chuẩn VN,
    văn phong tự nhiên, giữ cấu trúc CoT.

    Tối ưu: dịch CẢ record trong 1 API call (trả JSON) -> ít call hơn + model thấy toàn bộ
    ngữ cảnh nên dịch nhất quán hơn 3 call rời.

    CHI PHÍ: ~$0.04/mẫu với gpt-5.5 -> cân nhắc dịch SUBSET đã lọc, không dịch cả 40k.
    """

    SYSTEM = (
        "Bạn là chuyên gia dịch y khoa Anh->Việt. Quy tắc BẮT BUỘC:\n"
        "1. GIỮ NGUYÊN tên thuốc gốc Latin (Warfarin, Metformin...).\n"
        "2. GIỮ NGUYÊN mọi liều/số + đơn vị (5mg, 2.5 mg/kg...).\n"
        "3. Dịch thuật ngữ theo chuẩn y khoa Việt Nam, văn phong tự nhiên như bác sĩ Việt.\n"
        "4. GIỮ NGUYÊN cấu trúc/thứ tự các bước suy luận trong chain-of-thought.\n"
    )
    # prompt cho dịch cả record 1 lần
    _RECORD_INSTR = (
        "Dịch 3 trường sau sang tiếng Việt. Trả về DUY NHẤT một JSON object với đúng 3 khoá "
        '"question", "cot", "response" (không thêm chữ nào ngoài JSON):\n\n'
        "question: {q}\n\ncot: {c}\n\nresponse: {r}"
    )

    def __init__(self, model="gpt-5.5", api_key_env="OPENAI_API_KEY",
                 max_retries=4, rate_limit_s=0.0, temperature=0.2):
        import os
        from openai import OpenAI  # lazy import: chỉ cần khi thật sự dùng

        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"Chưa có API key ở biến môi trường {api_key_env}. "
                f"Trên Kaggle: os.environ['{api_key_env}'] = "
                f"UserSecretsClient().get_secret('{api_key_env}')"
            )
        self.client = OpenAI(api_key=key)
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_s = rate_limit_s
        self.temperature = temperature
        self.reset_usage()

    def _call(self, user_msg: str, force_json: bool = False) -> str:
        import time
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        # một số model reasoning không nhận temperature -> thử, lỗi thì bỏ
        try_kwargs = dict(kwargs, temperature=self.temperature)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                try:
                    resp = self.client.chat.completions.create(**try_kwargs)
                except Exception:
                    resp = self.client.chat.completions.create(**kwargs)  # bỏ temperature
                if self.rate_limit_s:
                    time.sleep(self.rate_limit_s)
                u = getattr(resp, "usage", None)
                if u is not None:
                    self.last_usage["in"] += getattr(u, "prompt_tokens", 0) or 0
                    self.last_usage["out"] += getattr(u, "completion_tokens", 0) or 0
                return resp.choices[0].message.content.strip()
            except Exception as e:  # phân loại lỗi
                msg = str(e)
                code = getattr(e, "code", "") or ""
                # content filter: retry vô ích (cùng prompt cùng bị từ chối) -> báo để BỎ QUA
                if code == "invalid_prompt" or "invalid_prompt" in msg or "usage policy" in msg:
                    raise ContentFiltered(msg) from e
                # rate limit / lỗi mạng -> backoff rồi thử lại
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API thất bại sau {self.max_retries} lần: {last_err}")

    def translate(self, text: str) -> str:
        """Dịch một chuỗi (dùng bởi quality-gate/fallback)."""
        if not text.strip():
            return ""
        return self._call(f"Dịch sang tiếng Việt, chỉ trả bản dịch:\n\n{text}")

    def translate_record(self, rec: dict) -> dict:
        """Dịch cả record trong 1 call, trả JSON -> nhất quán + tiết kiệm."""
        msg = self._RECORD_INSTR.format(
            q=rec.get("question", ""), c=rec.get("cot", ""), r=rec.get("response", ""))
        raw = self._call(msg, force_json=True)
        try:
            data = json.loads(raw)
            return {"question": data.get("question", "").strip(),
                    "cot": data.get("cot", "").strip(),
                    "response": data.get("response", "").strip()}
        except json.JSONDecodeError:
            # model không trả JSON hợp lệ -> fallback dịch từng field
            return {k: self.translate(v) if v else "" for k, v in rec.items()}


def get_translator(cfg: dict) -> Translator:
    engine = (cfg.get("translation", {}).get("engine") or "nllb").lower()
    if engine == "nllb":
        return NLLBTranslator()
    if engine in ("llm", "gpt", "gemini", "api"):
        return LLMTranslator(model=cfg["translation"].get("model", "gpt-5.5"))
    raise ValueError(f"engine không hỗ trợ: {engine}")


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"date": "", "tokens_today": 0, "done": 0,
            "kept": 0, "rejected": 0, "needs_human": 0}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _final_stats(state: dict, total: int) -> dict:
    return {"total": total, "done": state["done"], "kept": state["kept"],
            "rejected": state["rejected"], "needs_human": state["needs_human"],
            "filtered": state.get("filtered", 0),
            "tokens_today": state["tokens_today"]}


def translate_dataset(config_path: str, in_path: str, out_path: str,
                      report_path: str = "reports/translation_quality.md",
                      daily_token_budget: int | None = None,
                      translator: "Translator | None" = None) -> dict:
    """Dịch có ĐẾM TOKEN + TỰ DỪNG theo hạn mức ngày + RESUME.

    - Ghi incrementally (append) -> crash-safe; chạy lại lệnh này là TIẾP TỤC từ chỗ dừng.
    - Chỉ LLM tiêu token; NLLB offline -> usage=0 -> không bao giờ chạm budget (dịch hết 1 lần).
    - State ở <out>.state.json: {date, tokens_today, done, kept, rejected, needs_human}.
    - Sang ngày mới: tokens_today tự reset 0, GIỮ nguyên tiến độ 'done'.
    - Mẫu bị loại/cần review ghi ra <out>.rejected.jsonl / <out>.needs_human.jsonl.
    """
    from datetime import date

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    translator = translator or get_translator(cfg)
    gate = QualityGate(cfg)
    budget = (daily_token_budget
              or cfg.get("translation", {}).get("daily_token_budget")
              or 2_400_000)   # margin dưới 2.5M free tier

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    rej_path = out.with_suffix(".rejected.jsonl")
    hum_path = out.with_suffix(".needs_human.jsonl")
    state_path = out.with_suffix(".state.json")

    state = _load_state(state_path)
    today = date.today().isoformat()
    if state.get("date") != today:      # ngày mới -> reset token, giữ tiến độ
        state["date"] = today
        state["tokens_today"] = 0

    src_recs = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    done = state["done"]
    if done >= len(src_recs):
        print(f"[translate] Đã dịch xong toàn bộ {len(src_recs)} mẫu.")
        return _final_stats(state, len(src_recs))

    f_keep = open(out, "a", encoding="utf-8")
    f_rej = open(rej_path, "a", encoding="utf-8")
    f_hum = open(hum_path, "a", encoding="utf-8")
    f_flt = open(out.with_suffix(".filtered.jsonl"), "a", encoding="utf-8")
    stopped = False
    try:
        for i in range(done, len(src_recs)):
            if state["tokens_today"] >= budget:
                stopped = True
                print(f"[translate] Chạm hạn mức ngày ({budget:,} token). Dừng ở mẫu "
                      f"{i}/{len(src_recs)}. Chạy lại lệnh này (ngày mai) để tiếp tục.")
                break
            rec = src_recs[i]
            if hasattr(translator, "reset_usage"):
                translator.reset_usage()
            try:
                tgt = translator.translate_record(rec)
            except ContentFiltered as e:
                # mẫu bị content filter chặn -> ghi riêng, BỎ QUA, đi tiếp (không sập)
                f_flt.write(json.dumps({**rec, "_filtered": str(e)[:200]},
                                       ensure_ascii=False) + "\n"); f_flt.flush()
                state["filtered"] = state.get("filtered", 0) + 1
                state["done"] = i + 1
                _save_state(state_path, state)
                continue
            res = gate.evaluate(rec, tgt)
            row = json.dumps({**tgt, "_gate": res.scores, "_reasons": res.reasons},
                             ensure_ascii=False) + "\n"
            if res.needs_human:
                f_hum.write(row); state["needs_human"] += 1
            elif res.passed:
                f_keep.write(row); state["kept"] += 1
            else:
                f_rej.write(row); state["rejected"] += 1
            f_keep.flush(); f_rej.flush(); f_hum.flush()

            state["tokens_today"] += (translator.last_usage.get("in", 0)
                                      + translator.last_usage.get("out", 0))
            state["done"] = i + 1
            _save_state(state_path, state)   # persist mỗi mẫu -> crash-safe
    finally:
        f_keep.close(); f_rej.close(); f_hum.close(); f_flt.close()

    stats = _final_stats(state, len(src_recs))
    stats["stopped_at_budget"] = stopped
    _write_report(report_path, state, budget, len(src_recs))
    print(f"[translate] {stats}")
    return stats


def _write_report(path, state, budget, total):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pct = 100 * state["done"] / max(total, 1)
    lines = ["# Translation Quality Report\n",
             f"- tiến độ: {state['done']}/{total} ({pct:.1f}%)",
             f"- kept: {state['kept']}",
             f"- rejected: {state['rejected']}",
             f"- needs_human (có liều + nghi ngờ): {state['needs_human']}",
             f"- filtered (content filter chặn): {state.get('filtered', 0)}",
             f"- tokens_today: {state['tokens_today']:,} / budget {budget:,}\n",
             "Chi tiết mẫu bị loại / cần review tay:",
             "- `<out>.rejected.jsonl`",
             "- `<out>.needs_human.jsonl`"]
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
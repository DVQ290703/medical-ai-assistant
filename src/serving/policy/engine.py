"""Policy — Điều phối rules -> QUYẾT ĐỊNH hành động cho 1 query.

Khác Guard (kiểm tra), Policy quyết định LÀM GÌ. Chạy TRƯỚC retrieve/LLM:
  - emergency        -> action=escalate (trả thông điệp 115 ngay, KHÔNG gọi LLM)
  - out_of_scope     -> action=refuse   (nếu refuse_out_of_scope=true trong prompt.yaml)
  - còn lại          -> action=answer   (kèm cờ need_doctor để chèn disclaimer sau)

Trả PolicyDecision để orchestrator thực thi. Đây là "action layer" README mô tả.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

from src.serving.policy import rules
from src.serving.policy.escalation import escalation_message


REFUSE_MSG = ("Xin lỗi, tôi là trợ lý thông tin y tế nên chỉ hỗ trợ các câu hỏi về sức khoẻ, "
              "bệnh, thuốc, triệu chứng... Bạn vui lòng đặt câu hỏi trong phạm vi này.")


@dataclass
class PolicyDecision:
    action: str          # escalate | refuse | answer
    message: str = ""    # nội dung trả ngay (escalate/refuse); rỗng nếu action=answer
    need_doctor: bool = False


def _refuse_out_of_scope(path: str = "configs/prompt.yaml") -> bool:
    if not os.path.exists(path):
        return True
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return bool(y.get("refuse_out_of_scope", True))


def decide(query: str, in_conversation: bool = False) -> PolicyDecision:
    """in_conversation: đang giữa hội thoại (đã có lượt trước). Khi True, KHÔNG refuse câu
    ngắn tiếp nối (vd 'trẻ 3 tuổi', '15kg') — tự nó không có dấu hiệu y tế nhưng là câu
    trả lời cho câu bot vừa hỏi. Refuse out-of-scope chỉ áp cho câu MỞ ĐẦU cuộc hội thoại.
    """
    # 1. Cấp cứu -> escalate (ưu tiên tuyệt đối — xét MỌI lượt, kể cả giữa hội thoại)
    emg = escalation_message(query)
    if emg:
        return PolicyDecision(action="escalate", message=emg)

    # 2. Ngoài phạm vi y tế -> refuse (chỉ khi là câu MỞ ĐẦU, không phải câu tiếp nối)
    if not in_conversation and _refuse_out_of_scope() and rules.is_out_of_scope(query):
        return PolicyDecision(action="refuse", message=REFUSE_MSG)

    # 3. Trả lời bình thường (đánh dấu need_doctor để chèn disclaimer phù hợp)
    return PolicyDecision(action="answer", need_doctor=rules.need_doctor(query))

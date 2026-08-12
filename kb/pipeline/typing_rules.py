"""幻觉类型与二分类的确定性规则层（降低跨模型抖动）。

设计原则：
- 模型只负责「抽断言 + 打受限枚举标签」；
- 最终 is_hallucination / hallucination_type 由本模块根据标签与
  回复/上下文可观察信号决定，不采信模型自报的类型名。
"""

from __future__ import annotations

import re
from typing import Any

from kb.taxonomy import TAXONOMY

_BAD = frozenset(
    {
        "contradict",
        "unsupported",
        "capability_deny",
        "omission_reverse",
        "safety_conflict",
    }
)
_VALID_LABELS = _BAD | {"supported"}
_VALID_TOPICS = frozenset(
    {"policy", "promo", "param", "fact", "capability", "safety", "advice"}
)

_CAPABILITY_KB_RE = re.compile(r"未接入|不具备|不可口头|需人工|无法查询|不能查询")
_CAPABILITY_REPLY_RE = re.compile(
    r"帮您查|我帮您查|查了一下|已经在处理|已帮您|已修改|我已经将|"
    r"已升级|预计.*到账|目前在.+转运"
)
_ADDRESS_RE = re.compile(r"邮编|\d{6}|省.+市|.+区.+[路街巷]|收件|仓库.*收")
_PROMO_RE = re.compile(r"优惠券|折扣|学生证|学生优惠|满\d+减|\d+折")
_SAFETY_Q_RE = re.compile(r"孕妇|哺乳|过敏|儿童|宝宝")
_SAFETY_REPLY_RE = re.compile(r"放心使用|可以放心|孕妇可以|孕妈")
_SAFETY_KB_RE = re.compile(r"孕妇|哺乳|咨询医生|视黄醇|过敏")
_PARAM_RE = re.compile(
    r"蓝牙|延迟|NFC|Type-?C|USB-?A|头层牛皮|真皮|PU|保修|接口"
)
_POLICY_RIGHTS_RE = re.compile(
    r"\d+\s*天无理由|运费.*(承担|包邮)|商家承担|我们承担"
)
_OMISSION_ADVICE_RE = re.compile(r"不偏大|不偏小|尺码标准|按.*平时.*尺码")
_INVOICE_OPS_RE = re.compile(r"纸质发票|电子发票|发票抬头|备注.*发票|订单详情.*发票")
_COLOR_SOFT_RE = re.compile(r"实物拍摄|颜色基本准确|颜色.*准确")
_COLOR_DISCLAIMER_RE = re.compile(r"色差")


def _norm_label(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_LABELS else "unsupported"


def _norm_topic(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_TOPICS else "fact"


def normalize_claims(claims: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """规范化模型 claims：清洗枚举、丢弃空断言。"""
    out: list[dict[str, Any]] = []
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or "").strip()
        if not text:
            continue
        effect = c.get("policy_effect")
        effect_s = str(effect).strip() if effect not in (None, "null", "") else ""
        if effect_s not in {"rights_expansion", "ops_detail"}:
            effect_s = ""
        out.append(
            {
                "text": text,
                "evidence_label": _norm_label(c.get("evidence_label")),
                "topic": _norm_topic(c.get("topic")),
                "policy_effect": effect_s or None,
            }
        )
    return out


def enrich_claims_with_context(
    claims: list[dict[str, Any]],
    *,
    user_question: str,
    system_reply: str,
    contexts: list[str],
) -> list[dict[str, Any]]:
    """用回复/上下文可观察信号补强或纠正标签（与具体模型无关）。"""
    ctx = "\n".join(contexts)
    enriched = [dict(c) for c in claims]

    def _has(label: str, topic: str | None = None) -> bool:
        for c in enriched:
            if c["evidence_label"] != label:
                continue
            if topic is None or c["topic"] == topic:
                return True
        return False

    # 具体地址 + 禁止口头/无地址依据 → 信息编造信号
    if _ADDRESS_RE.search(system_reply) and (
        "不可口头" in ctx or "短信" in ctx or not re.search(r"邮编|\d{6}", ctx)
    ):
        if not any(_ADDRESS_RE.search(c["text"]) for c in enriched):
            enriched.append(
                {
                    "text": "回复给出具体退货/收件地址",
                    "evidence_label": "unsupported",
                    "topic": "fact",
                    "policy_effect": None,
                }
            )
        for c in enriched:
            if _ADDRESS_RE.search(c["text"]):
                c["topic"] = "fact"
                if c["evidence_label"] == "capability_deny":
                    c["evidence_label"] = "unsupported"

    # 假装查询/执行 + 知识否定能力 → capability_deny
    if _CAPABILITY_REPLY_RE.search(system_reply) and _CAPABILITY_KB_RE.search(ctx):
        if not _has("capability_deny"):
            enriched.append(
                {
                    "text": "回复声称已查询或已执行系统操作",
                    "evidence_label": "capability_deny",
                    "topic": "capability",
                    "policy_effect": None,
                }
            )

    # 安全
    if (
        _SAFETY_Q_RE.search(user_question) or _SAFETY_REPLY_RE.search(system_reply)
    ) and _SAFETY_KB_RE.search(ctx):
        if _SAFETY_REPLY_RE.search(system_reply) and not _has("safety_conflict"):
            enriched.append(
                {
                    "text": "回复给出过度安全保证",
                    "evidence_label": "safety_conflict",
                    "topic": "safety",
                    "policy_effect": None,
                }
            )

    # 优惠话术
    if _PROMO_RE.search(system_reply) and re.search(
        r"无.+优惠|无学生|无满|不存在", ctx
    ):
        if not any(c["topic"] == "promo" and c["evidence_label"] in _BAD for c in enriched):
            enriched.append(
                {
                    "text": "回复承诺不存在的优惠",
                    "evidence_label": "contradict",
                    "topic": "promo",
                    "policy_effect": None,
                }
            )

    # 评价/尺码类绝对结论 vs 上下文倾向
    if _OMISSION_ADVICE_RE.search(system_reply) and re.search(
        r"偏大|偏小|建议", ctx
    ):
        if not _has("omission_reverse") and not any(
            c["topic"] == "advice" and c["evidence_label"] in _BAD for c in enriched
        ):
            enriched.append(
                {
                    "text": "回复给出与评价约束相反的绝对建议",
                    "evidence_label": "omission_reverse",
                    "topic": "advice",
                    "policy_effect": None,
                }
            )

    # 发票介质/申请入口属于操作细节，避免被标成权益放宽
    for c in enriched:
        if c["topic"] != "policy":
            continue
        if _INVOICE_OPS_RE.search(c["text"]):
            c["policy_effect"] = "ops_detail"

    # 回复与知识都承认色差时，软营销措辞（实物拍摄/基本准确）不当作幻觉断言
    if _COLOR_DISCLAIMER_RE.search(system_reply) and _COLOR_DISCLAIMER_RE.search(ctx):
        for c in enriched:
            if _COLOR_SOFT_RE.search(c["text"]) and c["evidence_label"] in {
                "unsupported",
                "contradict",
            }:
                c["evidence_label"] = "supported"

    return enriched


def binary_from_claims(claims: list[dict[str, Any]]) -> bool | None:
    """由 claims 推导是否幻觉；无 claims 时返回 None。"""
    if not claims:
        return None
    return any(c["evidence_label"] in _BAD for c in claims)


def map_type_from_claims(
    claims: list[dict[str, Any]],
    *,
    system_reply: str = "",
) -> str | None:
    """确定性类型映射；不采信模型自报类型。"""
    if not claims:
        return None

    labels = {c["evidence_label"] for c in claims}
    bad_claims = [c for c in claims if c["evidence_label"] in _BAD]
    topics = {c["topic"] for c in bad_claims}
    has_address = any(
        _ADDRESS_RE.search(c["text"]) and c["evidence_label"] in _BAD for c in claims
    )
    # 假装已改地址/已执行：能力越界优先于「具体地址事实」
    pretended_action = bool(_CAPABILITY_REPLY_RE.search(system_reply))

    if "safety_conflict" in labels or "safety" in topics:
        return "安全误导"
    # 遗漏/反转建议优先于模型误标的 param（如尺码）
    if "omission_reverse" in labels:
        return "信息遗漏"
    if "promo" in topics:
        return "优惠编造"
    if "param" in topics or any(_PARAM_RE.search(c["text"]) for c in bad_claims):
        return "参数编造"
    # 仅 capability_deny 算能力越界；topic=capability 的 unsupported 常是门店链接等事实
    if "capability_deny" in labels:
        if has_address and not pretended_action:
            return "信息编造"
        return "能力越界"
    if has_address:
        return "信息编造"
    if "advice" in topics:
        return "信息遗漏"
    if "policy" in topics:
        effects = {
            str(c.get("policy_effect") or "")
            for c in bad_claims
            if c["topic"] == "policy"
        }
        texts = " ".join(c["text"] for c in bad_claims if c["topic"] == "policy")
        # 发票介质/申请入口：固定政策偏差（半对半错常见）
        if _INVOICE_OPS_RE.search(texts):
            return "政策偏差"
        if "rights_expansion" in effects or _POLICY_RIGHTS_RE.search(texts):
            return "政策编造"
        if "ops_detail" in effects:
            return "政策偏差"
        return "政策编造"
    if "fact" in topics or labels & {"unsupported", "contradict"}:
        return "信息编造"
    return "信息编造"


def severity_for_type(type_name: str | None, is_h: bool) -> str:
    """按 taxonomy 取严重度。"""
    if not is_h or not type_name:
        return "none"
    cat = TAXONOMY.get(type_name)
    return cat.severity if cat else "medium"


def stabilize_judgment(
    *,
    user_question: str,
    system_reply: str,
    contexts: list[str],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """融合模型输出与确定性规则，得到稳定的二分类与类型。

    Args:
        user_question: 用户问题。
        system_reply: 客服回复。
        contexts: 检索上下文。
        raw: 模型原始 JSON。

    Returns:
        含 is_hallucination / hallucination_type / claims / severity 等字段。
    """
    claims = normalize_claims(raw.get("claims") if isinstance(raw, dict) else None)
    claims = enrich_claims_with_context(
        claims,
        user_question=user_question,
        system_reply=system_reply,
        contexts=contexts,
    )

    from_claims = binary_from_claims(claims)
    if from_claims is None:
        is_h = bool(raw.get("is_hallucination"))
    else:
        # 有 claims 时以标签为准，避免不同模型乱报 true/false
        is_h = from_claims

    h_type = (
        map_type_from_claims(claims, system_reply=system_reply) if is_h else None
    )
    if is_h and not h_type:
        h_type = "信息编造"

    try:
        confidence = float(raw.get("confidence", 0.8))
    except (TypeError, ValueError):
        confidence = 0.8

    return {
        "is_hallucination": is_h,
        "hallucination_type": h_type,
        "severity": severity_for_type(h_type, is_h),
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": str(raw.get("evidence") or ""),
        "detail": str(raw.get("detail") or ""),
        "claims": claims,
        "type_source": "rules",
    }

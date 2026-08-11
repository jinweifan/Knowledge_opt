"""客服回复幻觉分类体系。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HallucinationCategory:
    """幻觉类型定义。"""

    name: str
    severity: str  # critical / high / medium / low
    severity_score: int  # 4=critical ... 1=low
    definition: str
    examples: str


# 判定原则：以知识库为唯一事实源。
# 回复中出现与知识库矛盾、知识库明确否定、或系统明确不具备的能力/信息，即视为幻觉。
TAXONOMY: dict[str, HallucinationCategory] = {
    "安全误导": HallucinationCategory(
        name="安全误导",
        severity="critical",
        severity_score=4,
        definition=(
            "回复在健康/安全相关问题上给出与知识库相悖或过度肯定的建议，"
            "可能造成人身伤害或健康风险。"
        ),
        examples="孕妇可用、可放心食用/使用，而知识库要求咨询医生或标注风险成分。",
    ),
    "政策编造": HallucinationCategory(
        name="政策编造",
        severity="high",
        severity_score=3,
        definition="杜撰或大幅放宽退货、发票、发货等业务政策，核心条款与知识库冲突。",
        examples="把7天无理由说成30天且商家包运费。",
    ),
    "政策偏差": HallucinationCategory(
        name="政策偏差",
        severity="medium",
        severity_score=2,
        definition="政策主体正确或部分正确，但关键细节（时效、渠道、范围）与知识库不符。",
        examples="发票支持电子但误说支持纸质；发货时效/快递公司说错。",
    ),
    "参数编造": HallucinationCategory(
        name="参数编造",
        severity="high",
        severity_score=3,
        definition="编造或篡改产品规格、材质、功能、接口等客观参数。",
        examples="蓝牙版本、材质、NFC、充电接口等与知识库不符或知识库未标注却肯定支持。",
    ),
    "优惠编造": HallucinationCategory(
        name="优惠编造",
        severity="high",
        severity_score=3,
        definition="杜撰不存在的优惠券、折扣、学生价等促销信息。",
        examples="满300减50、学生9折等知识库明确不存在的活动。",
    ),
    "信息编造": HallucinationCategory(
        name="信息编造",
        severity="high",
        severity_score=3,
        definition="编造具体事实性信息（地址、门店、品牌关系等），知识库无依据或明确否定。",
        examples="杜撰退货地址、线下门店、与其他品牌隶属关系。",
    ),
    "能力越界": HallucinationCategory(
        name="能力越界",
        severity="high",
        severity_score=3,
        definition=(
            "知识库标明系统不具备某查询/操作能力，回复却假装已查询、已执行或给出具体结果。"
        ),
        examples="未接物流/退款接口却报进度；未接改单接口却称已改地址；假装升级工单。",
    ),
    "信息遗漏": HallucinationCategory(
        name="信息遗漏",
        severity="low",
        severity_score=1,
        definition=(
            "知识库含关键约束/评价信息，回复省略后给出相反或误导性结论。"
            "边界相对模糊：仅省略但结论仍保守时可不判幻觉。"
        ),
        examples="知识库反馈偏大半码，回复却说尺码完全标准。",
    ),
}

CATEGORY_NAMES: list[str] = list(TAXONOMY.keys())

# 与 ground_truth 类型名对齐，便于评估时做类型对照
GT_TYPE_ALIASES = {
    "政策编造": "政策编造",
    "政策偏差": "政策偏差",
    "参数编造": "参数编造",
    "优惠编造": "优惠编造",
    "信息编造": "信息编造",
    "能力越界": "能力越界",
    "安全误导": "安全误导",
    "信息遗漏": "信息遗漏",
}


def taxonomy_markdown() -> str:
    """将分类体系格式化为 Markdown 表格。

    Returns:
        Markdown 字符串。
    """
    lines = [
        "| 类型 | 严重程度 | 定义 |",
        "|---|---|---|",
    ]
    for cat in TAXONOMY.values():
        lines.append(
            f"| {cat.name} | {cat.severity} ({cat.severity_score}) | {cat.definition} |"
        )
    return "\n".join(lines)

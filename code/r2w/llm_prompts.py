"""R2W FINAL 的真实表示抽取契约。"""

KV = """你是 R2W 键值记忆抽取器。根据下列对话抽取至多 {max_keys} 个可检索事实。
每个 value 必须是包含 key 的原句片段，最多 {max_value_words} 个词；不得编造。
key 与 value 必须保持对话的原语言，不得翻译；第一人称事实必须能由说话人字段确定主体。
严格输出 JSON：{{"items":[{{"key":"...","value":"..."}}]}}。
说话人：{speaker}
对话：{text}"""

EVENT = """你是 R2W 事件记忆抽取器。为下列对话给出至多 {max_items} 个可检索事件。
time 必须是以 fallback_time 为锚点归一化后的绝对时间：
- yesterday、last Saturday 等表达要换算为明确日历日期；
- last week 等区间要写成“week before <fallback date>”这类带明确日期锚点的形式；
- 原文没有时间表达时直接使用 fallback_time；
- time 中不得保留 yesterday、today、last week、last Saturday 等无日期锚点的相对表达。
event 必须忠实概括原文、每项最多 {max_words} 个词，保持原语言，不得翻译或编造。
按检索价值排序；没有事件时 items 为空。严格输出 JSON：
{{"items":[{{"time":"...","event":"..."}}]}}。
说话人：{speaker}
fallback_time：{fallback_time}
对话：{text}"""

GRAPH = """你是 R2W 实体关系抽取器。从下列单轮对话抽取至多 {max_items} 条明确提到的关系。
关系必须由原文支持并保持对话的原语言，不得翻译或编造。把 I、me、my 等第一人称指代解析为说话人姓名。
每条必须带可检索的 value 节点、来源时间窗；不清楚的 valid_to 使用 null。
严格输出 JSON：{{"relations":[{{"subject":"...","relation":"...","object":"...",
"value":"...","valid_from":"...","valid_to":null}}]}}。
说话人：{speaker}
对话：{text}"""

READER = """当前日期：{date}
问题：{question}
检索到的记忆正文（索引卡不会展示给你）：
{context}
只能根据正文回答。cites 必须列出所有实际用于答案的正文编号；没有答案时回答 unknown 且 cites=[]。
严格输出 JSON：{{"answer":"...","cites":[0]}}。"""

FACT_SPLIT = """你是 R2W 事实核验器。把下列记忆正文拆成最小、独立、可核验的事实。
不得补充正文没有表达的信息；说话人、日期、数字和否定必须保留。元信息块可以作为事实来源。
严格输出 JSON：{{"facts":["..."]}}。
记忆正文：{payload}"""

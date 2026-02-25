import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ai_cloud_plugin import SiliconFlowAI
from constants import FILE_CLUSTER_PROMPT_, SUFFIX_NEED2_CLASSIFY
from tokenization import _read_file_content, get_token


@dataclass
class ClusterComputation:
    category_map: Dict[str, List[str]]
    preview_payload: Dict[str, str]
    raw_response: str


class LLMFileClusterPlanner:
    """负责将文件描述交给 LLM 并解析聚类结果"""

    def __init__(
        self,
        base_folder: str,
        llm_client: Optional[SiliconFlowAI] = None,
        max_snippet_chars: int = 600,
        preview_examples: int = 3,
    ) -> None:
        self.base_folder = base_folder
        self.max_snippet_chars = max(120, max_snippet_chars)
        self.preview_examples = max(1, preview_examples)
        self.llm = llm_client or SiliconFlowAI(
            system_prompt=FILE_CLUSTER_PROMPT_,
            stream=False,
            max_tokens=4096,
        )

    def cluster_files(self, file_paths: Sequence[str]) -> ClusterComputation:
        records = self._collect_records(file_paths)
        if not records:
            raise ValueError("未收集到可供聚类的文件")
        prompt = self._build_prompt(records)
        response = self.llm.chat_with_ai(prompt)
        if not response:
            raise RuntimeError("LLM 返回内容为空，无法完成聚类")
        clusters, category_map = self._parse_response(response, records)
        preview_payload = self._build_preview_payload(clusters, category_map, records)
        return ClusterComputation(category_map=category_map, preview_payload=preview_payload, raw_response=response)

    def _collect_records(self, file_paths: Sequence[str]) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        for idx, file_path in enumerate(sorted(file_paths)):
            if not os.path.isfile(file_path):
                continue
            file_id = f"F{idx + 1:03d}"
            rel_path = os.path.relpath(file_path, self.base_folder)
            rel_path = rel_path.replace("\\", "/")
            snippet = self._extract_snippet(file_path)
            if not snippet:
                snippet = "内容不可读取，请结合文件名与路径判断主题。"
            try:
                stat = os.stat(file_path)
            except OSError:
                continue
            records.append(
                {
                    "id": file_id,
                    "path": file_path,
                    "rel_path": rel_path,
                    "name": os.path.basename(file_path),
                    "ext": os.path.splitext(file_path)[1] or "",
                    "size": stat.st_size,
                    "snippet": snippet,
                }
            )
        return records

    def _extract_snippet(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUFFIX_NEED2_CLASSIFY:
            return ""
        try:
            content = _read_file_content(file_path)
        except Exception as exc:  # pragma: no cover - 容错打印
            print(f"LLM聚类读取文件失败: {file_path}: {exc}")
            return ""
        tokens = get_token(content)
        snippet = " ".join(tokens)
        snippet = re.sub(r"\s+", " ", snippet)
        snippet = snippet.replace("```", "` ` `")
        return snippet[: self.max_snippet_chars]

    def _build_prompt(self, records: Sequence[Dict[str, str]]) -> str:
        lines = [
            "下面是待聚类的文件清单。请根据文件内容摘要与文件名，为这些文件生成合适的聚类分类。",
            "输出必须是 JSON 数组，每个元素包含 category(类别名)、reason(简短说明)、files(文件ID列表)。",
            "仅使用我提供的文件ID（如 F001），不要自行编造路径。",
            "分类必须体现业务/主题语义，如“校园项目”“客户合同”“公司制度”，禁止使用“PDF文件”“Excel文档”等纯文件格式类别。",
            "如果某些文件确实无法归到其它类别，才放入名为 '未分类' 的类别，并在 reason 中说明原因。",
            "文件列表：",
        ]
        for record in records:
            lines.extend(
                [
                    f"{record['id']}：",
                    f"- 文件名: {record['name']}",
                    f"- 相对路径: {record['rel_path']}",
                    f"- 扩展名: {record['ext'] or '无'} | 大小: {record['size']} 字节",
                    f"- 内容摘要: {record['snippet']}",
                ]
            )
        lines.append(
            "请直接返回 JSON 数组，不要添加额外解释或前后缀。示例：[{\"category\":\"项目文档\",\"reason\":\"描述项目规划\",\"files\":[\"F001\",\"F002\"]}]"
        )
        return "\n".join(lines)

    def _parse_response(self, response: str, records: Sequence[Dict[str, str]]):
        cleaned = self._strip_code_fence(response.strip())
        json_obj = self._try_load_json(cleaned)
        if isinstance(json_obj, dict) and "clusters" in json_obj:
            clusters = json_obj.get("clusters")
        else:
            clusters = json_obj
        if not isinstance(clusters, list):
            raise ValueError("LLM 返回格式不正确，期望为 JSON 数组")
        id_to_record = {record["id"]: record for record in records}
        category_map: Dict[str, List[str]] = {}
        assigned_ids = set()
        normalized_clusters = []
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            raw_category = str(cluster.get("category", "")).strip() or "LLM未命名类别"
            reason = str(cluster.get("reason", "")).strip()
            file_ids = self._normalize_file_ids(cluster.get("files"))
            resolved_files = []
            preview_files = []
            for file_id in file_ids:
                if file_id in assigned_ids:
                    continue
                record = id_to_record.get(file_id)
                if not record:
                    continue
                assigned_ids.add(file_id)
                resolved_files.append(record["path"])
                preview_files.append(record["name"])
            if resolved_files:
                category_map.setdefault(raw_category, []).extend(resolved_files)
            normalized_clusters.append(
                {
                    "category": raw_category,
                    "reason": reason,
                    "files": preview_files,
                }
            )
        unassigned_ids = set(id_to_record.keys()) - assigned_ids
        if unassigned_ids:
            fallback_paths = [id_to_record[file_id]["path"] for file_id in unassigned_ids]
            category_map.setdefault("未分类", []).extend(fallback_paths)
            normalized_clusters.append(
                {
                    "category": "未分类",
                    "reason": "LLM 未覆盖，自动补充",
                    "files": [id_to_record[file_id]["name"] for file_id in unassigned_ids],
                }
            )
        return normalized_clusters, category_map

    def _build_preview_payload(
        self,
        clusters: Sequence[Dict[str, Sequence[str]]],
        category_map: Dict[str, List[str]],
        records: Sequence[Dict[str, str]],
    ) -> Dict[str, str]:
        total_files = len(records)
        lines = [f"共{total_files}个文件，LLM建议{len(clusters)}个分类："]
        for cluster in clusters:
            files = cluster.get("files") or []
            lines.append(
                f"- {cluster.get('category', '未命名类别')}（{len(files)} 个）: {cluster.get('reason', '未提供原因') or '未提供原因'}"
            )
            for name in list(files)[: self.preview_examples]:
                lines.append(f"    · {name}")
            if len(files) > self.preview_examples:
                lines.append("    · ...")
        summary = f"是否按照上述LLM聚类结果移动{total_files}个文件？"
        detail_text = "\n".join(lines)
        return {"summary": summary, "detail": detail_text}

    def _strip_code_fence(self, text: str) -> str:
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*", "", text)
            text = text.rstrip("`")
        return text.strip()

    def _try_load_json(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\[.*\]|\{.*\})", text, re.S)
            if match:
                return json.loads(match.group(1))
            raise

    def _normalize_file_ids(self, files_field) -> List[str]:
        result: List[str] = []
        if isinstance(files_field, str):
            result.append(files_field.strip())
        elif isinstance(files_field, list):
            for item in files_field:
                if isinstance(item, str):
                    result.append(item.strip())
                elif isinstance(item, dict):
                    for key in ("id", "file_id", "fileId", "fileID"):
                        if item.get(key):
                            result.append(str(item[key]).strip())
                            break
        result = [fid for fid in result if fid]
        # 去重保持顺序
        seen = set()
        deduped = []
        for fid in result:
            if fid not in seen:
                deduped.append(fid)
                seen.add(fid)
        return deduped

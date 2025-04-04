# core/workflow_descriptor.py

from dataclasses import dataclass
from typing import List

@dataclass
class WorkflowDescriptor:
    name: str                     # имя JSON-файла
    output_node_ids: List[str]   # список нод, в которых ожидаются финальные результаты
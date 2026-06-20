"""
Shared pipeline data types.

Author: Gayatri Malladi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "metadata": self.metadata,
        }

    @property
    def is_poisoned(self) -> bool:
        # New format: poisoned docs have _id starting with "IPI_"
        # Old format: had _poisoned in metadata
        # Visual/OCR and newer curated corpora use is_poisoned / attack_family.
        return bool(
            self.doc_id.startswith("IPI_")
            or self.metadata.get("_poisoned", False)
            or self.metadata.get("is_poisoned", False)
            or self.metadata.get("attack_family")
            or self.metadata.get("security_family")
        )

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


from openviking.storage.index_action import IndexAction


_UPDATE_FIELD_ALLOWLIST = frozenset(
    {"md5", "content", "abstract", "updated_at", "active_count", "tags", "search_tags"}
)


@dataclass
class EmbeddingMsg:
    """Durable embedding-queue message for model or index-only work.

    ``context_data`` holds context identity and normal scalar data for
    ``UPSERT``/``MERGE``. ``record_ids``/``update_fields`` make delete and scalar
    updates explicit rather than encoding them as special embedding payloads.
    """

    message: Optional[Union[str, List[Dict[str, Any]]]]
    context_data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    telemetry_id: str = ""
    action: IndexAction = IndexAction.UPSERT
    record_ids: List[str] = field(default_factory=list)
    update_fields: Dict[str, Any] = field(default_factory=dict)
    queue_enqueued_at: float = 0.0

    def __init__(
        self,
        message: Optional[Union[str, List[Dict[str, Any]]]],
        context_data: Dict[str, Any],
        telemetry_id: str = "",
        action: IndexAction | str = IndexAction.UPSERT,
        record_ids: Optional[List[str]] = None,
        update_fields: Optional[Dict[str, Any]] = None,
        queue_enqueued_at: float = 0.0,
    ):
        self.id = str(uuid4())
        self.message = message
        self.context_data = context_data
        self.telemetry_id = telemetry_id
        self.action = IndexAction(action)
        self.record_ids = list(record_ids or [])
        self.update_fields = dict(update_fields or {})
        self.queue_enqueued_at = max(float(queue_enqueued_at or 0.0), 0.0)
        self._validate()

    def _validate(self) -> None:
        if self.action is IndexAction.NONE:
            if self.message is not None or self.record_ids or self.update_fields:
                raise ValueError("none action cannot carry work")
            return
        if self.action in {IndexAction.UPSERT, IndexAction.MERGE}:
            if not isinstance(self.message, (str, list)):
                raise ValueError(f"{self.action.value} requires an embedding message")
            return
        if self.message is not None:
            raise ValueError(f"{self.action.value} does not accept an embedding message")
        if not self.record_ids:
            raise ValueError(f"{self.action.value} requires record_ids")
        if self.action is IndexAction.DELETE:
            if self.update_fields:
                raise ValueError("delete does not accept update_fields")
            return
        if len(self.record_ids) != 1:
            raise ValueError("update_fields requires exactly one record id")
        unknown = set(self.update_fields) - _UPDATE_FIELD_ALLOWLIST
        if unknown:
            raise ValueError(f"update_fields contains forbidden fields: {sorted(unknown)}")
        if not self.update_fields:
            raise ValueError("update_fields must not be empty")

    @classmethod
    def for_update_fields(
        cls,
        *,
        record_id: str,
        fields: Dict[str, Any],
        context_data: Dict[str, Any],
        telemetry_id: str = "",
    ) -> "EmbeddingMsg":
        return cls(
            message=None,
            context_data=context_data,
            telemetry_id=telemetry_id,
            action=IndexAction.UPDATE_FIELDS,
            record_ids=[record_id],
            update_fields=fields,
        )

    @classmethod
    def for_delete(
        cls,
        *,
        record_ids: List[str],
        context_data: Dict[str, Any],
        telemetry_id: str = "",
    ) -> "EmbeddingMsg":
        return cls(
            message=None,
            context_data=context_data,
            telemetry_id=telemetry_id,
            action=IndexAction.DELETE,
            record_ids=record_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert embedding message to dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert embedding message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingMsg":
        """Create an embedding message object from dictionary."""
        obj = EmbeddingMsg(
            message=data["message"],
            context_data=data["context_data"],
            telemetry_id=data.get("telemetry_id", ""),
            action=data.get("action", IndexAction.UPSERT.value),
            record_ids=data.get("record_ids"),
            update_fields=data.get("update_fields"),
            queue_enqueued_at=data.get("queue_enqueued_at", 0.0),
        )
        obj.id = data.get("id", obj.id)
        return obj

    @classmethod
    def from_json(cls, json_str: str) -> "EmbeddingMsg":
        """Safely create object from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

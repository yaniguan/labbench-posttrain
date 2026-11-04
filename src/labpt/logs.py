"""MCAP-style envelopes in plain JSONL; not the binary MCAP format."""
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from .actions import StrictModel


class LogMessage(StrictModel):
    topic: str = Field(min_length=1)
    timestamp: int = Field(ge=0)  # Unix nanoseconds
    message: dict[str, Any]


class LogWriter:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def write(self, topic, message, timestamp=None):
        row = LogMessage(topic=topic, timestamp=time.time_ns() if timestamp is None else timestamp, message=message)
        self.file.write(row.model_dump_json() + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_log(path, topic=None):
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = LogMessage.model_validate_json(line)
                if topic is None or row.topic == topic:
                    yield row.model_dump()

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

@dataclass
class MemoryItem:
    title: str
    description: str
    content: str
    embedding: list[float]
    source: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return (
            f"## {self.title}\n"
            f"_When to use_: {self.description}\n"
            f"{self.content}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

class Bank:

    def __init__(self, items: list[MemoryItem] | None = None):
        self.items = list(items or [])

    def add(self, items):
        self.items.extend(items)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Bank(n_items={len(self.items)})"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for item in self.items:
                f.write(json.dumps(item.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Bank":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Bank file not found: {path}")

        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue

                data = json.loads(line)
                items.append(MemoryItem(**data))

        return cls(items)

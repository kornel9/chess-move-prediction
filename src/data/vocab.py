"""UCI move vocabulary with reserved special tokens."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable


@dataclass
class Vocab:
    """Maps UCI move strings to integer ids.

    IDs are assigned in this order: PAD=0, START=1, END=2, UNK=3, then UCI
    moves seen in the input games in sorted order. The on-disk format is a
    single JSON file with the ordered token list under ``id_to_token``.
    """

    token_to_id: dict[str, int]
    id_to_token: list[str]

    PAD: ClassVar[str] = "<PAD>"
    START: ClassVar[str] = "<START>"
    END: ClassVar[str] = "<END>"
    UNK: ClassVar[str] = "<UNK>"
    SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = (PAD, START, END, UNK)

    @classmethod
    def build_from_games(cls, games: Iterable[Iterable[str]]) -> "Vocab":
        """Build a vocab from an iterable of games, each a sequence of UCI moves."""
        seen: set[str] = set()
        for game in games:
            seen.update(game)
        ordered = list(cls.SPECIAL_TOKENS) + sorted(seen)
        token_to_id = {tok: i for i, tok in enumerate(ordered)}
        return cls(token_to_id=token_to_id, id_to_token=ordered)

    def encode(self, move: str) -> int:
        """Return the id for ``move``, or the ``<UNK>`` id if unseen."""
        return self.token_to_id.get(move, self.token_to_id[self.UNK])

    def decode(self, idx: int) -> str:
        """Return the token at position ``idx``."""
        return self.id_to_token[idx]

    def save(self, path: Path | str) -> None:
        """Persist the vocab as JSON at ``path``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"id_to_token": self.id_to_token}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "Vocab":
        """Load a vocab previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_token = list(data["id_to_token"])
        token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def start_id(self) -> int:
        return self.token_to_id[self.START]

    @property
    def end_id(self) -> int:
        return self.token_to_id[self.END]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

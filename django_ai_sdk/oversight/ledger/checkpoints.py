"""Checkpoints. A Merkle root over the chain, written as an optionally signed note.

Publish each checkpoint somewhere the database writers cannot change. Notifiers get
every new one as an "oversight.checkpoint" event. After that, rewriting earlier
history makes the chain disagree with the published root.

The tree follows RFC 6962 over the entry hashes. The note is a C2SP signed note
(https://c2sp.org/tlog-checkpoint), so standard transparency-log tools can read it.
There are no inclusion or consistency proofs. `ledger_verify` recomputes each root
from the chain instead.
"""

from __future__ import annotations

import base64
import hashlib
import uuid
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ImproperlyConfigured
from django.db import transaction
from django.utils import timezone

from django_ai_sdk.errors import ErrorCode, UserError
from django_ai_sdk.vault.crypto import decode_key_b64

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from django_ai_sdk.oversight.ledger.models import Checkpoint

# Namespace for checkpoint event ids, so the same checkpoint always gets the same id.
CHECKPOINT_EVENTS = uuid.UUID("6c3a4f0e-1b7d-4e52-8f39-0d2b5a9e7c14")
ED25519_ALG = b"\x01"


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """The largest power of two smaller than n."""
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """RFC 6962 Merkle Tree Hash of `leaves` (raw leaf data, hashed here)."""
    if not leaves:
        return hashlib.sha256(b"").digest()
    return _root([leaf_hash(leaf) for leaf in leaves])


def _root(hashes: Sequence[bytes]) -> bytes:
    if len(hashes) == 1:
        return hashes[0]
    k = _split(len(hashes))
    return node_hash(_root(hashes[:k]), _root(hashes[k:]))


# Signed notes (https://c2sp.org/signed-note).


def private_key(raw_b64: str) -> Any:
    """Load the Ed25519 key in `raw_b64`. Raise ImproperlyConfigured if unusable."""
    raw = decode_key_b64(raw_b64, "AI_SDK_LEDGER['SIGNING_KEY']")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise ImproperlyConfigured(
            "AI_SDK_LEDGER['SIGNING_KEY'] is set, but the cryptography package is not "
            "installed: no checkpoint can be signed."
        ) from exc
    return Ed25519PrivateKey.from_private_bytes(raw)


def _public_bytes(key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


def key_hash(name: str, public_key: bytes) -> bytes:
    """The 4-byte key hash a signature line starts with."""
    return hashlib.sha256(name.encode() + b"\n" + ED25519_ALG + public_key).digest()[:4]


def checkpoint_body(origin: str, size: int, root: bytes) -> str:
    return f"{origin}\n{size}\n{base64.b64encode(root).decode()}\n"


def sign_note(body: str, *, name: str, private_key_b64: str) -> str:
    """Append one Ed25519 signature line to `body`."""
    key = private_key(private_key_b64)
    signature = key.sign(body.encode())
    blob = base64.b64encode(key_hash(name, _public_bytes(key)) + signature).decode()
    return f"{body}\n— {name} {blob}\n"


def verifier_key(name: str, private_key_b64: str) -> str:
    """The public verifier key to publish, as `name+keyhash+base64(alg||pub)`."""
    public = _public_bytes(private_key(private_key_b64))
    return (
        f"{name}+{key_hash(name, public).hex()}+{base64.b64encode(ED25519_ALG + public).decode()}"
    )


def verify_note(note: str, verifier: str) -> bool:
    """Check `note` has a valid signature by the `verifier` key string.

    Raises ValueError when `verifier` is malformed.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    name, _hash, key_b64 = verifier.split("+", 2)
    key_bytes = base64.b64decode(key_b64)
    if key_bytes[:1] != ED25519_ALG:
        return False
    public = key_bytes[1:]
    body, _, signatures = note.partition("\n\n")
    body += "\n"
    for line in signatures.splitlines():
        if not line.startswith(f"— {name} "):
            continue
        try:
            blob = base64.b64decode(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        if blob[:4] != key_hash(name, public):
            continue
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(blob[4:], body.encode())
        except InvalidSignature:
            return False
        return True
    return False


# Checkpoints of this deployment's chain.


def prefix_roots(hashes: Iterable[bytes], sizes: Iterable[int]) -> dict[int, bytes]:
    """The RFC 6962 root of the first n `hashes` for each n in `sizes`, in one pass.

    Keeps the perfect subtrees built so far; a prefix's root folds them right to left.
    A size past the end of `hashes` is left out.
    """
    wanted = set(sizes)
    roots: dict[int, bytes] = {}
    if 0 in wanted:
        roots[0] = merkle_root([])
    stack: list[tuple[int, bytes]] = []  # (height, hash), heights strictly falling
    for count, data in enumerate(hashes, 1):
        height, node = 0, leaf_hash(data)
        while stack and stack[-1][0] == height:
            _, left = stack.pop()
            height, node = height + 1, node_hash(left, node)
        stack.append((height, node))
        if count in wanted:
            root = stack[-1][1]
            for _, left in reversed(stack[:-1]):
                root = node_hash(left, root)
            roots[count] = root
    return roots


def chain_roots(sizes: Iterable[int]) -> dict[int, bytes]:
    """`prefix_roots` of this deployment's chain, reading it once."""
    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    sizes = list(sizes)
    if not sizes:
        return {}
    hashes = (
        LedgerEntry.objects.order_by("seq")
        .values_list("entry_hash", flat=True)[: max(sizes)]
        .iterator(chunk_size=10_000)
    )
    return prefix_roots((bytes.fromhex(h) for h in hashes), sizes)


def chain_root(size: int) -> bytes | None:
    """The Merkle root of the first `size` entries, or None if there are fewer."""
    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    hashes = LedgerEntry.objects.order_by("seq").values_list("entry_hash", flat=True)[:size]
    found = [bytes.fromhex(h) for h in hashes]
    return merkle_root(found) if len(found) == size else None


def checkpoint_matches_chain(checkpoint: Checkpoint) -> bool:
    """Check the chain still gives the checkpoint's root."""
    root = chain_root(checkpoint.tree_size)
    return root is not None and root.hex() == checkpoint.root_hash


def note_size_root(note: str) -> tuple[int, str]:
    """The tree size and root (hex) a checkpoint note states. Raises ValueError."""
    _origin, size, root = note.split("\n")[:3]
    return int(size), base64.b64decode(root, validate=True).hex()


def signed_by(checkpoint: Checkpoint, verifier: str) -> bool:
    """Check `verifier` signed the note and the note matches the row's size and root.

    Only the note is signed, so the tree_size and root_hash columns must agree with it.
    """
    try:
        stated = note_size_root(checkpoint.note) == (checkpoint.tree_size, checkpoint.root_hash)
    except ValueError:
        return False
    return stated and verify_note(checkpoint.note, verifier)


class ChainTruncated(UserError):
    """The chain ends before its head, or no longer matches its last checkpoint. A 409."""

    code = ErrorCode.CONFLICT


def write_checkpoint() -> Checkpoint | None:
    """Checkpoint the current chain, signed when SIGNING_KEY is set. Synchronous.

    Returns None when nothing was chained since the last one. Raises ChainTruncated
    when the chain ends before its head. There is one checkpoint per tree size, so a
    concurrent writer gets the existing one instead of an error. New checkpoints go to
    every notifier as an "oversight.checkpoint" event after commit.
    """
    from django_ai_sdk.oversight.ledger.conf import origin, setting
    from django_ai_sdk.oversight.ledger.models import Checkpoint, LedgerHead
    from django_ai_sdk.oversight.persist import enqueue_notifications

    head = LedgerHead.objects.filter(id=1).first()
    size = head.seq if head else 0
    last = Checkpoint.objects.order_by("-tree_size").first()
    if size == 0 or (last is not None and last.tree_size >= size):
        return None
    roots = chain_roots([size, *([last.tree_size] if last else [])])
    root = roots.get(size)
    if root is None:
        # Fewer entries than the head counts: the newest ones were deleted.
        raise ChainTruncated("The ledger chain ends before its head; run ledger_verify.")
    if last is not None and roots[last.tree_size].hex() != last.root_hash:
        # Signing now would vouch for a chain someone rewrote since the last checkpoint.
        raise ChainTruncated(
            f"The chain no longer matches the checkpoint at {last.tree_size}; run ledger_verify."
        )
    key = setting("SIGNING_KEY")
    body = checkpoint_body(origin(), size, root)
    note = sign_note(body, name=origin(), private_key_b64=key) if key else body
    with transaction.atomic():
        checkpoint, created = Checkpoint.objects.get_or_create(
            tree_size=size, defaults={"root_hash": root.hex(), "note": note, "signed": bool(key)}
        )
        if created:
            event = checkpoint_event(checkpoint)
            transaction.on_commit(lambda: enqueue_notifications([(event, [])]))
    return checkpoint


def checkpoint_event(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid5(CHECKPOINT_EVENTS, f"{checkpoint.tree_size}:{checkpoint.root_hash}")),
        "type": "oversight.checkpoint",
        "tree_size": checkpoint.tree_size,
        "root_hash": checkpoint.root_hash,
        "note": checkpoint.note,
        "signed": checkpoint.signed,
        "at": (checkpoint.created_at or timezone.now()).isoformat(),
    }


__all__ = [
    "ChainTruncated",
    "checkpoint_body",
    "checkpoint_event",
    "chain_root",
    "checkpoint_matches_chain",
    "merkle_root",
    "private_key",
    "sign_note",
    "signed_by",
    "verifier_key",
    "verify_note",
    "write_checkpoint",
]

"""The hash chain. Append-only, verifiable, and blind to content.

    entry_hash = sha256(seq, kind, ref, payload_hash, prev_hash)

Each chained row is hashed as a fixed set of its fields (`KINDS`) in canonical JSON.
Editing a row breaks its payload hash; editing or dropping an entry breaks every later
`prev_hash`. Content is only hashed by its keyed digest, so shredding it changes nothing
here. Dropping the newest entries is caught by `LedgerHead`, unless the head was rolled
back too: only a checkpoint stored out of the writers' reach protects against that.

A turn is chained when its gate ends and again at close. Only its newest entry has
to match the row. Fields that change after close on purpose are not hashed: labels,
severity, reflect state, `error`, `status` set by the sweep, and `actor_id` cleared
when the user is deleted.

Every deleted row leaves a trace in the chain. Retention chains a "prune" entry for
each run of entries whose rows it deletes. Forgetting a scope chains a "forget"
entry for the shredded key and an "erase" entry for each run of content entries
deleted with it. verify only accepts a missing row under one of these. Anyone who
can write the chain can append a prune or erase entry, so check them against
retention and the nearby forget entries.

verify reports content it cannot open. Shredded content needs its forget entry.
Content under a master key not configured here is "unreadable", which is not a
chain failure. Content that does not open under a key that is here was altered.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from django.db import transaction

from django_ai_sdk.oversight.custody import THREAD_SCOPE, TURN_SCOPE
from django_ai_sdk.oversight.ledger.models import GENESIS
from django_ai_sdk.oversight.ledger.schemas import MAX_LISTED, VerifyOut
from django_ai_sdk.oversight.models import APP_LABEL as OVERSIGHT
from django_ai_sdk.utils import as_uuid
from django_ai_sdk.vault.crypto import canonical, stored_bytes
from django_ai_sdk.vault.models import APP_LABEL as VAULT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from django.db import models

    from django_ai_sdk.oversight.ledger.models import LedgerEntry


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _entry_digest(seq: int, kind: str, ref: str, payload_hash: str, prev_hash: str) -> str:
    return _sha256(f"{seq}\n{kind}\n{ref}\n{payload_hash}\n{prev_hash}")


@dataclass(frozen=True)
class Kind:
    """A chained row kind: its model, and the fields its payload hashes.

    Ids hash as text and datetimes as ISO 8601 (`vault.crypto.canonical`).
    """

    app_label: str
    model: str
    fields: tuple[str, ...]
    # Chained again on update. Only the newest entry must match the row.
    rechained: bool = False
    # Payload keys that are not plain fields.
    extra: Callable[[Any], dict[str, Any]] | None = None

    def payload(self, row: Any) -> dict[str, Any]:
        found = {name: getattr(row, name) for name in self.fields}
        return found | self.extra(row) if self.extra else found


def _digest(row: Any) -> dict[str, Any]:
    # Skip the ciphertext. It is re-encrypted when judges and reviewers add to it.
    stored = row.stored if isinstance(row.stored, dict) else {}
    return {"digest": stored.get("digest")}


# Entry kinds of chained rows.
TURN = "turn"
ANSWER = "answer"
MATCH = "match"
CONTENT = "content"
ACCESS = "access"
OVERRIDE = "override"
FORGET = "forget"

_CHANGE = ("id", "version", "reason", "changed_by_id", "changed_at")

# Never change the hashed fields of an existing kind. Add a new kind instead.
# verify() recomputes old payloads from these, so any change breaks the chain for
# every row written before it. Fields that change on purpose are left out (see the
# module docstring), and so is an access record's `reader_label`, which is cut to
# the id when the reader is deleted.
# The switches app sits on top of the ledger and may be missing, so its label is
# not imported.
KINDS: dict[str, Kind] = {
    TURN: Kind(
        OVERSIGHT,
        "Turn",
        (
            "id",
            "created_at",
            "channel",
            "agent_id",
            "agent_fingerprint",
            "policy",
            "thread_id",
            "user_message_id",
            "message_id",
            "parent_turn_id",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "cost",
            "outcome",
            "report_outcome",
            "expires_at",
        ),
        rechained=True,
    ),  # fmt: skip
    ANSWER: Kind(
        OVERSIGHT,
        "LensAnswer",
        (
            "id",
            "turn_id",
            "lens",
            "stage",
            "answerer",
            "model",
            "value",
            "confidence",
            "spans",
            "trail",
            "error",
            "reviewer",
        ),
    ),  # fmt: skip
    MATCH: Kind(
        OVERSIGHT,
        "RuleMatch",
        ("id", "turn_id", "rule", "stage", "mode", "actions", "acted", "condition"),
    ),
    CONTENT: Kind(
        OVERSIGHT, "TurnContent", ("id", "scope", "capture", "expires_at"), extra=_digest
    ),
    ACCESS: Kind(
        OVERSIGHT,
        "AccessRecord",
        ("id", "reader_id", "operation", "turn_ids", "content", "query", "created_at"),
    ),
    OVERRIDE: Kind(
        "django_ai_sdk_oversight_switches",
        "Override",
        (*_CHANGE, "kind", "policy", "target", "value"),
    ),
    FORGET: Kind(
        VAULT,
        "DataKey",
        ("id", "scope", "shredded_at"),
        extra=lambda key: {"kept": key.wrapped is not None},
    ),
}

# Span entries. Their ref is "first-last", a run of entries whose rows were deleted.
# They have no row of their own, so they are not in KINDS.
PRUNE = "prune"  # deleted by retention
ERASE = "erase"  # content deleted when its scope was forgotten
# The kinds each span entry may excuse. Retention deletes a turn with its answers,
# matches and content, and access records on their own schedule.
COVERS: dict[str, frozenset[str]] = {
    PRUNE: frozenset({TURN, ANSWER, MATCH, CONTENT, ACCESS}),
    ERASE: frozenset({CONTENT}),
}
# Oversight's vault scopes. A shredded key in one of them needs a forget entry.
_SCOPES = (THREAD_SCOPE, TURN_SCOPE)


def payload_hash(kind: str, row: models.Model) -> str:
    return _sha256(stored_bytes(KINDS[kind].payload(row)))


def _span(ref: str) -> tuple[int, int]:
    first, _, last = ref.partition("-")
    return int(first), int(last)


def append_many(items: list[tuple[str, models.Model]]) -> list[LedgerEntry]:
    """Chain `items` in order. Call synchronously inside transaction.atomic().

    Rows are read back first, so we hash exactly what verify will read. The lock on
    the head row keeps the sequence gapless. Call this last in the transaction so
    other writers wait on the lock as briefly as possible.
    """
    _require_atomic("append_many")
    fresh = _read_back(items)
    return _chain([(kind, str(row.pk), payload_hash(kind, row)) for kind, row in fresh])


def _read_back(items: list[tuple[str, models.Model]]) -> list[tuple[str, models.Model]]:
    by_kind: dict[str, list[str]] = {}
    for kind, row in items:
        by_kind.setdefault(kind, []).append(str(row.pk))
    found: dict[tuple[str, str], models.Model] = {}
    for kind, refs in by_kind.items():
        found.update(_load(kind, refs))
    missing = [(kind, row.pk) for kind, row in items if (kind, str(row.pk)) not in found]
    if missing:
        raise RuntimeError(f"Cannot chain rows that are not written: {missing}")
    return [(kind, found[(kind, str(row.pk))]) for kind, row in items]


def _chain(items: list[tuple[str, str, str]]) -> list[LedgerEntry]:
    """Append (kind, ref, payload hash) items while holding the head row's lock."""
    from django_ai_sdk.oversight.ledger.models import LedgerEntry, LedgerHead

    if not items:
        return []
    head, _ = LedgerHead.objects.select_for_update().get_or_create(id=1)
    seq, previous = head.seq, head.entry_hash
    entries: list[LedgerEntry] = []
    for kind, ref, digest in items:
        seq += 1
        entry_hash = _entry_digest(seq, kind, ref, digest, previous)
        entries.append(
            LedgerEntry(
                seq=seq,
                kind=kind,
                ref=ref,
                payload_hash=digest,
                prev_hash=previous,
                entry_hash=entry_hash,
            )
        )
        previous = entry_hash
    LedgerEntry.objects.bulk_create(entries)
    head.seq, head.entry_hash = seq, previous
    head.save(update_fields=["seq", "entry_hash"])
    return entries


def _require_atomic(what: str) -> None:
    if not transaction.get_connection().in_atomic_block:
        raise RuntimeError(f"ledger.{what} must run inside transaction.atomic()")


def prune(
    turn_ids: Iterable[Any] = (),
    *,
    content_ids: Iterable[Any] = (),
    access_ids: Iterable[Any] = (),
) -> list[LedgerEntry]:
    """Chain "prune" entries for rows retention is about to delete.

    Covers `turn_ids` with their answers, matches and content, `content_ids` that
    expire before their turn, and expired `access_ids`. Writes one entry per run of
    consecutive seqs. Without it, verify reports the missing rows. Call inside the
    deleting transaction.
    """
    from django_ai_sdk.oversight.models import LensAnswer, RuleMatch, Turn

    _require_atomic("prune")
    ids = list(turn_ids)
    contents = [str(pk) for pk in content_ids]
    refs: dict[str, list[Any]] = {CONTENT: contents, ACCESS: [str(pk) for pk in access_ids]}
    if ids:
        refs[TURN] = ids
        refs[ANSWER] = list(LensAnswer.objects.filter(turn__in=ids).values_list("pk", flat=True))
        refs[MATCH] = list(RuleMatch.objects.filter(turn__in=ids).values_list("pk", flat=True))
        contents.extend(
            str(pk)
            for pk in Turn.objects.filter(pk__in=ids, content__isnull=False).values_list(
                "content_id", flat=True
            )
        )
    return _cover(PRUNE, refs)


def erase(content_ids: Iterable[Any]) -> list[LedgerEntry]:
    """Chain "erase" entries for content deleted with a forgotten scope.

    Call inside the deleting transaction.
    """
    _require_atomic("erase")
    return _cover(ERASE, {CONTENT: [str(pk) for pk in content_ids]})


def _cover(kind: str, refs: dict[str, list[Any]]) -> list[LedgerEntry]:
    """Chain one `kind` entry per run of consecutive entries for `refs`."""
    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    seqs: list[int] = []
    for entry_kind, pks in refs.items():
        if pks:
            found = LedgerEntry.objects.filter(kind=entry_kind, ref__in=[str(pk) for pk in pks])
            seqs.extend(found.values_list("seq", flat=True))
    spans = [f"{first}-{last}" for first, last in _merge((seq, seq) for seq in seqs)]
    return _chain([(kind, ref, _span_hash(ref)) for ref in spans])


def verify(*, batch: int = 2_000) -> VerifyOut:
    """Walk the whole chain, recomputing every link and payload.

    A row may only be missing if a later "prune" entry covers it (or "erase", for
    content). A turn's row must match its newest entry. Content must decrypt and
    give the same digest. Shredded content needs its key's forget entry. Content
    under a master key not configured here counts as unreadable.

    Every shredded oversight key must be chained, and the chain must reach the head
    as it was when the walk began.

    Synchronous and read-only. Run it from a command or via sync_to_async.
    """
    from django_ai_sdk.oversight.ledger.models import LedgerEntry, LedgerHead

    result = VerifyOut(head=GENESIS)
    previous = GENESIS
    expected_seq = 1
    head = LedgerHead.objects.filter(id=1).first()
    spans = {kind: _covered(kind) for kind in COVERS}
    keys = _Keys()
    query = LedgerEntry.objects.order_by("seq")
    last = 0
    while True:
        entries = list(query.filter(seq__gt=last)[:batch])
        if not entries:
            break
        rows, newest = _rows_for(entries)
        keys.load([row for (kind, _), row in rows.items() if kind == CONTENT])
        for entry in entries:
            result.checked += 1
            if entry.seq != expected_seq:
                result.fail(entry.seq, "gap", expected=expected_seq)
            expected_seq = entry.seq + 1
            if entry.prev_hash != previous:
                result.fail(entry.seq, "broken link", expected=previous, found=entry.prev_hash)
            digest = _entry_digest(
                entry.seq, entry.kind, entry.ref, entry.payload_hash, entry.prev_hash
            )
            if digest != entry.entry_hash:
                result.fail(entry.seq, "entry altered")
            _check_row(result, entry, rows, newest, spans, keys)
            previous = entry.entry_hash
        last = entries[-1].seq
    result.head = previous
    _unchained_shreds(result)
    _reaches_head(result, head, walked=last)
    return result


def _load(kind: str, refs: list[str]) -> dict[tuple[str, str], Any]:
    """The rows of `kind` with pks `refs`, keyed by (kind, ref). None of an app not installed."""
    from django.apps import apps

    spec = KINDS[kind]
    try:
        model = apps.get_model(spec.app_label, spec.model)
    except LookupError:
        return {}  # So its rows count as missing.
    return {(kind, str(row.pk)): row for row in model._default_manager.filter(pk__in=refs)}


def _rows_for(entries: list[LedgerEntry]) -> tuple[dict[tuple[str, str], Any], dict[str, int]]:
    """Load the rows behind `entries`, keyed by (kind, ref), and each turn's newest seq."""
    from django.db.models import Max

    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    by_kind: dict[str, list[str]] = {}
    for entry in entries:
        by_kind.setdefault(entry.kind, []).append(entry.ref)
    rows: dict[tuple[str, str], Any] = {}
    newest: dict[str, int] = {}
    for kind, refs in by_kind.items():
        spec = KINDS.get(kind)
        if spec is None:
            continue
        rows.update(_load(kind, refs))
        if spec.rechained:
            newest.update(
                LedgerEntry.objects.filter(kind=kind, ref__in=refs)
                .values("ref")
                .annotate(last=Max("seq"))
                .values_list("ref", "last")
            )
    return rows, newest


class _Keys:
    """Vault key state for verify: key rows, configured master keys, chained shreds."""

    def __init__(self) -> None:
        from django.core.exceptions import ImproperlyConfigured

        from django_ai_sdk.vault.keys import provider

        try:
            self.configured = set(provider().key_ids())
        except ImproperlyConfigured:
            self.configured = set()
        self.rows: dict[str, Any] = {}
        self.forgotten: set[str] = set()

    def load(self, contents: list[Any]) -> None:
        from django_ai_sdk.oversight.ledger.models import LedgerEntry
        from django_ai_sdk.vault import key_rows

        envelopes = [r.stored.get("vault") for r in contents if isinstance(r.stored, dict)]
        found = key_rows([e for e in envelopes if e is not None])
        self.rows.update(found)
        shredded = [pk for pk, row in found.items() if row.shredded]
        if shredded:
            self.forgotten.update(
                LedgerEntry.objects.filter(kind=FORGET, ref__in=shredded).values_list(
                    "ref", flat=True
                )
            )


def _check_row(
    result: VerifyOut,
    entry: LedgerEntry,
    rows: dict[tuple[str, str], Any],
    newest: dict[str, int],
    spans: dict[str, tuple[list[int], list[int]]],
    keys: _Keys,
) -> None:
    spec = KINDS.get(entry.kind)
    row = rows.get((entry.kind, entry.ref))
    if entry.kind in COVERS:
        try:
            altered = _span_hash(entry.ref) != entry.payload_hash
        except ValueError:
            altered = True
        if altered:
            result.fail(entry.seq, "record altered", kind=entry.kind, ref=entry.ref)
        return
    if spec is None:
        result.fail(entry.seq, "unknown kind", kind=entry.kind)
        return
    if row is None:
        if entry.kind in COVERS[PRUNE] and _within(spans[PRUNE], entry.seq):
            result.pruned += 1
        elif entry.kind in COVERS[ERASE] and _within(spans[ERASE], entry.seq):
            result.forgotten += 1
        else:
            result.fail(entry.seq, "record missing", kind=entry.kind, ref=entry.ref)
        return
    if spec.rechained and newest.get(entry.ref, entry.seq) != entry.seq:
        return  # The turn was chained again later.
    if payload_hash(entry.kind, row) != entry.payload_hash:
        result.fail(entry.seq, "record altered", kind=entry.kind, ref=entry.ref)
    elif entry.kind == CONTENT:
        _check_content(result, entry, row, keys)


def _within(spans: tuple[list[int], list[int]], seq: int) -> bool:
    starts, ends = spans
    i = bisect_right(starts, seq) - 1
    return i >= 0 and seq <= ends[i]


def _check_content(result: VerifyOut, entry: LedgerEntry, row: Any, keys: _Keys) -> None:
    """Check that a content row decrypts and still gives its chained digest.

    Shredded content counts as forgotten if its key's forget entry is chained.
    Content under a master key not configured here counts as unreadable.
    """
    from django_ai_sdk.oversight.ledger.custody import content_digest
    from django_ai_sdk.vault import VaultError, decrypt_json, is_envelope

    def fail(problem: str) -> None:
        result.fail(entry.seq, problem, kind=entry.kind, ref=entry.ref)

    stored = row.stored
    if not isinstance(stored, dict) or "vault" not in stored:
        return fail("content not encrypted")
    envelope = stored["vault"]
    if not is_envelope(envelope) or envelope["scope"] != row.scope:
        return fail("content altered")
    key = keys.rows.get(envelope["key"])
    if key is None:
        return fail("key missing")
    if key.scope != row.scope:
        return fail("content altered")
    if key.shredded:
        if str(key.pk) in keys.forgotten:
            result.forgotten += 1
            return None
        return fail("shred not chained")
    if key.key_id not in keys.configured:
        result.cannot_read(entry.seq, entry.ref, key.key_id)
        return None
    try:
        content = decrypt_json(envelope, rows=keys.rows)
        digest = content_digest(row.scope, content, row=key)
    except VaultError:
        return fail("content altered")
    if digest != stored.get("digest"):
        return fail("content altered")
    return None


def _unchained_shreds(result: VerifyOut) -> None:
    """Fail on shredded oversight keys without a forget entry.

    These were shredded by hand instead of through custody.
    """
    from django.db.models import Q

    from django_ai_sdk.oversight.ledger.models import LedgerEntry
    from django_ai_sdk.vault.models import DataKey

    scoped = Q()
    for prefix in _SCOPES:
        scoped |= Q(scope__startswith=prefix)
    chained = LedgerEntry.objects.filter(kind=FORGET).values_list("ref", flat=True)
    shredded = (
        DataKey.objects.filter(scoped, shredded_at__isnull=False)
        .exclude(pk__in=[pk for ref in chained if (pk := as_uuid(ref))])
        .order_by("shredded_at")
    )
    for key in shredded[:MAX_LISTED]:
        result.fail(0, "shred not chained", kind=FORGET, ref=str(key.pk), scope=key.scope)


def _reaches_head(result: VerifyOut, head: Any, *, walked: int) -> None:
    """Check the chain still reaches the head read before the walk.

    The head is read first so entries appended during the walk do not count.
    Deleting the newest entries leaves the head pointing past the end. Rolling the
    head back too is not caught here, only by an external checkpoint.
    """
    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    if head is None:
        if walked > 0:
            result.fail(walked, "head missing")
        return
    if head.seq == 0:
        return
    tip = LedgerEntry.objects.filter(seq=head.seq).values_list("entry_hash", flat=True).first()
    if tip is None:
        result.fail(head.seq, "chain ends early", head=head.entry_hash, last=walked)
    elif tip != head.entry_hash:
        result.fail(head.seq, "head mismatch", head=head.entry_hash, found=tip)


def _covered(kind: str) -> tuple[list[int], list[int]]:
    """Merge the seq runs covered by `kind` span entries.

    Returns (starts, ends). A span entry only covers entries before itself, so it
    cannot excuse a later deletion.
    """
    from django_ai_sdk.oversight.ledger.models import LedgerEntry

    spans: list[tuple[int, int]] = []
    for seq, ref in LedgerEntry.objects.filter(kind=kind).values_list("seq", "ref"):
        try:
            first, last = _span(ref)
        except ValueError:
            continue  # The walk reports a bad ref as altered.
        if first <= last < seq:
            spans.append((first, last))
    runs = _merge(spans)
    return [first for first, _ in runs], [last for _, last in runs]


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge (first, last) spans that overlap or touch into sorted runs."""
    runs: list[tuple[int, int]] = []
    for first, last in sorted(spans):
        if runs and first <= runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], max(runs[-1][1], last))
        else:
            runs.append((first, last))
    return runs


def _span_hash(ref: str) -> str:
    """Payload hash of a span entry. Raises ValueError when `ref` is not "first-last"."""
    first, last = _span(ref)
    return _sha256(canonical({"from": first, "to": last}))


__all__ = [
    "ACCESS",
    "ANSWER",
    "CONTENT",
    "ERASE",
    "FORGET",
    "KINDS",
    "MATCH",
    "OVERRIDE",
    "PRUNE",
    "TURN",
    "append_many",
    "erase",
    "payload_hash",
    "prune",
    "verify",
]

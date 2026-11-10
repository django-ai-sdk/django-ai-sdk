---
title: Audit-grade oversight
type: docs
weight: 12
---

**Encrypt what the oversight record keeps, forget a thread by destroying its key, and hash-chain the record so an edit or a delete behind its back shows.**

Two opt-in apps on top of [oversight](/oversight/), installed together:

- **The vault** (`django_ai_sdk.vault`) keeps data under a key per scope (`thread:42`) and forgets a scope by destroying its key. It knows nothing of oversight, and any app of yours may use it.
- **The ledger** (`django_ai_sdk.oversight.ledger`) needs the vault. It encrypts every turn's content in the vault, chains every row of the record into a hash chain, writes checkpoints over the chain, and names who asked a turn by a pseudonym.

Without them, core keeps content in the clear (masked as `CAPTURE` says), deletes it to forget, chains nothing and names who asked by their user id. Core makes no encryption or tamper-evidence claims.

## Install

```bash
pip install 'django-ai-sdk[ledger]'
python manage.py vault_keys     # prints a new master key
python manage.py ledger_keys    # prints a signing key, its verifier key and a pseudonym secret
```

```python
INSTALLED_APPS = [
    ...,
    "django_ai_sdk",
    "django_ai_sdk.oversight",
    "django_ai_sdk.vault",
    "django_ai_sdk.oversight.ledger",
]

AI_SDK_VAULT = {"MASTER_KEYS": [env("VAULT_MASTER_KEY")]}
AI_SDK_LEDGER = {
    "SIGNING_KEY": env("LEDGER_SIGNING_KEY"),
    "PSEUDONYM_SECRET": env("LEDGER_PSEUDONYM_SECRET"),
}
```

Run `manage.py migrate`. On start the ledger installs `LedgerCustody` (`django_ai_sdk.oversight.custody.install`): adding the app is the only switch. It refuses to start without `django_ai_sdk.oversight` and `django_ai_sdk.vault` in `INSTALLED_APPS`. Keep the master key, the signing key and the pseudonym secret out of the database and out of its backups: whoever holds the database and the master key reads everything the vault keeps.

Schedule two commands next to `oversight_prune`:

```bash
python manage.py ledger_checkpoint   # hourly or daily: a checkpoint over what was chained since the last
python manage.py ledger_verify       # daily: walks the chain, exits non-zero on a problem
```

With the app installed, `get_oversight_router()` also serves its routes, next to the review routes. Leave them out with its `exclude=`.

## What changes with the ledger installed

| | Core | With the ledger |
| --- | --- | --- |
| A turn's content (question, answer, history, sources, tool calls, judges' reasons, reviewers' notes) | Kept in `TurnContent.stored` as captured | Encrypted in the vault under its scope's key: `{"vault": <envelope>, "digest": <keyed digest>, "aside": <keyed digest>}` |
| Deleting a thread | Deletes its content rows | Shreds the thread's key, chains a `forget` entry, deletes its content rows under an `erase` entry |
| A later turn in a deleted thread | Records content again | Records no content: the scope stays forgotten |
| Deleting a user | Clears `actor_id`, deletes the content of their turns without a thread | The same, for turns under their pseudonym and turns recorded before the ledger under their user id; each such turn's key (`turn:<id>`) is shredded and chained |
| Who asked (`Turn.actor_id`) | The user's primary key | A pseudonym, `u-` and 16 characters; revealed only for a recorded reason |
| Turns, answers, matches, content, access records, switches | Written | Written and chained, in the same transaction |
| Retention (`oversight_prune`) | Deletes in batches | Deletes in batches and chains a `prune` entry for each run it deleted: turns with their rows, content that expired ahead of its turn, and access records past `RETENTION["metadata"]` |

What capture keeps does not change: `CAPTURE` still decides what the content holds, and the rows (the turn, its answers and matches) still hold nothing that quotes the turn. Content kept before the ledger was installed stays as core kept it, in the clear and unchained, until it expires or its thread is deleted.

## The vault

The vault has [its own page](/vault/): keeping data per scope, shredding, master keys and rotation, and its settings. The ledger keeps every turn's content there, under the scope `thread:<id>`, or `turn:<id>` for a turn without a thread.

## The chain

Every row of the record is chained when it is written, in the transaction that writes it: turns (when the gate ends and again at close), lens answers, rule matches, content, access records and, with the [switches app](/oversight-switches/), switches. An entry is `sha256(seq, kind, ref, payload_hash, prev_hash)`: the row's id, a hash of a fixed set of its fields, and the entry before it. The chain holds hashes only, no content, and no foreign key into the record: deleting a row never deletes its entry, which is how a missing row shows. A turn's content is committed to by its keyed digest, which leaves out the judges' reasons and reviewers' notes, so what they add later does not move it. Those two are committed to by a keyed digest of their own (`aside`), chained in an `aside` entry with the content and again each time a judge or reviewer adds to them; only the newest `aside` entry has to match. Shredding the key changes nothing in the chain. A content entry also names the turn that points at it, so content cannot be moved to another turn or replaced by a row that was never chained.

Fields that change after close by design are not hashed: a turn's labels, severity, reflect state, `error`, `status` (the sweep's `interrupted`) and `actor_id` (cleared when the user is deleted); an access record's `reader_label`; a switch's `active`.

One row (`LedgerHead`) holds the tip, and its lock orders writers: one gapless sequence, whatever runs at once.

### Verify

`ledger_verify` (or `POST /oversight/ledger/verify/`) walks the chain and recomputes every link and every row, then looks the other way: every turn, answer, match, content row and access record written since the chain began must have an entry. It checks every checkpoint against the chain too.

| Problem | What it means |
| --- | --- |
| `record altered` | A row no longer hashes to its entry: an outcome, a value or a match was changed. |
| `record missing` | A chained row is gone, and no `prune` (retention) or `erase` (forgetting) entry covers it. |
| `record not chained` | A row written since the chain began has no entry: it was inserted behind the app's back. |
| `turn chained too often` | A turn has more than its two entries (gate and close): one was appended to rewrite it. |
| `checkpoint mismatch` | The chain no longer gives a checkpoint's root: history before it was rewritten. |
| `published checkpoint mismatch`, `published checkpoint badly signed` | A note passed with `--checkpoint` does not match the chain, or is not signed by `--verifier`. |
| `content altered` | Content does not open, or does not give its digest, under a key that is here. |
| `quotes altered` | The judges' reasons or reviewers' notes in the content no longer give the digest its newest `aside` entry commits to. |
| `quotes not chained` | The content holds judges' reasons or reviewers' notes, and no `aside` entry commits to them. |
| `content not encrypted` | A chained content row holds plain content instead of its envelope. |
| `key missing` | A content row's data key row is gone. |
| `shred not chained` | A data key of an oversight scope was shredded without its `forget` entry. |
| `entry altered`, `broken link`, `gap`, `unknown kind` | The chain itself was edited. |
| `chain ends early`, `head mismatch`, `head missing` | The newest entries were dropped, or the head disagrees with its entry or is gone. |

Its `state` is `ok`, `failed` (any problem), or `unreadable`: content kept under a master key that is not configured here went unchecked. That is not a failure of the chain: the command exits 1 on a problem and 2 when the only finding is unreadable content, unless given `--allow-unreadable`. Verify also counts what was `pruned` and `forgotten`.

A `prune` or `erase` entry is itself only a claim: anyone who can write the chain can append one. Check what was pruned against retention, and what was erased against the `forget` entry beside it.

### Checkpoints

The chain proves order; a checkpoint holds its end. `ledger_checkpoint` (`write_checkpoint()` from Python, `POST /oversight/ledger/checkpoints/` over HTTP) writes a `Checkpoint`: an RFC 6962 Merkle root over every entry so far, as a [C2SP signed note](https://c2sp.org/tlog-checkpoint), signed with `SIGNING_KEY` (Ed25519) when one is set. One checkpoint per tree size: two writers at once get the same one. Each new checkpoint goes to every notifier in `NOTIFIERS` as an event, from the task worker:

```json
{"id": "…", "type": "oversight.checkpoint", "tree_size": 1204, "root_hash": "…", "note": "…", "signed": true, "at": "…"}
```

Publish it where the deployment's database writers cannot change it (a webhook into another system, a log you do not own). Someone who can write the database can drop the newest entries and roll the head back with them; walking the chain cannot see that, but the published checkpoint no longer matches.

`ledger_verify` checks every checkpoint against the chain in one pass, so a rewrite shows in every checkpoint after it, even when a newer one was written over the rewrite. `write_checkpoint` refuses (409) when the chain no longer gives the last checkpoint's root, so a schedule does not sign over a rewrite. With `--verifier <key>` (as `ledger_keys` prints it) every signed checkpoint's signature is checked, and the newest checkpoint must be signed. `--checkpoint FILE` checks a note you published elsewhere, so deleting the checkpoint rows does not hide a rewrite; pass it the newest you kept. There are no inclusion or consistency proofs.

## Forgetting

Deleting a thread shreds its key: everything kept under `thread:<id>` stops opening at once, wherever it is stored, and nothing is kept under it again. The turns stay with their answers, matches and reviews, which hold no text of the turn. The `forget` and `erase` entries record that it happened.

Deleting a user shreds the key of each turn they asked without a thread and clears their pseudonym from all their turns, as [Forgetting](/oversight/#forgetting) describes for core.

Shredding reaches no further than core's forgetting ([what deleting does not reach](/oversight/#forgetting)); a backup taken before still holds the wrapped key, readable with the master key until the backup itself is gone.

## Who asked

With the ledger installed, `Turn.actor_id` is a pseudonym: `u-` and 80 bits of an HMAC of the user's primary key under `PSEUDONYM_SECRET`, as 16 base32 characters. The same user always has the same one, so a reviewer can tell turns by one person apart. Nothing stores the way back: revealing computes each user's pseudonym until one matches. Whoever holds the secret can do that, so guard it like the master key. Without `PSEUDONYM_SECRET` it is derived from `SECRET_KEY` (`ai_sdk.oversight.ledger.W002`), and rotating either renames everyone on the record from then on.

To rotate it, set a list with the new secret first: `"PSEUDONYM_SECRET": [new, old]`. New turns use the first; deleting a user, `reveal` and refusing self-review find turns under every one listed. Without the setting, `SECRET_KEY_FALLBACKS` serve the same way. Drop a former secret once the turns named under it have expired. Turns recorded before the ledger was installed name their user by id, and are forgotten as core forgets them.

A pseudonym does not hide a turn in a thread: the turn names the thread, and the thread names its user.

`LedgerService.reveal(turn_id, user=, reason=)` (`POST /oversight/ledger/turns/{id}/reveal/` with `{"reason": "…"}`) tells who asked a turn in the reader's scope. It takes a reason of at least ten characters and its own permission, and writes a chained access record with the reason before it answers. A user who was deleted is not revealed, and the read is still recorded.

## Routes and permissions

| Operation | Route |
| --- | --- |
| `VIEW_OVERSIGHT_LEDGER` | `GET /oversight/ledger/checkpoints/`, `POST /oversight/ledger/verify/` |
| `CHECKPOINT_OVERSIGHT` | `POST /oversight/ledger/checkpoints/` (201, or 204 when nothing was chained since the last) |
| `REVEAL_OVERSIGHT_ACTOR` | `POST /oversight/ledger/turns/{id}/reveal/` |

The [manual's permission table](/manual/oversight/#review-api) names the model permission behind each operation; the router signs readers in as the review routes do. None of the review permissions reaches these. Writing a checkpoint, verifying and revealing write an access record (`checkpoint`, `verify`, `reveal`), chained like any other; listing checkpoints, and a checkpoint request with nothing new to cover, write none. From Python, `LedgerService` does the same with the same checks:

```python
from django_ai_sdk.oversight.ledger import LedgerService

result = await LedgerService.verify(user=request.user)   # VerifyOut
assert result.state == "ok"
```

`CheckpointOut`, `VerifyOut`, `RevealIn` and `RevealOut` are what the routes take and return.

## Ledger settings

`AI_SDK_LEDGER`, one dict:

| Key | Default | What |
| --- | --- | --- |
| `SIGNING_KEY` | `""` | base64 Ed25519 private key that signs checkpoints. Empty: unsigned (`ai_sdk.oversight.ledger.W001`). |
| `ORIGIN` | `""` | The log's name: a checkpoint's origin line, and the name it is signed under, which the verifier key carries too. Empty: the first `ALLOWED_HOSTS` entry and `/oversight`. |
| `PSEUDONYM_SECRET` | `""` | At least 32 characters, or a list of such secrets with the current one first ([Who asked](#who-asked)). Empty: derived from `SECRET_KEY` and `SECRET_KEY_FALLBACKS` (`ai_sdk.oversight.ledger.W002`). |

## Commands

| Command | What |
| --- | --- |
| `vault_keys` | Prints a new master key, and the ids of the configured ones. |
| `vault_rewrap [--batch 500]` | Wraps every data key under the current master key. |
| `ledger_keys` | Without a `SIGNING_KEY`: prints a new one, its verifier key and a new `PSEUDONYM_SECRET`. With one: its verifier key, to publish. |
| `ledger_checkpoint` | Writes a checkpoint over what was chained since the last, and sends it to the notifiers. |
| `ledger_verify [--json] [--verifier KEY] [--checkpoint FILE] [--allow-unreadable]` | Walks the chain and checks every checkpoint, and each published note given; exits 1 on a problem, 2 on unreadable content only. |

## Checks

| Id | When |
| --- | --- |
| `ai_sdk.vault.E001` | The vault has no usable master key: `MASTER_KEYS` empty or not base64 of 32 bytes, the cryptography package missing, or a `KEY_PROVIDER` that does not import. |
| `ai_sdk.vault.E002` | Database check: data keys are wrapped under a master key id no configured key has. What they keep does not open. |
| `ai_sdk.vault.E012` | `AI_SDK_VAULT` has an unknown key, or a value of the wrong kind. |
| `ai_sdk.vault.W001` | Deploy check, against a database: data keys are still wrapped under a previous master key. Run `vault_rewrap` before dropping it. |
| `ai_sdk.oversight.ledger.E001` | `SIGNING_KEY` is set but cannot sign (not base64 of 32 bytes, or the cryptography package is missing). |
| `ai_sdk.oversight.ledger.E012` | `AI_SDK_LEDGER` has an unknown key, a value of the wrong kind, or a `PSEUDONYM_SECRET` shorter than 32 characters. |
| `ai_sdk.oversight.ledger.W001` | Deploy check: no `SIGNING_KEY`, so checkpoints are unsigned. |
| `ai_sdk.oversight.ledger.W002` | Deploy check: no `PSEUDONYM_SECRET`, so pseudonyms are derived from `SECRET_KEY`. |
| `ai_sdk.oversight.ledger.W003` | Deploy check, against a database: the chain grew and no checkpoint was written for two days. |

With the ledger installed, core's `ai_sdk.oversight.W001` (full capture kept in the clear) no longer fires.

## Removing them

The ledger can go on its own; the vault only after it. To remove the ledger, first run `manage.py migrate django_ai_sdk_oversight_ledger zero` (this drops the chain and the checkpoints), then take it out of `INSTALLED_APPS`. Content the ledger kept stays encrypted in `TurnContent` and core does not open it, so readers see no question or answer on those turns until `oversight_prune` deletes it; content kept from then on is in the clear again. Turns the ledger recorded keep their pseudonym, which core cannot compute: deleting their user no longer clears it or forgets their content without a thread, so run `oversight_forget_actor` for users deleted later, before you remove the ledger, or let those turns expire. To remove the vault too, migrate `django_ai_sdk_vault` to zero after the ledger: that destroys every data key, and with them all the vault kept. Core never imports either app, and the SDK and oversight run the same without them.

## Limitations

- **The database and the master key together read everything.** Encryption at rest protects against a copy of the database alone, not against a process that holds both.
- **Entries are hashes, not signatures.** Someone who can write the database can append entries that verify, or rehash the chain from any point. Every checkpoint written before the rewrite then disagrees, and verify reports it, but they can delete those checkpoint rows too: only a checkpoint published outside the database, checked with `--checkpoint`, catches a rewrite since then for sure. Publish each one.
- **Only a published checkpoint protects the end of the chain.** A writer who drops the newest entries and rolls the head back is caught only by a checkpoint kept outside the database.
- **A turn never closed can be given a close.** A turn has two entries, at its gate and at its close. One that was interrupted has only the first, so an appended entry can still rewrite it.
- **Not hashed:** a turn's `actor_id` and `status` (both change by design), and review items. A reviewer's answer is chained as a lens answer and their read as an access record; the item's own status is not.
- **Prune and erase entries are claims.** See [Verify](#verify).
- **Backups keep wrapped keys** until they expire themselves. So do Postgres's WAL and point-in-time archives, replicas, and dead rows until `VACUUM`: a shredded key's wrapped bytes live there until those are gone.
- **Free text in chained rows cannot be erased.** A reveal's reason, a lens answer's `error` and `trail` (a provider's error can echo the input) are hashed into the chain in the clear and stay for the metadata retention.
- **Rotating a master key does not rotate data keys.** `vault_rewrap` wraps each data key again; a data key that already leaked stays valid for its scope.
- **Reads trust the database.** Content is read as stored; plain content where an envelope belongs, or another envelope of the same scope, shows until `ledger_verify` runs. Verify is the control, so schedule it.
- **Oversight's scopes are `thread:` and `turn:`.** An app that keeps its own data in the vault under those names shares their keys and is shredded with them; use a prefix of your own.
- **No unwrap audit trail.** Opening content is not recorded per key; reads through the review API are, as access records.
- **Content from before the ledger** stays as core kept it, unchained.
- **One lock orders every chained write.** Each turn, each switch and each read through the review API (it writes a chained access record) takes the `LedgerHead` row lock for the end of its transaction, so they queue behind each other there.
- **Verify walks the whole chain.** `POST /oversight/ledger/verify/` walks and decrypts every entry in one request, as `ledger_verify` does. On a long chain, run the command from a scheduler instead, and keep the route for small deployments.
- **Checkpoints read every entry hash.** `ledger_checkpoint` and verify stream all entry hashes once each time, so their cost grows with the chain.

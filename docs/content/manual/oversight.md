---
title: Oversight
type: docs
weight: 127
---

The [Oversight guide](/oversight/) starts with a quickstart and covers the ideas. This page documents what core oversight ships: the classes, settings, stages, models, seams and checks.

Core records turns and creates review items. Reading the record and resolving review items is the review API, an opt-in layer above core ([Review API](#review-api)). Runtime switches for operators are an opt-in app on the `OverlaySource` seam below ([Runtime switches](/oversight-switches/)), and defining whole policies at runtime another one on the policy loader seam ([Runtime authoring](/oversight-authoring/)); an encrypted, chained record is two opt-in apps on the `custody` seam, the vault and the ledger ([Audit-grade oversight](/oversight-ledger/)).

## Installing

Add `"django_ai_sdk.oversight"` to `INSTALLED_APPS` and run `migrate`. The app needs no extra to run. Content is kept in the clear, masked as `CAPTURE` says.

Without the app, nothing checks a turn. The core check `ai_sdk.E001` is an Error when an agent class still asks for a policy (it sets `oversight` to one, or overrides `get_oversight()`) or `AI_SDK_OVERSIGHT["DEFAULT_POLICY"]` is set, so a missing app does not leave those agents unguarded in silence.

- `pip install "django-ai-sdk[typesafe]"` adds the client for the optional `TypeSafe` judge (`django_ai_sdk.oversight.judges.TypeSafe`), a hosted commercial service.

## Settings

`AI_SDK_OVERSIGHT`, one dict, every key optional (`django_ai_sdk.oversight.conf.setting`). A dict value merges one level deep with its default, so `{"DEADLINE_MS": {"gate": 3000}}` keeps the other stages' deadlines. Every key is checked: an unknown key, or a value of the wrong kind, is an error that says how to fix it (`ai_sdk.oversight.E012`).

| Key | Default | Meaning |
| --- | --- | --- |
| `ENABLED` | `True` | Kill switch: off leaves every agent unguarded, whatever its policy says. |
| `DEFAULT_POLICY` | `""` | The policy of agents whose `oversight` is `None`, and of runtime agents whose base class names none. Empty: they are unguarded. |
| `CAPTURE` | `"masked"` | The most content the record keeps (`none`, `metadata`, `masked`, `full`); a policy may keep less, never more. |
| `STREAM_MODE` | `"auto"` | `live` streams the answer as it is written; `auto` holds it until finish passed it only while an enforced rule may mask or withhold it there, decided per turn. A policy may set its own. |
| `UNGUARDED_MODEL_CALLS` | `"refuse"` | A model called through the SDK's generators where a turn is required (every workflow step) but outside one: `refuse` raises `UnguardedModelCall` before the call, `warn` logs it once per place, `allow` lets it be ([Model calls outside a turn](/oversight/#model-calls-outside-a-turn)). |
| `JUDGES` | `{}` | Named judges: `{"alias": {"class": ..., **kwargs}}`, or the name of another alias. A class without a dot is one of `django_ai_sdk.oversight.judges`. |
| `DEADLINE_MS` | `{"gate": 1000, "tool": 1000, "finish": 2000}` | Milliseconds each inline stage may add: the gate before the model starts, the tool stage before each tool call (and masking each tool result), finish once the answer is complete. `None`: no limit. Room for patterns, checks and a fast classifier; a model judge (`LLMJudge`) usually needs longer. A judge still running at the deadline is abandoned: the lens fails (`FAIL_MODE`), keeps the answer it has for the record, and reflect asks again later. |
| `MAX_INPUT_CHARS` | `100000` | A longer question is refused at the gate, unread. With history the server did not store (`gate_history`), the question and the history judged with it count together. `None`: no limit. |
| `CLIENT_DETAIL` | `"actions"` | What `data-oversight` tells the chat client. `actions`: the stage, outcome and message when an enforced rule acted in a way the person sees (block, restrict_tools, mask) or its notice tells them, no rule names. `full`: rule names and every stage's answers too, for a developer view, each value as the record keeps it (see [Models](#models)). |
| `FAIL_MODE` | `"closed"` | What a failed lens means: its check or judge raised, timed out, missed the deadline or was handed more than it reads (too long). `closed` treats an enforced rule that blocks, masks or takes tools away as matched when a lens it reads failed, withholds a tool result it cannot check, and refuses a turn whose policy cannot be chosen or loaded. Rules that only alert, queue or notify never fire on a failure; rules in report mode only record it. `open` lets the turn through and records the failure, and runs a turn whose policy cannot load unguarded. A policy may set its own (`fail_mode`); anything but `"open"` is closed. |
| `RETENTION` | `{"content": 90, "metadata": 730}` | Days a turn's content (`TurnContent`, with the reasons and notes kept in it) and its metadata (the turn, its answers and matches) are kept, as each row's `expires_at`. A policy may set either. `oversight_prune` deletes expired rows ([Retention](#retention)); access records are kept for the `metadata` days set here. |
| `NOTIFIERS` | `[]` | Where notifying matches are sent: instances, dotted paths, or `{"class": "<dotted path>", **kwargs}` (unlike `JUDGES`, always the full path: `"django_ai_sdk.oversight.notifiers.WebhookNotifier"`). One that cannot be built is logged and left out. |
| `RUNTIME_CACHE_SECONDS` | `10` | How long each process caches what operators change at runtime ([switches](/oversight-switches/), [authored policies](/oversight-authoring/)); `0` reads them every turn. |

Outside the dict:

| Setting | Default | Meaning |
| --- | --- | --- |
| `AI_SDK_MODEL_PRICES` | `{}` | Prices the judges' answers (`Answer.cost`) ([Settings](/manual/settings/)). |
| `TYPESAFE_API_KEY` | unset | The `TypeSafe` judge's key; without it that judge abstains (`ai_sdk.oversight.W005`). |
| `TYPESAFE_MODEL` | unset | A TypeSafe model to pin; unset leaves the choice to `typesafe-sdk`. `TypeSafe(model=...)` overrides it. |

## Agents and runs

| On `Agent` | Meaning |
| --- | --- |
| `oversight` | A `Policy` subclass (registered when the agent class is defined), a registered policy's name, `None` (the default: `DEFAULT_POLICY`) or `False` (no oversight). Anything else (an instance, `True`, `""`) raises `ImproperlyConfigured` when the class is defined. |
| `async get_oversight(*, channel, user, **kwargs)` | The policy for this turn; returns `oversight` unless overridden. It may read the database; keep `**kwargs`, as more about the turn may be passed later. Channels: `chat`, `run`, `api`, `workflow:<key>:<step>` (a stored workflow's slug or a registered workflow's name), `workflow:adhoc:<step>` (any other definition), or what a caller passes. A policy only this returns is registered when a turn first uses it: `@register_policy` it, so the checks see it. |
| `oversight_required` | `True`: an overlay (a [runtime switch](/oversight-switches/)) may give the agent another policy but never switch its oversight off; an overlay's "off" is ignored and logged. Code (`oversight = False`, `run(oversight=False)`) still decides. Default `False`. |
| `run(messages, ..., *, channel="run", oversight=None)` | One turn under the policy. `None` is the agent's own; a class or a name replaces it for this call; `False` runs unguarded (trusted code only). Raises `Blocked`. |
| `as_view(...)` | Chat: one turn on channel `chat`. A refusal streams the policy's message. |

A turn's policy is, first match wins: `run(oversight=...)`; an overlay's assignment for the agent (a name, or "off" unless the agent sets `oversight_required`; `OverlaySource.agent_policies`, none in core); `get_oversight()`; `DEFAULT_POLICY` when that returned `None`. `False` anywhere in that order means no oversight. A runtime agent (an `AgentSettings` row) without an assignment of its own takes its base class's `oversight` (`AI_SDK_RUNTIME_AGENT_BASES`), then `DEFAULT_POLICY`. A hook that raises, or a policy that cannot load, refuses the turn (unless `FAIL_MODE` is `"open"`).

`django_ai_sdk.errors.Blocked(message, *, stage="", turn_id="")`, error code `blocked` (HTTP 400, like `content_filtered`, through Ninja's `register_error_handlers` and DRF's `django_ai_sdk.contrib.drf.exception_handler`, with `message` as the body's message). Raised by `Agent.run()` when the gate refuses the history or the question, or finish withholds the answer, and when a masked structured result no longer validates. `message` is the policy's text for the person asking. The run endpoints answer `{"result", "thread_id", "blocked"}` instead (`thread_id` is `""` for the stateless one).

Each agent runs under its own policy. A run inside an active turn joins it when it resolves the same policy, or none: no gate or finish of its own, its tool calls checked and recorded in the outer turn. Under another policy (its own, or an explicit `oversight=`) it opens a child turn, with its own gate, tool checks, finish and record, and `parent_turn_id` naming the outer turn. A subagent's delegation does the same with its task as the question; a refusal is the tool result the coordinator reads. A delegation run synchronously (Haystack's `run()`) cannot open a turn: when it would need one, it answers `UNGUARDABLE` instead. `run(oversight=False)` runs unguarded and does not join. A closed turn is never joined, even by a task that copied it.

`run()` judges what the caller sent on copies: only the last user message of the list is masked in place, so what the caller stores is the masked copy. History the server stored itself is judged on its own turn, not again: with a `thread_id` whose thread is stored, under a policy whose gate masks, the stored messages are the history (the thread run endpoints, as chat); a history that is the stored thread as stored stands. Any other history is judged by `TurnGuard.gate_history()` before the question.

A subclass's own `run()` is wrapped when the class is defined: the turn opens around the outermost `run()`, its arguments read as `Agent.run` takes them, and the `super().run()` inside joins. An override that answers without calling `super()` is judged too. `run_turn` in `django_ai_sdk.oversight.agent_run` is what `run()` wraps around the work; `turn` decides between joining and a child turn; `thread_history` is the stored-thread rule chat shares.

## Lenses

`django_ai_sdk.oversight.lenses`. Subclass one of the three kinds, or construct one with keyword arguments (an unknown keyword raises `ImproperlyConfigured`). The module's docstring states the rules they follow.

Every lens:

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | required | Key within the policy. Answers key on it, so renaming starts a new series. |
| `label` | `""` | A readable name; defaults to the name. |
| `answer` | per kind | `Flag`, `Choice`, `Score` or `Entities` (below), or its name in JSON. |
| `subject` | `Subject.INPUT` | `input`, `output`, `exchange`, `sources`, `tool_call`. |
| `stages` | by subject | Any of `gate`, `tool`, `finish`, `reflect`. Empty: where the subject is first there to judge (`lenses.DEFAULT_STAGES`): the question at `gate`, a tool call at `tool`, the answer, the exchange and the sources at `finish`. Gate reads the question, `tool` tool calls, finish the answer, the exchange and the sources, reflect all but tool calls (`lenses.STAGE_SUBJECTS`). |
| `sample` | `1.0` | Share of turns the reflect stage judges. Inline stages never sample. |
| `enabled` | `True` | Off: the lens never runs. |
| `needs_grounding` | `False` | Only asked when the turn has sources or tool results to judge against. |

Each kind adds its own, and only its own: an attribute declared on a lens whose kind or answer type does not read it is refused when the policy is checked.

| Kind | Answers | Own attributes |
| --- | --- | --- |
| `PatternLens` | `Entities` only, with spans | `patterns` (label to regexes; a plain list labels hits with the lens's name), `keywords` (label to whole words; the label with the most hits comes first), `verify` (label to a check a match must pass, such as a checksum), `flags` (`re.IGNORECASE`, for patterns and keywords), `exclusive` (each stretch of text gets one label, the first pattern's). What patterns find is a fact: hit and miss both have confidence `1`. |
| `CheckLens` | `Flag` by default | `validators`: functions `fn(text)` or `fn(text, view)`, or names registered with `register_validator`; each raises `Violation(message)` to say yes, and the messages are the answer's reason. `check(text, view)` runs them; override it to answer another type, returning the value, an `Answer` (with a reason or confidence) or `None` to abstain. A sync `check` runs in a thread. `timeout` (`30` s). |
| `JudgeLens` | `Flag` by default | `judges`: tried in order (instances, aliases, optional aliases such as `"judge?"`, or `{"class": ..., **kwargs}`); `min_confidence` (`0.75`: an answer below it asks the next judge); `question` (asked verbatim); `guidance` (scale anchors, edge cases); `reads_question` (`True`: a tool-call lens also reads the question that led to the call). |

| Answer | Value | Options, on the lens |
| --- | --- | --- |
| `Flag` | `bool` | `yes`, `no`: what each means |
| `Choice` | one of `labels` | `labels`, `criteria` (label to meaning), `default` (a label; what patterns that find none mean, the last label when unset) |
| `Score` | `0..1` | `low`, `middle`, `high`: what the ends and the middle mean |
| `Entities` | sorted list of labels, with spans | `labels` (when set, other values are dropped); `per_source` (a judge that rates source by source answers with the sources that hold, `items` each one's weight) |

An option the lens leaves out takes the default in `<Answer>.options`. A `Choice` whose `default` is set but not one of its `labels` is refused. Only an `Entities` lens says where in the text it found something, so only it can drive a `Mask`. On an `Entities` answer, `L(name).is_true` means it found something, and `is_false` that it found nothing.

`lens_from_dict(node)` builds a lens from JSON: `{"kind": "pattern" | "check" | "judge", "answer": "flag" | "choice" | "score" | "entities", ...attributes}`, or `{"preset": "pii", ...}` for a built-in one. A check lens there names registered validators only, so JSON never names code to run.

### Built-in lenses

`django_ai_sdk.oversight.presets`. Each has a default `name`; subclass it to name your own.

| Class | Kind | Answers |
| --- | --- | --- |
| `PIILens` | pattern | Personal data. `kinds` defaults to `patterns.DEFAULT_PII_KINDS`: `email`, `iban` (mod-97), `credit_card` (Luhn), `phone` (international format), `ip_address`. Country kinds (`COUNTRY_PII_KINDS`), only when named: `bsn` (11-check), `phone_nl`, `postcode_nl`. `patterns` adds your own, unchecked. |
| `SecretsLens` | pattern | `api_key`, `stripe_key`, `google_api_key`, `github_token`, `aws_key`, `slack_token`, `jwt`, `private_key`, `password_assignment` (`patterns.SECRET_PATTERNS`). |
| `RefusalLens` | judge | Answers that decline: `Patterns(REFUSAL_PATTERNS)` (English and Dutch), then `classifier?`. |
| `LengthLens` | check | A `Flag`: the subject is longer than `max_chars`. |
| `LanguageLens` | check | A `Flag`: answered in another language than asked. Abstains when unsure. For the subject's language as a `Choice`, write a check lens over `presets.guess_language`. |
| `EmptyRetrievalLens`, `ToolErrorsLens` | check | A tool call that found no sources (a `Flag`); the share of tool calls that failed (a `Score`: `L("tool_errors") > 0` when any did). |

The regex tables, checksums and the masking helpers (`mask`, `mask_all`, `mask_secrets`) are in `django_ai_sdk.oversight.patterns`.

## Judges

`django_ai_sdk.oversight.judges`. `answer(lens, view) -> Answer | None`, where `view` is the `TurnContext` the lens reads (`input`, `output`, `history`, `sources`, `tool_call`, the `answers` so far, and `text(subject)`); `None` abstains. An `Answer` has `value`, `confidence`, `rationale`, `spans`, `items`, `trail`, `error`, `cost`, `tokens`, `latency_ms`, `answerer` (who answered: a judge's name, `patterns` or `check`) and `model`. Every judge also takes `name=`, the name its answers are recorded under. The package exports the judges; helpers such as `base.resolve` stay in their modules.

Local (`external = ""`: nothing leaves the process):

| Class | Answers |
| --- | --- |
| `Patterns(patterns=(), *, keywords=(), verify=None, flags=re.IGNORECASE, exclusive=False, miss_confidence=0.5)` | Regexes and keywords, as a pattern lens reads them. A hit is sure; a miss is `miss_confidence` sure, by default below `min_confidence`, so the next judge is asked. A miss at or above it ends the lens, and the judges after it are never asked (`ai_sdk.oversight.W004`). |
| `LexicalSupport()` | Word-overlap groundedness (output) and context relevance (sources). Low confidence by design: a floor under the model judges. |
| `LocalClassifier(model=ENGLISH, *, file="onnx/model_quantized.onnx", positive="")` | An ONNX model from the Hugging Face Hub, run by onnxruntime in this process (the `local` extra, no torch). Zero-shot by default: yes/no (entails the lens's `yes`), label (the criterion it entails most, as "This text is about <criterion>." unless it starts with "This text") and score lenses, about 20 ms each on a CPU. `ENGLISH` (87 MB) and `MULTILINGUAL` (340 MB) are MIT-licensed NLI models; `positive=` makes a model trained for one question (a prompt-injection detector) answer yes/no lenses. Reads one text, the question or the answer, of at most 1,600 characters: lenses on sources or tool calls, lenses that need grounding and "taken from" go to the next judge. Steps aside until its model is loaded, loading it in the background; `judges.local.load()` loads it at startup. |

Remote (`external` names who receives the turn's content; secrets are masked first):

| Class | Answers with |
| --- | --- |
| `LLMJudge(model=None, *, llm=None, base_url="", api_key_env="", generation_kwargs=None, timeout=None)` | Any chat model. Subclass it and set `instructions` for another system prompt. `base_url` and `api_key_env` name a server of your own: the judge then speaks Chat Completions (Ollama, vLLM, LM Studio, llama.cpp), and abstains while the key's variable is unset. Without them, the default gateway (OpenAI Responses, `OPENAI_API_URL`). `llm`: an SDK generator factory or its dotted path, for another provider. `external = "model gateway"`, 45 s timeout. Structured output where the generator supports it, parsed JSON where not. |
| `TypeSafe(model=None)` | TypeSafe, a hosted commercial classification service. `external = "TypeSafe"`, 20 s timeout. Needs the `typesafe` extra and `TYPESAFE_API_KEY`, else abstains (`ai_sdk.oversight.W005`). Import it from `django_ai_sdk.oversight.judges`. Answers `Flag`, `Choice` and `Score` lenses, and an `Entities` lens with `per_source = True` by asking one yes/no per source: the value is the titles of the sources that answer yes, `items` each source's probability. One stage's questions about the same turn go out as one request. A failed request is an `unavailable` error, so `FAIL_MODE` applies and reflect asks again. |

`TypeSafe` is a `Classifier` (`judges.classifier`), the base for any model that answers typed questions: see the guide's [Your own judge](/oversight/#your-own-judge).

What each remote judge sends, per lens subject and with every limit, is in the guide's [Where turn content goes](/oversight/#where-turn-content-goes).

A judge lens's judges are asked in order until one is sure (its confidence reaches the lens's `min_confidence`). When none is, the last one that answered decides: confidences from different judges are on different scales and are never compared. Every other answer is kept in its `trail`. A chain that ends on an error or the deadline after only unsure answers counts as failed (`FAIL_MODE`), and keeps the unsure answer for the record.

A lens that judges the answer against its grounding (subject `sources`, or `needs_grounding = True` such as `groundedness`) reads the turn's retrieved sources *and* its tool results, and is not asked on a turn that has neither. A lens on the answer also reads the sources earlier turns of the thread found, after this turn's. Reflect asks no lens on the answer of a turn the gate blocked. Reflect reads the turn from its content; under `masked` capture it reads the question and answer from the thread's stored messages instead, where there are any, so its masks reach the stored answer.

## Rules and conditions

```python
Rule(name, *, when=Condition, then=Action | [Action], mode=Mode.REPORT,
     message="", notice="", description="", min_mode=Mode.OFF)
```

`min_mode` is a floor in code: an overlay mode (a [runtime switch](/oversight-switches/)) below it (`off` < `report` < `enforce`) is rejected, and so is `pick(..., mode=)` below it, and a `mode` below it when the rule is declared (both raise `ImproperlyConfigured`). `Rule(..., mode="enforce", min_mode="enforce")` stays enforced whatever an operator switches.

Conditions are built from `L(lens)`: `.is_true`, `.is_false`, `== value`, `!= value`, `> < >= <=`, `.in_(values)`, `.contains(label)`, `.missing`, `.present`, `.confident(x)`, `.unsure(x)`, combined with `&`, `|` and `~`. Each serializes to JSON (`{"all": [...]}`, `{"any": [...]}`, `{"not": ...}`, `{"lens", "op", "value"}`), which is what an authored policy stores. Every comparison on a lens without a usable answer is false; use `.missing` to test for that. A failed lens (its check or judge raised or timed out) is unknown, not false: a comparison on it is neither true nor false, so `~L("grounded").is_true` does not match just because "grounded" failed.

A rule is evaluated once every lens it reads has had its chance, so it can combine a gate lens and a reflect lens. A rule fires once per turn. `message` replaces the action's message when the rule blocks.

The answer is judged whole, at finish: nothing judges it while it streams. A tool call rule is judged on every call, so a refused call stays refused when the model retries it; the record keeps it once. `RestrictTools` at the tool stage refuses the calls it names (all, when it names none), and those tools stay away for the rest of the turn.

| Action | Enforced at | Outcome |
| --- | --- | --- |
| `Block(message="")` | gate, tool, finish | `blocked` |
| `RestrictTools(names=None)` (none = all) | gate, tool | `restricted` |
| `Mask(lenses=None, template="[{label}]")` | gate, finish, reflect | `masked` |
| `Queue(queue="default", priority=0)`, `Notify(to=None, severity="warning")`, `Alert(severity="warning")` | any | `flagged` |

`Alert` and `Notify` both mark the turn at their severity; `Notify` also sends the match to the notifiers. A severity is `info`, `warning`, `high` or `critical` (`Severity`, or its string); a blocked turn is at least `high`. `Block.DEFAULT_MESSAGE` is "This request can't be answered here."

`Mask` masks the spans of `lenses` (default: every lens the rule reads), each replaced by `template` with the label in capitals (`[EMAIL]`). Every lens it masks must be an `Entities` lens on text (not a `tool_call` lens: arguments are judged, never rewritten; block the call or take the tool away instead), or the policy is refused when it is built. Tool results are masked by the gate's mask rules. An enforced `Mask` that cannot point at what it should mask (a lens it reads failed while the policy fails closed, or answered without spans) blocks instead. Reflect masks only the stored message, and only when the stored text is the text it judged; it appends a `reflect` notice to the message.

## Policies

```python
class Policy:
    name: str
    label: str
    description: str
    extends: list[str]          # packs or policies
    lenses: list[Lens | type[Lens]]
    rules: list[Rule]
    capture: Capture | None     # none, metadata, masked, full; None inherits
    stream_mode: str | None     # auto or live; None inherits
    fail_mode: str | None       # closed or open; None inherits FAIL_MODE
    retention: dict[str, int]   # content, metadata (days); either alone
```

A policy is registered by `@register_policy` (in any module; an installed app's `oversight.py` is imported on startup), or by naming the class on an agent (`oversight = SupportPolicy`) or in `run(oversight=...)`. A policy that fails validation is not registered (`ai_sdk.oversight.E001`); its agents' turns are refused. Two classes that declare one name are an error (`ai_sdk.oversight.E010`): the first one registered keeps the name.

`pick(items, *names, mode=None)` takes single lenses or rules out of a pack instead of extending it whole: `pick(SecurityPack.lenses, "prompt_injection")`, `pick(SecurityPack.rules, "block_injection", mode=Mode.ENFORCE)`. They come back in the order named; rules are copies, so `mode` leaves the pack's own alone. An unknown name raises `ImproperlyConfigured`, and `mode` on lenses raises `TypeError`.

Every effective value has a source, in this order: the extended packs (`pack:<name>`), the class (`code`), then the overlay (`overlay`); a policy's `capture` capped by the deployment's `CAPTURE` says `settings`. The overlay only switches rule modes. A parent's switched modes reach the policies that extend it, with the source `overlay:<parent>`. A mode that cannot apply (an unknown rule or mode, or one below `min_mode`) is listed under `rejected` (`{"rule", "mode", "reason"}`), logged and ignored.

The overlay comes from an `OverlaySource` (`django_ai_sdk.oversight.resolve`). Core's is empty; an app that lets operators switch modes at runtime subclasses it and calls `set_overlay_source(source)`: `async rule_modes(names)` returns `{policy: {rule: mode}}` for the policy names asked, `async agent_policies()` every agent it assigns (agent id to a policy name, or `""` for off; an agent left out is up to code), and `ttl` is how long a resolution built with it, and its agent assignments, are reused (core: until settings change or a policy registers). When `rule_modes` or `agent_policies` raises, turns run the last good read, else the policy as code declares it. The [switches app](/oversight-switches/) installs one that reads its `Override` rows, with `ttl` = `RUNTIME_CACHE_SECONDS`.

An app can also register whole policies at runtime. It adds a loader with `resolve.add_policy_loader(loader)`: an `async` callable, awaited before policy lookups at most once per `RUNTIME_CACHE_SECONDS` (and again after `resolve.invalidate()`), that registers policies (`registry.register_policy`) and withdraws them (`registry.unregister_policy(name)`). A loader that raises is logged, and what it registered stays. `policy.build(name, policy_class=cls)` builds a class that is not registered yet, so a draft can be checked first. The [authoring app](/oversight-authoring/) is such a loader.

A policy that fails to load (a missing parent, a cycle) refuses its agents' turns; only `FAIL_MODE = "open"` leaves them unguarded. The reason is logged and kept (`resolve.broken_reason(name)`).

## Packs

`django_ai_sdk.oversight.packs`. Every pack rule is in report mode. A lens answered only by an optional alias is skipped where that alias is undefined.

| Pack | Lens | Kind, answer, subject | Stage | Answered by |
| --- | --- | --- | --- | --- |
| `security` | `prompt_injection` | judge, Flag, input | gate | `Patterns(INJECTION_PATTERNS)` (overrides, requests for the hidden prompt, role takeovers, fake delimiters; a hit is sure, a miss asks on), `classifier?` |
| | `pii_input` | pattern, Entities, input | gate | `PIILens` |
| | `secrets_input` | pattern, Entities, input | gate | `SecretsLens` |
| | `secrets_tool` | pattern, Entities, tool call | tool | `SecretsLens` |
| | `pii_tool` | pattern, Entities, tool call | tool | `PIILens` |
| | `pii_output` | pattern, Entities, output | finish | `PIILens` |
| | `exfiltration` | pattern, Entities, output | finish | `EXFILTRATION_PATTERNS`: Markdown images, links and reference links, and HTML images, that carry data in their URL's query |
| `safety` | `self_harm` | judge, Flag, input | gate | `classifier?` |
| | `harm_input` | judge, Score, input | gate | `classifier?` |
| | `harm_output` | judge, Score, output | finish | `classifier?`, `judge?` |
| `quality` | `context_relevance` | judge, Score, sources | reflect | `classifier?` |
| | `groundedness` | judge, Score, output, needs grounding | reflect | `classifier?` |
| | `taken_from` | judge, Entities, output, `per_source` | reflect | `classifier?` |
| | `answer_relevance` | judge, Score, exchange | reflect | `classifier?` |
| | `helpfulness` | judge, Score, exchange | reflect | `classifier?`, `judge?` |
| | `refusal` | judge, Flag, output | finish | `Patterns(REFUSAL_PATTERNS)` (a hit is sure, a miss asks on), `classifier?` |
| | `language_mismatch` | judge, Flag, exchange | finish | `classifier?` |
| | `empty_retrieval` | check, Flag, sources | finish | `EmptyRetrievalLens` |
| | `tool_errors` | check, Score, exchange | finish | `ToolErrorsLens` |

Rules: `security` has `block_injection`, `mask_secrets_in_question`, `mask_pii_in_question`, `refuse_secrets_in_tools`, `mask_pii_in_answer` and `block_exfiltration`; `safety` has `person_at_risk` (never blocks: queues at priority 100 and notifies at `critical`), `harmful_request` and `harmful_answer`; `quality` has `review_ungrounded`, `review_unsourced`, `review_irrelevant_context`, `note_empty_retrieval` and `note_tool_failures`. `default` extends all three.

## The guard

`TurnGuard` (`django_ai_sdk.oversight.guard`) is the seam every turn passes through. `Agent.run()`, chat and the SDK's entry points open it for you; open one yourself only for a turn the SDK does not run.

`await TurnGuard.start(agent, *, channel, user=None, thread_id=None, messages=None, input=None, oversight=None)` returns an inert guard unless a policy applies (every call then returns at once and allows everything); `oversight` replaces the agent's own as in `run()`, and `False` opens none. Then `gate_history(messages)` (history the server did not store; returns a `GateResult`), `gate()`, `add_text(chunk)` (streamed answer text), `tool_call(call)`, `mask_result(call_id, text)`, `set_sources(...)`, `add_tool_call(...)`, `add_usage(...)`, `finish(output)` and `close(error="")`, which writes the turn and never raises. `gate()` (and `gate_history()` when it refuses) writes the turn ahead, open, when it ends (`write_ahead()`); `close` updates it.

`unguarded()` suppresses guards inside a block (the SDK's judges and dry runs run under it); `guarding(guard)` makes one visible to `get_tools()`, the tool hooks and nested runs; `current_guard()` returns it, and `None` once it is closed. `OversightToolHook` (before each tool call) and `OversightToolResultHook` (after) are in `django_ai_sdk.agents.tool_agent.default_hooks(agent)`, so every tool agent, subagents included, reports to the active guard. Tool results that are not text (images, files) pass as they are.

A headless `agent.run(..., tools=True)` reports its tool calls, their results and its usage to the guard, so finish judges what the loop did. A run without tools reports no token usage, so its turn has no cost.

## Client notices

`data-oversight` stream parts (`DataEvent(data_type="oversight")`):

| Key | When |
| --- | --- |
| `turn_id`, `stage`, `outcome` | Always. `outcome` is `blocked`, `restricted`, `masked`, `flagged` or `allowed`. |
| `message` | The turn was blocked: the policy's text. |
| `replace` | Finish replaced the answer: the text to show instead. |
| `input` | The gate masked the question: the masked text. |
| `notes` | Enforced rules with a `notice` that acted, once a turn each: `kind` (the first of their actions by `actions.ACTION_ORDER` in `actions.USER_FACING`: `block`, `restrict_tools` or `mask`; else `"notice"`, when it only queued, alerted or notified), `also` (the other user-facing ones), `text`, `stage`. |
| `rules`, `detail` | Only with `CLIENT_DETAIL` `full`: rule names and every stage's answers. Each answer's `value` is the record's ([`LensAnswer`](#models)), so the client never reads what a mask took out. |

A refused question or withheld answer ends with finish reason `content-filter` (`django_ai_sdk.oversight.stream.CONTENT_FILTER`); `OpenAIProtocolHandler` sends `content_filter` and no `data-*` parts. The notices are stored on the message under `metadata["oversight"]`, with `replace` reduced to `"replaced": true`.

## Models

| Model | Holds |
| --- | --- |
| `Turn` | `status` (`open` from the end of the gate until it closes, `closed`, or `interrupted` when `oversight_sweep` found it never closed), channel, agent, agent fingerprint, policy, thread and message ids (no foreign keys), `actor_id` (the user's primary key, or with the [ledger](/oversight-ledger/#who-asked) a pseudonym; `""` without one), model, tokens, cost, latency, `outcome`, `report_outcome` (what report-mode rules would have made it; both blank while the turn is open, and when it closed on an error of its run with no rule acting), `error`; labels (the lens answers' values, below) and severity (denormalized); `expires_at`, when the policy's metadata retention runs out. |
| `TurnContent` | The content at the policy's capture level, as custody stored it (`stored`), with its scope (`thread:<id>`, or `turn:<id>` without a thread) and an expiry. Beside it: `texts`, by answer id, what each answer said in words (its reason, the names a judge gave for an `Entities` lens, its per-item scores by key, the same of the answers on its trail); and `notes`, by review item id, what reviewers wrote. Both are added as reflect's and reviewers' answers arrive, and go when the content goes. |
| `LensAnswer` | One answer to one lens: answerer (who answered), model, value, confidence, spans (offsets and labels), trail, error, cost, latency, reviewer. Nothing that can quote the turn: an `Entities` value keeps a label the lens declares, or a kind its patterns report (a pattern lens, or a `Patterns` judge), while any other name (a judge's, `TypeSafe`'s source titles, a person's) is `"[found]"`, one per name, so `.is_true` and counts still hold. The words are in the content's `texts`. |
| `RuleMatch` | A rule that matched: stage, mode, actions, whether it acted, the condition. Inline a rule may match more than once (the history and the question, each tool call); after close it matches once, whether reflect or a reviewer's answer made it match: both lock the turn before they look, and a unique constraint holds it. |
| `ReviewItem` | A turn waiting for a person, per queue and priority, created by a `Queue` action. Its resolution holds who closed it (a user id), the value as the record keeps it and whether they agreed; the reviewer's note is in the content's `notes`. Core creates items; the [review API](#review-api) resolves them. |
| `AccessRecord` | A read of the record, or an operator's change of it: the reader (`reader_id`, a user id, and `reader_label`, their username and id, cut to the id when the user is deleted), the `operation` (`AccessRecord.Operation`: the reads `list`, `view`, `reviews`, `review`, `verify` and `reveal`; the changes `checkpoint`, `switch` and `author`; or `prune` for a run of `oversight_prune`, with what it deleted in `query`), the ids of the turns it showed (`turn_ids`, a list read included), whether it showed content, and the query. When a list showed the content of some of its turns only (the content scope is narrower), `query["content_ids"]` names those. The same reader repeating the same read within a minute is one record. Counts are not recorded. |

The Django admin registers `Turn`, `LensAnswer`, `RuleMatch`, `ReviewItem` and `AccessRecord` read-only, without the content. It needs only metrics access, so a turn also leaves out who asked and its thread and message ids. Turns and their rows are narrowed by the metrics scope (`scope_turns`), as in the API; views there write no access record. The access records list both reads and changes (switches, authored versions, checkpoints and prune runs); filter by operation to see one kind. Reasons and notes are not in those rows, so the admin never shows them.

## Review API

An opt-in layer above core: core never imports it, and nothing changes when it is not mounted. `django_ai_sdk.oversight.services.OversightService` is the API; `get_oversight_router()` serves it over django-ninja. There are no Django REST framework views for it in this release: a DRF project calls `OversightService` from views of its own, which keeps its permissions and access records.

```python
from django_ai_sdk.contrib.ninja import get_oversight_router

api.add_router("/", get_oversight_router())
```

The router signs readers in with django-ninja's `django_auth`: the session, with Django's CSRF check on every POST. It does not take the API's auth, so an API that signs in another way passes its own: `get_oversight_router(auth=MyAuth())`.

| Route | Service | Operation |
| --- | --- | --- |
| `GET /oversight/turns/` | `list_turns(filters, user=, limit=, cursor=)` | metrics |
| `GET /oversight/turns/{id}/` | `get_turn(turn_id, user=)` | metrics; the content takes content |
| `GET /oversight/reviews/` | `list_reviews(user=, queue=, status=, limit=, cursor=)` | review |
| `GET /oversight/queues/` | `queues(user=)` | review |
| `POST /oversight/reviews/{id}/` | `resolve_review(item_id, answer, user=)` | review |
| `GET /oversight/metrics/` | `summarize(user=, days=, agent_id=, policy=)` | metrics |

The turns feed filters on `outcome`, `report_outcome` (turns a report-mode rule would have changed), `status`, `agent_id`, `channel` (a prefix), `policy`, `severity`, `label` with an optional `value`, `rule`, `review` (`open`, `reviewed`, `unreviewed`), `thread_id` (content: it takes `VIEW_OVERSIGHT_CONTENT`, 403 without, and reaches only turns in the content scope), `since` and `until`. Lists are newest first (reviews: highest priority, then oldest) and page by cursor: pass a page's `next_cursor` back as `cursor`; there is no total. On Postgres a GIN index on the labels answers the label filter.

Resolving an item (`ResolveIn`: `lens`, `value`, `note`, `agree` for a call on a rule, and `dismiss`) answers the item's lens, or `rule:<rule>` for an item a rule queued; a `lens` other than the item's own is refused, 400, unless the rule that queued the item reads it, so a reviewer answers only what their queue asks. For an item with neither, `lens` names a lens of the turn's policy, or `rule:<name>` for a rule that matched it. The answer is written as a `LensAnswer` with the reviewer's user id and answerer `reviewer`, the turn's labels follow it, rules it now makes match are recorded (and queue the turn when they queue), and the note goes into the turn's content. `dismiss` closes the item without an answer. Resolving claims the item with a conditional update: a second resolve answers 409. Nobody reviews a turn they asked (its `actor_id`, or that of the turn that delegated it, is theirs): resolving and dismissing it answer 403. A forgotten user's turns carry no actor, so this cannot hold for them. A turn still open (its answer is still being written) takes no answer, 409; its items may be dismissed.

A rule a reviewer's answer makes match sends its `Notify` actions once the answer is written, as reflect's do, with `stage` `"review"`: a reviewer's "this leaked" alerts. Its other actions are recorded and do nothing more: a `Queue` queues the turn, but a `Mask` masks nothing and no notice reaches the person's thread.

Permissions are the `oversight` domain of `AI_SDK_PERMISSIONS`. The default, `django_ai_sdk.oversight.permissions.OversightDefaultPermission`, takes a Django model permission per operation and grants nothing for `is_staff`:

| Operation | Model permission | Reads |
| --- | --- | --- |
| `VIEW_OVERSIGHT_METRICS` | `django_ai_sdk_oversight.view_turn` | Turns without their content, answers' values, rule matches, counts. |
| `VIEW_OVERSIGHT_CONTENT` | `django_ai_sdk_oversight.view_turncontent` | On top of metrics: the turn's content, the preview, who asked, thread and message ids, answers' reasons, spans and words, reviewers' notes. |
| `REVIEW_OVERSIGHT` | `django_ai_sdk_oversight.change_reviewitem` | The review queues, and resolving an item with an answer. The queue lists each item's turn row only to a reader who also holds metrics, and its content only to one who holds content. |
| `VIEW_OVERSIGHT_SWITCHES` | `django_ai_sdk_oversight_switches.view_switch` | The [runtime switches](/oversight-switches/) in force (the opt-in switches app). |
| `SWITCH_OVERSIGHT` | `django_ai_sdk_oversight_switches.change_switch` | Switching a rule's mode or an agent's policy. |
| `VIEW_OVERSIGHT_LEDGER` | `django_ai_sdk_oversight_ledger.view_checkpoint` | The [ledger's](/oversight-ledger/) checkpoints, and verifying the chain (the opt-in ledger app). |
| `CHECKPOINT_OVERSIGHT` | `django_ai_sdk_oversight_ledger.add_checkpoint` | Writing a checkpoint. |
| `REVEAL_OVERSIGHT_ACTOR` | `django_ai_sdk_oversight_ledger.reveal_actor` | Who asked a turn, for a recorded reason. |
| `VIEW_OVERSIGHT_AUTHORING` | `django_ai_sdk_oversight_authoring.view_authoredpolicy` | The [policies defined at runtime](/oversight-authoring/), their versions and what a definition may use (the opt-in authoring app). |
| `AUTHOR_OVERSIGHT` | `django_ai_sdk_oversight_authoring.change_authoredpolicy` | Saving, trying and withdrawing a policy defined at runtime. |

Operators give a reason where an act weakens oversight or exposes someone, and it is kept in the act's access record: a [switch](/oversight-switches/) that weakens oversight needs one (raising a mode does not), every [authored policy](/oversight-authoring/) version needs one, and revealing who asked a turn (the [ledger](/oversight-ledger/#who-asked)) needs ten characters or more.

A reviewer may hold review alone: they then work the queue from each item's queue, lens and rule, without the turn's row, and resolve it without seeing the question. Grant metrics, and content, for a reviewer who should judge the turn itself.

Which turns a reader reaches is narrowed by two hooks a subclass overrides, `scope_turns(user, operation, queryset)` and `scope_reviews(user, operation, queryset)`. A turn or item outside them is not found (404), and a turn outside the content scope is shown without its content. Every read decides through these querysets, a list or one turn alike, so narrow in the hooks (or `get_queryset_perms`): a `has_object_permission` of your own is not consulted for content. By default an item is in scope when its turn is:

```python
from django_ai_sdk.oversight.permissions import OversightDefaultPermission


class TeamOversight(OversightDefaultPermission):
    def scope_turns(self, user, operation, queryset):
        return queryset.filter(agent_id__in=agents_of(user))
```

An empty `AI_SDK_PERMISSIONS["oversight"]` falls back to the default, as for agents; a record open to every user says so with `["django_ai_sdk.permissions.AllowAll"]`.

Every read of turns or review items writes an `AccessRecord` with the ids of the turns it showed, and of those whose content it showed when that is not all of them.

What was forgotten reads back as gone. A turn whose thread or user was deleted, or whose content expired and was pruned, has `content_state` `"none"`, no preview, no answers' words and no reviewers' notes; a deleted user's turns have an empty `actor_id`; a pruned turn is not found. Content past its expiry and not yet pruned reads as `"expired"` on the turn, and as no preview and no notes in lists. Content kept under a master key not configured here (the [vault](/oversight-ledger/#the-vault)) reads as `"unreadable"`.

## Retention

Part of the review layer: `django_ai_sdk.oversight.retention.prune()`, run by `manage.py oversight_prune`. Core stamps every row with `expires_at` and deletes nothing. Schedule the command daily (cron, or your scheduler):

```bash
python manage.py oversight_prune --batch-size 500
```

It deletes, `--batch-size` rows per transaction so a backlog never holds one long lock:

- content rows past their `expires_at`, and with them what judges and reviewers wrote there; the turn stays, its content cleared;
- turns past theirs, with their answers, matches, review items and content. Retention wins over a review still open: the prune's access record counts them in `open_reviews`. Custody's `pruning(turn_ids)` is handed each batch first, in its transaction; when it raises, that batch stays and the run stops;
- access records older than `RETENTION["metadata"]` days. Custody's `pruning((), access_ids=ids)` is handed each batch first, in its transaction. This is the setting's value, not a policy's: when a policy keeps its turns longer, set the setting at least as long, or the record of who read them goes first.

It ends with an `AccessRecord` of operation `prune`, with the counts in `query`. `AccessRecord.last_pruned()` reads it, and so does the deploy check `W007`, which warns when there are turns more than two days old and no run finished in the last two days. It reads the database, so it runs with `--database`:

```bash
python manage.py check --deploy --database default
```

## Custody

`django_ai_sdk.oversight.custody` decides how a turn's content is kept, and how the record names who asked. A custody has seven methods, the `Custody` protocol; core installs `PlainCustody`:

| Method | What it is handed | `PlainCustody` |
| --- | --- | --- |
| `actor(user_pk) -> str` | A user's primary key: what `Turn.actor_id` holds for them. Pure, no query; `""` stays `""`. | The primary key |
| `store(scope, content) -> dict` | A turn's content: what its `TurnContent.stored` keeps. May raise `custody.Forgotten`: the turn is then recorded without content. | The content as it is |
| `load_many(items) -> list` | What `store` kept for each `(scope, stored)`, in order and in a fixed number of queries; None for one that no longer opens, instead of raising. | `stored` of each |
| `forget(scope)` | A scope to forget (`"thread:<id>"`, `"turn:<id>"`). | Deletes its `TurnContent` rows; the turns stay, `content` cleared |
| `chained(items)` | Rows just written, as `(kind, row)`: `turn`, `answer`, `match`, `access`, `switch`. A turn is handed each time its row is written (open at the end of the gate, then at close). Raising aborts the write. | Nothing |
| `pruning(turn_ids, *, access_ids=())` | Turns retention is about to delete with their answers, matches and content, or access records past the metadata retention. | Nothing |
| `expiring(content_ids)` | Content rows retention is about to delete ahead of their turns. | Nothing |

Every method runs in the caller's transaction. An app installs its own with `custody.install(custody)` in its `AppConfig.ready()`; `install` raises `TypeError` when the object lacks one of them. The [ledger app](/oversight-ledger/) installs `LedgerCustody`. `custody.actor_names(user_pk)` is every name the record may hold for a user: the installed custody's, and their primary key, as turns recorded before that custody was installed name them.

Deleting a thread calls `forget("thread:<id>")`. Deleting a user forgets the content of each turn they asked without a thread, then clears `actor_id` on all their turns (under every name of `actor_names`) and their name on the access records they left (`signals.forget_actor`). Under `PlainCustody` that is one delete of content rows per 500 turns; another custody is handed `forget("turn:<id>")` per turn. A failure there is logged and leaves the user's turns as they were, all or nothing ([Forgetting](/oversight/#forgetting) says what goes and what stays; [Commands](#commands) run it again).

## Testing

`django_ai_sdk.oversight.testing.dry_run(policy, *, input, output="", tool_calls=None, user=None) -> TurnResult` runs a policy (a registered name, or a draft built with `policy.build(name, policy_class=...)`) over one exchange through gate, the tool calls (each checked as the model's), finish and reflect (every reflect lens, whatever its sample), and records nothing. `TurnResult` has `outcome`, `report_outcome`, `ended_at` (the stage that refused, or `""`), `stages` (`StageResult`: `stage`, `answered`, `matched`, `blocked`, `message`, `text`, `tool`, and `answers` and `rules` by name) and `stage(name)`. An unknown or broken policy raises `LookupError`. Judges are real: a model judge is called.

## Tasks

`django_ai_sdk.oversight.tasks`, `django-tasks` on the `default` queue: `reflect_turn`, queued on commit when the policy has reflect lenses or a lens is to be asked again, and `send_notifications`, which sends a turn's notifications from the worker so a slow receiver never holds a stream open. What reflect does is in `django_ai_sdk.oversight.reflect`.

`Turn.reflect_state` follows reflection: `pending` when queued, `running` while a worker holds its claim, `done`, or `failed`. `areflect` claims the turn in one conditional update with a lease of `reflect.REFLECT_LEASE` (ten minutes) and counts `reflect_attempts`, so a doubled task does nothing. It writes its answers, matches, the stored answer's mask and notices, and `done`, in one transaction and only while the lease is still its own. A run that fails gives the claim back (`pending`); after `reflect.REFLECT_ATTEMPTS` (three) the turn is `failed` and its `error` says `reflect failed`. A lease that ran out (its worker died) may be claimed again.

## Commands

`oversight_sweep [--stale-after 30m] [--reflect-after 10m] [--limit 500] [--dry-run]` marks turns still `open` after `--stale-after` as `interrupted`, and queues `reflect_turn` again for turns `pending` longer than `--reflect-after` (their task was never queued) and turns `running` under a lease that ran out (their worker died), at most `--limit` of each, oldest first. A turn whose last attempt died is marked `failed`. Ages are a whole number with `s`, `m`, `h` or `d`. Schedule it; each run takes the next batch.

`oversight_prune [--batch-size 500]` applies retention; see [Retention](#retention).

`oversight_forget_actor <user id> [...]` forgets users on the record as their delete does (`signals.forget_actor`): for a delete whose forgetting failed and was logged. `oversight_forget_thread <thread id> [...]` forgets threads as their delete does, through the installed custody: for threads deleted without signals (a queryset `.update()`, raw SQL, another database). Both are safe to run again.

`oversight_sweep` and `workflows_sweep` share `django_ai_sdk.management.sweep.SweepCommand`: subclass it, return the stale rows from `stale(cutoff)` and set `marked` (or override `act`).

## Notifiers

`AI_SDK_OVERSIGHT["NOTIFIERS"]` lists `LogNotifier()` and `WebhookNotifier(url, *, secret="", timeout=5.0)` (both in `django_ai_sdk.oversight.notifiers`), or your own `Notifier` subclass (implement `async send(event)`), as instances, dotted paths, or `{"class": "<dotted path>", **kwargs}`. Every notifier takes `name` (by default its class name without `Notifier`, lowercased: `log`, `webhook`); `Notify(to=[...])` sends to those names only. A `Notify` action sends `{"id", "type": "oversight.match", "turn_id", "rule", "stage", "severity", "outcome", "agent", "channel"}`: no turn content. `id` is stable (`notifiers.event_id(turn_id, rule, stage)`): a repeated delivery carries the same one.

A `WebhookNotifier` sends the event's `id` as `webhook-id`. With a `secret` it signs like [Standard Webhooks](https://www.standardwebhooks.com/): headers `webhook-timestamp` and `webhook-signature` (`v1,` + base64 HMAC-SHA256 of `{id}.{timestamp}.{body}`).
To verify, recompute the HMAC over the raw body with the same secret (base64-decoded when it starts with `whsec_`) and compare in constant time.
Refuse timestamps more than a few minutes old and ids you have seen, so a captured request cannot be replayed:

```python
import base64, hashlib, hmac, time

def verified(headers, body: bytes, secret: bytes) -> bool:
    msg_id, stamp = headers["webhook-id"], headers["webhook-timestamp"]
    if abs(time.time() - int(stamp)) > 300:
        return False
    signed = f"{msg_id}.{stamp}.".encode() + body
    expected = "v1," + base64.b64encode(hmac.new(secret, signed, hashlib.sha256).digest()).decode()
    return any(hmac.compare_digest(expected, s) for s in headers["webhook-signature"].split())
```

A notifier's `send(event)` may raise: the error is logged and the turn goes on.

## System checks

| Id | When |
| --- | --- |
| `ai_sdk.oversight.E001` | A policy was not registered; the message says why (a lens without a name, a pattern lens without patterns, a check lens without validators, a judge lens without judges, an attribute its kind or answer type does not read, a duplicate lens or rule, a rule without `then`, a `Mask` over a lens that is not `Entities` or that judges a tool call, a `Choice` whose `default` is not a label). |
| `ai_sdk.oversight.E002` | A policy does not resolve (an unknown parent, a cycle, an inherited lens a rule cannot use). |
| `ai_sdk.oversight.E003` | A rule reads a lens its policy does not have. |
| `ai_sdk.oversight.E004` | A judge lens names an alias `JUDGES` does not define (optional `?` aliases excepted). |
| `ai_sdk.oversight.E005` | An alias cannot be built. |
| `ai_sdk.oversight.E006` | `DEFAULT_POLICY` names an unknown policy. |
| `ai_sdk.oversight.E007` | An enforced rule reads a lens no judge here can answer. |
| `ai_sdk.oversight.E009` | An agent class names a policy no app registers: every turn it runs is refused. |
| `ai_sdk.oversight.E010` | Two policy classes declare the same name; the first one registered keeps it. |
| `ai_sdk.oversight.E011` | An agent class's `oversight` names no policy (set after the class was defined). |
| `ai_sdk.oversight.E012` | `AI_SDK_OVERSIGHT` has an unknown key, or a value of the wrong kind; the hint says how to set it. |
| `ai_sdk.oversight.E013` | Deploy check: an enforced gate or tool rule that blocks, masks or takes tools away reads a lens that can ask an external judge (any judge with `external` set) with a timeout longer than that stage's deadline. That includes a judge behind patterns whose miss is unsure (`miss_confidence` below the lens's `min_confidence`): it is asked on every miss. A slow answer fails the lens, and the rule refuses the turn or the tool call. |
| `ai_sdk.oversight.W001` | Deploy check: `CAPTURE` is `"full"` and content is kept in the clear (`PlainCustody`). |
| `ai_sdk.oversight.W003` | Judge lenses of the policies in use (`DEFAULT_POLICY`, the ones agent classes name, and what they extend) have no model judge here, listed in one warning by policy: skipped on every turn, or answered by their `Patterns` alone ("patterns only"). |
| `ai_sdk.oversight.W004` | A judge lens lists judges it never asks: a `Patterns` judge before them is sure when it finds nothing. |
| `ai_sdk.oversight.W005` | An alias's judge cannot run here, such as `TypeSafe` without `TYPESAFE_API_KEY` or `typesafe-sdk`. |
| `ai_sdk.oversight.W006` | A lens is at a stage that never reads its subject (the question at `tool` or `finish`, the answer at `gate` or `tool`, a tool call anywhere but `tool`): it never runs there. |
| `ai_sdk.oversight.W007` | Deploy check, against a database (`check --deploy --database default`): there are turns more than two days old and `oversight_prune` has not finished in the last two days, so expired rows stay. |
| `ai_sdk.oversight.W008` | An enforced rule reads lenses whose judges are all optional aliases and none is defined: it never acts. |
| `ai_sdk.oversight.W011` | A policy in use masks or blocks the answer at finish but streams it live: the client has the text before finish judges it. Use `stream_mode = "auto"` or a client that applies the `replace` notice. |
| `ai_sdk.oversight.W013` | An agent with a policy streams chat through a `get_pipeline_adapter()` annotated to return something other than the SDK's `Stream`: the answer is never finished. |
| `ai_sdk.oversight.W014` | Deploy check: the `TASKS` backend runs reflect and notifications inline in the request (`ImmediateBackend`), never runs them (`DummyBackend`), or does not load. |

`W001`, `W007` and `W014` are checked only while oversight is enabled; `W001`, `W007`, `E013` and `W014` only by `manage.py check --deploy`, and `W007` only with `--database`.

The opt-in apps raise their own, under `ai_sdk.vault.*` and `ai_sdk.oversight.ledger.*` ([Audit-grade oversight](/oversight-ledger/#checks)), `ai_sdk.oversight.switches.*` ([Switches](/oversight-switches/)) and `ai_sdk.oversight.authoring.*` ([Authoring](/oversight-authoring/)).

In every app, the twelfth error id is the same check: its settings dict has an unknown key, or a value of the wrong kind.

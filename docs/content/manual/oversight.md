---
title: Oversight
sidebar:
  order: 127
---

The [Oversight guide](../oversight/oversight.md) starts with a quickstart and covers the ideas. This page documents what core oversight ships: the classes, settings, stages, models, seams and checks.

Core records turns and creates review items. The [review API](#review-api) reads the record and resolves review items, and is an opt-in layer above core. Operators write policies at runtime with the opt-in [authoring app](../oversight/oversight-authoring.md). A lens or rule declared in code changes only with a deploy.

## Imports

An app imports everything it uses from `django_ai_sdk.oversight`. Its submodules are internal. The declaration API loads with the package and the rest loads on first use, so a policy file imports no model client. The review API (`services`, `schemas`, `permissions`) and the [authoring app](../oversight/oversight-authoring.md) are opt-in layers with modules of their own.

| For | Names |
| --- | --- |
| Policies | `Policy`, `LensPack`, `RulePack`, `register_policy`, `unregister_policy`, `add_policy_loader`, `invalidate_policy`, `build_policy` (a policy resolved with its parents and settings, or a draft class with `policy_class=`), `ResolvedPolicy`, `Capture`, `Mode`, `Severity`, `Stage`, `Subject`, `Outcome` |
| Lenses | `Lens`, `JudgeLens`, `PatternLens`, `CheckLens`, `PIILens`, `SecretsLens`, `RefusalLens`, `LengthLens`, `LanguageLens`, `EmptyRetrievalLens`, `ToolErrorsLens`, `Flag`, `Choice`, `Score`, `Entities`, `Violation`, `register_validator`, `register_pii_kind`, `Answer`, `TextSpan` (a span of an `Entities` answer), `TurnContext` |
| Rules | `Rule`, `L`, `Block`, `Mask`, `RestrictTools`, `Queue`, `Alert`, `Notify` |
| Judges | `Judge`, `Classifier`, `certainty`, `PatternJudge`, `LLMJudge`, `LocalClassifier`, `load_local_classifier`, `TypeSafe`, `Maybe` (an alias a lens skips while `JUDGES` does not define it) |
| Packs | `SecurityPack` (`PromptInjection`, `Exfiltration`), `SafetyPack` (`SelfHarm`, `HarmQuestion`, `HarmAnswer`), `QualityPack` (`ContextRelevance`, `Groundedness`, `TakenFrom`, `AnswerRelevance`, `Helpfulness`), `DefaultPolicy`, `INJECTION_PATTERNS`, `EXFILTRATION_PATTERNS` |
| Patterns | `PII_PATTERNS`, `SECRET_PATTERNS`, `DEFAULT_PII_KINDS`, `mask`, `mask_all`, `mask_secrets` |
| Runs | `run_with_oversight`, `run_with_oversight_sync`, `TurnRequest`, `DEFAULT_CHANNEL` (`"default"`, the channel of a run that names none), `needs_own_turn`, `needs_own_turn_sync`, `apply_oversight` (a copy of an Agent with the hooks) |
| The guard | `TurnGuard`, `StageResult`, `current_guard`, `unguarded` |
| Testing | `dry_run`, `DryRunResult` |
| Notifiers | `Notifier`, `LogNotifier`, `WebhookNotifier`, `send_to_notifiers` (sends an event to every notifier, or the named ones) |
| Forgetting and retention | `forget_turns`, `forget_thread`, `forget_actor`, `prune` |

:::details[Plumbing for code that runs a turn or builds hooks by hand]
Code that runs a turn by hand, such as an adapter, imports its plumbing from the module that defines it:

- `django_ai_sdk.oversight.guard`: `guarding`, `closing_turn`, which closes a guard when its block ends, and `outlived_turn`, for code that runs after its turn closed.
- `django_ai_sdk.oversight.stream`: `StreamTurn`, `CONTENT_FILTER`.
- `django_ai_sdk.oversight.history`: `history_for`.
- `django_ai_sdk.oversight.judges.base`: `judge_label`, a spec's name.

Code that builds a Haystack Agent's hooks by hand imports them from `django_ai_sdk.oversight.plugin` (`build_oversight_hooks`, `merge_oversight_hooks`) and `django_ai_sdk.oversight.plugin.hooks` (`OversightHook`, `GateQuestionHook`, `StopClosedTurnHook`, `LimitToolChoiceHook`, `JudgeToolCallHook`, `MaskToolResultHook`, `RecordUsageHook`, `JudgeAnswerHook`). Apps use `apply_oversight` and `run_with_oversight` instead. A judge that asks another judge imports `ask_judge` and `build_judge` from `django_ai_sdk.oversight.judges.base`. `ask_judge` asks one judge with its timeout, and `build_judge` builds a judge from an alias or a spec.
:::

## Installing

Add `"django_ai_sdk.oversight"` to `INSTALLED_APPS` and run `migrate`. The app needs no extra. It keeps content in the clear, masked as `CAPTURE` says.

:::caution[Without the app]
Without the app, nothing checks a turn. The core check `ai_sdk.E001` is an Error when an agent class still asks for a policy, by setting `oversight` or overriding `get_oversight()`, or when `AI_SDK_OVERSIGHT["DEFAULT_POLICY"]` is set. A missing app therefore never leaves those agents unguarded without a warning.
:::

:badge[Optional extra]{variant="secondary"} The `typesafe` extra adds the client for the optional `TypeSafe` judge, a hosted commercial service:

::install[django-ai-sdk[typesafe]]{tools="uv pip"}

## Settings

`AI_SDK_OVERSIGHT` is one dict, and every key is optional (`django_ai_sdk.oversight.conf.setting`). A dict value merges one level deep with its default, so `{"DEADLINE_MS": {"gate": 3000}}` keeps the other stages' deadlines. The SDK checks every key. An unknown key, or a value of the wrong kind, is an error (`ai_sdk.oversight.E012`) that names the key and what it accepts, such as `AI_SDK_OVERSIGHT['FAIL_MODE'] must be one of 'closed', 'open'.` The defaults are `django_ai_sdk.oversight.conf.DEFAULTS`.

| Key | Default | Meaning |
| --- | --- | --- |
| `ENABLED` | `True` | Kill switch: off leaves every agent unguarded, whatever its policy says. |
| `DEFAULT_POLICY` | `""` | The policy of agents whose `oversight` is `None`, and of runtime agents whose base class names none. Empty: they are unguarded. |
| `CAPTURE` | `"masked"` | The most content the record keeps (`none`, `metadata`, `masked`, `full`); a policy may keep less, never more. |
| `STREAM_MODE` | `"auto"` | `live` streams the answer as it is written; `auto` holds it until finish passed it only while an enforced rule may mask or withhold it there, decided per turn. A policy may set its own. |
| `JUDGES` | `{}` | Named judges: `{"alias": {"class": ..., **kwargs}}`, a `Judge` instance, or the name of another alias. A class name without a dot is one of `django_ai_sdk.oversight.judges` (`"LLMJudge"`); otherwise it is a dotted path. |
| `DEADLINE_MS` | `{"gate": 1000, "tool": 1000, "finish": 2000}` | Milliseconds each inline stage may add: the gate before the model starts, the tool stage before each tool call (and masking each tool result), finish once the answer is complete. `None`: no limit. Room for patterns, checks and a fast classifier; a model judge (`LLMJudge`) usually needs longer. A judge still running at the deadline is abandoned: the lens fails (`FAIL_MODE`), keeps the answer it has for the record, and reflect asks again later. |
| `MAX_INPUT_CHARS` | `100000` | A longer question is blocked at the gate, unread. With history the server did not store (`gate_history`), the question and the history judged with it count together. `None`: no limit. |
| `MAX_OTHER_TEXT_CHUNKS` | `10` | A tool result, source or earlier message longer than a lens's judge reads (its `TEXT_CHARS`) is judged in chunks that judge reads, at most this many; a longer text is too long, as without chunking (`FAIL_MODE`). |
| `CLIENT_DETAIL` | `"actions"` | What `data-oversight` tells the chat client. `actions`: the stage, outcome and message when an enforced rule acted in a way the person sees (block, restrict_tools, mask) or its notice tells them, no rule names. `full`: rule names and every stage's answers too, for a developer view, each value as the record keeps it (see [Models](#models)). |
| `FAIL_MODE` | `"closed"` | What a failed lens means: its check or judge raised, timed out, missed the deadline or was handed more than it reads (too long). `closed` treats an enforced rule that blocks, masks or takes tools away as matched when a lens it reads failed and the lenses that did answer leave it undecided (an `all` with one test already false is decided), withholds a tool result it cannot check, and blocks a turn whose policy cannot be chosen or loaded. A rule matched on a failure only blocks, masks or takes tools away: it never alerts, queues or notifies, so an outage does not flood the review queue. Rules in report mode only record it. `open` lets the turn through and records the failure, and runs a turn whose policy cannot load unguarded. A policy may set its own (`fail_mode`); anything but `"open"` is closed. |
| `RETENTION` | `{"content": 90, "metadata": 730}` | Days a turn's content (`TurnContent`, with the reasons kept in it, and reviewers' notes) and its metadata (the turn, its answers, matches and review items) are kept, as each row's `expires_at`. A policy may set either. `oversight_prune` deletes expired rows ([Retention](#retention)). |
| `NOTIFIERS` | `[]` | Where notifying matches are sent: instances, class names, or `{"class": ..., **kwargs}`. As in `JUDGES`, a class name without a dot is one of `django_ai_sdk.oversight.notifiers` (`"LogNotifier"`); otherwise it is a dotted path. One that cannot be built is logged and left out. |
| `RUNTIME_CACHE_SECONDS` | `10` | How long each process caches [authored policies](../oversight/oversight-authoring.md); `0` reads them every turn. |
| `UNGUARDED_MODEL_CALLS` | `"refuse"` | A model called through the SDK's generators where a turn is required (every workflow step) but outside one: `refuse` raises `UnguardedModelCall` before the call, `warn` logs it once per place, `allow` lets it be. A step calls its model through `Agent.run()` or `self.turn()` (`django_ai_sdk.oversight.turns`). |

Outside the dict:

| Setting | Default | Meaning |
| --- | --- | --- |
| `AI_SDK_MODEL_PRICES` | `{}` | Prices the judges' answers (`Answer.cost`) ([Settings](settings.md)). |
| `TYPESAFE_API_KEY` | unset | The `TypeSafe` judge's key; without it that judge abstains (`ai_sdk.oversight.W005`). |
| `TYPESAFE_MODEL` | unset | A TypeSafe model to pin; unset leaves the choice to `typesafe-sdk`. `TypeSafe(model=...)` overrides it. |

## Agents and runs

| On `Agent` | Meaning |
| --- | --- |
| `oversight` | A `Policy` subclass (registered when the agent class is defined), a registered policy's name, `None` (the default: `DEFAULT_POLICY`) or `False` (no oversight). Anything else (an instance, `True`, `""`) raises `ImproperlyConfigured` when the class is defined. |
| `async get_oversight(*, channel, user, **kwargs)` | The policy for this turn; returns `oversight` unless overridden. It may read the database; keep `**kwargs`, as more about the turn may be passed later. Channels: `chat`, `api`, `subagent`, `workflow:<workflow>:<step>`, `workflow:adhoc:<step>` (a submitted definition), `dry_run`, a name the app passed, or `default` (`DEFAULT_CHANNEL`) when the caller passed none; `channel` is never `None`. A policy only this returns is registered when a turn first uses it: `@register_policy` it, so the checks see it. |
| `run(messages, ..., *, channel=None, oversight=None)` | One turn under the policy, on `channel` (a name the app picks; `None` records `default`). `None` is the agent's own; a class or a name replaces it for this call; `False` runs unguarded (trusted code only). Raises `Blocked`. |
| `as_view(...)` | Chat: one turn on channel `chat`. A block streams the policy's message. |

The SDK picks a turn's policy from this order, and the first match wins: `run(oversight=...)`, then `get_oversight()`, then `DEFAULT_POLICY` when `get_oversight()` returned `None`. `False` anywhere in that order means no oversight. A runtime agent, an `AgentSettings` row, without an assignment of its own takes its base class's `oversight` (`AI_SDK_RUNTIME_AGENT_BASES`), then `DEFAULT_POLICY`. A hook that raises, or a policy that cannot load, blocks the turn unless `FAIL_MODE` is `"open"`.

`django_ai_sdk.errors.Blocked(message, *, stage="", turn_id="")` has error code `blocked`. Like `content_filtered`, it becomes HTTP 400 through Ninja's `register_error_handlers` and DRF's `django_ai_sdk.contrib.drf.exception_handler`, with `message` as the body's message. `Agent.run()` raises it when the gate blocks the history or the question, when finish withholds the answer, and when a masked structured result no longer validates. `message` is the policy's text for the person asking. The run endpoints answer `{"result", "thread_id", "blocked"}` instead, and `thread_id` is `""` for the stateless one.

Each agent runs under its own policy. A run inside an active turn joins that turn when it resolves the same policy, or none. It then has no gate or finish of its own. The outer turn judges its question like a tool result, and masks it or withholds it whole. Its tool calls are checked and recorded in the outer turn. Where a turn is required, its model calls count as overseen once the outer gate ran.

Under another policy, its own or an explicit `oversight=`, the run opens a child turn with its own gate, tool checks, finish and record. `parent_turn_id` names the outer turn, and the child turn keeps the outer turn's user and the tools the outer turn took away. A subagent's delegation does the same with its task as the question. In a child turn, the subagent reads only that checked task, not the coordinator's conversation, and a block message is the tool result the coordinator reads. A delegation run synchronously (Haystack's `run()`) cannot open a turn, so when it would need one it answers `UNGUARDABLE` instead. `run(oversight=False)` runs unguarded and does not join. No run joins a closed turn, even a task that copied it. That task's tool calls are blocked and its tool results withheld.

`run()` judges copies of what the caller sent. It masks only the last user message of the list in place, so the caller stores the masked copy. History the server stored itself was judged on its own turn and is not judged again. With a `thread_id` whose thread is stored, under a policy whose gate masks, the stored messages are the history. The thread run endpoints work this way, as chat does. A history that matches the stored thread stays. `TurnGuard.gate_history()` judges any other history before the question.

The checks are Haystack hooks on the run (`django_ai_sdk.oversight.plugin`). A subclass's own `run()` that calls `super().run()` is checked. One that calls its own model is not, so run that model call through `run_with_oversight()`. `django_ai_sdk.oversight.history` holds the stored-thread rules that `run()` and chat share.

## Lenses

Subclass one of the three kinds, or construct one with keyword arguments. An unknown keyword raises `ImproperlyConfigured`. The docstring of `oversight/lenses.py` states the rules lenses follow.

Every lens:

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | required | Key within the policy. Answers key on it, so renaming starts a new series. |
| `label` | `""` | A readable name; defaults to the name. |
| `answer` | per kind | `Flag`, `Choice`, `Score` or `Entities` (below), or its name in JSON. |
| `subject` | `Subject.QUESTION` | `input`, `output`, `exchange`, `sources`, `tool_call`. |
| `stages` | by subject | Any of `gate`, `tool`, `finish`, `reflect`. Empty: where the subject is first there to judge (`lenses.DEFAULT_STAGES`): the question at `gate`, a tool call at `tool`, the answer, the exchange and the sources at `finish`. Gate reads the question, `tool` tool calls, finish the answer, the exchange and the sources, reflect all but tool calls (`lenses.STAGE_SUBJECTS`). |
| `sample` | `1.0` | Share of turns the reflect stage judges. Inline stages never sample. |
| `enabled` | `True` | Off: the lens never runs. |
| `needs_grounding` | `False` | Only asked when the turn has sources or tool results to judge against. |
| `judges_tool_results` | `True` | Whether the question can be asked of a tool result. A gate lens that a block or `restrict_tools` rule reads also judges every tool result (a rule in report mode records what it would have done to it); `False` keeps it to the question, for a lens about the person writing. A lens an enforced `Mask` rule reads masks tool results either way. |

Each kind adds its own:

| Kind | Answers | Own attributes |
| --- | --- | --- |
| `PatternLens` | `Entities` only, with spans | `patterns` (label to regexes; a plain list labels hits with the lens's name), `keywords` (label to whole words), `verify` (label to a check a match must pass, such as a checksum), `flags` (`re.IGNORECASE`, for patterns and keywords). Each stretch of text gets one label, the first pattern's. What patterns find is a fact: hit and miss both have confidence `1`. |
| `CheckLens` | `Flag` by default | `validators`: functions `fn(text)` or `fn(text, context)`, or names registered with `register_validator`; each raises `Violation(message)` to say yes, and the messages are the answer's reason. `check(text, context)` runs them; override it to answer another type, returning the value, an `Answer` (with a reason or confidence) or `None` to abstain. A sync `check` runs in a thread. `timeout` (`30` s). |
| `JudgeLens` | `Flag` by default | `judges`: tried in order (instances, aliases, optional aliases such as `Maybe("judge")`, or `{"class": ..., **kwargs}`); `min_confidence` (`0.75`: an answer below it asks the next judge); `question` (asked verbatim); `guidance` (scale anchors, edge cases); `reads_question` (`True`: a tool-call lens also reads the question that led to the call). |

| Answer | Value | Options, on the lens |
| --- | --- | --- |
| `Flag` | `bool` | `yes`, `no`: what each means |
| `Choice` | one of `labels` | `labels`, `criteria` (label to meaning), `default` (a label; what patterns that find none mean, the last label when unset) |
| `Score` | `0..1` | `low`, `middle`, `high`: what the ends and the middle mean |
| `Entities` | sorted list of labels, with spans | `labels` (when set, other values are dropped); `per_source` (a judge that rates source by source answers with the sources that hold, `items` each one's weight) |

An option the lens leaves out takes the default in `<Answer>.options`. The SDK refuses a `Choice` whose `default` is set but is not one of its `labels`. Only an `Entities` lens says where in the text it found something, so only it can drive a `Mask`. On an `Entities` answer, `L(name).is_true` means it found something, and `is_false` that it found nothing.

`lens_from_dict(node)` builds a lens from JSON, in one of two forms. `{"kind": "pattern" | "check" | "judge", "answer": "flag" | "choice" | "score" | "entities", ...attributes}` builds a new lens, and `{"preset": "pii", ...}` builds a built-in one. A check lens in JSON names registered validators only, so JSON never names code to run.

### Built-in lenses

Each has a default `name`; subclass it to name your own.

| Class | Kind | Answers |
| --- | --- | --- |
| `PIILens` | pattern | Personal data. `kinds` defaults to `patterns.DEFAULT_PII_KINDS`: `email`, `iban` (mod-97), `credit_card` (Luhn), `phone` (international format), `ip_address`. A host adds its own kinds with `register_pii_kind`, then names them. `patterns` adds your own, unchecked. See [Your own patterns](../oversight/oversight.md#your-own-patterns-countries-and-languages). |
| `SecretsLens` | pattern | `api_key`, `stripe_key`, `google_api_key`, `github_token`, `aws_key`, `slack_token`, `jwt`, `private_key`, `password_assignment` (`patterns.SECRET_PATTERNS`). |
| `RefusalLens` | judge | Answers that decline: `PatternJudge(REFUSAL_PATTERNS)` (English), then `Maybe("classifier")`. |
| `LengthLens` | check | A `Flag`: the subject is longer than `max_chars`. |
| `LanguageLens` | check | A `Flag`: answered in another language than asked. Abstains when unsure. For the subject's language as a `Choice`, write a check lens over `lenses.guess_language`. |
| `EmptyRetrievalLens`, `ToolErrorsLens` | check | A tool call that found no sources (a `Flag`); the share of tool calls that failed (a `Score`: `L("tool_errors") > 0` when any did). |

The regex tables, checksums and the masking helpers (`mask`, `mask_all`, `mask_secrets`) are in `oversight/patterns.py`. Import the tables and helpers from `django_ai_sdk.oversight`.

## Judges

A judge implements `answer(lens, context) -> Answer | None`. `context` is the `TurnContext` the lens reads: `input`, `output`, `history`, `sources`, `tool_call`, the `answers` so far, and `text(subject)`. Returning `None` abstains. A judge whose `missing()` says why it cannot run also abstains. A lens fails with `no judge can run here` when none of its judges can run or answer its kind of question (`answers(lens)`; a classifier answers no free `Entities` lens), and `FAIL_MODE` decides. An answer with neither a value nor an `error` counts as `no verdict`, so `FAIL_MODE` applies. An `Answer` has `value`, `confidence`, `rationale`, `spans`, `items`, `trail`, `error`, `cost`, `tokens`, `latency_ms`, `answerer` (who answered: a judge's name, `patterns` or `check`) and `model`. Every judge also takes `name=`, the name its answers are recorded under, and `timeout=`, the seconds before its answer is abandoned. The package exports the judges. Helpers such as `base.resolve` stay in their modules.

Local judges have `external = ""`, so nothing leaves the process:

| Class | Answers |
| --- | --- |
| `PatternJudge(patterns=(), *, keywords=(), verify=None, flags=re.IGNORECASE, miss_confidence=0.5, timeout=None)` | Regexes and keywords, as a pattern lens reads them. A hit is sure; a miss is `miss_confidence` sure, by default below `min_confidence`, so the next judge is asked. A miss at or above it ends the lens, and the judges after it are never asked. |
| `LocalClassifier(model=ENGLISH, *, file="onnx/model_quantized.onnx", positive="", timeout=None)` | An ONNX model from the Hugging Face Hub, run by onnxruntime in this process (the `local` extra, no torch). Zero-shot by default: yes/no (entails the lens's `yes`), label (the criterion it entails most, as "This text is about <criterion>." unless it starts with "This text") and score lenses, about 20 ms each on a CPU. `ENGLISH` (87 MB) and `MULTILINGUAL` (340 MB) are MIT-licensed NLI models; `positive=` makes a model trained for one question (a prompt-injection detector) answer yes/no lenses. Reads one text, the question or the answer, of at most 1,600 characters: lenses on sources or tool calls, lenses that need grounding and "taken from" go to the next judge. Until its model is loaded it answers `unavailable` and loads it in the background, so the next judge is asked, `FAIL_MODE` applies and reflect asks again; a failed load is logged and tried again a minute later. `judges.local.load()` loads it at startup. |

Remote judges set `external` to who receives the turn's content. Secrets are masked first:

| Class | Answers with |
| --- | --- |
| `LLMJudge(model=None, *, llm=None, base_url="", api_key_env="", generation_kwargs=None, timeout=None)` | Any chat model. Subclass it and set `instructions` for another system prompt. `base_url` and `api_key_env` name a server of your own: the judge then speaks Chat Completions (Ollama, vLLM, LM Studio, llama.cpp), and abstains while the key's variable is unset. Without them, the default gateway (OpenAI Responses, `OPENAI_API_URL`). `llm`: an SDK generator factory or its dotted path, for another provider. `external = "model gateway"`, 45 s timeout. Structured output where the generator supports it, parsed JSON where not. |
| `TypeSafe(model=None, *, timeout=None)` | TypeSafe, a hosted commercial classification service. `external = "TypeSafe"`, 20 s timeout. Needs the `typesafe` extra and `TYPESAFE_API_KEY`, else abstains (`ai_sdk.oversight.W005`). Answers `Flag`, `Choice` and `Score` lenses, and an `Entities` lens with `per_source = True` by asking one yes/no per source: the value is the titles of the sources that answer yes, `items` each source's probability. One stage's questions about the same turn go out as one request. A failed request is an `unavailable` error, so `FAIL_MODE` applies and reflect asks again. |

`TypeSafe` is a `Classifier` (`judges.classifier`), the base for any model that answers typed questions. See the guide's [Your own judge](../oversight/oversight.md#your-own-judge).

The guide's [Where turn content goes](../oversight/oversight.md#where-turn-content-goes) lists what each remote judge sends, per lens subject and with every limit.

A judge lens asks its judges in order until one is sure, which means its confidence reaches the lens's `min_confidence`. When none is sure, the last one that answered decides. Confidences from different judges are on different scales, so the SDK never compares them. The answer keeps every other answer in its `trail`. A chain that ends on an error or the deadline after only unsure answers counts as failed (`FAIL_MODE`), and keeps the unsure answer for the record.

A lens that judges the answer against its grounding reads the turn's retrieved sources *and* its tool results, and is not asked on a turn that has neither. Such a lens has subject `sources`, or `needs_grounding = True` like `groundedness`. A lens on the answer also reads the sources that earlier turns of the thread found, after this turn's. Reflect asks no lens on the answer of a turn the gate blocked. Reflect reads the turn from its content. Under `masked` capture it reads the question and answer from the thread's stored messages instead, where there are any, so its masks reach the stored answer.

## Rules and conditions

```python
Rule(name, *, when=Condition, then=Action | [Action], mode=Mode.REPORT,
     message="", notice="", description="")
```

Conditions are built from `L(lens)`: `.is_true`, `.is_false`, `== value`, `!= value`, `> < >= <=`, `.in_(values)`, `.contains(label)`, `.missing`, `.present`, `.confident(x)`, `.unsure(x)`, `.disputed` (two models asked for the lens answered differently; a yes/no or a label lens only), combined with `&`, `|` and `~`. Each serializes to JSON (`{"all": [...]}`, `{"any": [...]}`, `{"not": ...}`, `{"lens", "op", "value"}`). Every comparison on a lens without a usable answer is false. Use `.missing` to test for that. A failed lens, whose check or judge raised or timed out, is unknown, not false. A comparison on it is neither true nor false, so `~L("grounded").is_true` does not match only because "grounded" failed.

A rule is evaluated once every lens it reads has had its chance, so it can combine a gate lens and a reflect lens. A rule fires once per turn. `message` replaces the action's message when the rule blocks.

Finish judges the whole answer. Nothing judges it while it streams. A tool call rule is judged on every call, so a blocked call stays blocked when the model retries it, and the record keeps it once. A tool call's answer belongs to that call and is gone afterwards. A rule therefore cannot read a tool call lens next to a lens judged at finish or reflect. Give the tool call its own rule, or read what the turn's tools did in a check lens on the exchange. `RestrictTools` at the tool stage blocks the calls it names (all, when it names none), and those tools stay away for the rest of the turn.

| Action | Enforced at | Outcome |
| --- | --- | --- |
| `Block(message="")` | gate, tool, finish | `blocked` |
| `RestrictTools(names=None)` (none = all) | gate, tool | `restricted` |
| `Mask(lenses=None, template="[{label}]")` | gate, finish, reflect | `masked` |
| `Queue(queue="default")`, `Notify(to=None, severity="warning")`, `Alert(severity="warning")` | any | `flagged` |

`Alert` and `Notify` both mark the turn at their severity, and `Notify` also sends the match to the notifiers. While a rule's review is still open on an earlier turn of the same thread, a new match of that rule is recorded and queued but sends nothing to the notifiers. Once the review is resolved, the next match notifies again. When the model sends a blocked tool call again unchanged, it gets the same refusal at once. A changed call, such as a note without the part that was refused, is judged again. A severity is `info`, `warning`, `high` or `critical` (`Severity`, or its string). A blocked turn is at least `high`. `Block.DEFAULT_MESSAGE` is "This request can't be answered here."

`Mask` masks the spans of `lenses`, by default every lens the rule reads. It replaces each span with `template`, with the label in capitals (`[EMAIL]`). Every lens it masks must be an `Entities` lens on the question or the answer, so `input`, `output` or `exchange`. Otherwise the policy is refused when it is built. `sources` cannot be masked, and neither can `tool_call`: arguments are judged and never rewritten, so block the call or take the tool away instead. The gate's mask rules mask tool results.

:::caution
An enforced `Mask` that cannot point at what it should mask blocks instead. That happens when a lens it reads failed while the policy fails closed, or answered without spans.
:::

Reflect masks only the stored message, and only when the stored text is the text it judged. It appends a `reflect` notice to the message.

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

`@register_policy` registers a policy, in any module. The SDK imports an installed app's `oversight.py` on startup. Naming the class on an agent (`oversight = SupportPolicy`) or in `run(oversight=...)` also registers it. A policy that fails validation is not registered (`ai_sdk.oversight.E001`), and its agents' turns are blocked. Two classes that declare one name are an error (`ai_sdk.oversight.E010`), and the first one registered keeps the name.

`LensPack(lenses, *names, **overrides)` and `RulePack(rules, *names, mode=None, notice=None)` take single lenses or rules out of a pack instead of extending the whole pack, for example `LensPack(SecurityPack.lenses, "injection_question")`, `RulePack(SecurityPack.rules, "injection_in_question", mode=Mode.ENFORCE)`. With no names they take every one. They return copies in the order named, so `overrides` (lens attributes such as `stages`), `mode` and `notice` leave the pack's own lenses and rules unchanged. An unknown name, or an override a lens does not have, raises `ImproperlyConfigured`. When a policy class is built, it flattens its `lenses` and `rules` one level. A pack, or a list such as another policy's `rules`, puts its items in its place.

`ResolvedPolicy.sources` gives, for each lens (`lens:<name>`), rule (`rule:<name>`) and policy option (`policy.capture`, ...), the name of the policy that declared it. The extended policies come first and the class wins. A policy's `capture` capped by the deployment's `CAPTURE` says `settings`. A resolved policy is cached until settings change or a policy registers. `invalidate_policy(name)` clears it and every policy that extends it.

An app can register whole policies at runtime. It adds a loader with `add_policy_loader(loader)`. The loader is an `async` callable that registers policies (`register_policy`) and removes them (`unregister_policy`). The SDK awaits it before policy lookups at most once per `RUNTIME_CACHE_SECONDS`, and again after `invalidate_policy()`. The [authoring app](../oversight/oversight-authoring.md) is one such loader. A policy registered this way is built and checked like a policy in code. A loader is code and is trusted like code. Only the authoring app adds its own limits on what an authored policy may change.

A policy that fails to load, for example because of a missing parent or a cycle, blocks its agents' turns. Only `FAIL_MODE = "open"` leaves them unguarded. The reason is logged and kept in `registry.broken_reason(name)`.

## Packs

Every pack rule is in report mode. A lens answered only by an optional alias is skipped where that alias is undefined.

| Pack | Lens | Kind, answer, subject | Stage | Answered by |
| --- | --- | --- | --- | --- |
| `security` | `injection_question` | judge, Flag, input | gate | `PatternJudge(INJECTION_PATTERNS)` (overrides, requests for the hidden prompt, role takeovers, fake delimiters; a hit is sure, a miss asks on), `Maybe("classifier")` |
| | `pii_question` | pattern, Entities, input | gate | `PIILens` |
| | `secrets_question` | pattern, Entities, input | gate | `SecretsLens` |
| | `secrets_tool_call` | pattern, Entities, tool call | tool | `SecretsLens` |
| | `pii_tool_call` | pattern, Entities, tool call | tool | `PIILens` |
| | `pii_answer` | pattern, Entities, output | finish | `PIILens` |
| | `exfiltration_answer` | pattern, Entities, output | finish | `EXFILTRATION_PATTERNS`: Markdown images, links and reference links, and HTML images, that carry data in their URL's query |
| `safety` | `self_harm_question` | judge, Flag, input | gate | `Maybe("classifier")` |
| | `harm_question` | judge, Score, input | gate | `Maybe("classifier")` |
| | `harm_answer` | judge, Score, output | finish | `Maybe("classifier")`, `Maybe("judge")` |
| `quality` | `context_relevance` | judge, Score, sources | reflect | `Maybe("classifier")` |
| | `groundedness` | judge, Score, output, needs grounding | reflect | `Maybe("classifier")` |
| | `taken_from` | judge, Entities, output, `per_source` | reflect | `Maybe("classifier")` |
| | `answer_relevance` | judge, Score, exchange | reflect | `Maybe("classifier")` |
| | `helpfulness` | judge, Score, exchange | reflect | `Maybe("classifier")`, `Maybe("judge")` |
| | `refusal` | judge, Flag, output | finish | `PatternJudge(REFUSAL_PATTERNS)` (a hit is sure, a miss asks on), `Maybe("classifier")` |
| | `language` | check, Flag, exchange | finish | `LanguageLens` |
| | `empty_retrieval` | check, Flag, sources | finish | `EmptyRetrievalLens` |
| | `tool_errors` | check, Score, exchange | finish | `ToolErrorsLens` |

The packs have these rules. `security` has `injection_in_question`, `secrets_in_question`, `pii_in_question`, `secrets_in_tool_call`, `pii_in_answer` and `exfiltration_in_answer`; `safety` has `person_at_risk_in_question` (never blocks, but queues for review and notifies at `critical`), `harm_in_question` and `harm_in_answer`; `quality` has `ungrounded_answer`, `unsourced_answer`, `irrelevant_context`, `empty_retrieval` and `tool_failures`. `default` extends all three.

## The guard

Every turn passes through a `TurnGuard`. `Agent.run()`, chat and the SDK's entry points open it for you. Open one yourself only for a turn the SDK does not run.

`await TurnGuard.start(agent, *, channel=None, user=None, thread_id=None, messages=None, input=None, oversight=None)` returns an inert guard unless a policy applies. Every call on an inert guard returns at once and allows everything. `oversight` replaces the agent's own as in `run()`, and `False` opens no guard. The guard then has these methods:

::::trace[A turn through its guard]
:::step[gate_history(earlier, question=None)]{icon="code"}
`gate_history(earlier, question=None)` takes history the server did not store, and the question for its attachments, and returns a `StageResult`.
:::

:::step[gate()]{icon="code"}
:::

:::step[add_text(chunk)]{icon="code"}
`add_text(chunk)` takes streamed answer text.
:::

:::step[tool_call(call), mask_result(call_id, text), set_sources(...), add_tool_call(...), add_usage(...)]{icon="wrench"}
:::

:::step[finish(output)]{icon="code"}
:::

:::step[close(error="")]{icon="code"}
`close(error="")` writes the turn and never raises.
:::
::::

When `gate()` ends, it writes the turn ahead as open (`write_ahead()`), and so does `gate_history()` when it blocks. `close` updates that row.

`unguarded()` suppresses guards inside a block, and the SDK's judges and dry runs run under it. `oversight.guard.guarding(guard)` makes a guard visible to `get_tools()`, the tool hooks and nested runs. `current_guard()` returns that guard, or `None` once it is closed. The oversight plugin has `JudgeToolCallHook`, which runs before each tool call, and `MaskToolResultHook`, which runs after it. It also has `GateQuestionHook`, `StopClosedTurnHook`, `LimitToolChoiceHook`, `RecordUsageHook` and `JudgeAnswerHook`, all subclasses of `OversightHook`. `run_with_oversight` adds them to every Haystack Agent the SDK runs, so every run, subagents included, reports to its turn. Tool results that are not text, such as images and files, pass unchanged.

A headless `agent.run(...)` reports its tool calls, their results and its usage to the guard, so finish judges what the loop did, with or without tools.

## Client notices

`data-oversight` stream parts (`DataEvent(data_type="oversight")`):

| Key | When |
| --- | --- |
| `turn_id`, `stage`, `outcome` | Always. `outcome` is `blocked`, `restricted`, `masked`, `flagged` or `allowed`. |
| `message` | The turn was blocked: the policy's text. |
| `replace` | Finish replaced the answer: the text to show instead. |
| `input` | The gate masked the question: the masked text. |
| `notes` | Enforced rules with a `notice` that acted, once a turn each: `kind` (the first of their actions by `rules.ACTION_ORDER` in `rules.USER_FACING`: `block`, `restrict_tools` or `mask`; else `"notice"`, when it only queued, alerted or notified), `also` (the other user-facing ones), `text`, `stage`. |
| `rules`, `detail` | Only with `CLIENT_DETAIL` `full`: rule names and every stage's answers. Each answer's `value` is the record's ([`LensAnswer`](#models)), so the client never reads what a mask took out. |

A blocked question or withheld answer ends with finish reason `content-filter` (`oversight.stream.CONTENT_FILTER`). `OpenAIProtocolHandler` sends `content_filter` and no `data-*` parts. The SDK stores the notices on the message under `metadata["oversight"]`, with `replace` reduced to `"replaced": true`.

## Models

| Model | Holds |
| --- | --- |
| `Turn` | `status` (`open` from the end of the gate until it closes, `closed`, or `interrupted` when `oversight_sweep` found it never closed), channel, agent, agent fingerprint, policy, thread and message ids (no foreign keys), `actor_id` (the user's primary key, `""` without one), model, tokens, cost, latency, `outcome`, `report_outcome` (what report-mode rules would have made it; both blank while the turn is open, and when it closed on an error of its run with no rule acting), `error`; labels (the lens answers' values, below) and severity (denormalized); `expires_at`, when the policy's metadata retention runs out; `content_forgotten`, set when the turn was forgotten before it closed (open, or interrupted and still able to close), so it stores no content when it closes. |
| `TurnContent` | The content at the policy's capture level (`stored`), with an expiry: past it, the content reads as none (`load_content`) and reflect adds nothing to it. Beside it: `texts`, by answer id, what each answer said in words (its reason, the names a judge gave for an `Entities` lens, its per-item scores by key, the same of the answers on its trail). Reflect adds to it, and it goes when the content goes. |
| `LensAnswer` | One answer to one lens: answerer (who answered), model, value, confidence, spans (offsets and labels), trail, error, cost, latency. Nothing that can quote the turn: an `Entities` value keeps a label the lens declares, or a kind its patterns report (a pattern lens, or a `PatternJudge` judge), while any other name (a judge's, `TypeSafe`'s source titles, a person's) is `"[found]"`, one per name, so `.is_true` and counts still hold. The words are in the content's `texts`. |
| `RuleMatch` | A rule that matched: stage, mode, actions, whether it acted, the condition. Inline a rule may match more than once (the history and the question, each tool call); after close it matches once: reflect locks the turn before it looks. |
| `ReviewItem` | A turn a rule queued for a person, per queue and rule, created by a `Queue` action. When closed it holds who closed it (`resolved_by_id`, a user id), whether they agreed with the rule (`agree`) and their `note`. The note is content: forgetting the thread, or the content expiring, blanks it. Core creates items; the [review API](#review-api) closes them. |

The Django admin registers `Turn`, `LensAnswer`, `RuleMatch` and `ReviewItem` read-only, without the content, reviewers' notes, or who closed an item. It needs only metrics access: Django's view permission and `VIEW_OVERSIGHT_METRICS` from the `oversight` permission domain. A turn in the admin therefore also leaves out who asked and its thread and message ids. The metrics scope (`scope_turns`) narrows turns and their rows, as in the API.

## Review API

The review API is an opt-in layer above core. Core never imports it, and nothing changes when it is not mounted. `django_ai_sdk.oversight.services.OversightService` is the API, and `get_oversight_router()` serves it over django-ninja. This release has no Django REST framework views for it. A DRF project calls `OversightService` from views of its own, and the service still applies its permissions.

```python
from django_ai_sdk.contrib.ninja import get_oversight_router

api.add_router("/", get_oversight_router())
```

The router signs readers in with django-ninja's `django_auth`, which uses the session, with Django's CSRF check on every POST. It does not take the API's auth. An API that signs in another way passes its own, such as `get_oversight_router(auth=MyAuth())`.

| Route | Service | Operation |
| --- | --- | --- |
| `GET /oversight/turns/` | `list_turns(filters, user=, limit=, cursor=)` | metrics |
| `GET /oversight/turns/{id}/` | `get_turn(turn_id, user=)` | metrics; the content takes content |
| `GET /oversight/reviews/` | `list_reviews(user=, queue=, status=, limit=, cursor=)` | review |
| `GET /oversight/queues/` | `list_queues(user=)` | review |
| `POST /oversight/reviews/{id}/` | `resolve_review(item_id, verdict, user=)` | review |
| `GET /oversight/metrics/` | `get_metrics(user=, days=, agent_id=, policy=)` | metrics |

The turns feed filters on `outcome`, `report_outcome` (turns a report-mode rule would have changed), `status`, `agent_id`, `channel` (a prefix), `policy`, `severity`, `label` with an optional `value`, `rule`, `review` (`open`, `reviewed`, `unreviewed`), `thread_id`, `since` and `until`. `thread_id` is content, so it takes `VIEW_OVERSIGHT_CONTENT`, answers 403 without it, and reaches only turns in the content scope. Lists are newest first, except reviews, which are oldest first. Lists page by cursor: pass a page's `next_cursor` back as `cursor`. There is no total. On Postgres a GIN index on the labels answers the label filter.

Closing an item takes `ResolveIn` with `agree`, `note` and `dismiss`. It either says whether the rule that queued the turn was right (`agree` true or false), or dismisses the item (`dismiss`). The request must set exactly one of the two, or it gets 400. The item keeps the note as long as the turn keeps its content. A turn whose content was forgotten, expired, or never captured (`capture = "none"`) keeps no note. Closing claims the item with a conditional update, so a second close answers 409. Nobody reviews a turn they asked, which is a turn whose `actor_id` is theirs. A child turn carries the actor of the outer turn, however deep. Closing or dismissing such a turn answers 403. A forgotten user's turns carry no actor, so this rule cannot hold for them. A verdict changes nothing else: no label, no rule match and no notification.


Permissions are the `oversight` domain of `AI_SDK_PERMISSIONS`. The default, `django_ai_sdk.oversight.permissions.OversightDefaultPermission`, takes a Django model permission per operation and grants nothing for `is_staff`:

| Operation | Model permission | Reads |
| --- | --- | --- |
| `VIEW_OVERSIGHT_METRICS` | `django_ai_sdk_oversight.view_turn` | Turns without their content, answers' values, rule matches, counts. |
| `VIEW_OVERSIGHT_CONTENT` | `django_ai_sdk_oversight.view_turncontent` | On top of metrics: the turn's content, the preview, who asked, thread and message ids, answers' reasons, spans and words, reviewers' notes. |
| `REVIEW_OVERSIGHT` | `django_ai_sdk_oversight.change_reviewitem` | The review queues, and closing an item with a verdict. The queue lists each item's turn row only to a reader who also holds metrics, and its content only to one who holds content. |
| `VIEW_OVERSIGHT_AUTHORING` | `django_ai_sdk_oversight_authoring.view_authoredpolicy` | The [policies defined at runtime](../oversight/oversight-authoring.md) and what a definition may use (the opt-in authoring app). |
| `AUTHOR_OVERSIGHT` | `django_ai_sdk_oversight_authoring.change_authoredpolicy` | Saving, toggling, trying and deleting a policy defined at runtime. |

A reviewer may hold only the review permission. They then work the queue from each item's queue and rule, without the turn's row, and close items without seeing the question. Grant metrics and content to a reviewer who should judge the turn itself.

Two hooks that a subclass overrides narrow which turns a reader reaches: `scope_turns(user, operation, queryset)` and `scope_reviews(user, operation, queryset)`. A turn or item outside them is not found (404), and a turn outside the content scope is shown without its content. Every read, a list or one turn, decides through these querysets. Narrow in the hooks or in `get_queryset_perms`, because the SDK does not consult a `has_object_permission` of your own for content. By default an item is in scope when its turn is:

```python
from django_ai_sdk.oversight.permissions import OversightDefaultPermission


class TeamOversight(OversightDefaultPermission):
    def scope_turns(self, user, operation, queryset):
        return queryset.filter(agent_id__in=agents_of(user))
```

An empty `AI_SDK_PERMISSIONS["oversight"]` falls back to the default, as for agents. To open the record to every user, set it to `["django_ai_sdk.permissions.AllowAll"]`.


Forgotten content reads back as gone. A turn whose thread or user was deleted, or whose content expired and was pruned, has `content_state` `"none"`, no preview, no answers' words and no reviewers' notes. A deleted user's turns have an empty `actor_id`. A pruned turn is not found. Content past its expiry and not yet pruned reads as `"expired"` on the turn, and as no preview and no notes in lists.

## Forgetting

`forget_turns(turns)` takes a queryset of turns and locks them. It deletes their `TurnContent` rows, and the turns stay with `content` cleared. It blanks reviewers' notes on them and sets `content_forgotten` on turns that are not closed yet (open, or interrupted), so they store no content when they close. `forget_thread(thread_id)` forgets every turn in a thread. Both run in the caller's transaction.

Deleting a thread calls `forget_thread`. Deleting a user forgets each turn they asked without a thread, 500 turns at a time, then clears `actor_id` on all their turns (`content.forget_actor`). A failure there is logged and leaves all of the user's turns as they were. [Forgetting](../oversight/oversight.md#forgetting) says what goes and what stays, and the [commands](#commands) run it again.

## Retention

`prune()`, run by `manage.py oversight_prune`, deletes what expired. Core stamps every row with `expires_at`, and nothing is deleted until the command runs. Schedule it daily with cron or your scheduler:

```bash
python manage.py oversight_prune --batch-size 500
```

It deletes `--batch-size` rows per transaction, so a backlog never holds one long lock. It deletes:

- Content rows past their `expires_at`, and with them what judges wrote there and reviewers' notes on the turn. The turn stays, with its content cleared.
- Turns past their `expires_at`, with their answers, matches, review items and content. Retention wins over a review that is still open, and the command reports how many there were.

The deploy check `W007` warns when rows whose `expires_at` passed more than two days ago still exist. It reads the database, so it runs only with `--database`:

```bash
python manage.py check --deploy --database default
```

## Testing

`dry_run(policy, *, question, answer="", tool_calls=None, user=None) -> DryRunResult` runs a policy over one exchange and records nothing. Pass the policy by its registered name, or as a `ResolvedPolicy` from `build_policy(name, policy_class=Draft)` to try a draft before registering it. The exchange goes through the gate, the tool calls (each checked as if the model sent it), finish and reflect. Reflect asks every reflect lens, whatever its sample. `DryRunResult` has `outcome`, `report_outcome`, `blocked_stage` (the stage that blocked, or `""`), `stages` (`StageResult`: `stage`, `answered`, `matched`, `blocked`, `message`, `text`, `changed`, `notice`, `restrict`, `tool`, and `answers` and `rules` by name) and `stage(name)`. An unknown or broken policy raises `LookupError`. The judges are real, so a dry run calls a model judge.

## Tasks

`django_ai_sdk.oversight.tasks` has two `django-tasks` tasks on the `default` queue. `reflect_turn` is queued on commit when the policy has reflect lenses or a lens is to be asked again. `send_notifications` sends a turn's notifications from the worker, so a slow receiver never holds a stream open. `django_ai_sdk.oversight.reflection` holds what reflect does.

`Turn.reflect_state` tracks reflection: `pending` when queued, `running` while a worker holds its claim, then `done` or `failed`. `run_reflection` claims the turn in one conditional update with a lease of `reflection.REFLECT_LEASE` (ten minutes) and counts `reflect_attempts`, so a duplicate task does nothing. It writes its answers, matches, the stored answer's mask and notices, and `done`, in one transaction and only while it still holds the lease. A run that fails gives the claim back (`pending`). After `reflection.REFLECT_ATTEMPTS` (three) the turn is `failed` and its `error` says `reflect failed`. A worker may claim a lease again once it ran out because its worker died.

## Commands

`oversight_sweep [--stale-after 30m] [--reflect-after 10m] [--limit 500] [--dry-run]` marks turns still `open` after `--stale-after` as `interrupted`, and queues `reflect_turn` again for turns `pending` longer than `--reflect-after` (their task was never queued) and turns `running` under a lease that ran out (their worker died), at most `--limit` of each, oldest first. A turn whose last attempt died is marked `failed`. Ages are a whole number with `s`, `m`, `h` or `d`. Schedule the command, and each run takes the next batch.

`oversight_forget_thread <thread id> [...]` forgets threads as their delete does. Use it for threads deleted without signals, such as by a queryset `.update()`, raw SQL or another database. It is safe to run again.

`oversight_forget_actor <user id> [...]` forgets users as their delete does. Use it for a delete whose forgetting failed and was logged. It is safe to run again.

`oversight_prune [--batch-size 500]` deletes expired rows ([Retention](#retention)).

## Notifiers

`AI_SDK_OVERSIGHT["NOTIFIERS"]` lists notifiers: `LogNotifier()` and `WebhookNotifier(url, *, secret="", timeout=5.0)` from `django_ai_sdk.oversight.notifiers`, or your own `Notifier` subclass that implements `async send(event)`. Give each as an instance, a class name, or `{"class": ..., **kwargs}`, such as `"LogNotifier"`, `{"class": "WebhookNotifier", "url": ...}` or a dotted path to your own. Every notifier takes `name`, by default its class name without `Notifier` in lowercase (`log`, `webhook`). `Notify(to=[...])` sends to those names only. A `Notify` action sends `{"id", "type": "oversight.match", "turn_id", "rule", "stage", "severity", "outcome", "agent", "channel"}`, with no turn content. `id` is stable (`notifiers.event_id(turn_id, rule, stage)`), so a repeated delivery carries the same one.

A `WebhookNotifier` sends the event's `id` as `webhook-id`. With a `secret`, it signs like [Standard Webhooks](https://www.standardwebhooks.com/), with the headers `webhook-timestamp` and `webhook-signature`. The signature is `v1,` followed by the base64 HMAC-SHA256 of `{id}.{timestamp}.{body}`.

:::details[Verify a signature]
To verify, recompute the HMAC over the raw body with the same secret, base64-decoded when it starts with `whsec_`, and compare in constant time.
Refuse timestamps more than a few minutes old and ids you have seen, so nobody can replay a captured request:

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
:::

A notifier's `send(event)` may raise. The error is logged and the turn goes on.

## System checks

| Id | When |
| --- | --- |
| `ai_sdk.oversight.E001` | A policy was not registered; the message says why (a lens without a name, a pattern lens without patterns, a check lens without validators, a judge lens without judges, an attribute its kind or answer type does not read, a duplicate lens or rule, a rule without `then`, a `Mask` over a lens that is not `Entities` or does not read the question or the answer, a `Choice` whose `default` is not a label, a lens at a stage that never reads its subject (the question at `tool` or `finish`, the answer at `gate` or `tool`, a tool call anywhere but `tool`), a `sample` or `min_confidence` outside 0 to 1). |
| `ai_sdk.oversight.E002` | A policy does not resolve (an unknown parent, a cycle, a rule or `Mask` naming a lens the policy does not have, an inherited lens a rule cannot use). |
| `ai_sdk.oversight.E004` | A judge lens names an alias `JUDGES` does not define (`Maybe` aliases excepted). |
| `ai_sdk.oversight.E005` | An alias cannot be built. |
| `ai_sdk.oversight.E006` | `DEFAULT_POLICY` names an unknown policy. |
| `ai_sdk.oversight.E007` | An enforced rule reads a lens no judge here can answer. |
| `ai_sdk.oversight.W008` | An enforced rule reads lenses whose judges are all optional aliases, and none is defined here: the rule never acts. |
| `ai_sdk.oversight.E009` | An agent class names a policy no app registers: every turn it runs is blocked. |
| `ai_sdk.oversight.E010` | Two policy classes declare the same name; the first one registered keeps it. |
| `ai_sdk.oversight.E011` | An agent class's `oversight` names no policy (set after the class was defined). |
| `ai_sdk.oversight.E012` | `AI_SDK_OVERSIGHT` has an unknown key, or a value of the wrong kind; the message says what it accepts. |
| `ai_sdk.oversight.E013` | Deploy error: an enforced gate or tool rule that blocks, masks or takes tools away reads a lens with an external judge (any judge with `external` set) whose timeout is longer than that stage's deadline. A slow answer fails the lens, and the rule blocks the turn or the tool call. |
| `ai_sdk.oversight.E014` | A `Notify(to=[...])` names a notifier `NOTIFIERS` does not have (checked when `NOTIFIERS` is set). |
| `ai_sdk.oversight.W001` | Deploy check: `CAPTURE` is `"full"`, so content is kept unmasked. |
| `ai_sdk.oversight.W005` | An alias's judge cannot run here, such as `TypeSafe` without `TYPESAFE_API_KEY` or `typesafe-sdk`. |
| `ai_sdk.oversight.W007` | Deploy check, against a database (`check --deploy --database default`): rows whose `expires_at` passed more than two days ago still exist, so `oversight_prune` does not run. |
| `ai_sdk.oversight.W011` | A policy in use masks or blocks the answer at finish but streams it live: the client has the text before finish judges it. Use `stream_mode = "auto"` or a client that applies the `replace` notice. |
| `ai_sdk.oversight.W014` | Deploy check: the `TASKS` backend runs reflect and notifications inline in the request (`ImmediateBackend`), never runs them (`DummyBackend`), or does not load. |
| `ai_sdk.oversight.W015` | A policy in use has an enforced `Notify` rule, but `NOTIFIERS` is empty: nobody is told. |
| `ai_sdk.oversight.W017` | Deploy check, against a database (`check --deploy --database default`): turns have been open for more than a day, so `oversight_sweep` does not run. |
| `ai_sdk.oversight.W018` | An enforced rule that blocks or takes tools away reads a lens judged at the gate, tool stage or finish and one judged only at reflect. When the inline lens fails closed or leaves the rule undecided, the rule waits for reflect, where it can no longer block or take tools away. Judge those lenses inline too, or split the rule. |
| `ai_sdk.oversight.W019` | An enforced rule masks the question with a gate lens, but also reads a lens judged after the gate. The rule waits for that lens, so the model reads the question unmasked. Split the rule: one that masks on the gate lenses, and one for the others. |

`W001`, `W007` and `W014` are checked only while oversight is enabled. `W001`, `W007`, `E013` and `W014` are checked only by `manage.py check --deploy`, and `W007` only with `--database`.

The ids missing from this table (`E003`, `E008`, `W002`, `W003`, `W004`, `W006`, `W009`, `W010`, `W012`, `W013` and `W016`) are reserved. Earlier builds raised them, and an id never gets another meaning.

In every app, the twelfth error id is the same check: its settings dict has an unknown key, or a value of the wrong kind.

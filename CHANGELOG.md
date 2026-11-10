# Changelog

## 0.2.0a1 (unreleased)

### Added

- Oversight (`django_ai_sdk.oversight`): policies of lenses and rules that judge every agent turn at the gate, its tool calls and its finish, act on the result, and keep a masked record. See [Oversight](docs/content/oversight.md).
- Durable workflow runs: a run is claimed by one worker, keeps a heartbeat, is swept with `manage.py workflows_sweep` when its worker dies, and resumes with `WorkflowService.resume(run_id)` or `POST /workflows/runs/{id}/resume/`. See [Workflows](docs/content/manual/workflows.md).
- The oversight review API (`OversightService`, `get_oversight_router()`): the turns feed and one turn, review queues, per-operation model permissions with scope hooks, an access record of every read, forgetting on user and thread delete, and `manage.py oversight_prune`. See [Oversight](docs/content/manual/oversight.md).
- The opt-in switches app (`django_ai_sdk.oversight.switches`): change a rule's mode or an agent's policy at runtime, with a reason when it weakens oversight, above the floors code sets (`Rule(min_mode=)`, `Agent.oversight_required`). See [Switches](docs/content/oversight-switches.md).
- The opt-in vault (`django_ai_sdk.vault`): data encrypted under a key per scope, shredded to forget it, with master key rotation. And the opt-in ledger (`django_ai_sdk.oversight.ledger`): the oversight record hash-chained, content in the vault, pseudonyms for who asked, and signed checkpoints to publish. See [Vault](docs/content/vault.md) and [Audit-grade oversight](docs/content/oversight-ledger.md).
- Policies registered at runtime (`add_policy_loader`, `unregister_policy`), and the opt-in authoring app (`django_ai_sdk.oversight.authoring`): operators write policies as versioned definitions with a reason, on top of code policies whose lenses and rules they cannot change. See [Authored policies](docs/content/oversight-authoring.md).

### Upgrading from 0.1.x

- **Workflow tables.** The workflows app now owns one migration, `django_ai_sdk_workflows.0001_initial`, and the core app's `0003_move_workflows_to_workflows_app` is gone. Before you migrate, drop `django_ai_sdk_workflows`, `django_ai_sdk_workflow_runs` and `django_ai_sdk_workflow_run_steps`, and delete the `django_migrations` rows for `django_ai_sdk_workflows`. Stored workflows and run history do not carry over.
- **Settings.** `AI_SDK_WORKFLOW_STEPS` is now `AI_SDK_WORKFLOWS["STEPS"]`, and `AI_SDK_WORKFLOW_ACTIONS` is now `AI_SDK_WORKFLOWS["LISTENERS"]`. System check `ai_sdk.workflows.E005` reports the old names.
- **Names.** `WorkflowSettings` is now `Workflow`, `WorkflowRunStep` is now `StepRun`, and workflow actions are now listeners (`RunListener`).
- **Listener config.** A listener refuses any `config` unless it overrides `validate_config`, as a step does.
- **HTTP.** `POST /workflows/run/`, which ran a definition sent in the request, is removed. A registered or stored workflow runs by its key at `POST /workflows/{key}/run/`.
- **Permissions.** `is_staff` no longer grants anything for workflows. Running needs a logged-in user. Managing, and running a definition that is neither registered nor stored, needs the `change_workflow` model permission. Reading runs needs `view_workflowrun`.
- **Model calls in a step.** A step that calls a model through the SDK's generators outside a turn is now refused (`UNGUARDED_MODEL_CALLS = "refuse"`). Call the model through `self.turn()` or an `AgentStep`, or set it to `"warn"` while you move.

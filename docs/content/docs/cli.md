---
title: Command Line Interface
type: docs
weight: 45
---

The Django AI SDK includes a built-in management command for interacting with assistants directly from the terminal. This is useful for testing, debugging, and quick interactions without building a web interface.

## Quick Start

```bash
# List all registered assistants
python manage.py assistant list

# Start a chat session
python manage.py assistant chat <assistant_id>

# Chat with a partial UUID (first 8+ characters)
python manage.py assistant chat fa5cd75a

# Continue an existing conversation
python manage.py assistant chat fa5cd75a --thread-id <thread-uuid>
```

**UUID Prefix Matching:** The CLI accepts either a full UUID or the first 8+ characters. This is a convenience feature for quick testing — just copy the first few characters from `assistant list` output.

## Commands

### `list` - List Registered Assistants

Displays all assistants registered in the `AI_SDK_ASSISTANTS` setting.

```bash
python manage.py assistant list
```

**Output:**
```
Registered Assistants:
----------------------------------------------------------------------
  fa5cd75a...  Pirate Captain                 (gpt-4o-mini)
  3a8f2b91...  Customer Support Bot           (gpt-4)

----------------------------------------------------------------------
Use --verbose for detailed information
Total: 2 assistant(s)
```

**Options:**

- `--verbose, -v` — Show detailed information including description and class name

### `chat` - Interactive Chat Session

Starts an interactive chat session with an assistant.

```bash
python manage.py assistant chat <assistant_id> [options]
```

**Arguments:**

- `assistant_id` — Full UUID or first 8+ characters of the assistant's ID

**Options:**

| Option | Description |
|--------|-------------|
| `--thread-id, -t <uuid>` | Continue an existing conversation thread |
| `--no-history` | Don't show previous conversation history |
| `--debug` | Show DEBUG level logs from the SDK |

**Examples:**

```bash
# Basic chat
python manage.py assistant chat fa5cd75a

# Continue existing thread
python manage.py assistant chat fa5cd75a --thread-id abc123-...

# Start fresh without history
python manage.py assistant chat fa5cd75a --no-history

# Debug mode (show SDK logs)
python manage.py assistant chat fa5cd75a --debug
```

## Chat Commands

During an interactive chat session, these slash commands are available:

| Command | Description |
|---------|-------------|
| `/quit` | Exit the chat session |
| `/help` | Show available commands |
| `/clear` | Clear the conversation history from memory |

**Note:** `/clear` only clears the in-memory conversation. The thread in the database is preserved.

## Features

### Streaming Output

The CLI streams responses token-by-token for a real-time chat experience:

```
You: Tell me a joke
Assistant: Why did the pirate go to school? To improve his arrrrr-ticulation!
```

### Conversation Persistence

Each chat session automatically creates or uses a `Thread` object. Messages are saved to the database and can be resumed later using the `--thread-id` option.

### Clean Output

By default, SDK debug logs are suppressed to keep the chat interface clean. Use `--debug` to see detailed logs for troubleshooting.

## Use Cases

1. **Testing Assistants** — Quickly test assistant behavior without building a UI
2. **Debugging** — See raw streaming output and diagnose issues
3. **Demo Mode** — Show off assistant capabilities in terminal recordings
4. **Development** — Iterate on prompts and instructions rapidly

## Troubleshooting

### Assistant not found

```
Assistant not found: fa5cd75a
Run 'python manage.py assistant list' to see available assistants.
```

**Solution:** Check your `AI_SDK_ASSISTANTS` setting in `settings.py` and ensure the assistant class path is correct.

### No response appearing

If the assistant name appears but no response text shows up:

1. Check your API keys are configured correctly
2. Run with `--debug` to see SDK logs
3. Verify the assistant's `get_pipeline_adapter()` method is working

### History not loading

The `--thread-id` option requires a full UUID. Find thread IDs in your database or Django admin.

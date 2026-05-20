---
title: CLI Implementation
type: docs
weight: 107
---

Internal documentation for contributors working on the management command interface.

## File Location

```
django_ai_sdk/management/commands/assistant.py
```

## Architecture

The CLI is implemented as a standard Django management command with two subcommands: `list` and `chat`. The implementation prioritizes simplicity and maintainability over features.

### Key Principles

1. **Single File** — All logic in one file for easy understanding and modification
2. **Clear Separation** — Each subcommand has its own handler and helper methods
3. **Async-First** — Chat operations use async/await throughout
4. **Clean Output** — SDK logs suppressed by default, use `--debug` to enable

## Command Structure

```
Command (BaseCommand)
├── add_arguments()          # Define CLI interface
├── handle()                   # Route to subcommands
│
├── handle_list()             # LIST subcommand
│   ├── _list_verbose()       # Detailed output
│   └── _list_compact()       # Compact output
│
└── handle_chat()             # CHAT subcommand (async)
    ├── _configure_logging()   # Suppress logs
    ├── _get_assistant()      # Find by full/partial UUID
    ├── _print_chat_header()  # Display session info
    ├── _get_or_create_thread()  # Thread management
    ├── _show_history()       # Display previous messages
    └── _interactive_chat_loop()  # Main chat loop
        └── _process_user_message()
            └── _stream_response()
                ├── _convert_messages()
                └── _extract_content()
```

## Subcommands

### `list` Subcommand

**Purpose:** Display registered assistants from the registry.

**Implementation:**
```python
def handle_list(self, options):
    from django_ai_sdk.assistants.registry import registry
    # Check if assistants exist
    # Route to verbose or compact display
```

**Key Features:**
- Two display modes: compact (default) and verbose (`-v`)
- UUID prefix matching not needed here (shows full list)
- Falls back gracefully when no assistants registered

### `chat` Subcommand

**Purpose:** Interactive terminal chat with streaming responses.

**Flow:**
1. Configure logging (suppress unless `--debug`)
2. Find assistant by full or partial UUID
3. Display chat header
4. Get or create thread
5. Show history (unless `--no-history`)
6. Enter interactive loop

**Async Pattern:**
```python
async def handle_chat(self, options):
    # ... setup ...
    await self._interactive_chat_loop(assistant, thread_id)

async def _interactive_chat_loop(self, assistant, thread_id):
    while True:
        user_input = input("You: ").strip()
        await self._process_user_message(...)
```

## Key Implementation Details

### 1. UUID Matching

The CLI accepts full UUIDs or prefixes (8+ characters). The `_get_assistant()` method handles this:

```python
def _get_assistant(self, assistant_id: str, registry):
    # Try exact match first
    if assistant_id in registry:
        return registry.get(assistant_id)
    
    # Try prefix match (need at least 8 chars for safety)
    if len(assistant_id) >= 8:
        for full_id in registry.ids():
            if full_id.startswith(assistant_id):
                return registry.get(full_id)
    
    return None
```

**Why 8 characters?** — Balance between convenience (short to type) and uniqueness (low collision probability).

### 2. Streaming Output

The streaming implementation uses Python's `print()` with `flush=True` instead of `sys.stdout.write()` for immediate terminal output:

```python
async def _stream_response(self, assistant, messages, thread_id):
    print("Assistant: ", end="", flush=True)
    
    async for chunk in adapter.stream(chat_messages):
        content = self._extract_content(chunk)
        if content:
            print(content, end="", flush=True)  # Immediate output
            full_response += content
    
    print()  # Final newline
```

**Why print instead of sys.stdout.write?**
- `print()` with `flush=True` ensures immediate display
- Better handling of terminal buffering
- Avoids conflicts with debug logs writing to the same line

### 3. Content Extraction

The `_extract_content()` method handles multiple chunk types from different adapters:

```python
def _extract_content(self, chunk: Any) -> str:
    # Handle StreamEvent types (Haystack streaming)
    if hasattr(chunk, "type"):
        if chunk.type == "text_chunk":
            return chunk.content
        elif chunk.type == "reasoning_chunk":
            return ""  # Skip reasoning
    
    # Handle objects with content attribute
    if hasattr(chunk, "content"):
        return str(chunk.content)
    
    # Handle strings directly
    if isinstance(chunk, str):
        return chunk
    
    return ""
```

**Design:** Duck typing over isinstance checks for flexibility with different adapter implementations.

### 4. Log Suppression

The CLI suppresses SDK logs by default to keep chat output clean:

```python
def _configure_logging(self, logger, debug_mode: bool):
    logger.remove()  # Remove default handler
    level = "DEBUG" if debug_mode else "WARNING"
    logger.add(sys.stderr, level=level, ...)
```

**Why WARNING level by default?** — INFO logs still clutter the interface. Only warnings and errors should appear.

### 5. Message Conversion

The `_convert_messages()` method normalizes message formats:

```python
def _convert_messages(self, messages: list) -> list:
    chat_messages = []
    for msg in messages:
        if isinstance(msg, ChatMessage):
            chat_messages.append(msg)
        else:
            chat_messages.append(
                ChatMessage(role=msg["role"], content=msg["content"])
            )
    return chat_messages
```

**Purpose:** Handle both `ChatMessage` objects and dicts (from history loading).

## Error Handling

### Assistant Not Found

```python
if not assistant:
    raise CommandError(
        f"Assistant not found: {options['assistant_id']}\n"
        f"Run 'python manage.py assistant list' to see available assistants."
    )
```

### Streaming Errors

```python
try:
    async for chunk in adapter.stream(...):
        # ... process ...
except Exception as e:
    self.stdout.write(self.style.ERROR(f"\nError: {e}"))
```

### Keyboard Interrupt

```python
except KeyboardInterrupt:
    self.stdout.write(
        self.style.WARNING("\n\nInterrupted. Use /quit to exit properly.")
    )
    continue  # Return to prompt instead of exiting
```

## Testing Considerations

### What to Test

1. **UUID matching** — Full UUIDs, 8-char prefixes, invalid IDs
2. **Streaming** — Verify tokens appear immediately
3. **Commands** — /quit, /help, /clear functionality
4. **Error handling** — Missing assistants, API failures
5. **Log suppression** — Verify --debug flag works

### Manual Testing Checklist

```bash
# List assistants
python manage.py assistant list
python manage.py assistant list -v

# Chat with full and partial UUID
python manage.py assistant chat <full-uuid>
python manage.py assistant chat <8-char-prefix>

# Thread operations
python manage.py assistant chat <id> --thread-id <existing-thread>
python manage.py assistant chat <id> --no-history

# Debug mode
python manage.py assistant chat <id> --debug

# Chat commands
# - Type /help
# - Type /clear  
# - Type /quit
# - Press Ctrl+C (should not crash)
```

## Future Enhancements

Potential improvements (not currently implemented):

1. **Export conversations** — Save chat history to file
2. **Multi-turn templates** — Load initial prompts from files
3. **Thread listing** — `assistant threads` to list existing threads
4. **Configuration** — `.aidkrc` file for default assistant/model
5. **Non-interactive mode** — `echo "prompt" | assistant chat <id>`

## Code Style

- Use type hints for function signatures
- Keep imports inside methods (lazy loading)
- Use `self.style` for colored output
- Group related methods with section comments
- Maximum 20-30 lines per method

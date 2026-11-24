#!/usr/bin/env python3
"""
Graphviz diagram generator for Django AI SDK documentation.

Generates the diagrams used by the docs. Output goes to
docs/content/images/graphs/, next to the pages that show them.

Run with: uv run python docs/graph.py
"""

import graphviz
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "content" / "images" / "graphs"

# Neutral shadcn grays from the docs theme: black for the main path, gray for the rest.
ACCENT = "#171717"  # neutral-900: primary-flow borders
ACCENT_FILL = "#ffffff"
ACCENT_TEXT = "#171717"
CARD = "#fafafa"  # neutral-50: diagram background, a light card in both site modes
SURFACE = "#ffffff"  # node fill
SUBTLE = "#f5f5f5"  # neutral-100 (muted): clusters, notes
BORDER = "#d4d4d4"  # neutral-300
LINE = "#737373"  # neutral-500: edges
TEXT = "#171717"  # neutral-900
MUTED_TEXT = "#737373"  # neutral-500 (muted-foreground): notes, edge labels

FONT = "Helvetica,Arial,sans-serif"  # the site uses the system sans stack

# The page column is ~720 CSS px (7.5in at 96 px/in). Every diagram is padded to
# this width, so all of them show at the same scale.
COLUMN_WIDTH_IN = 7.5


def create_graph(name: str, **attrs) -> graphviz.Digraph:
    """Create a new directed graph with the docs-site styling."""
    dot = graphviz.Digraph(
        name=name,
        format="svg",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": CARD,
            "fontname": FONT,
            "fontsize": "11",
            "fontcolor": MUTED_TEXT,
            "nodesep": "0.25",
            "ranksep": "0.35",
            "splines": "ortho",
            "pad": "0.2",
            **attrs.get("graph_attr", {}),
        },
        node_attr={
            "fontname": FONT,
            "fontsize": "11",
            "fontcolor": TEXT,
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": SURFACE,
            "color": BORDER,
            "penwidth": "0.8",
            "margin": "0.12,0.06",
            "height": "0.3",
        },
        edge_attr={
            "fontname": FONT,
            "fontsize": "10",
            "fontcolor": MUTED_TEXT,
            "color": LINE,
            "penwidth": "0.8",
            "arrowhead": "normal",
            "arrowsize": "0.5",
        },
    )
    return dot


def add_component(dot, name, label, **attrs) -> str:
    """Add a neutral component node."""
    dot.node(name, label=label, **attrs)
    return name


def add_flow_node(dot, name, label, **attrs) -> str:
    """Add a primary-flow node in the accent color."""
    dot.node(
        name, label=label, fillcolor=ACCENT_FILL, color=ACCENT, fontcolor=ACCENT_TEXT, **attrs
    )
    return name


def add_note(dot, name, label, **attrs) -> str:
    """Add a muted annotation note."""
    dot.node(
        name,
        label=label,
        shape="note",
        style="filled",
        fillcolor=SUBTLE,
        color=BORDER,
        fontcolor=MUTED_TEXT,
        fontsize="10",
        **attrs,
    )
    return name


def style_cluster(c, label) -> None:
    """Style a cluster as a muted rounded panel."""
    c.attr(
        label=label,
        style="rounded,filled",
        fillcolor=SUBTLE,
        color=BORDER,
        penwidth="0.8",
        fontsize="10",
        fontcolor=MUTED_TEXT,
        margin="10",
    )


def fit_to_column(dot: graphviz.Digraph) -> None:
    """Pad a narrow diagram out to the column width, so all display at one scale."""
    width_in = float(dot.pipe(format="plain").split()[2])
    side = max((COLUMN_WIDTH_IN - width_in) / 2, 0.2)
    dot.graph_attr["pad"] = f"{side:.3f},0.2"


def generate_overview_architecture() -> graphviz.Digraph:
    """End-to-end request flow through an Agent."""
    dot = create_graph("overview_architecture")

    user = add_flow_node(dot, "user", "Client\nPOST /chat (Vercel protocol)")

    agent = add_flow_node(
        dot,
        "agent",
        "Agent\n• as_view()\n• get_pipeline_adapter()\n• get_run_adapter()\n• get_tools()",
    )

    stream = add_flow_node(
        dot,
        "stream",
        "Stream\n• haystack Pipeline\n• ToolAgent + generator\n• storage_adapter",
    )

    run = add_flow_node(
        dot,
        "run",
        "Run\n• generator\n• response_format\n(structured output)",
    )

    tools = add_component(
        dot,
        "tools",
        "Tools\n• haystack Tool objects\n• integrations\n• RAG retrieval\n• artifacts",
    )

    storage = add_component(
        dot,
        "storage",
        "Storage\nMemoryStorageAdapter / DbStorageAdapter\nThreads • Messages",
    )

    protocol = add_component(
        dot,
        "protocol",
        "Protocol Handler\nVercelProtocolHandler (default)\nStreamEvents → SSE",
    )

    dot.edge("user", "agent")
    dot.edge("agent", "stream", xlabel="streaming")
    dot.edge("agent", "run", xlabel="non-streaming")
    dot.edge("agent", "tools", style="dashed")
    dot.edge("agent", "storage", style="dashed")
    dot.edge("stream", "protocol", style="dashed")
    dot.edge("run", "protocol", style="dashed")
    dot.edge("protocol", "user", xlabel="SSE", style="dashed")

    return dot


def generate_data_flow() -> graphviz.Digraph:
    """Step-by-step lifecycle of a streaming chat request."""
    dot = create_graph(
        "data_flow", graph_attr={"rankdir": "TB"}
    )

    step1 = add_flow_node(dot, "step1", "1. USER REQUEST\nPOST /chat (protocol messages)")
    step2 = add_flow_node(
        dot,
        "step2",
        "2. AGENT LAYER\nas_view(messages, thread_id, user)\n• check CHAT permissions\n• protocol_handler.to_chat_messages()\n• store last user message",
    )
    step3 = add_flow_node(
        dot,
        "step3",
        "3. ADAPTER LAYER\nget_pipeline_adapter(thread_id, user)\n• get_tools() (class + integrations + RAG)\n• build ToolAgent pipeline\n• return Stream",
    )
    step4 = add_flow_node(dot, "step4", "4. HAYSTACK PIPELINE\ngenerator streams chunks")
    step5 = add_flow_node(
        dot,
        "step5",
        "5. EVENT NORMALIZATION\nchunks → StreamEvents",
    )
    step6 = add_flow_node(
        dot,
        "step6",
        "6. PROTOCOL CONVERSION\nevents → Vercel protocol SSE parts",
    )
    step7 = add_component(
        dot,
        "step7",
        "7. STORAGE\nStreamWriter → ChatMessage\nsame message_id everywhere",
    )
    step8 = add_component(
        dot,
        "step8",
        "8. POST-PROCESSING\ncitations • suggestions",
    )

    dot.edge("step1", "step2")
    dot.edge("step2", "step3")
    dot.edge("step3", "step4")
    dot.edge("step4", "step5")
    dot.edge("step5", "step6")
    dot.edge("step3", "step7", style="dashed")
    dot.edge("step6", "step8", style="dashed")
    for step, side in (("step3", "step7"), ("step6", "step8")):
        with dot.subgraph() as same:
            same.attr(rank="same")
            same.node(step)
            same.node(side)

    return dot


def generate_subagent_flow() -> graphviz.Digraph:
    """Coordinator delegating to a subagent, in call order."""
    dot = create_graph(
        "subagent_flow", graph_attr={"rankdir": "TB"}
    )

    coordinator = add_flow_node(
        dot,
        "coordinator",
        "Coordinator's ToolAgent loop\nits own Haystack Agent, its own tools",
    )
    tool_call = add_flow_node(
        dot, "tool_call", 'LLM calls a subagent tool\ne.g. "research_planner"'
    )
    component_tool = add_flow_node(
        dot,
        "component_tool",
        "ComponentTool\ncomponent = SubagentStreamFilter\n(built agent, name, agent_id)",
    )

    with dot.subgraph(name="cluster_subagent") as c:
        style_cluster(c, "Subagent's OWN Haystack Agent loop  (built once, in build_subagent())")
        span = add_note(
            c,
            "span",
            "tracing span:\ndjango_ai_sdk.subagent.run",
        )
        own_loop = add_flow_node(
            c,
            "own_loop",
            "own tools · own hooks (default_hooks)\nown max_agent_steps · own generator",
        )
        recovery = c.node(
            "recovery",
            "no response yet?\nONE more tool-less call\n(FINAL_PROMPT)",
        )
        flush = c.node(
            "flush",
            "any tool result never streamed?\nflushed through the callback now",
        )
        c.edge("span", "own_loop")
        c.edge("own_loop", "recovery", style="dashed", xlabel="loop exits")
        c.edge("recovery", "flush")

    response = add_flow_node(
        dot,
        "response",
        "subagent_response(messages)\nresponse text (or a digest)\n+ Sources consulted",
    )
    tool_result = add_flow_node(
        dot,
        "tool_result",
        'ToolCallResult on the COORDINATOR\'s\nconversation, tagged "handoff"\n(coordinator\'s loop continues from here)',
    )
    persist = add_component(
        dot,
        "persist",
        'Stream persists everything\ntool_call["agent"/"handoff"] stored',
    )
    replay = add_component(
        dot,
        "replay",
        "Next turn: replay_handoff()\nreal assistant(tool_calls) + tool(result)\nnot flattened to text",
    )

    dot.edge("coordinator", "tool_call")
    dot.edge("tool_call", "component_tool")
    dot.edge("component_tool", "span", lhead="cluster_subagent")
    dot.edge("flush", "response", ltail="cluster_subagent")
    dot.edge("response", "tool_result")
    dot.edge("tool_result", "persist")
    dot.edge("persist", "replay")

    dot.attr(compound="true")

    return dot


def generate_headless_tool_run_flow() -> graphviz.Digraph:
    """Agent.run(tools=True): the unattended, non-streaming equivalent."""
    dot = create_graph(
        "headless_tool_run_flow", graph_attr={"rankdir": "TB"}
    )

    call = add_flow_node(dot, "call", "await agent.run(messages, tools=True)")

    agent_run = add_flow_node(
        dot,
        "agent_run",
        "Agent.run()\nresolved = response_format\nadapter = get_run_adapter()",
    )

    gate = add_note(
        dot,
        "gate",
        "tools=True\nAND response_format is None",
    )

    own_tools = add_flow_node(
        dot,
        "own_tools",
        "Run.run(tools=...)\ntools = await self.get_tools()\nhooks = default_hooks(self)",
    )

    build_agent = add_flow_node(
        dot,
        "build_agent",
        "Haystack Agent(\n  generator, tools,\n  max_agent_steps=self.max_agent_steps,\n  hooks=hooks,\n)\nrun_agent() adds oversight",
    )

    run_async = add_flow_node(
        dot, "run_async", "run_agent(agent, ...)\nno streaming callback"
    )

    result = add_flow_node(
        dot,
        "result",
        'result["messages"][-1].text\nthe final answer, plain string',
    )

    not_run = add_component(
        dot,
        "not_run",
        "Run.run(...)\nplain / structured: the same\nrun_agent(), with no tools",
    )

    shared_note = add_note(
        dot,
        "shared_note",
        "default_hooks() + run_agent()\nare the SAME assembly build_subagent()\nuses for a streamed delegation",
    )

    dot.edge("call", "agent_run")
    dot.edge("agent_run", "gate")
    dot.edge("gate", "own_tools", xlabel="yes")
    dot.edge("gate", "not_run", xlabel="no", style="dashed")
    dot.edge("own_tools", "build_agent")
    dot.edge("build_agent", "run_async")
    dot.edge("run_async", "result")
    dot.edge("build_agent", "shared_note", style="dotted", arrowhead="none")
    with dot.subgraph() as same:
        same.attr(rank="same")
        same.node("build_agent")
        same.node("shared_note")

    return dot


def generate_adapter_flow() -> graphviz.Digraph:
    """Stream/Run adapters over haystack components."""
    dot = create_graph("adapter_flow")

    agent = add_flow_node(
        dot,
        "agent",
        "Agent\nget_pipeline_adapter() / get_run_adapter()",
    )

    stream = add_flow_node(
        dot,
        "stream",
        "Stream\n• pipeline: haystack.Pipeline\n• generator\n• stream() → StreamEvents",
    )

    run = add_flow_node(
        dot,
        "run",
        "Run\n• generator\n• run() → structured output\n(response_format)",
    )

    tool_agent = add_component(
        dot,
        "tool_agent",
        "haystack ToolAgent\n• tools: haystack.Tool[]\n• system_prompt",
    )

    generator = add_component(
        dot,
        "generator",
        "Generator\ndjango_ai_sdk.generators (multi-vendor)",
    )

    note_stream = add_note(dot, "note_stream", "Stream requires a haystack.Pipeline")
    note_run = add_note(dot, "note_run", "Run requires a Runnable generator")

    dot.edge("agent", "stream", xlabel="streaming")
    dot.edge("agent", "run", xlabel="non-streaming")
    dot.edge("stream", "tool_agent")
    dot.edge("tool_agent", "generator")
    dot.edge("run", "generator", style="dashed")
    dot.edge("stream", "note_stream", style="dotted", arrowhead="none")
    dot.edge("run", "note_run", style="dotted", arrowhead="none")
    for node, note in (("stream", "note_stream"), ("run", "note_run")):
        with dot.subgraph() as same:
            same.attr(rank="same")
            same.node(node)
            same.node(note)

    return dot


def generate_id_generation() -> graphviz.Digraph:
    """Message id generated once in the adapter and reused everywhere."""
    dot = create_graph("id_generation", graph_attr={"rankdir": "TB"})

    dot.attr(compound="true")

    with dot.subgraph(name="cluster_flow") as c:
        style_cluster(c, "Stream.stream()")

        gen = add_flow_node(
            c,
            "gen",
            "message_id = uuid.uuid4()\n← GENERATED",
        )
        start = c.node("start", "yield MessageStartEvent\n(message_id=...)")
        sse = c.node("sse", 'SSE Stream\n{"messageId": "..."}')
        writer = c.node("writer", "StreamWriter\n(message_id=...)")
        msg = c.node("msg", "ChatMessage\n(id=message_id)")
        storage = c.node("storage", "Storage.save()\nsame ID everywhere")

        c.edge("gen", "start")
        c.edge("start", "sse")
        c.edge("sse", "writer")
        c.edge("writer", "msg")
        c.edge("msg", "storage")

    return dot


def generate_storage_architecture() -> graphviz.Digraph:
    """Storage layer: base interface + implementations."""
    dot = create_graph("storage_architecture")

    with dot.subgraph(name="cluster_storage") as c:
        style_cluster(c, "Storage Layer")

        memory = add_component(
            c,
            "memory",
            "MemoryStorageAdapter\n(In-Memory)",
        )
        db = add_component(c, "db", "DbStorageAdapter\n(Django ORM)")
        base = add_flow_node(c, "base", "BaseStorageAdapter\n(Abstract)")

        c.edge("memory", "base", style="dashed")
        c.edge("db", "base", style="dashed")

    format_node = add_note(
        dot,
        "format",
        "ChatMessage\nsingle format for history, rating, threads",
    )

    dot.edge("base", "format", style="dashed")

    return dot


def generate_rag_architecture() -> graphviz.Digraph:
    """RAG pipeline: provider, memories, retrieval tools, citations."""
    dot = create_graph("rag_architecture")

    agent = add_flow_node(
        dot,
        "agent",
        "Agent\n• rag_provider = RAGProvider()\n• get_rag_queryset(memory_id)\n• get_rag_documents(memory_id)\n• get_rag_pipeline(memory_id)",
    )

    provider = add_flow_node(
        dot,
        "provider",
        "RAGProvider\n• caches pipelines per agent + memory\n• warmup() / reindex() / clear_rag_cache()",
    )

    pipelines = add_flow_node(
        dot,
        "pipelines",
        "RAG Pipelines\nBM25QueryExpanderRAG\nChromaDBQueryExpanderRAG\nQdrantBM25HybridRAG",
    )

    memories = add_component(
        dot,
        "memories",
        "MemoryService\nthread ↔ memory links",
    )

    retrieval = add_component(
        dot,
        "retrieval",
        "get_rag_tools()\nretrieval tools on the agent",
    )

    citations = add_component(
        dot,
        "citations",
        "CitationRegistry + CitationFormatter\nsource numbers streamed with the answer",
    )

    dot.edge("agent", "provider")
    dot.edge("agent", "memories", style="dashed")
    dot.edge("provider", "pipelines")
    dot.edge("memories", "pipelines", style="dashed", constraint="false")
    dot.edge("pipelines", "retrieval", style="dashed")
    dot.edge("retrieval", "citations", style="dashed")

    return dot


def generate_testing_pyramid() -> graphviz.Digraph:
    """Test pyramid: unit / integration / e2e."""
    dot = create_graph("testing_pyramid", graph_attr={"rankdir": "TB"})

    e2e = add_component(
        dot,
        "e2e",
        "E2E Tests\nfull agent flow",
        width="2",
    )

    integration = add_component(
        dot,
        "integration",
        "Integration Tests\nAdapter + Storage + Protocol",
        width="3",
    )

    unit = add_flow_node(
        dot,
        "unit",
        "Unit Tests\nindividual components",
        width="4",
    )

    dot.edge("integration", "e2e", style="invis")
    dot.edge("unit", "integration", style="invis")

    dot.graph_attr["rankdir"] = "BT"

    return dot


DIAGRAMS = {
    "overview_architecture": generate_overview_architecture,
    "data_flow": generate_data_flow,
    "adapter_flow": generate_adapter_flow,
    "subagent_flow": generate_subagent_flow,
    "headless_tool_run_flow": generate_headless_tool_run_flow,
    "id_generation": generate_id_generation,
    "storage_architecture": generate_storage_architecture,
    "rag_architecture": generate_rag_architecture,
    "testing_pyramid": generate_testing_pyramid,
}


def generate_all():
    """Generate every diagram as SVG next to the docs pages."""
    print("Generating Django AI SDK documentation diagrams...")
    print(f"Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    failed = []

    for name, generator_func in DIAGRAMS.items():
        try:
            print(f"Generating {name}...", end=" ")
            dot = generator_func()
            fit_to_column(dot)
            path = OUTPUT_DIR / name
            dot.render(str(path), cleanup=True)
            generated.append(name)
            print("✓")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"✗ ({e})")

    print()
    print(f"Generated {len(generated)} diagrams to {OUTPUT_DIR}")

    if failed:
        print()
        print(f"Failed to generate {len(failed)} diagrams:")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")

    print()
    print("Done!")


if __name__ == "__main__":
    generate_all()

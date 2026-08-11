#!/usr/bin/env python3
"""
Graphviz diagram generator for Django AI SDK documentation.

Generates the diagrams used by the Hugo site. Output goes to
docs/static/images/graphs/ so pages can reference them as
/images/graphs/<name>.png.

Run with: uv run python docs/graph.py
"""

import graphviz
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "static" / "images" / "graphs"


def create_graph(name: str, **attrs) -> graphviz.Digraph:
    """Create a new directed graph with default styling."""
    dot = graphviz.Digraph(
        name=name,
        format="png",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "transparent",
            "fontname": "Helvetica,Arial,sans-serif",
            "nodesep": "0.5",
            "ranksep": "0.75",
            "splines": "ortho",
            "pad": "0.5",
            "dpi": "150",
            **attrs.get("graph_attr", {}),
        },
        node_attr={
            "fontname": "Helvetica,Arial,sans-serif",
            "fontsize": "11",
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#f5f5f5",
            "color": "#333333",
            "margin": "0.3,0.2",
        },
        edge_attr={
            "fontname": "Helvetica,Arial,sans-serif",
            "fontsize": "10",
            "color": "#666666",
            "arrowhead": "vee",
            "arrowsize": "0.8",
        },
    )
    return dot


def add_component(dot, name, label, fillcolor="#f5f5f5", **attrs) -> str:
    """Add a component node with consistent styling."""
    dot.node(name, label=label, fillcolor=fillcolor, **attrs)
    return name


def add_flow_node(dot, name, label, fillcolor="#e3f2fd", **attrs) -> str:
    """Add a flow/step node with blue styling."""
    dot.node(name, label=label, fillcolor=fillcolor, color="#1565c0", fontcolor="#1565c0", **attrs)
    return name


def add_note(dot, name, label, **attrs) -> str:
    """Add an annotation note node."""
    dot.node(
        name,
        label=label,
        shape="note",
        fillcolor="#fff9c4",
        color="#f57f17",
        fontsize="9",
        **attrs,
    )
    return name


def generate_overview_architecture() -> graphviz.Digraph:
    """End-to-end request flow through an Agent."""
    dot = create_graph("overview_architecture")

    user = add_flow_node(dot, "user", "Client\nPOST /chat (Vercel protocol)")

    agent = add_component(
        dot,
        "agent",
        "Agent\n• as_view()\n• get_pipeline_adapter()\n• get_run_adapter()\n• get_tools()",
        fillcolor="#fff3e0",
        color="#e65100",
        fontcolor="#e65100",
    )

    stream = add_component(
        dot,
        "stream",
        "Stream\n• haystack Pipeline\n• ToolAgent + generator\n• storage_adapter",
        fillcolor="#e8f5e9",
        color="#2e7d32",
        fontcolor="#2e7d32",
    )

    run = add_component(
        dot,
        "run",
        "Run\n• generator\n• response_format\n(structured output)",
        fillcolor="#e8f5e9",
        color="#2e7d32",
        fontcolor="#2e7d32",
    )

    tools = add_component(
        dot,
        "tools",
        "Tools\n• haystack Tool objects\n• integrations\n• RAG retrieval\n• artifacts",
        fillcolor="#fce4ec",
        color="#c2185b",
        fontcolor="#c2185b",
    )

    storage = add_component(
        dot,
        "storage",
        "Storage\nMemoryStorageAdapter / DbStorageAdapter\nThreads • Messages",
        fillcolor="#f3e5f5",
        color="#7b1fa2",
        fontcolor="#7b1fa2",
    )

    protocol = add_component(
        dot,
        "protocol",
        "Protocol Handler\nVercelProtocolHandler (default)\nStreamEvents → SSE",
        fillcolor="#e0f2f1",
        color="#00695c",
        fontcolor="#00695c",
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
        "data_flow", graph_attr={"rankdir": "TB", "nodesep": "0.6", "ranksep": "1.0"}
    )

    step1 = add_flow_node(dot, "step1", "1. USER REQUEST\nPOST /chat (protocol messages)")
    step2 = add_component(
        dot,
        "step2",
        "2. AGENT LAYER\nas_view(messages, thread_id, user)\n• protocol_handler.to_chat_messages()\n• store last user message",
        fillcolor="#fff3e0",
    )
    step3 = add_component(
        dot,
        "step3",
        "3. ADAPTER LAYER\nget_pipeline_adapter(thread_id, user)\n• get_tools() (class + integrations + RAG)\n• build ToolAgent pipeline\n• return Stream",
        fillcolor="#e8f5e9",
    )
    step4 = add_flow_node(dot, "step4", "4. HAYSTACK PIPELINE\ngenerator streams chunks")
    step5 = add_component(
        dot,
        "step5",
        "5. EVENT NORMALIZATION\nchunks → StreamEvents",
        fillcolor="#e1f5fe",
    )
    step6 = add_component(
        dot,
        "step6",
        "6. PROTOCOL CONVERSION\nevents → Vercel protocol SSE parts",
        fillcolor="#e0f2f1",
    )
    step7 = add_component(
        dot,
        "step7",
        "7. STORAGE\nStreamWriter → ChatMessage\nsame message_id everywhere",
        fillcolor="#fce4ec",
    )
    step8 = add_component(
        dot,
        "step8",
        "8. POST-PROCESSING\ncitations • suggestions",
        fillcolor="#f3e5f5",
    )

    dot.edge("step1", "step2")
    dot.edge("step2", "step3")
    dot.edge("step3", "step4")
    dot.edge("step4", "step5")
    dot.edge("step5", "step6")
    dot.edge("step3", "step7", style="dashed", constraint="false")
    dot.edge("step6", "step8", style="dashed", constraint="false")

    return dot


def generate_adapter_flow() -> graphviz.Digraph:
    """Stream/Run adapters over haystack components."""
    dot = create_graph("adapter_flow")

    agent = add_component(
        dot,
        "agent",
        "Agent\nget_pipeline_adapter() / get_run_adapter()",
        fillcolor="#fff3e0",
        color="#e65100",
    )

    stream = add_component(
        dot,
        "stream",
        "Stream\n• pipeline: haystack.Pipeline\n• generator\n• stream() → StreamEvents",
        fillcolor="#e8f5e9",
        color="#2e7d32",
    )

    run = add_component(
        dot,
        "run",
        "Run\n• generator\n• run() → structured output\n(response_format)",
        fillcolor="#e8f5e9",
        color="#2e7d32",
    )

    tool_agent = add_component(
        dot,
        "tool_agent",
        "haystack ToolAgent\n• tools: haystack.Tool[]\n• system_prompt",
        fillcolor="#e1f5fe",
        color="#1565c0",
    )

    generator = add_component(
        dot,
        "generator",
        "Generator\nOpenAIChatGenerator (OpenAI-compatible)",
        fillcolor="#f3e5f5",
        color="#7b1fa2",
    )

    note_stream = add_note(dot, "note_stream", "Stream requires a haystack.Pipeline")
    note_run = add_note(dot, "note_run", "Run requires a Runnable generator")

    dot.edge("agent", "stream", xlabel="streaming")
    dot.edge("agent", "run", xlabel="non-streaming")
    dot.edge("stream", "tool_agent")
    dot.edge("tool_agent", "generator")
    dot.edge("run", "generator", style="dashed")
    dot.edge("stream", "note_stream", style="dotted", arrowhead="none", constraint="false")
    dot.edge("run", "note_run", style="dotted", arrowhead="none", constraint="false")

    return dot


def generate_id_generation() -> graphviz.Digraph:
    """Message id generated once in the adapter and reused everywhere."""
    dot = create_graph("id_generation", graph_attr={"rankdir": "TB"})

    dot.attr(compound="true")

    with dot.subgraph(name="cluster_flow") as c:
        c.attr(
            label="Stream.stream()",
            style="rounded",
            fillcolor="#f5f5f5",
            color="#666666",
        )

        gen = c.node(
            "gen",
            "message_id = uuid.uuid4()\n← GENERATED",
            fillcolor="#fff9c4",
            color="#f57f17",
            fontcolor="#f57f17",
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
        c.attr(label="Storage Layer", style="rounded", fillcolor="#f5f5f5", color="#666666")

        memory = add_component(
            c,
            "memory",
            "MemoryStorageAdapter\n(In-Memory)",
            fillcolor="#e3f2fd",
            color="#1565c0",
        )
        db = add_component(
            c, "db", "DbStorageAdapter\n(Django ORM)", fillcolor="#e8f5e9", color="#2e7d32"
        )
        base = add_component(
            c, "base", "BaseStorageAdapter\n(Abstract)", fillcolor="#fff3e0", color="#e65100"
        )

        c.edge("memory", "base", style="dashed")
        c.edge("db", "base", style="dashed")

    format_node = add_component(
        dot,
        "format",
        "ChatMessage\nsingle format for history, rating, threads",
        fillcolor="#fce4ec",
        color="#c2185b",
        fontcolor="#c2185b",
        shape="note",
    )

    dot.edge("base", "format", style="dashed", color="#c2185b")

    return dot


def generate_rag_architecture() -> graphviz.Digraph:
    """RAG pipeline: provider, memories, retrieval tools, citations."""
    dot = create_graph("rag_architecture")

    agent = add_component(
        dot,
        "agent",
        "Agent\n• rag_provider = RAGProvider()\n• get_rag_queryset(memory_id)\n• get_rag_documents(memory_id)\n• get_rag_pipeline(memory_id)",
        fillcolor="#fff3e0",
        color="#e65100",
    )

    provider = add_component(
        dot,
        "provider",
        "RAGProvider\n• caches pipelines per agent + memory\n• warmup() / reindex() / clear_rag_cache()",
        fillcolor="#e3f2fd",
        color="#1565c0",
    )

    pipelines = add_component(
        dot,
        "pipelines",
        "RAG Pipelines\nBM25QueryExpanderRAG\nChromaDBQueryExpanderRAG\nQdrantBM25HybridRAG",
        fillcolor="#e8f5e9",
        color="#2e7d32",
    )

    memories = add_component(
        dot,
        "memories",
        "MemoryService\nthread ↔ memory links",
        fillcolor="#f3e5f5",
        color="#7b1fa2",
    )

    retrieval = add_component(
        dot,
        "retrieval",
        "get_rag_tools()\nretrieval tools on the agent",
        fillcolor="#fce4ec",
        color="#c2185b",
    )

    citations = add_component(
        dot,
        "citations",
        "CitationRegistry + CitationFormatter\nsource numbers streamed with the answer",
        fillcolor="#fff9c4",
        color="#f57f17",
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
        fillcolor="#fce4ec",
        color="#c2185b",
        width="2",
    )

    integration = add_component(
        dot,
        "integration",
        "Integration Tests\nAdapter + Storage + Protocol",
        fillcolor="#fff3e0",
        color="#e65100",
        width="3",
    )

    unit = add_component(
        dot,
        "unit",
        "Unit Tests\nindividual components",
        fillcolor="#e8f5e9",
        color="#2e7d32",
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
    "id_generation": generate_id_generation,
    "storage_architecture": generate_storage_architecture,
    "rag_architecture": generate_rag_architecture,
    "testing_pyramid": generate_testing_pyramid,
}


def generate_all():
    """Generate all diagram images into the docs static dir."""
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

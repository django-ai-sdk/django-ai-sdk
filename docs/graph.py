#!/usr/bin/env python3
"""
Graphviz diagram generator for Django AI SDK documentation.

This script generates all visual diagrams for both:
1. The website (docs/static/images/graphs/)
2. The manual (manual/graphs/)

Run with: python manual/graph.py
"""

import graphviz
from pathlib import Path
from typing import Dict, Any, List

# Output directories
MANUAL_GRAPHS_DIR = Path(__file__).parent / "graphs"
DOCS_GRAPHS_DIR = Path(__file__).parent.parent / "docs" / "static" / "images" / "graphs"


def create_graph(name: str, **attrs) -> graphviz.Digraph:
    """Create a new directed graph with default styling."""
    dot = graphviz.Digraph(
        name=name,
        format="png",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
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


def add_component(
    dot: graphviz.Digraph, name: str, label: str, fillcolor: str = "#f5f5f5", **attrs
) -> str:
    """Add a component node with consistent styling."""
    dot.node(
        name,
        label=label,
        fillcolor=fillcolor,
        **attrs,
    )
    return name


def add_flow_node(
    dot: graphviz.Digraph,
    name: str,
    label: str,
    fillcolor: str = "#e3f2fd",
    textcolor: str = "#1565c0",
    bordercolor: str = "#1565c0",
    **attrs,
) -> str:
    """Add a flow/step node with blue styling."""
    dot.node(
        name,
        label=label,
        fillcolor=fillcolor,
        fontcolor=textcolor,
        color=bordercolor,
        **attrs,
    )
    return name


def generate_overview_architecture() -> graphviz.Digraph:
    """Generate README.md overview architecture diagram."""
    dot = create_graph("overview_architecture")

    # User request
    user = add_flow_node(dot, "user", "User Request\nPOST /api/chat")

    # Assistant
    assistant = add_component(
        dot,
        "assistant",
        "Assistant\n• as_view()\n• get_pipeline_adapter()\n• get_storage_adapter()",
        fillcolor="#fff3e0",
        color="#e65100",
        fontcolor="#e65100",
    )

    # Adapter
    adapter = add_component(
        dot,
        "adapter",
        "Adapter\nOpenAIAdapter\nHaystackAdapter\nOpenAIAgentAdapter",
        fillcolor="#e8f5e9",
        color="#2e7d32",
        fontcolor="#2e7d32",
    )

    # Storage
    storage = add_component(
        dot,
        "storage",
        "Storage\nMemory/Database\n• Stores messages\n• Retrieves history",
        fillcolor="#fce4ec",
        color="#c2185b",
        fontcolor="#c2185b",
    )

    # AI Provider
    provider = add_component(
        dot,
        "provider",
        "AI Provider\nOpenAI / Haystack\nStreaming Response",
        fillcolor="#f3e5f5",
        color="#7b1fa2",
        fontcolor="#7b1fa2",
    )

    # Protocol
    protocol = add_component(
        dot,
        "protocol",
        "Protocol Handler\nVercel Protocol\nConverts → SSE → Frontend",
        fillcolor="#e0f2f1",
        color="#00695c",
        fontcolor="#00695c",
    )

    # Edges
    dot.edge("user", "assistant")
    dot.edge("assistant", "adapter")
    dot.edge("assistant", "storage", style="dashed", color="#c2185b")
    dot.edge("adapter", "provider")
    dot.edge("provider", "protocol")

    return dot


def generate_data_flow() -> graphviz.Digraph:
    """Generate ARCHITECTURE.md data flow diagram."""
    dot = create_graph(
        "data_flow", graph_attr={"rankdir": "TB", "nodesep": "0.6", "ranksep": "1.0"}
    )

    # Steps as nodes
    step1 = add_flow_node(dot, "step1", "1. USER REQUEST\nPOST /api/chat")
    step2 = add_component(
        dot,
        "step2",
        "2. ASSISTANT LAYER\n• Convert protocol → ChatMessage\n• Store last user message\n• Get pipeline adapter",
        fillcolor="#fff3e0",
    )
    step3 = add_component(
        dot,
        "step3",
        "3. ADAPTER LAYER\n• Generate UUID\n• RAG: retrieve() → inject context\n• Call AI provider",
        fillcolor="#e8f5e9",
    )
    step4 = add_flow_node(dot, "step4", "4. AI PROVIDER\nStreaming chunks")
    step5 = add_component(
        dot,
        "step5",
        "5. EVENT NORMALIZATION\nChunks → StreamEvents",
        fillcolor="#e1f5fe",
    )
    step6 = add_component(
        dot,
        "step6",
        "6. PROTOCOL CONVERSION\nEvents → Vercel Protocol\nSSE format",
        fillcolor="#e0f2f1",
    )
    step7 = add_component(
        dot,
        "step7",
        "7. STORAGE\nStreamWriter → ChatMessage\nSame UUID everywhere!",
        fillcolor="#fce4ec",
    )

    # Connect steps
    dot.edge("step1", "step2")
    dot.edge("step2", "step3")
    dot.edge("step3", "step4")
    dot.edge("step4", "step5")
    dot.edge("step5", "step6")
    dot.edge("step3", "step7", style="dashed", constraint="false")

    return dot


def generate_adapter_flow() -> graphviz.Digraph:
    """Generate ADAPTERS.md adapter flow diagram."""
    dot = create_graph("adapter_flow")

    assistant = add_component(
        dot,
        "assistant",
        "Assistant\nget_pipeline_adapter()",
        fillcolor="#fff3e0",
        color="#e65100",
    )
    adapter = add_component(
        dot,
        "adapter",
        "Adapter\n• stream()\n• get_messages()\n• emit events",
        fillcolor="#e8f5e9",
        color="#2e7d32",
    )
    provider = add_component(
        dot,
        "provider",
        "AI Provider\nOpenAI/Haystack",
        fillcolor="#f3e5f5",
        color="#7b1fa2",
    )

    # Add action labels
    dot.edge("assistant", "adapter")
    dot.edge("adapter", "provider")

    # Add annotation nodes
    dot.node(
        "gen_id",
        "Generate ID",
        shape="note",
        fillcolor="#fff9c4",
        color="#f57f17",
        fontsize="9",
    )
    dot.node(
        "format",
        "Format events",
        shape="note",
        fillcolor="#fff9c4",
        color="#f57f17",
        fontsize="9",
    )

    dot.edge("adapter", "gen_id", style="dotted", arrowhead="none", constraint="false")
    dot.edge("adapter", "format", style="dotted", arrowhead="none", constraint="false")

    return dot


def generate_id_generation() -> graphviz.Digraph:
    """Generate ADAPTERS.md ID generation flow diagram."""
    dot = create_graph("id_generation", graph_attr={"rankdir": "TB"})

    dot.attr(compound="true")

    # Create a cluster for the flow
    with dot.subgraph(name="cluster_flow") as c:
        c.attr(
            label="Adapter.stream()",
            style="rounded",
            fillcolor="#f5f5f5",
            color="#666666",
        )

        gen = c.node(
            "gen",
            "message_id = uuid.uuid4()\n← GENERATE",
            fillcolor="#fff9c4",
            color="#f57f17",
            fontcolor="#f57f17",
        )
        start = c.node("start", "yield MessageStartEvent\n(message_id=...)")
        sse = c.node("sse", 'SSE Stream\n{"messageId": "..."}')
        writer = c.node("writer", "StreamWriter\n(message_id=...)")
        msg = c.node("msg", "ChatMessage\n(id=message_id)")
        storage = c.node("storage", "Storage.save()\nSame ID everywhere!")

        c.edge("gen", "start")
        c.edge("start", "sse")
        c.edge("sse", "writer")
        c.edge("writer", "msg")
        c.edge("msg", "storage")

    return dot


def generate_storage_architecture() -> graphviz.Digraph:
    """Generate STORAGE.md storage architecture diagram."""
    dot = create_graph("storage_architecture")

    # Storage Layer cluster
    with dot.subgraph(name="cluster_storage") as c:
        c.attr(
            label="Storage Layer", style="rounded", fillcolor="#f5f5f5", color="#666666"
        )

        memory = add_component(
            c,
            "memory",
            "MemoryStorage\n(In-Memory)",
            fillcolor="#e3f2fd",
            color="#1565c0",
        )
        db = add_component(
            c, "db", "DbStorage\n(Django ORM)", fillcolor="#e8f5e9", color="#2e7d32"
        )
        base = add_component(
            c, "base", "BaseStorage\n(Abstract)", fillcolor="#fff3e0", color="#e65100"
        )

        c.edge("memory", "base", style="dashed")
        c.edge("db", "base", style="dashed")

    # ChatMessage format
    format_node = add_component(
        dot,
        "format",
        "Universal ChatMessage Format\n{id, role, content, model, ...}",
        fillcolor="#fce4ec",
        color="#c2185b",
        fontcolor="#c2185b",
        shape="note",
    )

    dot.edge("base", "format", style="dashed", color="#c2185b")

    return dot


def generate_id_consistency() -> graphviz.Digraph:
    """Generate STORAGE.md ID consistency flow diagram."""
    dot = create_graph("id_consistency", graph_attr={"rankdir": "TB", "ranksep": "0.8"})

    step1 = add_flow_node(dot, "step1", '1. User Request\n"What is the pirate code?"')
    step2 = add_component(
        dot,
        "step2",
        "2. Adapter Layer\nmessage_id = uuid\n← GENERATED ONCE",
        fillcolor="#fff9c4",
        color="#f57f17",
    )
    step3 = add_flow_node(
        dot, "step3", '3. Streaming to Frontend\nSSE: {"messageId": "..."}'
    )
    step4 = add_component(
        dot,
        "step4",
        "4. Storage Layer\nMessage(id=uuid)\nSame ID in database!",
        fillcolor="#e8f5e9",
        color="#2e7d32",
    )
    step5 = add_flow_node(
        dot, "step5", "5. API Endpoints\nGET /threads/.../rate\nSame ID for rating!"
    )

    dot.edge("step1", "step2")
    dot.edge("step2", "step3")
    dot.edge("step3", "step4")
    dot.edge("step4", "step5")

    return dot


def generate_rag_architecture() -> graphviz.Digraph:
    """Generate RAG.md architecture diagram."""
    dot = create_graph("rag_architecture")

    assistant = add_component(
        dot,
        "assistant",
        "Assistant\n• get_rag_queryset()\n• get_rag_documents()\n• get_rag_pipeline()\n• rag_provider",
        fillcolor="#fff3e0",
        color="#e65100",
    )

    provider = add_component(
        dot,
        "provider",
        "RAG Provider\n• warmup()\n• get_rag_instance()\n• build_tool()\n• clear_cache()",
        fillcolor="#e3f2fd",
        color="#1565c0",
    )

    instance = add_component(
        dot,
        "instance",
        "RAG Instance\n• warmup()\n• retrieve(query)\n• format_context()\n• as_tool()",
        fillcolor="#e8f5e9",
        color="#2e7d32",
    )

    adapter = add_component(
        dot,
        "adapter",
        "Adapter\n• Context injection\n• Tool calling",
        fillcolor="#f3e5f5",
        color="#7b1fa2",
    )

    dot.edge("assistant", "provider")
    dot.edge("provider", "instance")
    dot.edge("instance", "adapter")

    return dot


def generate_testing_pyramid() -> graphviz.Digraph:
    """Generate TESTING.md test pyramid diagram."""
    dot = create_graph("testing_pyramid", graph_attr={"rankdir": "TB"})

    # Pyramid levels
    e2e = add_component(
        dot,
        "e2e",
        "E2E Tests\nFull assistant flow",
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
        "Unit Tests\nIndividual components",
        fillcolor="#e8f5e9",
        color="#2e7d32",
        width="4",
    )

    dot.edge("integration", "e2e", style="invis")
    dot.edge("unit", "integration", style="invis")

    # Align them
    dot.graph_attr["rankdir"] = "BT"

    return dot


def generate_directory_structure() -> graphviz.Digraph:
    """Generate directory structure tree diagram."""
    dot = graphviz.Digraph(
        name="directory_structure",
        format="png",
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "white",
            "fontname": "Courier,monospace",
            "dpi": "150",
        },
        node_attr={
            "fontname": "Courier,monospace",
            "fontsize": "10",
            "shape": "none",
        },
    )

    tree_text = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD ALIGN="LEFT" BGCOLOR="#f5f5f5">django_ai_sdk/</TD></TR>
<TR><TD ALIGN="LEFT">├── __init__.py</TD></TR>
<TR><TD ALIGN="LEFT">├── assistant.py</TD></TR>
<TR><TD ALIGN="LEFT">├── common.py</TD></TR>
<TR><TD ALIGN="LEFT">├── events.py</TD></TR>
<TR><TD ALIGN="LEFT">├── adapters/</TD></TR>
<TR><TD ALIGN="LEFT">│   ├── openai.py</TD></TR>
<TR><TD ALIGN="LEFT">│   └── haystack.py</TD></TR>
<TR><TD ALIGN="LEFT">├── protocols/</TD></TR>
<TR><TD ALIGN="LEFT">│   └── vercel.py</TD></TR>
<TR><TD ALIGN="LEFT">├── storage/</TD></TR>
<TR><TD ALIGN="LEFT">│   ├── memory.py</TD></TR>
<TR><TD ALIGN="LEFT">│   └── db.py</TD></TR>
<TR><TD ALIGN="LEFT">├── rags/</TD></TR>
<TR><TD ALIGN="LEFT">│   ├── bm25.py</TD></TR>
<TR><TD ALIGN="LEFT">│   └── haystack/</TD></TR>
<TR><TD ALIGN="LEFT">└── tests/</TD></TR>
</TABLE>
>"""

    manual_tree = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD ALIGN="LEFT" BGCOLOR="#e3f2fd">manual/</TD></TR>
<TR><TD ALIGN="LEFT">├── README.md</TD></TR>
<TR><TD ALIGN="LEFT">├── ARCHITECTURE.md</TD></TR>
<TR><TD ALIGN="LEFT">├── RAG.md</TD></TR>
<TR><TD ALIGN="LEFT">├── ADAPTERS.md</TD></TR>
<TR><TD ALIGN="LEFT">├── STORAGE.md</TD></TR>
<TR><TD ALIGN="LEFT">└── TESTING.md</TD></TR>
</TABLE>
>"""

    dot.node("sdk", tree_text)
    dot.node("manual", manual_tree)

    return dot


# Dictionary of all diagrams to generate
DIAGRAMS: dict[str, Any] = {
    "overview_architecture": generate_overview_architecture,
    "data_flow": generate_data_flow,
    "adapter_flow": generate_adapter_flow,
    "id_generation": generate_id_generation,
    "storage_architecture": generate_storage_architecture,
    "id_consistency": generate_id_consistency,
    "rag_architecture": generate_rag_architecture,
    "testing_pyramid": generate_testing_pyramid,
    "directory_structure": generate_directory_structure,
}


def generate_all():
    """Generate all diagram images to both manual and docs directories."""
    print("Generating Django AI SDK documentation diagrams...")
    print(f"Manual output: {MANUAL_GRAPHS_DIR}")
    print(f"Docs output: {DOCS_GRAPHS_DIR}")
    print()

    # Ensure output directories exist
    MANUAL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    manual_generated = []
    docs_generated = []
    failed = []

    for name, generator_func in DIAGRAMS.items():
        try:
            print(f"Generating {name}...", end=" ")
            dot = generator_func()

            # Generate to manual directory
            manual_path = MANUAL_GRAPHS_DIR / name
            dot.render(str(manual_path), cleanup=True)
            manual_generated.append(name)

            # Generate to docs directory
            docs_path = DOCS_GRAPHS_DIR / name
            dot.render(str(docs_path), cleanup=True)
            docs_generated.append(name)

            print("✓")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"✗ ({e})")

    print()
    print(f"Generated {len(manual_generated)} diagrams to manual/graphs/")
    print(f"Generated {len(docs_generated)} diagrams to docs/static/images/graphs/")

    if failed:
        print()
        print(f"Failed to generate {len(failed)} diagrams:")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")

    print()
    print("Done!")


if __name__ == "__main__":
    generate_all()

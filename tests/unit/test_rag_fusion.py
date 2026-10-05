"""Multi-query results are fused by rank, and the user's own words count verbatim."""

from __future__ import annotations

from haystack.dataclasses import Document

from django_ai_sdk.rags.components import MultiQueryDeduplicationMixin

fuse = MultiQueryDeduplicationMixin.fuse_and_rank


def ranked(*docs: Document) -> dict[str, list[Document]]:
    """A retrieval result: `docs` best first, scored like a retriever would."""
    return {
        "documents": [
            Document(id=d.id, content=d.content, meta=d.meta, score=1 / (i + 1))
            for i, d in enumerate(docs)
        ]
    }


CVS = [
    Document(id=f"cv{i}", content=f"Consultant with Oracle experience, project {i}")
    for i in range(20)
]
SABRE_CV = Document(id="sabre", content="Travel consultant: years of SABRE reservations work")


def test_an_exact_rare_term_lifts_a_document_only_one_query_found_low() -> None:
    # As in the "Test Me" CVs: only the plain query found the SABRE chunk, at rank 7,
    # while the expanded queries found other CVs.
    original = ranked(*CVS[:6], SABRE_CV, *CVS[6:13])
    variants = [ranked(*CVS[:14]) for _ in range(3)]

    docs = fuse([original, *variants], ["SABRE", "a", "b", "c"], top_k=5, query="SABRE")

    assert docs[0].id == "sabre"


def test_what_several_queries_find_outranks_one_querys_top_hit() -> None:
    once, everywhere = CVS[0], CVS[1]
    results = [ranked(once, everywhere), ranked(everywhere), ranked(everywhere)]

    docs = fuse(results, ["x1", "x2", "x3"], top_k=2)

    assert [d.id for d in docs] == ["cv1", "cv0"]


def test_words_in_most_candidates_do_not_count_as_exact_terms() -> None:
    # "consultant" is in every CV: matching it says nothing, so it can't reorder them.
    results = [ranked(*CVS[:5]), ranked(*CVS[:5])]

    docs = fuse(results, ["q", "q2"], top_k=5, query="consultant")

    assert [d.id for d in docs] == [d.id for d in CVS[:5]]


def test_hits_below_min_score_are_left_out() -> None:
    docs = fuse([ranked(*CVS[:4])], ["q"], top_k=5, min_score=0.4)

    assert [d.id for d in docs] == ["cv0", "cv1"]  # scores 1, 0.5; not 0.33, 0.25


def incident(month: str, number: str) -> Document:
    """A chunk of an incident report: its ID in the text, its file name in the meta."""
    inc = f"INC-2025-{month}-{number}"
    return Document(
        id=f"{month}-{number}",
        content=f"Incident {inc}: a cooling unit failed in the data centre.",
        meta={"name": f"incident_2025-{month}_{inc}.pdf"},
    )


INCIDENTS = [incident(m, n) for m in ("01", "02", "03", "11") for n in ("001", "002")]


def test_an_identifier_matches_whole_not_every_document_with_the_same_parts():
    # Ranked by the embeddings, the asked-for incident is last among the "-002"s.
    others = [d for d in INCIDENTS if d.id != "03-002"]
    results = [ranked(*others, incident("03", "002"))]

    docs = fuse(results, ["INC-2025-03-002"], top_k=3, query="INC-2025-03-002")

    assert docs[0].id == "03-002"


def test_a_file_name_finds_that_file_though_its_text_does_not_contain_it():
    target = Document(
        id="t",
        content="A cooling unit failed.",
        meta={"name": "incident_2025-03_INC-2025-03-002.pdf"},
    )
    # The other incidents' text and names share every part of the name but the whole.
    results = [ranked(*[d for d in INCIDENTS if d.id != "03-002"], target)]

    docs = fuse(results, ["q"], top_k=3, query="incident_2025-03_INC-2025-03-002.pdf")

    assert docs[0].id == "t"

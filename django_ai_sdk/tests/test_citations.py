"""Simple tests for citation formatter and registry."""

from django_ai_sdk.citations import (
    CitationRegistry,
    DefaultCitationFormatter,
    NumberedSource,
)


class TestCitationRegistry:
    """Test CitationRegistry monotonic counter and source tracking."""

    def test_next_index_starts_at_1(self):
        """First citation should be [1], not [0]."""
        registry = CitationRegistry()
        assert registry.next_index == 1

    def test_monotonic_increment(self):
        """Indices increment correctly across multiple adds."""
        registry = CitationRegistry()

        # Add 3 sources
        sources1 = [
            NumberedSource(index=1, title="doc1", content="text1"),
            NumberedSource(index=2, title="doc2", content="text2"),
            NumberedSource(index=3, title="doc3", content="text3"),
        ]
        registry.add(sources1)

        # Next index should be 4
        assert registry.next_index == 4

        # Add 2 more
        sources2 = [
            NumberedSource(index=4, title="doc4", content="text4"),
            NumberedSource(index=5, title="doc5", content="text5"),
        ]
        registry.add(sources2)

        # Next index should be 6
        assert registry.next_index == 6

    def test_all_sources_returns_list(self):
        """all_sources returns all added sources in order."""
        registry = CitationRegistry()
        source1 = NumberedSource(index=1, title="first", content="content1")
        source2 = NumberedSource(index=2, title="second", content="content2")

        registry.add([source1, source2])
        all_sources = registry.all_sources

        assert len(all_sources) == 2
        assert all_sources[0].title == "first"
        assert all_sources[1].title == "second"

    def test_all_sources_is_defensive_copy(self):
        """Modifying returned list doesn't affect registry."""
        registry = CitationRegistry()
        source = NumberedSource(index=1, title="test", content="content")
        registry.add([source])

        sources = registry.all_sources
        sources.clear()  # Try to mutate

        # Registry should still have the source
        assert len(registry.all_sources) == 1

    def test_len_returns_counter(self):
        """__len__ returns total number of sources added."""
        registry = CitationRegistry()
        registry.add([
            NumberedSource(index=1, title="a", content="x"),
            NumberedSource(index=2, title="b", content="y"),
        ])

        assert len(registry) == 2


class TestDefaultCitationFormatter:
    """Test document formatting and title generation."""

    def test_format_returns_tuple(self):
        """format() returns (llm_text, numbered_sources)."""
        formatter = DefaultCitationFormatter()
        docs = [{"content": "test", "meta": {}}]

        text, sources = formatter.format(docs, start_index=1)

        assert isinstance(text, str)
        assert isinstance(sources, list)
        assert len(sources) == 1

    def test_title_from_file_name(self):
        """Title prefers file_name from metadata."""
        formatter = DefaultCitationFormatter()
        docs = [{"content": "content", "meta": {"file_name": "report.pdf"}}]

        _, sources = formatter.format(docs, start_index=1)

        assert sources[0].title == "report.pdf"

    def test_title_with_page_number(self):
        """Title adds page number if split_id and page_number exist."""
        formatter = DefaultCitationFormatter()
        docs = [
            {
                "content": "content",
                "meta": {"file_name": "report.pdf", "split_id": 0, "page_number": 3},
            }
        ]

        _, sources = formatter.format(docs, start_index=1)

        assert sources[0].title == "report.pdf · p3"

    def test_title_with_section_marker(self):
        """Title adds section marker if split_id but no page_number."""
        formatter = DefaultCitationFormatter()
        docs = [
            {
                "content": "content",
                "meta": {"file_name": "notes.txt", "split_id": 1},  # 0-indexed, so §2
            }
        ]

        _, sources = formatter.format(docs, start_index=1)

        assert sources[0].title == "notes.txt · §2"

    def test_title_fallback_to_document_n(self):
        """Title falls back to 'Document N' if no metadata."""
        formatter = DefaultCitationFormatter()
        docs = [{"content": "content", "meta": {}}]

        _, sources = formatter.format(docs, start_index=1)

        assert sources[0].title == "Document 1"

    def test_start_index_offset(self):
        """Indices start from start_index, not 1."""
        formatter = DefaultCitationFormatter()
        docs = [
            {"content": "first", "meta": {}},
            {"content": "second", "meta": {}},
        ]

        _, sources = formatter.format(docs, start_index=5)

        assert sources[0].index == 5
        assert sources[1].index == 6

    def test_metadata_preserved(self):
        """Metadata dict is extracted and preserved."""
        formatter = DefaultCitationFormatter()
        docs = [
            {
                "content": "content",
                "meta": {
                    "file_name": "doc.pdf",
                    "page_number": 1,
                    "split_id": 0,
                    "custom_field": "ignored",
                },
            }
        ]

        _, sources = formatter.format(docs, start_index=1)

        # Only file_name, page_number, split_id are kept
        assert "file_name" in sources[0].metadata
        assert "page_number" in sources[0].metadata
        assert "split_id" in sources[0].metadata
        assert "custom_field" not in sources[0].metadata

    def test_xml_output_contains_source_tags(self):
        """LLM-visible output contains <source id='N'> tags."""
        formatter = DefaultCitationFormatter()
        docs = [{"content": "test content", "meta": {"file_name": "doc.txt"}}]

        text, _ = formatter.format(docs, start_index=1)

        assert '<source id="1">' in text
        assert "test content" in text
        assert "</source>" in text

    def test_empty_documents_list(self):
        """Empty documents list returns valid empty output."""
        formatter = DefaultCitationFormatter()

        text, sources = formatter.format([], start_index=1)

        assert isinstance(text, str)
        assert sources == []

    def test_subclass_custom_template(self):
        """Subclasses can override RAG_TEMPLATE."""

        class CustomFormatter(DefaultCitationFormatter):
            RAG_TEMPLATE = "CUSTOM: cite as [N]"

        formatter = CustomFormatter()
        docs = [{"content": "test", "meta": {}}]

        text, _ = formatter.format(docs, start_index=1)

        assert "CUSTOM: cite as [N]" in text

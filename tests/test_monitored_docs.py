"""
Tests for app.ingestion.monitored_docs — the regulatory document registry.

These are pure unit tests with no network or DB calls.
"""

import pytest
from app.ingestion.monitored_docs import (
    get_monitored_docs,
    get_monitored_doc,
    list_all_countries,
    get_all_docs,
)

# All 15 countries mandated by the PRD
PRD_COUNTRIES = {"US", "EU", "UK", "UA", "IN", "CA", "AU", "JP", "CN", "BR", "KR", "CH", "MX", "RU", "SA"}


class TestGetMonitoredDocs:
    def test_all_15_countries_have_docs(self):
        """Every PRD country must have at least one monitored document."""
        for country in PRD_COUNTRIES:
            docs = get_monitored_docs(country)
            assert len(docs) >= 1, f"No monitored docs for country {country}"

    def test_country_case_insensitive(self):
        """get_monitored_docs should be case-insensitive."""
        assert get_monitored_docs("us") == get_monitored_docs("US")
        assert get_monitored_docs("Eu") == get_monitored_docs("EU")

    def test_unknown_country_returns_empty(self):
        """Unknown countries return an empty list, not an error."""
        assert get_monitored_docs("ZZ") == []
        assert get_monitored_docs("") == []

    def test_each_doc_has_required_fields(self):
        """Every document entry must have all three required fields."""
        required = {"document_id", "source_url", "regulation_name"}
        for doc in get_all_docs():
            missing = required - doc.keys()
            assert not missing, f"Doc {doc.get('document_id')} missing fields: {missing}"

    def test_document_ids_are_unique(self):
        """document_id values must be globally unique across all countries."""
        all_ids = [d["document_id"] for d in get_all_docs()]
        assert len(all_ids) == len(set(all_ids)), "Duplicate document_ids found"

    def test_source_urls_are_https(self):
        """All source URLs must use HTTPS (government sites)."""
        for doc in get_all_docs():
            url = doc["source_url"]
            assert url.startswith("https://"), (
                f"source_url for {doc['document_id']} does not use HTTPS: {url}"
            )

    def test_source_urls_are_non_empty(self):
        """No doc should have an empty source_url."""
        for doc in get_all_docs():
            assert doc["source_url"].strip(), (
                f"Empty source_url for {doc['document_id']}"
            )

    def test_regulation_names_are_non_empty(self):
        """Every regulation_name must be a non-empty string."""
        for doc in get_all_docs():
            assert doc["regulation_name"].strip(), (
                f"Empty regulation_name for {doc['document_id']}"
            )


class TestGetMonitoredDoc:
    def test_lookup_existing_doc(self):
        """Returns the correct document given country + document_id."""
        doc = get_monitored_doc("US", "US_FDA_QMSR_2026")
        assert doc is not None
        assert doc["document_id"] == "US_FDA_QMSR_2026"
        assert "source_url" in doc
        assert "ecfr.gov" in doc["source_url"]

    def test_returns_none_for_unknown_document_id(self):
        """Returns None when document_id is not in the registry."""
        assert get_monitored_doc("US", "NONEXISTENT") is None

    def test_returns_none_for_unknown_country(self):
        """Returns None when country is not in the registry."""
        assert get_monitored_doc("ZZ", "US_FDA_21CFR820") is None

    def test_ukraine_docs_present(self):
        """Ukraine (UA) must have at least one document (the target market)."""
        docs = get_monitored_docs("UA")
        assert len(docs) >= 1
        doc_ids = [d["document_id"] for d in docs]
        assert any("UA" in did for did in doc_ids)


class TestListAllCountries:
    def test_returns_all_15_countries(self):
        """list_all_countries must return exactly the 15 PRD countries."""
        countries = set(list_all_countries())
        assert countries == PRD_COUNTRIES

    def test_returns_sorted_list(self):
        """list_all_countries must return a sorted list."""
        result = list_all_countries()
        assert result == sorted(result)


class TestGetAllDocs:
    def test_returns_flat_list(self):
        """get_all_docs returns a flat list of dicts with a 'country' key."""
        docs = get_all_docs()
        assert isinstance(docs, list)
        assert len(docs) >= 15  # at least one per country

    def test_each_doc_has_country_field(self):
        """Every entry from get_all_docs must include the 'country' field."""
        for doc in get_all_docs():
            assert "country" in doc, f"Missing 'country' in {doc.get('document_id')}"
            assert doc["country"] in PRD_COUNTRIES

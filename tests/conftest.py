"""
Shared test fixtures and configuration.
"""


import pytest


def pytest_addoption(parser):
    parser.addoption("--server-url", action="store", default=None, help="Base URL for integration tests (e.g. http://localhost:8000)")


def pytest_configure(config):
    """Set ORTHOLINK_ACCEPTANCE_BASE_URL from --server-url before tests load."""
    try:
        url = config.getoption("--server-url", default=None)
        if url:
            import os
            os.environ["ORTHOLINK_ACCEPTANCE_BASE_URL"] = url
    except ValueError:
        pass


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, request):
    """Set test environment variables for all tests."""
    import os
    from pathlib import Path
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DEBUG", "true")
    # Integration tests need real OpenAI for embeddings/DVA; leave OPENAI_API_KEY from env
    if "integration" not in request.keywords:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-jwt-secret-for-unit-tests-only")
    # Use real FAISS index for integration tests so they see populated data
    if "integration" in request.keywords:
        backend_root = Path(__file__).resolve().parent.parent
        monkeypatch.setenv("FAISS_INDEX_PATH", str(backend_root / "data" / "embeddings"))
    else:
        monkeypatch.setenv("FAISS_INDEX_PATH", "/tmp/test-faiss-index")

    # Clear cached settings so each test gets fresh config
    from app.core.config import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()

"""
Tests for RBAC middleware.
"""


from app.middleware.rbac import ROLE_HIERARCHY, get_role_level


class TestRoleHierarchy:
    """Test role hierarchy: admin > reviewer > viewer."""

    def test_admin_highest(self):
        assert get_role_level("admin") == 3

    def test_reviewer_middle(self):
        assert get_role_level("reviewer") == 2

    def test_viewer_lowest(self):
        assert get_role_level("viewer") == 1

    def test_unknown_role_zero(self):
        assert get_role_level("unknown") == 0

    def test_admin_gt_reviewer(self):
        assert get_role_level("admin") > get_role_level("reviewer")

    def test_reviewer_gt_viewer(self):
        assert get_role_level("reviewer") > get_role_level("viewer")

    def test_admin_gt_viewer(self):
        assert get_role_level("admin") > get_role_level("viewer")

    def test_case_insensitive(self):
        assert get_role_level("Admin") == get_role_level("admin")
        assert get_role_level("VIEWER") == get_role_level("viewer")

    def test_hierarchy_completeness(self):
        """Ensure all expected roles are in the hierarchy."""
        assert "admin" in ROLE_HIERARCHY
        assert "reviewer" in ROLE_HIERARCHY
        assert "viewer" in ROLE_HIERARCHY
        assert len(ROLE_HIERARCHY) == 3

"""Budget model, quota evaluation, API, and enforcement tests (issues #369/#466)."""

from datetime import UTC, datetime, timedelta

import pytest

from app.extensions import db
from app.models import (
    AuditLog,
    Budget,
    Conversation,
    Notification,
    NotificationPreference,
    UsageRecord,
    Workspace,
)
from app.services import audit, budgets


def _workspace(user, name="Workspace"):
    workspace = Workspace(user_id=user.id, name=name)
    db.session.add(workspace)
    db.session.commit()
    return workspace


def _conversation(user, workspace=None):
    conversation = Conversation(
        user_id=user.id, title="Test", workspace_id=workspace.id if workspace else None
    )
    db.session.add(conversation)
    db.session.commit()
    return conversation


def _usage(user, *, tokens=0, cost=0.0, workspace=None, status="success", created_at=None):
    record = UsageRecord(
        user_id=user.id,
        workspace_id=workspace.id if workspace else None,
        source="chat",
        total_tokens=tokens,
        cost_estimate=cost,
        status=status,
        created_at=created_at or datetime.now(UTC),
    )
    db.session.add(record)
    db.session.commit()
    return record


def _token_budget(user, limit, *, threshold=0.8):
    return budgets.upsert_budget(
        scope="user",
        scope_id=user.id,
        token_limit=limit,
        warn_threshold=threshold,
    )


class TestQuotaEvaluation:
    def test_period_start_is_first_of_utc_month(self):
        start = budgets.period_start(datetime(2026, 10, 20, 13, 45, tzinfo=UTC))
        assert start == datetime(2026, 10, 1, tzinfo=UTC)
        assert budgets.next_period_start(start) == datetime(2026, 11, 1, tzinfo=UTC)
        assert budgets.next_period_start(datetime(2026, 12, 5, tzinfo=UTC)) == datetime(
            2027, 1, 1, tzinfo=UTC
        )

    def test_token_quota_warns_then_blocks(self, app, make_user):
        user = make_user()
        _token_budget(user, 100, threshold=0.8)
        _usage(user, tokens=79)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.allowed
        assert not status.warned

        _usage(user, tokens=1)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.allowed
        assert status.warned

        _usage(user, tokens=20)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert not status.allowed
        assert status.limits_hit == ("tokens",)

    def test_cost_quota_is_independent_of_tokens(self, app, make_user):
        user = make_user()
        budgets.upsert_budget(scope="user", scope_id=user.id, cost_limit=1.0, warn_threshold=0.5)
        _usage(user, tokens=1_000_000, cost=0.49)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.allowed
        assert not status.warned

        _usage(user, tokens=0, cost=0.02)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.allowed
        assert status.warned

        _usage(user, tokens=0, cost=0.5)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert not status.allowed
        assert status.limits_hit == ("cost",)

    def test_both_quotas_report_every_limit_hit(self, app, make_user):
        user = make_user()
        budgets.upsert_budget(scope="user", scope_id=user.id, token_limit=10, cost_limit=0.10)
        _usage(user, tokens=10, cost=0.10)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert not status.allowed
        assert status.limits_hit == ("tokens", "cost")
        decision = budgets.check_budgets(user)
        assert not decision.allowed
        assert set(decision.limits_hit) == {"tokens", "cost"}
        assert "token limit" in decision.message
        assert "cost limit" in decision.message

    def test_only_token_quota_configured_ignores_cost(self, app, make_user):
        user = make_user()
        _token_budget(user, 5)
        _usage(user, tokens=1, cost=1_000.0)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.allowed

    def test_period_boundary_excludes_previous_month(self, app, make_user):
        user = make_user()
        _token_budget(user, 10)
        start = budgets.period_start()
        _usage(user, tokens=9, created_at=start - timedelta(seconds=1))
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.tokens_used == 0
        assert status.allowed

    def test_denied_usage_does_not_count(self, app, make_user):
        user = make_user()
        _token_budget(user, 10)
        _usage(user, tokens=10, status="denied")
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert status.tokens_used == 0
        assert status.allowed

    def test_zero_limit_blocks_immediately(self, app, make_user):
        user = make_user()
        _token_budget(user, 0)
        status = budgets.evaluate_budget(budgets.budgets_for(user)[0])
        assert not status.allowed
        assert status.limits_hit == ("tokens",)


class TestBudgetScopes:
    def test_workspace_budget_only_applies_with_workspace(self, app, make_user):
        user = make_user()
        workspace = _workspace(user)
        budgets.upsert_budget(scope="workspace", scope_id=workspace.id, token_limit=10)
        _usage(user, tokens=10, workspace=workspace)

        assert budgets.check_budgets(user).allowed
        decision = budgets.check_budgets(user, workspace.id)
        assert not decision.allowed
        assert decision.statuses[0].scope == "workspace"

    def test_user_and_workspace_budgets_are_both_enforced(self, app, make_user):
        user = make_user()
        workspace = _workspace(user)
        _token_budget(user, 100)
        budgets.upsert_budget(scope="workspace", scope_id=workspace.id, token_limit=5)
        _usage(user, tokens=5, workspace=workspace)

        decision = budgets.check_budgets(user, workspace.id)
        assert not decision.allowed
        assert {status.scope for status in decision.statuses if not status.allowed} == {"workspace"}

    def test_most_restrictive_budget_blocks(self, app, make_user):
        user = make_user()
        workspace = _workspace(user)
        _token_budget(user, 5)
        budgets.upsert_budget(scope="workspace", scope_id=workspace.id, cost_limit=1.0)
        _usage(user, tokens=6, cost=0.1, workspace=workspace)

        decision = budgets.check_budgets(user, workspace.id)
        assert not decision.allowed
        assert decision.limits_hit == ("tokens",)

    def test_one_budget_per_scope_is_upserted(self, app, make_user):
        user = make_user()
        _token_budget(user, 10)
        _token_budget(user, 20)
        assert Budget.query.filter_by(user_id=user.id).count() == 1
        assert Budget.query.filter_by(user_id=user.id).one().token_limit == 20


class TestBudgetApi:
    def test_user_budget_can_be_configured_and_listed(self, client, make_user, login):
        make_user()
        login()

        response = client.put(
            "/billing/api/budgets",
            json={"scope": "user", "token_limit": 500, "cost_limit": 2.5},
        )
        assert response.status_code == 200, response.get_json()
        body = response.get_json()["budget"]
        assert body["scope"] == "user"
        assert body["token_limit"] == 500
        assert body["cost_limit"] == 2.5

        listing = client.get("/billing/api/budgets").get_json()
        assert len(listing["budgets"]) == 1
        assert listing["budgets"][0]["token_limit"] == 500

    def test_both_quota_types_can_be_set(self, client, make_user, login):
        make_user()
        login()
        response = client.put(
            "/billing/api/budgets",
            json={"scope": "user", "token_limit": 0, "cost_limit": 0},
        )
        assert response.status_code == 200
        body = response.get_json()["budget"]
        assert body["token_limit"] == 0
        assert body["cost_limit"] == 0

    def test_requires_at_least_one_limit(self, client, make_user, login):
        make_user()
        login()
        response = client.put("/billing/api/budgets", json={"scope": "user"})
        assert response.status_code == 400
        assert "Set a token limit" in response.get_json()["error"]

    def test_rejects_negative_and_invalid_threshold(self, client, make_user, login):
        make_user()
        login()
        assert (
            client.put(
                "/billing/api/budgets",
                json={"scope": "user", "token_limit": -1},
            ).status_code
            == 400
        )
        assert (
            client.put(
                "/billing/api/budgets",
                json={"scope": "user", "token_limit": 1, "warn_threshold": 1.5},
            ).status_code
            == 400
        )

    def test_workspace_budget_requires_owner(self, client, make_user, login, db):
        owner = make_user()
        other = make_user("other", "other@example.com")
        workspace = _workspace(owner)

        login()
        response = client.put(
            "/billing/api/budgets",
            json={"scope": "workspace", "workspace_id": workspace.id, "token_limit": 10},
        )
        assert response.status_code == 200

        login("other@example.com")
        response = client.put(
            "/billing/api/budgets",
            json={"scope": "workspace", "workspace_id": workspace.id, "token_limit": 20},
        )
        # Non-members get the same 404 as an unknown workspace (no oracle).
        assert response.status_code == 404
        assert other.id != owner.id

    def test_workspace_contributor_cannot_configure_budget(self, client, make_user, login, db):
        owner = make_user()
        contributor = make_user("contributor", "contributor@example.com")
        workspace = _workspace(owner)
        from app.models import WorkspaceMember

        db.session.add(
            WorkspaceMember(
                workspace_id=workspace.id,
                user_id=contributor.id,
                role="contributor",
                status="active",
            )
        )
        db.session.commit()

        login("contributor@example.com")
        response = client.put(
            "/billing/api/budgets",
            json={"scope": "workspace", "workspace_id": workspace.id, "token_limit": 20},
        )
        assert response.status_code == 403

    def test_delete_requires_matching_scope(self, client, make_user, login):
        user = make_user()
        budget = _token_budget(user, 10)

        login()
        assert client.delete(f"/billing/api/budgets/{budget.id}").status_code == 200
        assert db.session.get(Budget, budget.id) is None

    def test_delete_rejects_another_users_budget(self, client, make_user, login):
        owner = make_user()
        make_user("intruder", "intruder@example.com")
        budget = _token_budget(owner, 10)

        login("intruder@example.com")
        assert client.delete(f"/billing/api/budgets/{budget.id}").status_code == 403


class TestBudgetAudit:
    def test_create_records_bounded_diff_and_actor(self, client, make_user, login):
        user = make_user()
        login()

        response = client.put(
            "/billing/api/budgets",
            json={"scope": "user", "token_limit": 500, "cost_limit": 2.5},
        )
        assert response.status_code == 200
        budget = Budget.query.one()

        entry = AuditLog.query.filter_by(action=audit.BUDGET_CREATED).one()
        assert entry.user_id == user.id
        assert entry.target_type == "budget"
        assert entry.target_id == budget.id
        metadata = entry.metadata_dict
        assert metadata["scope"] == "user"
        assert metadata["scope_id"] == user.id
        assert metadata["diff"]["token_limit"] == {"before": None, "after": 500}
        assert metadata["diff"]["cost_limit"] == {"before": None, "after": 2.5}
        # The service owns the audit write; the route must not add a second row.
        assert AuditLog.query.filter_by(action=audit.BUDGET_CREATED).count() == 1
        assert AuditLog.query.filter_by(action=audit.BUDGET_UPDATED).count() == 0

    def test_update_records_only_changed_fields(self, client, make_user, login):
        make_user()
        login()
        client.put(
            "/billing/api/budgets",
            json={
                "scope": "user",
                "token_limit": 100,
                "cost_limit": 1.0,
                "warn_threshold": 0.8,
            },
        )
        client.put(
            "/billing/api/budgets",
            json={
                "scope": "user",
                "token_limit": 200,
                "cost_limit": 1.0,
                "warn_threshold": 0.5,
            },
        )

        entry = AuditLog.query.filter_by(action=audit.BUDGET_UPDATED).one()
        diff = entry.metadata_dict["diff"]
        assert diff["token_limit"] == {"before": 100, "after": 200}
        assert diff["warn_threshold"] == {"before": 0.8, "after": 0.5}
        assert "cost_limit" not in diff

    def test_zero_limits_are_preserved_in_diff(self, client, make_user, login):
        make_user()
        login()
        client.put(
            "/billing/api/budgets",
            json={"scope": "user", "token_limit": 0, "cost_limit": 0},
        )

        diff = AuditLog.query.filter_by(action=audit.BUDGET_CREATED).one().metadata_dict["diff"]
        assert diff["token_limit"] == {"before": None, "after": 0}
        assert diff["cost_limit"] == {"before": None, "after": 0}

    def test_unknown_payload_fields_are_not_audited(self, client, make_user, login):
        make_user()
        login()
        response = client.put(
            "/billing/api/budgets",
            json={
                "scope": "user",
                "token_limit": 10,
                "api_key": "sk-secret",
                "password": "hunter2",
                "token": "top-secret",
            },
        )
        assert response.status_code == 200

        entry = AuditLog.query.filter_by(action=audit.BUDGET_CREATED).one()
        metadata = entry.metadata_dict
        assert set(metadata) == {"scope", "scope_id", "diff"}
        serialized = str(metadata)
        assert "sk-secret" not in serialized
        assert "hunter2" not in serialized
        assert "top-secret" not in serialized

    def test_direct_service_upsert_is_audited(self, app, make_user):
        user = make_user()
        budget = budgets.upsert_budget(
            scope="user",
            scope_id=user.id,
            token_limit=42,
            actor=user,
        )

        entry = AuditLog.query.filter_by(action=audit.BUDGET_CREATED).one()
        assert entry.user_id == user.id
        assert entry.target_id == budget.id

    def test_validation_failure_writes_no_audit(self, client, make_user, login):
        make_user()
        login()
        response = client.put("/billing/api/budgets", json={"scope": "user"})
        assert response.status_code == 400
        assert (
            AuditLog.query.filter(
                AuditLog.action.in_([audit.BUDGET_CREATED, audit.BUDGET_UPDATED])
            ).count()
            == 0
        )


class TestNotifications:
    def test_warning_and_block_notifications_are_emitted_once(self, app, make_user):
        user = make_user()
        _token_budget(user, 10, threshold=0.5)
        _usage(user, tokens=5)

        decision = budgets.check_budgets(user)
        assert decision.warned
        created = budgets.notify_decision(user, decision, level="warning")
        db.session.commit()
        assert len(created) == 1
        assert created[0].type == "budget_alert"

        # Deduplicated for the same budget/period/level.
        assert budgets.notify_decision(user, decision, level="warning") == []
        assert Notification.query.filter_by(type="budget_alert").count() == 1

    def test_usage_recording_triggers_warning_notification(self, app, make_user):
        from app.services import usage_ledger

        user = make_user()
        _token_budget(user, 10, threshold=0.5)
        usage_ledger.record_usage(source="chat", user=user, total_tokens=6)

        assert Notification.query.filter_by(type="budget_alert").count() == 1

    def test_budget_alert_preference_suppresses_delivery(self, app, make_user):
        user = make_user()
        db.session.add(NotificationPreference(user_id=user.id, budget_alerts=False))
        db.session.commit()
        _token_budget(user, 10, threshold=0.5)
        _usage(user, tokens=5)

        decision = budgets.check_budgets(user)
        assert decision.warned
        assert budgets.notify_decision(user, decision, level="warning") == []
        db.session.commit()
        assert Notification.query.filter_by(type="budget_alert").count() == 0

    def test_block_notification_mentions_limits(self, app, make_user):
        user = make_user()
        _token_budget(user, 1)
        _usage(user, tokens=1)
        decision = budgets.check_budgets(user)
        budgets.notify_decision(user, decision, level="blocked")
        db.session.commit()
        notification = Notification.query.filter_by(type="budget_alert").one()
        assert notification.payload["limits_hit"] == ["tokens"]
        assert notification.payload["level"] == "blocked"


class TestEnforcement:
    def test_blocked_call_returns_429_with_limit(self, client, make_user, login):
        user = make_user()
        conversation = _conversation(user)
        _token_budget(user, 1)
        _usage(user, tokens=1)
        login()

        response = client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "hello"},
        )
        assert response.status_code == 429
        body = response.get_json()
        assert body["kind"] == "budget_exceeded"
        assert body["limits_hit"] == ["tokens"]
        assert response.headers["Retry-After"]
        assert db.session.get(Conversation, conversation.id).messages == []

    def test_allowed_call_succeeds_and_warns_via_header(self, client, make_user, login):
        user = make_user()
        conversation = _conversation(user)
        _token_budget(user, 1000, threshold=0.01)
        _usage(user, tokens=10)
        login()

        response = client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "hello"},
        )
        assert response.status_code == 201, response.get_json()
        assert response.headers.get("X-Budget-Warning") == "approaching_limit"

    def test_disabled_enforcement_allows_call(self, client, make_user, login, app):
        app.config["BUDGET_ENFORCEMENT_ENABLED"] = False
        user = make_user()
        conversation = _conversation(user)
        _token_budget(user, 1)
        _usage(user, tokens=100)
        login()

        response = client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "hello"},
        )
        assert response.status_code == 201

    def test_evaluation_error_fails_closed(self, client, make_user, login, monkeypatch):
        user = make_user()
        conversation = _conversation(user)
        _token_budget(user, 10)
        login()

        def boom(*args, **kwargs):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(budgets, "usage_totals", boom)
        response = client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "hello"},
        )
        assert response.status_code == 503
        assert response.get_json()["kind"] == "budget_check_failed"

    def test_user_budget_applies_to_workspace_conversation(self, client, make_user, login):
        user = make_user()
        workspace = _workspace(user)
        _token_budget(user, 1)
        _usage(user, tokens=1)
        conversation = _conversation(user, workspace)
        login()

        response = client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "hello"},
        )
        assert response.status_code == 429
        assert response.get_json()["limits_hit"] == ["tokens"]


class TestBudgetValidationUnit:
    def test_period_must_be_monthly(self):
        with pytest.raises(budgets.BudgetValidationError):
            budgets.validate_period("weekly")

    def test_scope_must_be_known(self, app, make_user):
        user = make_user()
        with pytest.raises(budgets.BudgetValidationError):
            budgets.upsert_budget(scope="team", scope_id=user.id, token_limit=1)

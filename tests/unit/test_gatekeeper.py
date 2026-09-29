"""Unit tests for Gatekeeper — all 7 event types, no Redis required."""

import time
import uuid
import pytest
from unittest.mock import MagicMock

from sentinel_engine.tokens import (
    BehaviorToken, EventType, HandsState, Posture, Severity, Velocity
)
from sentinel_engine.gatekeeper import Gatekeeper


def make_cfg(**overrides) -> dict:
    base = {
        "tenant_id": "test",
        "hysteresis_frames": 1,
        "fall_confirmation_frames": 1,
        "loitering_threshold_s": 1,
        "cooldowns": {
            "FALL_DETECTED": 0,
            "RUNNING_IN_ZONE": 0,
            "HANDS_RAISED": 0,
            "AGGRESSIVE_BEHAVIOR": 0,
            "UNKNOWN_PERSON": 0,
            "LOITERING": 0,
            "ZONE_INTRUSION": 0,
        },
    }
    base.update(overrides)
    return base


def make_token(
    person_id=1,
    posture=Posture.STANDING,
    velocity=Velocity.WALKING,
    velocity_value=1.0,
    hands_state=HandsState.LOWERED,
    zone_id=None,
    zone_name=None,
) -> BehaviorToken:
    return BehaviorToken(
        person_id=person_id,
        camera_id="cam_01",
        tenant_id="test",
        bbox=[100, 100, 200, 400],
        keypoints=None,
        posture=posture,
        velocity=velocity,
        velocity_value=velocity_value,
        hands_state=hands_state,
        zone_id=zone_id,
        zone_name=zone_name,
        trace_id=uuid.uuid4(),
    )


def make_gatekeeper(rules=None) -> Gatekeeper:
    cfg = make_cfg()
    zones = []
    if rules:
        zones = [{"id": "zone_01", "name": "Test Zone", "rules": rules, "points": [[0,0],[640,0],[640,480],[0,480]]}]
    gk = Gatekeeper(cfg, redis_client=None)
    gk.load_zone_rules(zones)
    return gk


class TestFallDetected:
    def test_fall_triggers_critical(self):
        gk = make_gatekeeper()
        events = gk.evaluate(make_token(posture=Posture.FALLEN))
        fall_events = [e for e in events if e.event_type == EventType.FALL_DETECTED]
        assert len(fall_events) == 1
        assert fall_events[0].severity == Severity.CRITICAL

    def test_standing_no_fall(self):
        gk = make_gatekeeper()
        events = gk.evaluate(make_token(posture=Posture.STANDING))
        falls = [e for e in events if e.event_type == EventType.FALL_DETECTED]
        assert len(falls) == 0


class TestRunningInZone:
    def test_running_in_zone_triggers(self):
        gk = make_gatekeeper(rules=["no_running"])
        token = make_token(velocity=Velocity.RUNNING, velocity_value=6.0, zone_id="zone_01", zone_name="Test Zone")
        events = gk.evaluate(token)
        running = [e for e in events if e.event_type == EventType.RUNNING_IN_ZONE]
        assert len(running) == 1
        assert running[0].severity == Severity.MEDIUM

    def test_walking_no_trigger(self):
        gk = make_gatekeeper(rules=["no_running"])
        events = gk.evaluate(make_token(velocity=Velocity.WALKING, velocity_value=1.5, zone_id="zone_01"))
        running = [e for e in events if e.event_type == EventType.RUNNING_IN_ZONE]
        assert len(running) == 0


class TestHandsRaised:
    def test_hands_raised_triggers_high(self):
        gk = make_gatekeeper()
        events = gk.evaluate(make_token(hands_state=HandsState.RAISED))
        raised = [e for e in events if e.event_type == EventType.HANDS_RAISED]
        assert len(raised) == 1
        assert raised[0].severity == Severity.HIGH


class TestAggressiveBehavior:
    def test_running_with_raised_hands_triggers(self):
        gk = make_gatekeeper()
        token = make_token(
            velocity=Velocity.RUNNING,
            velocity_value=6.0,
            hands_state=HandsState.RAISED,
        )
        events = gk.evaluate(token)
        aggressive = [e for e in events if e.event_type == EventType.AGGRESSIVE_BEHAVIOR]
        assert len(aggressive) == 1
        assert aggressive[0].severity == Severity.HIGH


class TestUnknownPerson:
    def test_unidentified_triggers(self):
        gk = make_gatekeeper(rules=["authorized_only"])
        token = make_token(zone_id="zone_01", zone_name="Test Zone")
        events = gk.evaluate(token)
        unknown = [e for e in events if e.event_type == EventType.UNKNOWN_PERSON]
        assert len(unknown) == 1
        assert unknown[0].severity == Severity.MEDIUM


class TestLoitering:
    def test_loitering_triggers_after_threshold(self):
        cfg = make_cfg(loitering_threshold_s=0)
        gk = Gatekeeper(cfg, redis_client=None)
        zones = [{"id": "zone_01", "name": "Test Zone", "rules": ["no_loitering"], "points": []}]
        gk.load_zone_rules(zones)
        token = make_token(velocity=Velocity.STATIONARY, zone_id="zone_01", zone_name="Test Zone")
        events = gk.evaluate(token)
        loiter = [e for e in events if e.event_type == EventType.LOITERING]
        assert len(loiter) == 1
        assert loiter[0].severity == Severity.LOW


class TestZoneIntrusion:
    def test_intrusion_triggers_high(self):
        # Zone intrusion fires on "authorized_only" rule after 5 confirmation frames
        gk = make_gatekeeper(rules=["authorized_only"])
        token = make_token(zone_id="zone_01", zone_name="Test Zone")
        events = []
        # Evaluate 5 times to pass confirmation frame threshold
        for _ in range(5):
            events = gk.evaluate(token)
        intrusion = [e for e in events if e.event_type == EventType.ZONE_INTRUSION]
        assert len(intrusion) == 1
        assert intrusion[0].severity == Severity.HIGH


class TestCooldown:
    def test_cooldown_suppresses_duplicate(self):
        cfg = make_cfg()
        cfg["cooldowns"]["FALL_DETECTED"] = 60
        gk = Gatekeeper(cfg, redis_client=None)
        gk.load_zone_rules([])
        token = make_token(posture=Posture.FALLEN)
        events1 = gk.evaluate(token)
        events2 = gk.evaluate(token)
        falls1 = [e for e in events1 if e.event_type == EventType.FALL_DETECTED]
        falls2 = [e for e in events2 if e.event_type == EventType.FALL_DETECTED]
        assert len(falls1) == 1
        assert len(falls2) == 0  # suppressed by cooldown


class TestIsIdentifiedReset:
    def test_new_person_id_starts_unidentified(self):
        """New person ID must start with fresh unidentified state."""
        gk = make_gatekeeper(rules=["authorized_only"])
        # Person 99 has never been seen — must be unidentified
        token = make_token(person_id=99, zone_id="zone_01", zone_name="Test Zone")
        events = gk.evaluate(token)
        unknown = [e for e in events if e.event_type == EventType.UNKNOWN_PERSON]
        assert len(unknown) == 1

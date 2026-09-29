"""
Gatekeeper — deterministic rule engine.

Evaluates BehaviorTokens against all 7 rules and publishes
SentinelEvents to Redis Streams.

Rules:
  1. FALL_DETECTED       (CRITICAL) — fallen posture for N consecutive frames
  2. RUNNING_IN_ZONE     (MEDIUM)   — running in zone with "no_running" rule
  3. HANDS_RAISED        (HIGH)     — hands raised for N frames
  4. AGGRESSIVE_BEHAVIOR (HIGH)     — fast movement + raised hands
  5. UNKNOWN_PERSON      (MEDIUM)   — person in "authorized_only" zone, unidentified
  6. LOITERING           (LOW)      — stationary in zone beyond threshold
  7. ZONE_INTRUSION      (HIGH)     — unidentified in "authorized_only" zone

Bug fixes from existing code:
  - LOITERING and AGGRESSIVE_BEHAVIOR now fully implemented (were missing)
  - is_identified resets when tracker ID is seen for first time in a new
    period (handled by StateManager clearing stale tracks)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import redis as redis_lib

from .tokens import (
    BehaviorToken,
    EVENT_SEVERITY,
    EventType,
    HandsState,
    Posture,
    SentinelEvent,
    Severity,
    Velocity,
)

logger = logging.getLogger(__name__)


@dataclass
class _PersonGateState:
    """Gatekeeper state for a single tracked person."""
    # Hysteresis counters — frames in state before event fires
    state_counters: dict[str, int] = field(default_factory=dict)
    # Last event fire times by type
    last_event_times: dict[str, float] = field(default_factory=dict)
    # Identity state
    is_identified: bool = False
    identity_name: Optional[str] = None
    last_identity_check: float = 0.0
    # Loitering: first time seen stationary in zone
    loiter_start: Optional[float] = None


class Gatekeeper:
    """
    Deterministic rule engine that evaluates BehaviorTokens and
    publishes SentinelEvents to Redis Streams.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        redis_client: Optional[redis_lib.Redis] = None,
    ) -> None:
        self._cfg = cfg
        self._redis = redis_client
        self._tenant_id = cfg.get("tenant_id", "default")

        self._cooldowns: dict[str, float] = cfg.get("cooldowns", {
            "FALL_DETECTED": 10.0,
            "RUNNING_IN_ZONE": 5.0,
            "HANDS_RAISED": 3.0,
            "AGGRESSIVE_BEHAVIOR": 15.0,
            "UNKNOWN_PERSON": 30.0,
            "LOITERING": 60.0,
            "ZONE_INTRUSION": 10.0,
        })

        self._hysteresis = int(cfg.get("hysteresis_frames", 3))
        self._fall_frames = max(self._hysteresis, int(cfg.get("fall_confirmation_frames", 5)))
        self._loitering_threshold = float(cfg.get("loitering_threshold_s", 30.0))

        # Zone rules: zone_id → list of rule strings
        self._zone_rules: dict[str, list[str]] = {}

        # Per-person gate state
        self._states: dict[int, _PersonGateState] = {}

        # In-memory callbacks (for MVP without Redis)
        self._callbacks: list[Callable[[SentinelEvent], None]] = []

    def load_zone_rules(self, zones: list[dict]) -> None:
        """Load zone rules from zones config format."""
        for zone in zones:
            zid = zone.get("id")
            rules = zone.get("rules", [])
            if zid:
                self._zone_rules[zid] = rules
        logger.info(f"Loaded rules for {len(self._zone_rules)} zones")

    def set_zone_rules(self, zone_id: str, rules: list[str]) -> None:
        self._zone_rules[zone_id] = rules

    def register_callback(self, cb: Callable[[SentinelEvent], None]) -> None:
        """Register callback for triggered events (used when Redis unavailable)."""
        self._callbacks.append(cb)

    def update_identity(
        self,
        person_id: int,
        name: Optional[str],
        is_known: bool,
    ) -> None:
        """
        Update identity info after FaceWorker identification.

        Bug fix: is_identified is tracked per-person-id. When the tracker
        assigns a previously-used ID to a new person, StateManager clears
        the stale state, and the Gatekeeper will create a fresh _PersonGateState
        (is_identified=False) on the next evaluate() call.
        """
        state = self._states.get(person_id)
        if state is None:
            return
        state.is_identified = is_known
        state.identity_name = name
        state.last_identity_check = time.time()

    def evaluate(self, token: BehaviorToken) -> list[SentinelEvent]:
        """
        Evaluate a BehaviorToken against all 7 rules.

        Returns:
            List of SentinelEvents triggered (may be empty).
        """
        person_id = token.person_id
        now = time.time()

        if person_id not in self._states:
            self._states[person_id] = _PersonGateState()
        state = self._states[person_id]

        triggered: list[SentinelEvent] = []

        event = self._check_fall(token, state, now)
        if event:
            triggered.append(event)

        event = self._check_running_in_zone(token, state, now)
        if event:
            triggered.append(event)

        event = self._check_hands_raised(token, state, now)
        if event:
            triggered.append(event)

        event = self._check_aggressive_behavior(token, state, now)
        if event:
            triggered.append(event)

        event = self._check_unknown_person(token, state, now)
        if event:
            triggered.append(event)

        event = self._check_loitering(token, state, now)
        if event:
            triggered.append(event)

        event = self._check_zone_intrusion(token, state, now)
        if event:
            triggered.append(event)

        # Publish all triggered events
        for evt in triggered:
            self._publish(evt)

        return triggered

    def cleanup_stale_states(self, active_ids: set[int]) -> None:
        """Remove state for persons no longer tracked."""
        stale = set(self._states.keys()) - active_ids
        for pid in stale:
            del self._states[pid]

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _check_fall(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        key = "fall"
        if token.posture == Posture.FALLEN:
            count = self._inc(key, state)
            if count >= self._fall_frames and self._cooldown_ok(EventType.FALL_DETECTED, state, now):
                self._fire(EventType.FALL_DETECTED, state, now)
                return self._make_event(
                    token, EventType.FALL_DETECTED, now,
                    requires_vlm=True,
                    requires_face_id=True,
                    metadata={"posture": "fallen", "confirmation_frames": count},
                )
        else:
            self._reset(key, state)
        return None

    def _check_running_in_zone(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        key = "running_zone"
        if token.velocity == Velocity.RUNNING and token.zone_id:
            zone_rules = self._zone_rules.get(token.zone_id, [])
            if "no_running" in zone_rules:
                count = self._inc(key, state)
                if count >= self._hysteresis and self._cooldown_ok(EventType.RUNNING_IN_ZONE, state, now):
                    self._fire(EventType.RUNNING_IN_ZONE, state, now)
                    return self._make_event(
                        token, EventType.RUNNING_IN_ZONE, now,
                        requires_vlm=False,
                        requires_face_id=True,
                        metadata={"velocity": token.velocity_value, "zone": token.zone_name},
                    )
            else:
                self._reset(key, state)
        else:
            self._reset(key, state)
        return None

    def _check_hands_raised(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        key = "hands_raised"
        if token.hands_state == HandsState.RAISED:
            count = self._inc(key, state)
            if count >= self._hysteresis and self._cooldown_ok(EventType.HANDS_RAISED, state, now):
                self._fire(EventType.HANDS_RAISED, state, now)
                return self._make_event(
                    token, EventType.HANDS_RAISED, now,
                    requires_vlm=True,
                    requires_face_id=True,
                    metadata={"hands_state": "raised"},
                )
        else:
            self._reset(key, state)
        return None

    def _check_aggressive_behavior(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        """
        Aggressive behavior = fast movement (RUNNING velocity) + RAISED hands.
        Both conditions must be true simultaneously for N frames.
        """
        key = "aggressive"
        if token.velocity == Velocity.RUNNING and token.hands_state == HandsState.RAISED:
            count = self._inc(key, state)
            if count >= self._hysteresis and self._cooldown_ok(EventType.AGGRESSIVE_BEHAVIOR, state, now):
                self._fire(EventType.AGGRESSIVE_BEHAVIOR, state, now)
                return self._make_event(
                    token, EventType.AGGRESSIVE_BEHAVIOR, now,
                    requires_vlm=True,
                    requires_face_id=True,
                    metadata={
                        "velocity": token.velocity_value,
                        "hands_state": "raised",
                    },
                )
        else:
            self._reset(key, state)
        return None

    def _check_unknown_person(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        """
        Unknown person = in authorized_only zone + not yet identified.
        Triggers to prompt face identification.
        """
        key = "unknown_person"
        if token.zone_id:
            zone_rules = self._zone_rules.get(token.zone_id, [])
            if "authorized_only" in zone_rules and not state.is_identified:
                count = self._inc(key, state)
                if count >= self._hysteresis and self._cooldown_ok(EventType.UNKNOWN_PERSON, state, now):
                    self._fire(EventType.UNKNOWN_PERSON, state, now)
                    return self._make_event(
                        token, EventType.UNKNOWN_PERSON, now,
                        requires_vlm=False,
                        requires_face_id=True,
                        metadata={"zone": token.zone_name, "is_identified": False},
                    )
            else:
                self._reset(key, state)
        else:
            self._reset(key, state)
        return None

    def _check_loitering(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        """
        Loitering = stationary in a zone with "no_loitering" rule beyond threshold.
        """
        key = "loitering"
        if token.zone_id and token.velocity == Velocity.STATIONARY:
            zone_rules = self._zone_rules.get(token.zone_id, [])
            if "no_loitering" in zone_rules:
                if state.loiter_start is None:
                    state.loiter_start = now
                elapsed = now - state.loiter_start
                if elapsed >= self._loitering_threshold and self._cooldown_ok(EventType.LOITERING, state, now):
                    self._fire(EventType.LOITERING, state, now)
                    return self._make_event(
                        token, EventType.LOITERING, now,
                        requires_vlm=False,
                        requires_face_id=True,
                        metadata={"loiter_seconds": round(elapsed, 1), "zone": token.zone_name},
                    )
            else:
                state.loiter_start = None
        else:
            state.loiter_start = None
            self._reset(key, state)
        return None

    def _check_zone_intrusion(
        self, token: BehaviorToken, state: _PersonGateState, now: float
    ) -> Optional[SentinelEvent]:
        """
        Zone intrusion = unidentified person in authorized_only zone.
        Higher severity than UNKNOWN_PERSON — fires after confirmation frames.
        """
        key = "zone_intrusion"
        if token.zone_id:
            zone_rules = self._zone_rules.get(token.zone_id, [])
            if "authorized_only" in zone_rules and not state.is_identified:
                count = self._inc(key, state)
                # Use higher hysteresis for intrusion (more serious)
                intrusion_frames = max(self._hysteresis, 5)
                if count >= intrusion_frames and self._cooldown_ok(EventType.ZONE_INTRUSION, state, now):
                    self._fire(EventType.ZONE_INTRUSION, state, now)
                    return self._make_event(
                        token, EventType.ZONE_INTRUSION, now,
                        requires_vlm=True,
                        requires_face_id=True,
                        metadata={"zone": token.zone_name, "confirmation_frames": count},
                    )
            else:
                self._reset(key, state)
        else:
            self._reset(key, state)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inc(self, key: str, state: _PersonGateState) -> int:
        state.state_counters[key] = state.state_counters.get(key, 0) + 1
        return state.state_counters[key]

    def _reset(self, key: str, state: _PersonGateState) -> None:
        state.state_counters[key] = 0

    def _cooldown_ok(
        self, event_type: EventType, state: _PersonGateState, now: float
    ) -> bool:
        last = state.last_event_times.get(event_type.value, 0.0)
        cooldown = self._cooldowns.get(event_type.value, 5.0)
        return (now - last) >= cooldown

    def _fire(self, event_type: EventType, state: _PersonGateState, now: float) -> None:
        state.last_event_times[event_type.value] = now

    def _make_event(
        self,
        token: BehaviorToken,
        event_type: EventType,
        now: float,
        requires_vlm: bool,
        requires_face_id: bool,
        metadata: dict,
    ) -> SentinelEvent:
        severity = EVENT_SEVERITY[event_type]
        return SentinelEvent(
            event_type=event_type,
            severity=severity,
            person_id=token.person_id,
            camera_id=token.camera_id,
            tenant_id=token.tenant_id,
            token=token,
            trace_id=token.trace_id,
            timestamp=now,
            zone_id=token.zone_id,
            requires_vlm=requires_vlm,
            requires_face_id=requires_face_id,
            metadata=metadata,
        )

    def _publish(self, event: SentinelEvent) -> None:
        """Publish event to Redis Stream or in-memory callbacks."""
        stream_key = f"sentinel:events:{event.tenant_id}"
        payload = event.to_json()

        if self._redis is not None:
            try:
                self._redis.xadd(stream_key, {"event": payload, "trace_id": str(event.trace_id)})
                logger.debug(
                    f"Published {event.event_type.value} to {stream_key}",
                    extra={"trace_id": str(event.trace_id)},
                )
            except redis_lib.RedisError as e:
                logger.error(f"Failed to publish event to Redis: {e} — falling back to callbacks")
                for cb in self._callbacks:
                    try:
                        cb(event)
                    except Exception as cb_err:
                        logger.error(f"Callback error: {cb_err}")
        else:
            for cb in self._callbacks:
                try:
                    cb(event)
                except Exception as cb_err:
                    logger.error(f"Callback error: {cb_err}")

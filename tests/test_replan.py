"""Tests for the replan reason description (Feature 05).

This module describes a context change; it does not decide whether to replan --
that is `should_replan()` in Optimization (Member 2). These tests pin the
wording the banner shows and the noise thresholds that stop sensor jitter
reading as weather.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from features.dashboard.replan import (
    DIRECTION_DOWN,
    DIRECTION_UP,
    describe_context_change,
    replan_reason,
)


def context(rain=0.0, visibility=5000.0, temperature=24.0, dust="low",
            congestion="low", hardness=0.5, slope=2.0, machine_condition="good"):
    return SimpleNamespace(
        weather=SimpleNamespace(rain_mm=rain, visibility_m=visibility, temperature_c=temperature,
                                dust_level=dust),
        traffic=SimpleNamespace(congestion=congestion),
        site=SimpleNamespace(hardness=hardness, slope=slope),
        machine=SimpleNamespace(machine_condition=machine_condition),
    )


def test_identical_contexts_report_no_change():
    assert describe_context_change(context(), context()) == ()
    assert replan_reason(()) == "no material change in operating context"


def test_rain_increase_is_reported_with_direction():
    changes = describe_context_change(context(rain=0.0), context(rain=2.4))
    assert len(changes) == 1
    assert changes[0].key == "rain_mm"
    assert changes[0].direction == DIRECTION_UP


def test_visibility_drop_is_a_decrease():
    changes = describe_context_change(context(visibility=5000.0), context(visibility=400.0))
    assert changes[0].direction == DIRECTION_DOWN


@pytest.mark.parametrize(
    "field,before,after",
    [("rain", 0.0, 0.2), ("visibility", 5000.0, 4950.0), ("temperature", 24.0, 25.0),
     ("hardness", 0.50, 0.52), ("slope", 2.0, 2.5)],
)
def test_movements_below_the_noise_threshold_are_ignored(field, before, after):
    """Sensor jitter must not read as a change in the weather."""
    assert describe_context_change(
        context(**{field: before}), context(**{field: after})
    ) == ()


def test_rain_jitter_below_threshold_is_ignored():
    assert describe_context_change(context(rain=0.0), context(rain=0.2)) == ()


def test_rain_above_threshold_is_reported():
    assert len(describe_context_change(context(rain=0.0), context(rain=0.6))) == 1


def test_ordered_categories_get_a_direction():
    worse = describe_context_change(context(congestion="low"), context(congestion="high"))
    better = describe_context_change(context(congestion="high"), context(congestion="low"))
    assert worse[0].direction == DIRECTION_UP
    assert better[0].direction == DIRECTION_DOWN


def test_machine_condition_degrading_is_a_decrease():
    changes = describe_context_change(
        context(machine_condition="good"), context(machine_condition="poor")
    )
    assert changes[0].direction == DIRECTION_DOWN


def test_the_architecture_example_reads_correctly():
    """'Reason: rain increased + congestion increased' from the source document."""
    changes = describe_context_change(
        context(rain=0.0, congestion="low"), context(rain=3.0, congestion="high")
    )
    assert replan_reason(changes) == "rain increased + congestion increased"


def test_detail_includes_both_values():
    changes = describe_context_change(context(rain=0.0), context(rain=3.0))
    assert changes[0].detail() == "rain increased (0.0 to 3.0)"


def test_missing_sub_contexts_do_not_raise():
    """A Passport session has no site or weather attached."""
    bare = SimpleNamespace()
    assert describe_context_change(bare, bare) == ()
    assert describe_context_change(bare, context(rain=5.0)) == ()


def test_none_contexts_are_handled():
    assert describe_context_change(None, context()) == ()
    assert describe_context_change(context(), None) == ()


def test_several_changes_are_all_reported():
    changes = describe_context_change(
        context(rain=0.0, visibility=5000.0, dust="low", hardness=0.5),
        context(rain=4.0, visibility=300.0, dust="high", hardness=0.9),
    )
    assert {c.key for c in changes} == {
        "rain_mm", "visibility_m", "dust_level", "soil_hardness_index",
    }


def test_payload_round_trips():
    change = describe_context_change(context(rain=0.0), context(rain=3.0))[0]
    assert change.to_dict() == {
        "key": "rain_mm", "label": "rain", "previous": 0.0,
        "current": 3.0, "direction": DIRECTION_UP,
    }

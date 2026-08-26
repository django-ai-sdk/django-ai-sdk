"""Cron schedule arithmetic, the only schedule kind automations support."""

from datetime import UTC, datetime

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_ai_sdk.automations.schedule import Cron, timezone_available


def dt(*args):
    return datetime(*args, tzinfo=UTC)


class TestCron:
    def test_next_occurrence(self):
        schedule = Cron("0 9 * * *")
        assert schedule.next_after(dt(2026, 8, 16, 10, 0)) == dt(2026, 8, 17, 9, 0)

    def test_weekday_range(self):
        # Saturday 09:00 -> the next weekday occurrence is Monday.
        schedule = Cron("0 9 * * 1-5")
        assert schedule.next_after(dt(2026, 8, 15, 12, 0)) == dt(2026, 8, 17, 9, 0)

    def test_validates_eagerly(self):
        # A typo must fail on import, not at 3am.
        with pytest.raises(ImproperlyConfigured):
            Cron("not a cron expression")

    def test_repr_names_the_timezone(self):
        # UTC is a decision, so it should be visible wherever the schedule is shown.
        assert "UTC" in str(Cron("0 9 * * *"))


class TestCronTimezone:
    def test_local_time_survives_a_dst_boundary(self):
        # The reason the timezone exists. Amsterdam is UTC+2 in summer and UTC+1 in
        # winter, so a fixed-UTC reading of "09:00" would drift by an hour.
        schedule = Cron("0 9 * * *", tz="Europe/Amsterdam")

        summer = schedule.next_after(dt(2026, 7, 1, 0, 0))
        winter = schedule.next_after(dt(2026, 12, 1, 0, 0))

        assert summer == dt(2026, 7, 1, 7, 0)
        assert winter == dt(2026, 12, 1, 8, 0)

    def test_utc_is_the_default(self):
        assert Cron("0 9 * * *").next_after(dt(2026, 7, 1, 0, 0)) == dt(2026, 7, 1, 9, 0)

    def test_repr_names_the_zone(self):
        assert "Europe/Amsterdam" in str(Cron("0 9 * * *", tz="Europe/Amsterdam"))

    def test_an_unknown_zone_falls_back_to_utc_rather_than_raising(self):
        # A misspelled zone must not stop a site from booting; W002 reports it.
        schedule = Cron("0 9 * * *", tz="Mars/Olympus_Mons")
        assert schedule.next_after(dt(2026, 7, 1, 0, 0)) == dt(2026, 7, 1, 9, 0)

    def test_zone_is_part_of_identity(self):
        assert Cron("0 9 * * *", tz="UTC") != Cron("0 9 * * *", tz="Europe/Amsterdam")


class TestTimezoneAvailable:
    def test_known_zone(self):
        assert timezone_available("Europe/Amsterdam")

    def test_unknown_zone(self):
        assert not timezone_available("Mars/Olympus_Mons")

from __future__ import annotations

import argparse

from scripts.prune import session


def test_record_args_default_split_is_overridable():
    parser = argparse.ArgumentParser()
    session.add_record_args(parser, default_split="held_out")
    assert parser.parse_args([]).split == "held_out"
    assert parser.parse_args(["--split", "calibration"]).split == "calibration"


def test_max_records_defaults_to_all():
    parser = argparse.ArgumentParser()
    session.add_record_args(parser)
    assert parser.parse_args([]).max_records is None

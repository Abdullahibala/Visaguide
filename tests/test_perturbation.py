import importlib.util
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "judge_perturbation", Path(__file__).parents[1] / "scripts" / "judge_perturbation.py")
jp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jp)


def test_corrupt_number_changes_a_number_but_not_citation_ids():
    answer = "You may work 140 full days per year [S2][S3]."
    new, marker = jp.corrupt_number(answer, random.Random(0))
    assert "[S2][S3]" in new
    assert marker == "210" and "210 full days" in new


def test_corrupt_number_keeps_thousands_format():
    new, marker = jp.corrupt_number("You need 11,904 euros [S1].", random.Random(0))
    assert marker == "17,856" and "17,856 euros" in new


def test_corrupt_number_none_without_numbers():
    assert jp.corrupt_number("Apply online [S1].", random.Random(0)) is None


def test_detected_requires_the_corrupted_claim_to_be_flagged():
    claims = [{"claim": "You may work 210 full days per year.", "supported_by_any": False},
              {"claim": "Apply online.", "supported_by_any": True}]
    assert jp.detected(claims, "210")
    claims[0]["supported_by_any"] = True
    assert not jp.detected(claims, "210")  # flagged elsewhere / not at all doesn't count


def test_detected_matches_paraphrased_addition():
    marker = ("The responsible office usually decides within two weeks of receiving the "
              "application [S1].")
    claims = [{"claim": "The office usually decides within two weeks of receiving an application.",
               "supported_by_any": False}]
    assert jp.detected(claims, marker)

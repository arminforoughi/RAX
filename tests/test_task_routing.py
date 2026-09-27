"""A placement phrase must place, whichever box it is typed into.

THE BUG THIS PINS. On a real run the operator asked for "red on green". The arm drove
to the red cube, grasped it, reported DONE and folded home — never going near the green
one. The log says why:

    PICK START  target='red'  query='red cube, green cube'
    ... PICK SUCCESS ... [DONE] red cube picked

Nothing had failed. "red on green" had gone into the DETECTION QUERY box, where it was
read as a vocabulary rather than as an instruction, and Start ran a plain pick whose
target is "whatever the query names first". The pick half worked perfectly; the place
half was never asked for.

Two things had to be true for the phrase to work, and neither was:
  * Start has to notice the phrase names a destination (reads_as_a_task).
  * The detector has to be looking for BOTH objects, or the destination can never
    reach the 2D map and the task aborts for want of it (task_vocabulary).

    pytest tests/test_task_routing.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "src", REPO / "examples" / "mission_server"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture(scope="module")
def S():
    try:
        import mission_server
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"mission_server did not import: {exc}")
    return mission_server


# --- reads_as_a_task: telling an instruction from a vocabulary ----------------------

@pytest.mark.parametrize("phrase", [
    "red on green",
    "red cube on green cube",
    "green onto red",
    "red > green",
    "red -> green",
    "green on red, blue on green",
])
def test_a_placement_phrase_is_recognised_as_a_task(S, phrase):
    steps = S.reads_as_a_task(phrase)
    assert steps, f"{phrase!r} names a destination and must read as a task"


@pytest.mark.parametrize("vocab", [
    "red cube, green cube",
    "pen, book, mug",
    "red cube",
    "",
])
def test_an_ordinary_vocabulary_is_not_mistaken_for_a_task(S, vocab):
    assert S.reads_as_a_task(vocab) is None, f"{vocab!r} is a class list, not a task"


def test_the_exact_query_from_the_failing_run_is_still_a_vocabulary(S):
    """'red cube, green cube' is what the failing run had set; it must stay a pick."""
    assert S.reads_as_a_task("red cube, green cube") is None


# --- task_vocabulary: both objects have to be detectable ---------------------------

def test_the_vocabulary_covers_the_destination_as_well_as_the_pick(S):
    vocab = S.task_vocabulary(S.reads_as_a_task("red on green"))
    assert "red" in vocab and "green" in vocab, vocab


def test_bare_colour_words_expand_to_the_cube_class_they_mean(S):
    """A one-word colour is a weak YOLO-World prompt; 'red cube' is a real class."""
    vocab = S.task_vocabulary(S.reads_as_a_task("red on green"))
    assert vocab == "red cube, green cube", vocab


def test_a_non_colour_object_is_left_alone(S):
    vocab = S.task_vocabulary(S.reads_as_a_task("cup on book"))
    assert "cup" in vocab and "book" in vocab
    assert "cup cube" not in vocab, vocab


def test_a_multi_step_task_lists_every_object_once(S):
    vocab = S.task_vocabulary(S.reads_as_a_task("green on red, blue on green"))
    assert vocab.count("green cube") == 1, vocab
    for want in ("red cube", "green cube", "blue cube"):
        assert want in vocab, f"{want} missing from {vocab!r}"


# --- the parse itself still rejects what it always rejected ------------------------

def test_a_step_missing_its_destination_is_refused(S):
    with pytest.raises(S.Abort):
        S.parse_task("red on")


def test_a_phrase_with_no_separator_is_refused_by_parse_task(S):
    with pytest.raises(S.Abort):
        S.parse_task("red cube")


# --- the intent must survive the detector reporting its vocabulary back -------------
# Found on the live server, not here: /setquery correctly expanded "red on green" to
# "red cube, green cube", and then the detector thread wrote that vocabulary back into
# state["query"] (on_query_change). Start read state["query"], saw no destination, and
# ran a plain pick — the original bug, restored a few seconds after being fixed.

def test_the_asked_for_phrase_survives_the_detector_overwriting_the_query(S):
    st = {"intent": "red on green", "query": "red cube, green cube"}
    assert S.pending_instruction(st) == "red on green"
    assert S.reads_as_a_task(S.pending_instruction(st)), "Start must still see a task"


def test_a_plain_query_is_used_when_no_task_was_asked_for(S):
    st = {"intent": None, "query": "red cube, green cube"}
    assert S.pending_instruction(st) == "red cube, green cube"
    assert S.reads_as_a_task(S.pending_instruction(st)) is None


def test_an_empty_state_yields_an_empty_instruction(S):
    assert S.pending_instruction({}) == ""
    assert S.reads_as_a_task(S.pending_instruction({})) is None

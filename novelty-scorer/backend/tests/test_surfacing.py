"""Redundancy-aware surfacing: sentence labels, closest ids, net_new_ratio."""

import pytest

from scoring import surfacing
from scoring.corpus_cache import get_sentence_index

TOPIC = "remote_work"

# Sentences with no counterpart in the corpus (the held-out T1 case's mechanism, reworded).
NOVEL = [
    "Payroll teams now file withholding returns in states where the company has no office at all.",
    "Permanent-establishment rules can make a single remote engineer create corporate tax exposure abroad.",
    "Some employers quietly publish lists of countries where staff are not allowed to live while working.",
]


@pytest.fixture(scope="module")
def ctx(topics):
    t = topics[TOPIC]
    return {"index": get_sentence_index(TOPIC), "ids": [s.id for s in t.submissions], "subs": t.submissions}


def corpus_sentence(ctx, sid, k=0):
    sub = next(s for s in ctx["subs"] if s.id == sid)
    return surfacing.segment(sub.content)[k]


def test_segment_splits_sentences_and_merges_fragments():
    assert surfacing.segment("One two three. Four five six! Seven eight nine?") == [
        "One two three.", "Four five six!", "Seven eight nine?"]
    assert surfacing.segment("A full sentence here. Ok.") == ["A full sentence here. Ok."]
    assert surfacing.segment("   ") == []


def test_thresholds_are_derived_from_corpus(ctx):
    idx = ctx["index"]
    assert 0.0 < idx.novel_max < idx.redundant_min <= 1.0


@pytest.mark.parametrize("sid", ["s20", "s30", "s12"])
def test_verbatim_corpus_sentence_is_redundant_with_correct_closest_id(ctx, sid):
    sent = corpus_sentence(ctx, sid)
    out = surfacing.surface(sent, ctx["index"], ctx["ids"])["sentences"]
    assert len(out) == 1
    assert out[0]["label"] == "redundant"
    assert out[0]["closest_id"] == sid
    assert out[0]["redundancy"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("sent", NOVEL)
def test_clearly_novel_sentence_is_labelled_novel(ctx, sent):
    out = surfacing.surface(sent, ctx["index"], ctx["ids"])["sentences"]
    assert out[0]["label"] == "novel", out[0]


def test_net_new_ratio_is_monotonic_in_novel_sentences(ctx):
    redundant = [corpus_sentence(ctx, sid) for sid in ("s20", "s30", "s12")]
    ratios = []
    for k in range(len(NOVEL) + 1):  # replace k copied sentences with k novel ones
        text = " ".join(NOVEL[:k] + redundant[k:])
        ratios.append(surfacing.surface(text, ctx["index"], ctx["ids"])["net_new_ratio"])
    assert all(b >= a for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[0] == 0.0 and ratios[-1] == 1.0, ratios


def test_leave_one_out_excludes_own_submission(ctx):
    sid = "s20"
    i = ctx["ids"].index(sid)
    sub = ctx["subs"][i]
    out = surfacing.surface(sub.content, ctx["index"], ctx["ids"], exclude_owner=i)
    assert all(x["closest_id"] != sid for x in out["sentences"])

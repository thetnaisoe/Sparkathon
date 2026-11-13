import pytest

from app.pipeline.research.retrieve import chunk, rank


def _text(n_words: int) -> str:
    # irregular whitespace on purpose: offsets must still map back exactly
    return "  ".join(f"w{i}" + ("\n" if i % 7 == 0 else "") for i in range(n_words))


def test_chunk_offsets_map_back_to_original_text():
    text = _text(1000)
    passages = chunk(text, target_tokens=400, overlap=50, doc="d1")
    assert len(passages) == 3
    for p in passages:
        assert text[p.start : p.end] == p.text
        assert p.doc == "d1"
        assert len(p.text.split()) <= 400
    # consecutive passages overlap by 50 words, and the last one reaches the end
    assert passages[0].text.split()[-50:] == passages[1].text.split()[:50]
    assert passages[-1].text.split()[-1] == "w999"
    assert passages[0].start == 0 or text[: passages[0].start].strip() == ""


def test_chunk_short_and_empty_text():
    assert chunk("") == []
    assert chunk("   \n ") == []
    [p] = chunk("  only a few words here ")
    assert p.text == "only a few words here"
    with pytest.raises(ValueError):
        chunk("x", target_tokens=0)


def test_rank_puts_relevant_passage_first():
    passages = [
        *chunk("The weather in Hamburg was rainy for most of the autumn season.", doc="a"),
        *chunk(
            "There were 2,450 machine building companies in Germany in 2023, "
            "according to the statistics office.",
            doc="b",
        ),
        *chunk("Football results from the weekend league matches were surprising.", doc="c"),
        *chunk("Companies in the food sector reported higher energy prices.", doc="d"),
    ]
    top = rank(passages, "number of machine building companies in Germany", k=2)
    assert top[0].doc == "b"
    assert top[0].score > 0
    assert all(p.doc != "c" for p in top)  # no shared term -> left out


def test_rank_edge_cases():
    passages = chunk("some words about turbines", doc="a")
    assert rank([], "turbines") == []
    assert rank(passages, "the of and") == []  # only stop words
    assert rank(passages, "turbines", k=0) == []
    assert rank(passages, "football") == []

from src.data.vocab import Vocab


SAMPLE_GAMES = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["e2e4", "c7c5"],
]


def test_specials_have_fixed_ids():
    v = Vocab.build_from_games(SAMPLE_GAMES)
    assert v.token_to_id[Vocab.PAD] == 0
    assert v.token_to_id[Vocab.START] == 1
    assert v.token_to_id[Vocab.END] == 2
    assert v.token_to_id[Vocab.UNK] == 3
    assert v.pad_id == 0
    assert v.start_id == 1
    assert v.end_id == 2
    assert v.unk_id == 3


def test_encode_decode_roundtrip():
    v = Vocab.build_from_games(SAMPLE_GAMES)
    for move in ["e2e4", "d2d4", "g1f3", "b8c6"]:
        assert v.decode(v.encode(move)) == move


def test_unknown_move_returns_unk_id():
    v = Vocab.build_from_games(SAMPLE_GAMES)
    assert v.encode("z9z9") == v.unk_id
    assert v.decode(v.encode("z9z9")) == Vocab.UNK


def test_no_duplicate_tokens_after_build():
    v = Vocab.build_from_games(SAMPLE_GAMES)
    assert len(v) == len(set(v.id_to_token))


def test_special_tokens_are_first_four_in_order():
    v = Vocab.build_from_games(SAMPLE_GAMES)
    assert v.id_to_token[:4] == list(Vocab.SPECIAL_TOKENS)


def test_save_and_load_roundtrip(tmp_path):
    v = Vocab.build_from_games(SAMPLE_GAMES)
    path = tmp_path / "vocab.json"
    v.save(path)

    loaded = Vocab.load(path)
    assert loaded.id_to_token == v.id_to_token
    assert loaded.token_to_id == v.token_to_id
    assert len(loaded) == len(v)
    assert loaded.pad_id == 0
    assert loaded.unk_id == 3

from lib.data.preprocessing import starts_for_length


def test_starts_for_length():
    starts = starts_for_length(length=2048, window=1024, step=256)
    assert starts == [0, 256, 512, 768, 1024]

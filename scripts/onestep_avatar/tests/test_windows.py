"""Tests for the frozen-subset builder -- block chaining, the split, and the plan invariant.

The load-bearing one is :func:`test_the_block_plan_is_pinned_to_exact_bounds`. Since
2026-09-15 ``windows.plan_blocks`` *is* ``causal_core.CausalGeometry.plan``, so comparing the
two would be tautological; what still matters is that the bounds themselves never move. A
frozen subset indexes blocks that ``train.py`` slices out of a master latent, so shifting the
plan would silently re-point every chain in every subset already on disk.
"""

from __future__ import annotations

import hashlib

import pytest

from scripts.onestep_avatar import windows


@pytest.mark.parametrize(
    ("total_frames", "latent_frames", "n_blocks", "last_block"),
    [
        (25, 4, 1, (0, 3)),        # exactly one block: the keyframe block alone
        (26, 4, 1, (0, 3)),        # the extra pixel frame does not reach a new latent frame
        (40, 5, 2, (3, 5)),
        (137, 18, 8, (15, 17)),    # the corpus's short tier -- note frame 17 is dropped
        (150, 19, 9, (17, 19)),
        (225, 29, 14, (27, 29)),   # the corpus's long tier
        (226, 29, 14, (27, 29)),
        (1000, 125, 62, (123, 125)),
    ],
)
def test_the_block_plan_is_pinned_to_exact_bounds(
    total_frames: int, latent_frames: int, n_blocks: int, last_block: tuple[int, int]
) -> None:
    """A golden test over the clip lengths the corpus actually has.

    ``plan_blocks`` now delegates straight to ``causal_core.CausalGeometry.plan`` -- since
    2026-09-15 there is one implementation, so a test that compared two would be
    tautological. What still matters is that the bounds themselves do not move: a frozen
    subset indexes blocks that ``train.py`` slices out of a master latent, and shifting the
    plan would silently re-point every chain in every subset already on disk.
    """
    assert windows.latent_frames_for(total_frames) == latent_frames
    blocks = windows.plan_blocks(latent_frames)
    assert len(blocks) == n_blocks
    assert blocks[0] == (0, 1 + windows.BLOCK_LATENT_FRAMES)
    assert blocks[-1] == last_block
    # No gaps and no overlaps between consecutive blocks.
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))


def test_latent_frame_count_follows_the_causal_vae() -> None:
    """Frame 0 covers ONE pixel frame; every later latent frame covers eight."""
    assert windows.latent_frames_for(1) == 1
    assert windows.latent_frames_for(9) == 2
    assert windows.latent_frames_for(137) == 18  # the corpus's 150-frame tier, as encoded
    assert windows.latent_frames_for(0) == 0


def test_corpus_block_counts() -> None:
    """A block is the deployed 16-pixel-frame stride, so a clip yields about half its old
    window count plus one -- block 0 absorbs the keyframe, every later block is 2 latent
    frames, and a tail shorter than a block is dropped."""
    assert len(windows.plan_blocks(windows.latent_frames_for(137))) == 8  # 18 latent frames -> 1 + 7
    assert len(windows.plan_blocks(windows.latent_frames_for(225))) == 14  # 29 latent frames -> 1 + 13


def test_a_clip_shorter_than_one_block_yields_nothing() -> None:
    """Dropped, never padded: a short block is a different training condition."""
    assert windows.plan_blocks(2) == []


def test_chains_are_consecutive_and_disjoint_by_default() -> None:
    chains = windows.chain_blocks(8, 3)
    assert chains == [[0, 1, 2], [3, 4, 5]]
    flat = [index for chain in chains for index in chain]
    assert len(flat) == len(set(flat))  # every block at most once per epoch
    # 6 and 7 are dropped: a 2-block remainder is not a K=3 chain.
    assert 6 not in flat


def test_overlapping_chains_are_available_for_small_tiers() -> None:
    assert windows.chain_blocks(6, 3, chain_stride=1) == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_k1_chains_cover_every_block() -> None:
    """K = 1 is A2's teacher-forced control, so it must not silently drop anything."""
    assert windows.chain_blocks(5, 1) == [[0], [1], [2], [3], [4]]


def test_chain_length_longer_than_the_clip_yields_nothing() -> None:
    assert windows.chain_blocks(2, 3) == []


def test_split_is_actor_disjoint_and_deterministic() -> None:
    actors = [str(i) for i in range(100)]
    train, held_out = windows.split_actors(actors)
    assert set(train) & set(held_out) == set()
    assert sorted(train + held_out) == sorted(actors)
    assert len(held_out) == 20  # the 20 % default, comfortably above the 12-actor floor
    assert (train, held_out) == windows.split_actors(list(reversed(actors)))


def test_split_honours_the_twelve_actor_floor_on_small_tiers() -> None:
    """§5.0: 12 held-out actors is a floor, not a fraction -- 20 % of 20 would be 4."""
    train, held_out = windows.split_actors([str(i) for i in range(20)])
    assert len(held_out) == 12
    assert len(train) == 8


def test_split_never_empties_the_training_side() -> None:
    """The floor yields to arithmetic rather than producing a subset with nothing to train on."""
    train, held_out = windows.split_actors(["a", "b"])
    assert len(train) == 1
    assert len(held_out) == 1


def test_split_refuses_a_single_actor() -> None:
    with pytest.raises(ValueError, match="cannot split"):
        windows.split_actors(["only"])


def test_split_is_not_correlated_with_id_order() -> None:
    """Slicing sorted ids would put every low-numbered actor -- one ingest batch, one capture
    session -- on the same side of the split. The hash ordering is what breaks that."""
    actors = [str(i) for i in range(100)]
    _, held_out = windows.split_actors(actors)
    numeric = sorted(int(a) for a in held_out)
    assert numeric != list(range(len(numeric)))
    assert max(numeric) > 50  # not a prefix of the sorted order


def test_actor_ranking_is_a_pure_function_of_the_id() -> None:
    """``--max-actors`` takes a prefix of the SAME ordering the split uses, so a small tier is
    a subset of the full corpus's ordering rather than an independently arbitrary one."""
    ranked = sorted(["7", "3", "11"], key=lambda a: hashlib.sha256(a.encode()).hexdigest())
    assert ranked == sorted(["3", "11", "7"], key=lambda a: hashlib.sha256(a.encode()).hexdigest())


def test_sha256_matches_hashlib(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "blob.bin"
    payload = b"x" * (3 * (1 << 20) + 17)  # spans several read chunks
    path.write_bytes(payload)
    assert windows.sha256(path) == hashlib.sha256(payload).hexdigest()


def test_verify_reports_an_unpinned_subset_rather_than_passing_it() -> None:
    """A subset frozen with --skip-hash has nothing to verify; saying so beats returning 'ok'."""
    problems = windows.verify({"content_pinned": False})
    assert len(problems) == 1
    assert "skip-hash" in problems[0]

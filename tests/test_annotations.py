import numpy as np

from segagent.data.annotations import (
    Contour,
    ReaderAnnotation,
    cluster_annotations,
    rasterize_consensus_masks,
)


def _annotation(reader_index: int, points: list[tuple[float, float]]) -> ReaderAnnotation:
    contour = Contour(
        sop_uid="slice-1",
        z_position=0.0,
        inclusion=True,
        points_xy=np.asarray(points, dtype=np.float64),
    )
    return ReaderAnnotation(reader_index, f"nodule-{reader_index}", (contour,))


def test_consensus_requires_ceil_half_of_three_readers() -> None:
    shared = [(1, 1), (3, 1), (3, 3), (1, 3)]
    remote = [(5, 5), (6, 5), (6, 6), (5, 6)]
    semantic, instances = rasterize_consensus_masks(
        [[_annotation(0, shared), _annotation(1, shared), _annotation(2, remote)]],
        {"slice-1": 0},
        (1, 8, 8),
        agreement_fraction=0.5,
    )

    assert semantic[0, 2, 2] == 1
    assert semantic[0, 5, 5] == 0
    assert instances[0, 2, 2] == 1


def test_clustering_never_puts_two_annotations_from_same_reader_together() -> None:
    shared = [(1, 1), (3, 1), (3, 3), (1, 3)]
    annotations = tuple(
        _annotation(reader, shared)
        for reader in (0, 0, 1, 2, 3)
    )

    clusters = cluster_annotations(annotations, slice_spacing_mm=1.0)

    assert sorted(len(cluster) for cluster in clusters) == [1, 4]
    assert all(
        len({annotation.reader_index for annotation in cluster}) == len(cluster)
        for cluster in clusters
    )

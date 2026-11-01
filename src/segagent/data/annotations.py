"""LIDC-IDRI XML parsing, annotation clustering, and consensus masks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from skimage.draw import polygon


@dataclass(frozen=True)
class Contour:
    sop_uid: str
    z_position: float
    inclusion: bool
    points_xy: np.ndarray


@dataclass(frozen=True)
class ReaderAnnotation:
    reader_index: int
    nodule_id: str
    contours: tuple[Contour, ...]

    def contour_matrix(self, slice_spacing_mm: float) -> np.ndarray:
        points = []
        for contour in self.contours:
            z_index = contour.z_position / slice_spacing_mm
            for x_value, y_value in contour.points_xy:
                points.append((y_value, x_value, z_index))
        return np.asarray(points, dtype=np.float64)

    def centroid(self, slice_spacing_mm: float) -> np.ndarray:
        return self.contour_matrix(slice_spacing_mm).mean(axis=0)


@dataclass(frozen=True)
class AnnotationDocument:
    series_uid: str
    archive_name: str
    annotations: tuple[ReaderAnnotation, ...]
    selection_reason: str
    candidates: tuple[str, ...]


def _local_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local_name(child) == name]


def _first_text(element: ElementTree.Element, name: str, default: str = "") -> str:
    children = _children(element, name)
    if not children:
        return default
    return (children[0].text or "").strip()


def _descendant_texts(element: ElementTree.Element, name: str) -> list[str]:
    return [
        (child.text or "").strip()
        for child in element.iter()
        if _local_name(child) == name
    ]


def _reading_session_hash(root: ElementTree.Element) -> str:
    payload = b"".join(
        ElementTree.tostring(element)
        for element in root.iter()
        if _local_name(element) == "readingSession"
    )
    return hashlib.sha256(payload).hexdigest()


def _select_revision(candidates: list[tuple[str, bytes]]) -> tuple[str, bytes, str]:
    if len(candidates) == 1:
        name, payload = candidates[0]
        return name, payload, "only_revision"

    corrections = [item for item in candidates if "correction" in item[0].lower()]
    if len(corrections) == 1:
        name, payload = corrections[0]
        return name, payload, "explicit_correction"

    hashes = {
        _reading_session_hash(ElementTree.fromstring(payload))
        for _, payload in candidates
    }
    if len(hashes) == 1:
        name, payload = sorted(candidates, key=lambda item: item[0])[0]
        return name, payload, "identical_reading_sessions"

    names = ", ".join(sorted(name for name, _ in candidates))
    raise ValueError(f"Conflicting XML revisions require a manifest: {names}")


def _parse_annotation_document(
    series_uid: str,
    archive_name: str,
    payload: bytes,
    selection_reason: str,
    candidates: tuple[str, ...],
) -> AnnotationDocument:
    root = ElementTree.fromstring(payload)
    annotations: list[ReaderAnnotation] = []
    sessions = [element for element in root.iter() if _local_name(element) == "readingSession"]
    for reader_index, session in enumerate(sessions):
        for nodule in _children(session, "unblindedReadNodule"):
            if not _children(nodule, "characteristics"):
                continue
            contours: list[Contour] = []
            for roi in _children(nodule, "roi"):
                points = [
                    (float(_first_text(edge, "xCoord")), float(_first_text(edge, "yCoord")))
                    for edge in _children(roi, "edgeMap")
                ]
                if len(points) < 3:
                    continue
                contours.append(
                    Contour(
                        sop_uid=_first_text(roi, "imageSOP_UID"),
                        z_position=float(_first_text(roi, "imageZposition")),
                        inclusion=_first_text(roi, "inclusion", "TRUE").upper() == "TRUE",
                        points_xy=np.asarray(points, dtype=np.float64),
                    )
                )
            if contours:
                annotations.append(
                    ReaderAnnotation(
                        reader_index=reader_index,
                        nodule_id=_first_text(nodule, "noduleID"),
                        contours=tuple(contours),
                    )
                )
    return AnnotationDocument(
        series_uid=series_uid,
        archive_name=archive_name,
        annotations=tuple(annotations),
        selection_reason=selection_reason,
        candidates=candidates,
    )


def load_annotation_documents(archive_path: Path) -> dict[str, AnnotationDocument]:
    """Load one deterministic CT annotation document for each Series UID."""
    grouped: dict[str, list[tuple[str, bytes]]] = {}
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".xml"):
                continue
            payload = archive.read(name)
            root = ElementTree.fromstring(payload)
            series_uids = _descendant_texts(root, "SeriesInstanceUid")
            if len(series_uids) != 1 or not series_uids[0]:
                continue
            grouped.setdefault(series_uids[0], []).append((name, payload))

    documents: dict[str, AnnotationDocument] = {}
    for series_uid, candidates in grouped.items():
        name, payload, reason = _select_revision(candidates)
        documents[series_uid] = _parse_annotation_document(
            series_uid,
            name,
            payload,
            reason,
            tuple(sorted(item[0] for item in candidates)),
        )
    return documents


def cluster_annotations(
    annotations: tuple[ReaderAnnotation, ...],
    slice_spacing_mm: float,
    tolerance: float | None = None,
    factor: float = 0.9,
    minimum_tolerance: float = 0.1,
) -> list[list[ReaderAnnotation]]:
    """Cluster contours by boundary distance with one annotation per reader."""
    if not annotations:
        return []
    if len(annotations) == 1:
        return [[annotations[0]]]

    matrices = [annotation.contour_matrix(slice_spacing_mm) for annotation in annotations]
    distances = np.zeros((len(annotations), len(annotations)), dtype=np.float64)
    for left in range(len(annotations)):
        for right in range(left + 1, len(annotations)):
            distance = float(cdist(matrices[left], matrices[right]).min())
            distances[left, right] = distances[right, left] = distance

    original_tolerance = slice_spacing_mm if tolerance is None else tolerance
    current_tolerance = original_tolerance
    while True:
        _, component_ids = connected_components(distances <= current_tolerance, directed=False)
        counts = np.bincount(component_ids)
        if counts.max(initial=0) <= 4:
            break
        current_tolerance *= factor
        if current_tolerance < minimum_tolerance:
            # Touching nodules can form a connected component containing two
            # annotations from the same reader. Resolve these rare cases with
            # deterministic constrained single-linkage: a physical nodule can
            # contain at most one annotation from each reader.
            parents = list(range(len(annotations)))
            readers = [{annotation.reader_index} for annotation in annotations]

            def find(index: int) -> int:
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            edges = sorted(
                (distances[left, right], left, right)
                for left in range(len(annotations))
                for right in range(left + 1, len(annotations))
                if distances[left, right] <= original_tolerance
            )
            for _, left, right in edges:
                left_root, right_root = find(left), find(right)
                if left_root == right_root or readers[left_root] & readers[right_root]:
                    continue
                parents[right_root] = left_root
                readers[left_root] |= readers[right_root]
            roots = [find(index) for index in range(len(annotations))]
            root_to_component = {
                root: component for component, root in enumerate(sorted(set(roots)))
            }
            component_ids = np.asarray(
                [root_to_component[root] for root in roots], dtype=np.int64
            )
            break

    clusters: list[list[ReaderAnnotation]] = []
    for component_id in sorted(set(component_ids.tolist())):
        cluster = [
            annotations[index]
            for index, value in enumerate(component_ids)
            if value == component_id
        ]
        clusters.append(cluster)
    return sorted(
        clusters,
        key=lambda group: float(
            np.mean([annotation.centroid(slice_spacing_mm)[2] for annotation in group])
        ),
    )


def eligible_clusters(
    document: AnnotationDocument,
    slice_spacing_mm: float,
    minimum_readers: int,
    tolerance: float | None = None,
) -> list[list[ReaderAnnotation]]:
    clusters = cluster_annotations(document.annotations, slice_spacing_mm, tolerance=tolerance)
    return [
        cluster
        for cluster in clusters
        if len({annotation.reader_index for annotation in cluster}) >= minimum_readers
    ]


def rasterize_consensus_masks(
    clusters: list[list[ReaderAnnotation]],
    sop_to_slice: dict[str, int],
    volume_shape_zyx: tuple[int, int, int],
    agreement_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize semantic and instance masks for eligible annotation clusters."""
    semantic = np.zeros(volume_shape_zyx, dtype=np.uint8)
    instances = np.zeros(volume_shape_zyx, dtype=np.uint16)
    for instance_id, cluster in enumerate(clusters, start=1):
        missing_sop_uids = {
            contour.sop_uid
            for annotation in cluster
            for contour in annotation.contours
            if contour.sop_uid not in sop_to_slice
        }
        if missing_sop_uids:
            raise ValueError(
                f"Annotation references {len(missing_sop_uids)} SOP UIDs outside the CT series"
            )
        valid_contours = [
            contour
            for annotation in cluster
            for contour in annotation.contours
            if contour.sop_uid in sop_to_slice
        ]
        if not valid_contours:
            continue
        z_values = [sop_to_slice[contour.sop_uid] for contour in valid_contours]
        x_values = [value for contour in valid_contours for value in contour.points_xy[:, 0]]
        y_values = [value for contour in valid_contours for value in contour.points_xy[:, 1]]
        z_min, z_max = min(z_values), max(z_values)
        y_min = max(0, math.floor(min(y_values)))
        y_max = min(volume_shape_zyx[1] - 1, math.ceil(max(y_values)))
        x_min = max(0, math.floor(min(x_values)))
        x_max = min(volume_shape_zyx[2] - 1, math.ceil(max(x_values)))
        local_shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
        votes = np.zeros(local_shape, dtype=np.uint8)

        for annotation in cluster:
            reader_mask = np.zeros(local_shape, dtype=bool)
            ordered_contours = sorted(
                annotation.contours,
                key=lambda contour: not contour.inclusion,
            )
            for contour in ordered_contours:
                local_z = sop_to_slice[contour.sop_uid] - z_min
                rows, columns = polygon(
                    contour.points_xy[:, 1] - y_min,
                    contour.points_xy[:, 0] - x_min,
                    shape=local_shape[1:],
                )
                if contour.inclusion:
                    reader_mask[local_z, rows, columns] = True
                else:
                    reader_mask[local_z, rows, columns] = False
            votes += reader_mask.astype(np.uint8)

        required_votes = math.ceil(len(cluster) * agreement_fraction)
        consensus = votes >= required_votes
        target = np.s_[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        semantic[target] |= consensus.astype(np.uint8)
        local_instances = instances[target]
        local_instances[consensus] = instance_id
        instances[target] = local_instances
    return semantic, instances

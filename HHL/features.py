from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import hog

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (200, 200)
COLOR_HIST_BINS = (8, 8, 8)
COLOR_HIST_LENGTH = int(np.prod(COLOR_HIST_BINS))
LANDMARK_COUNT = 468
LANDMARK_VECTOR_LENGTH = LANDMARK_COUNT * 2

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


def _compute_hog_vector_length(image_size: tuple[int, int]) -> int:
    width, height = image_size
    cells_x = width // HOG_PIXELS_PER_CELL[0]
    cells_y = height // HOG_PIXELS_PER_CELL[1]
    blocks_x = max(cells_x - HOG_CELLS_PER_BLOCK[0] + 1, 0)
    blocks_y = max(cells_y - HOG_CELLS_PER_BLOCK[1] + 1, 0)
    features_per_block = (
        HOG_CELLS_PER_BLOCK[0] * HOG_CELLS_PER_BLOCK[1] * HOG_ORIENTATIONS
    )
    return blocks_x * blocks_y * features_per_block


HOG_VECTOR_LENGTH = _compute_hog_vector_length(DEFAULT_IMAGE_SIZE)


@dataclass(frozen=True)
class FeatureBundle:
    hog: np.ndarray
    color_hist: np.ndarray
    landmark: np.ndarray

    def as_python_lists(self) -> dict[str, list[float]]:
        return {
            "hog": self.hog.astype(np.float64).tolist(),
            "color_hist": self.color_hist.astype(np.float64).tolist(),
            "landmark": self.landmark.astype(np.float64).tolist(),
        }


def ensure_bgr_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is empty.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3:
        raise ValueError("Input image must be a 2D or 3D array.")

    channels = image.shape[2]
    if channels == 3:
        return image
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported image format with {channels} channels.")


def read_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return ensure_bgr_image(image)


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Uploaded image is empty.")

    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return ensure_bgr_image(image)


class FeatureExtractor:
    def __init__(
        self,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.min_detection_confidence = min_detection_confidence
        self.hog_vector_length = _compute_hog_vector_length(image_size)
        self._face_mesh: Any | None = None
        self._face_mesh_init_lock = threading.Lock()
        self._face_mesh_process_lock = threading.Lock()

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        bgr_image = ensure_bgr_image(image)
        return cv2.resize(bgr_image, self.image_size, interpolation=cv2.INTER_AREA)

    def extract_color_hist(self, image: np.ndarray) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv_image],
            [0, 1, 2],
            None,
            list(COLOR_HIST_BINS),
            [0, 180, 0, 256, 0, 256],
        )
        histogram = cv2.normalize(
            histogram,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
        )
        return histogram.flatten().astype(np.float64)

    def extract_hog(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(
            gray_image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            feature_vector=True,
        )
        return np.asarray(features, dtype=np.float64)

    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            with self._face_mesh_process_lock:
                results = self._get_face_mesh().process(rgb_image)
        except Exception as exc:
            LOGGER.warning("MediaPipe landmark extraction failed: %s", exc)
            return np.zeros(LANDMARK_VECTOR_LENGTH, dtype=np.float64)

        if not results.multi_face_landmarks:
            return np.zeros(LANDMARK_VECTOR_LENGTH, dtype=np.float64)

        face_landmarks = results.multi_face_landmarks[0].landmark
        vector = np.zeros(LANDMARK_VECTOR_LENGTH, dtype=np.float64)

        for index, landmark in enumerate(face_landmarks[:LANDMARK_COUNT]):
            vector[2 * index] = landmark.x
            vector[(2 * index) + 1] = landmark.y

        return vector

    def extract(self, image: np.ndarray) -> FeatureBundle:
        processed_image = self.preprocess(image)
        return FeatureBundle(
            hog=self.extract_hog(processed_image),
            color_hist=self.extract_color_hist(processed_image),
            landmark=self.extract_landmarks(processed_image),
        )

    def extract_from_path(self, image_path: str | Path) -> FeatureBundle:
        return self.extract(read_image(image_path))

    def _get_face_mesh(self) -> Any:
        if self._face_mesh is None:
            with self._face_mesh_init_lock:
                if self._face_mesh is None:
                    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=self.min_detection_confidence,
                    )
        return self._face_mesh


_DEFAULT_EXTRACTOR: FeatureExtractor | None = None
_DEFAULT_EXTRACTOR_LOCK = threading.Lock()


def get_default_extractor() -> FeatureExtractor:
    global _DEFAULT_EXTRACTOR

    if _DEFAULT_EXTRACTOR is None:
        with _DEFAULT_EXTRACTOR_LOCK:
            if _DEFAULT_EXTRACTOR is None:
                _DEFAULT_EXTRACTOR = FeatureExtractor()

    return _DEFAULT_EXTRACTOR


def extract_features(image: np.ndarray) -> FeatureBundle:
    return get_default_extractor().extract(image)


def extract_features_from_path(image_path: str | Path) -> FeatureBundle:
    return get_default_extractor().extract_from_path(image_path)

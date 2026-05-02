from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import local_binary_pattern

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (200, 200)

# Color Histogram config
COLOR_HIST_BINS = (8, 8, 8)
COLOR_HIST_LENGTH = int(np.prod(COLOR_HIST_BINS))  # 512

# LBP config
LBP_P = 8   # Number of neighbors
LBP_R = 1   # Radius
LBP_LENGTH = 256  # 2^P = 256 bins (full histogram)

# Geometric Ratios config
GEOMETRIC_RATIOS_LENGTH = 6

# MediaPipe Face Mesh key landmark indices
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_FACE_TOP = 10
_CHIN = 152
_FACE_LEFT = 234
_FACE_RIGHT = 454
_NOSE_BRIDGE = 6
_NOSE_TIP = 1
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_CHEEKBONE_LEFT = 93
_CHEEKBONE_RIGHT = 323


def _landmark_distance(landmarks, idx1: int, idx2: int) -> float:
    """Tính khoảng cách Euclidean giữa 2 landmark (tọa độ chuẩn hóa x, y)."""
    p1, p2 = landmarks[idx1], landmarks[idx2]
    return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))


def _landmark_midpoint(landmarks, idx1: int, idx2: int) -> tuple[float, float]:
    """Tính điểm giữa của 2 landmark."""
    p1, p2 = landmarks[idx1], landmarks[idx2]
    return ((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)


@dataclass(frozen=True)
class FeatureBundle:
    color_hist: np.ndarray
    lbp: np.ndarray
    geometric_ratios: np.ndarray

    def as_python_lists(self) -> dict[str, list[float]]:
        return {
            "color_hist": self.color_hist.astype(np.float64).tolist(),
            "lbp": self.lbp.astype(np.float64).tolist(),
            "geometric_ratios": self.geometric_ratios.astype(np.float64).tolist(),
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
        """Trích xuất lược đồ màu HSV (8x8x8 = 512 bins)."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv_image], [0, 1, 2], None,
            list(COLOR_HIST_BINS), [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(histogram, histogram)
        return histogram.flatten().astype(np.float64)

    def extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """Trích xuất LBP histogram (P=8, R=1, full 256 bins)."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_image = local_binary_pattern(gray_image, LBP_P, LBP_R, method='default')
        histogram, _ = np.histogram(
            lbp_image.ravel(), bins=LBP_LENGTH, range=(0, LBP_LENGTH),
        )
        histogram = histogram.astype(np.float64)
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        return histogram

    def extract_geometric_ratios(self, image: np.ndarray) -> np.ndarray:
        """Trích xuất 6 tỷ lệ hình học từ facial landmarks.

        R1: Khoảng cách 2 tâm mắt / Chiều rộng mặt
        R2: Chiều cao mắt / Chiều rộng mắt (trung bình 2 mắt)
        R3: Chiều dài sống mũi / Chiều cao mặt
        R4: Chiều rộng miệng / Chiều rộng mặt
        R5: Khoảng cách 2 gò má / Chiều rộng mặt
        R6: Chiều cao mặt / Chiều rộng mặt
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            with self._face_mesh_process_lock:
                results = self._get_face_mesh().process(rgb_image)
        except Exception as exc:
            LOGGER.warning("MediaPipe landmark extraction failed: %s", exc)
            return np.zeros(GEOMETRIC_RATIOS_LENGTH, dtype=np.float64)

        if not results.multi_face_landmarks:
            return np.zeros(GEOMETRIC_RATIOS_LENGTH, dtype=np.float64)

        landmarks = results.multi_face_landmarks[0].landmark
        ratios = np.zeros(GEOMETRIC_RATIOS_LENGTH, dtype=np.float64)

        try:
            face_width = _landmark_distance(landmarks, _FACE_LEFT, _FACE_RIGHT)
            face_height = _landmark_distance(landmarks, _FACE_TOP, _CHIN)

            if face_width < 1e-10 or face_height < 1e-10:
                return ratios

            # R1: Khoảng cách 2 tâm mắt / Chiều rộng mặt
            left_eye_center = _landmark_midpoint(landmarks, _LEFT_EYE_OUTER, _LEFT_EYE_INNER)
            right_eye_center = _landmark_midpoint(landmarks, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER)
            eye_distance = float(np.sqrt(
                (left_eye_center[0] - right_eye_center[0]) ** 2 +
                (left_eye_center[1] - right_eye_center[1]) ** 2
            ))
            ratios[0] = eye_distance / face_width

            # R2: Độ mở mắt (trung bình 2 mắt)
            left_h = _landmark_distance(landmarks, _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM)
            left_w = _landmark_distance(landmarks, _LEFT_EYE_OUTER, _LEFT_EYE_INNER)
            right_h = _landmark_distance(landmarks, _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM)
            right_w = _landmark_distance(landmarks, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER)
            if left_w > 1e-10 and right_w > 1e-10:
                ratios[1] = (left_h / left_w + right_h / right_w) / 2.0

            # R3: Chiều dài sống mũi / Chiều cao mặt
            nose_length = _landmark_distance(landmarks, _NOSE_BRIDGE, _NOSE_TIP)
            ratios[2] = nose_length / face_height

            # R4: Chiều rộng miệng / Chiều rộng mặt
            mouth_width = _landmark_distance(landmarks, _MOUTH_LEFT, _MOUTH_RIGHT)
            ratios[3] = mouth_width / face_width

            # R5: Khoảng cách 2 gò má / Chiều rộng mặt
            cheekbone_dist = _landmark_distance(landmarks, _CHEEKBONE_LEFT, _CHEEKBONE_RIGHT)
            ratios[4] = cheekbone_dist / face_width

            # R6: Chiều cao mặt / Chiều rộng mặt
            ratios[5] = face_height / face_width

        except (IndexError, ZeroDivisionError) as exc:
            LOGGER.warning("Error computing geometric ratios: %s", exc)

        return ratios

    def extract(self, image: np.ndarray) -> FeatureBundle:
        processed_image = self.preprocess(image)
        return FeatureBundle(
            color_hist=self.extract_color_hist(processed_image),
            lbp=self.extract_lbp(processed_image),
            geometric_ratios=self.extract_geometric_ratios(processed_image),
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

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
import tkinter as tk
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterator
from dotenv import load_dotenv

import numpy as np
import psycopg2
from PIL import Image, ImageOps, ImageTk
from psycopg2.pool import SimpleConnectionPool

from features import (
    COLOR_HIST_LENGTH,
    HOG_VECTOR_LENGTH,
    LANDMARK_VECTOR_LENGTH,
    FeatureExtractor,
)

# Nạp cấu hình từ file .env
load_dotenv()

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-10
TOP_K = 5
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE_TYPES = [
    ("Tệp hình ảnh", "*.jpg *.jpeg *.png *.bmp"),
    ("Tất cả các tệp", "*.*"),
]
GENDER_LABELS = {0: "Nam", 1: "Nữ"}
RACE_LABELS = {
    0: "Da trắng",
    1: "Da đen",
    2: "Châu Á",
    3: "Ấn Độ",
    4: "Khác",
}
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str | None
    host: str
    port: int
    database: str
    user: str
    password: str

@dataclass(frozen=True)
class SearchMetadata:
    image_id: int
    image_path: str
    age: int
    gender: int
    race: int

@dataclass(frozen=True)
class SearchIndex:
    metadata: list[SearchMetadata]
    hog: np.ndarray
    color_hist: np.ndarray
    landmark: np.ndarray

    @property
    def size(self) -> int:
        return len(self.metadata)

@dataclass(frozen=True)
class SearchResult:
    metadata: SearchMetadata
    total_similarity: float
    hog_similarity: float
    color_similarity: float
    landmark_similarity: float

class SearchIndexCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._index: SearchIndex | None = None
        self._loaded_at = 0.0

    def get(self) -> SearchIndex:
        now = time.time()
        with self._lock:
            should_reload = self._index is None
            if self.ttl_seconds > 0 and (now - self._loaded_at) >= self.ttl_seconds:
                should_reload = True

            if should_reload:
                self._index = load_search_index()
                self._loaded_at = now

            return self._index

    def invalidate(self) -> None:
        with self._lock:
            self._index = None
            self._loaded_at = 0.0

def load_env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default

def get_database_config() -> DatabaseConfig:
    return DatabaseConfig(
        dsn=load_env_value("DATABASE_URL"),
        host=load_env_value("PGHOST", "POSTGRES_HOST", default="localhost") or "localhost",
        port=int(load_env_value("PGPORT", "POSTGRES_PORT", default="5432") or "5432"),
        database=load_env_value("PGDATABASE", "POSTGRES_DB", default="postgres") or "postgres",
        user=load_env_value("PGUSER", "POSTGRES_USER", default="postgres") or "postgres",
        password=load_env_value("PGPASSWORD", "POSTGRES_PASSWORD", default="") or "",
    )

_POOL_LOCK = threading.Lock()
_DB_POOL: SimpleConnectionPool | None = None
_FEATURE_EXTRACTOR = FeatureExtractor()
_SEARCH_CACHE = SearchIndexCache(
    ttl_seconds=int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "300"))
)

def resolve_image_path(image_path: str) -> Path:
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate

def get_db_pool() -> SimpleConnectionPool:
    global _DB_POOL
    if _DB_POOL is None:
        with _POOL_LOCK:
            if _DB_POOL is None:
                config = get_database_config()
                if config.dsn:
                    _DB_POOL = SimpleConnectionPool(
                        minconn=1,
                        maxconn=int(os.getenv("PG_MAX_CONNECTIONS", "5")),
                        dsn=config.dsn,
                    )
                else:
                    _DB_POOL = SimpleConnectionPool(
                        minconn=1,
                        maxconn=int(os.getenv("PG_MAX_CONNECTIONS", "5")),
                        host=config.host,
                        port=config.port,
                        dbname=config.database,
                        user=config.user,
                        password=config.password,
                    )
    return _DB_POOL

@contextmanager
def get_db_connection() -> Iterator:
    pool = get_db_pool()
    connection = pool.getconn()
    try:
        yield connection
    finally:
        pool.putconn(connection)

def load_search_index() -> SearchIndex:
    sql = """
    SELECT
        fi.id,
        fi.image_path,
        fi.age,
        fi.gender,
        fi.race,
        f.hog,
        f.color_hist,
        f.landmark
    FROM face_images AS fi
    INNER JOIN image_features AS f ON f.image_id = fi.id
    ORDER BY fi.id;
    """
    metadata: list[SearchMetadata] = []
    hog_vectors: list[np.ndarray] = []
    color_vectors: list[np.ndarray] = []
    landmark_vectors: list[np.ndarray] = []

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

    for row in rows:
        image_id, image_path, age, gender, race, hog, color_hist, landmark = row
        hog_vector = np.asarray(hog, dtype=np.float64)
        color_vector = np.asarray(color_hist, dtype=np.float64)
        landmark_vector = np.asarray(landmark, dtype=np.float64)

        if hog_vector.size != HOG_VECTOR_LENGTH:
            LOGGER.warning("Bỏ qua image_id=%s vì độ dài HOG là %s.", image_id, hog_vector.size)
            continue
        if color_vector.size != COLOR_HIST_LENGTH:
            LOGGER.warning("Bỏ qua image_id=%s vì độ dài lược đồ màu là %s.", image_id, color_vector.size)
            continue
        if landmark_vector.size != LANDMARK_VECTOR_LENGTH:
            LOGGER.warning("Bỏ qua image_id=%s vì độ dài landmark là %s.", image_id, landmark_vector.size)
            continue

        metadata.append(SearchMetadata(
            image_id=int(image_id), image_path=str(image_path),
            age=int(age), gender=int(gender), race=int(race),
        ))
        hog_vectors.append(hog_vector)
        color_vectors.append(color_vector)
        landmark_vectors.append(landmark_vector)

    if not metadata:
        return SearchIndex([], np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)))

    return SearchIndex(metadata, np.vstack(hog_vectors), np.vstack(color_vectors), np.vstack(landmark_vectors))

def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float64)
    candidates = np.asarray(matrix, dtype=np.float64)
    numerator = candidates @ query
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    return numerator / ((candidate_norms * query_norm) + EPSILON)

def min_max_normalize(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.size == 0: return scores
    score_min, score_max = np.min(scores), np.max(scores)
    return (scores - score_min) / ((score_max - score_min) + EPSILON)

def search_similar_images(query_path: Path, top_k: int = TOP_K) -> tuple[list[SearchResult], float]:
    start_time = time.perf_counter()
    query_features = _FEATURE_EXTRACTOR.extract_from_path(query_path)
    search_index = _SEARCH_CACHE.get()

    if search_index.size == 0:
        raise ValueError("Không tìm thấy ảnh đã lập chỉ mục trong cơ sở dữ liệu.")

    h_sim = cosine_similarity(query_features.hog, search_index.hog)
    c_sim = cosine_similarity(query_features.color_hist, search_index.color_hist)
    l_sim = cosine_similarity(query_features.landmark, search_index.landmark)

    h_norm = min_max_normalize(h_sim)
    c_norm = min_max_normalize(c_sim)
    l_norm = min_max_normalize(l_sim)

    # Trọng số: 0.5 Landmark, 0.3 HOG, 0.2 Color
    total_similarity = (0.5 * l_norm) + (0.3 * h_norm) + (0.2 * c_norm)
    top_indices = np.argsort(total_similarity)[::-1][: min(top_k, search_index.size)]
    results = [SearchResult(
        search_index.metadata[int(i)], float(total_similarity[i]),
        float(h_norm[i]), float(c_norm[i]), float(l_norm[i])
    ) for i in top_indices]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return results, elapsed_ms

def gender_to_text(value: int) -> str:
    return GENDER_LABELS.get(value, str(value))

def race_to_text(value: int) -> str:
    return RACE_LABELS.get(value, str(value))

class ChildFaceRetrievalApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Hệ thống Tìm kiếm Khuôn mặt Trẻ em")
        self.geometry("1100x750")
        self.minsize(950, 650)
        self.configure(bg="#eef3f9")

        self.selected_image_path: Path | None = None
        self.query_photo: ImageTk.PhotoImage | None = None
        self.result_photos: list[ImageTk.PhotoImage] = []
        self.is_searching = False

        self.status_var = tk.StringVar(value="Chọn ảnh để bắt đầu.")
        self.elapsed_var = tk.StringVar(value="Thời gian tìm kiếm: -- ms")
        self.file_var = tk.StringVar(value="Chưa chọn ảnh.")
        self.result_title_var = tk.StringVar(value=f"{TOP_K} Ảnh tương đồng nhất")

        self._configure_styles()
        self._build_layout()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        for name, color in [("Hog", "#4f6ef7"), ("Color", "#3fb576"), ("Landmark", "#5ac2f2")]:
            style.configure(
                f"{name}.Horizontal.TProgressbar",
                troughcolor="#e8edf7",
                background=color,
                bordercolor="#e8edf7",
                lightcolor=color,
                darkcolor=color,
                thickness=10,
            )

    def _build_layout(self) -> None:
        # Header ở trên cùng
        header = tk.Frame(self, bg="#2f5fd0", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg="#2f5fd0")
        header_inner.pack(fill="both", expand=True, padx=20, pady=5)

        tk.Label(header_inner, text="Hệ thống Tìm kiếm Khuôn mặt Trẻ em", font=("Segoe UI", 16, "bold"), fg="white", bg="#2f5fd0").pack(side="left")
        tk.Label(header_inner, textvariable=self.elapsed_var, font=("Segoe UI", 10), fg="#dbe6ff", bg="#2f5fd0").pack(side="right")

        # Container chính để chia cột
        main_container = tk.Frame(self, bg="#eef3f9")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        main_container.grid_columnconfigure(0, weight=0) # Cột trái cố định
        main_container.grid_columnconfigure(1, weight=1) # Cột phải mở rộng
        main_container.grid_rowconfigure(0, weight=1)

        # ------------------------------------------
        # CỘT TRÁI: ĐIỀU KHIỂN & PREVIEW (QUERY PANEL)
        # ------------------------------------------
        left_panel = tk.Frame(main_container, bg="#eef3f9", width=320)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.grid_propagate(False)

        # Card điều khiển
        control_card = tk.Frame(left_panel, bg="white", bd=1, relief="solid", highlightbackground="#d7dfec", highlightthickness=1)
        control_card.pack(fill="x", pady=(0, 10))

        tk.Label(control_card, text="Truy vấn hệ thống", bg="white", fg="#18202d", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=15, pady=(15, 2))
        
        btn_row = tk.Frame(control_card, bg="white")
        btn_row.pack(fill="x", padx=15, pady=10)
        self.choose_button = tk.Button(btn_row, text="Chọn ảnh", command=self.choose_image, bg="#2f5fd0", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10)
        self.choose_button.pack(side="left")
        self.search_button = tk.Button(btn_row, text="Tìm kiếm", command=self.start_search, state="disabled", bg="#2f5fd0", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10)
        self.search_button.pack(side="left", padx=5)

        # Thông tin file và trạng thái
        info_box = tk.Frame(control_card, bg="#f7f9fc", highlightbackground="#e3e9f3", highlightthickness=1)
        info_box.pack(fill="x", padx=15, pady=(0, 15))
        tk.Label(info_box, textvariable=self.file_var, bg="#f7f9fc", fg="#334155", font=("Segoe UI", 8), anchor="w", wraplength=280).pack(fill="x", padx=8, pady=5)
        tk.Label(info_box, textvariable=self.status_var, bg="#f7f9fc", fg="#2f5fd0", font=("Segoe UI", 8, "italic"), anchor="w").pack(fill="x", padx=8, pady=(0, 5))

        # Card Xem trước ảnh Query
        preview_card = tk.Frame(left_panel, bg="white", bd=1, relief="solid", highlightbackground="#d7dfec", highlightthickness=1)
        preview_card.pack(fill="both", expand=True)

        tk.Label(preview_card, text="Ảnh truy vấn", bg="white", fg="#18202d", font=("Segoe UI", 12, "bold")).pack(pady=(15, 5))
        query_frame = tk.Frame(preview_card, bg="#f4f7fb", bd=1, relief="solid", highlightbackground="#e3e9f3", highlightthickness=1)
        query_frame.pack(padx=15, pady=10)
        self.query_image_label = tk.Label(query_frame, bg="#f4f7fb", text="Chưa có ảnh", fg="#718096")
        self.query_image_label.pack(padx=5, pady=5)

        # ------------------------------------------
        # CỘT PHẢI: KẾT QUẢ (RESULTS PANEL)
        # ------------------------------------------
        results_panel = tk.Frame(main_container, bg="white", bd=1, relief="solid", highlightbackground="#d7dfec", highlightthickness=1)
        results_panel.grid(row=0, column=1, sticky="nsew")

        results_header = tk.Frame(results_panel, bg="white")
        results_header.pack(fill="x", padx=15, pady=15)
        tk.Label(results_header, textvariable=self.result_title_var, bg="white", fg="#1f2937", font=("Segoe UI", 14, "bold")).pack(side="left")

        # Vùng Scroll cho kết quả
        results_host = tk.Frame(results_panel, bg="white")
        results_host.pack(fill="both", expand=True, padx=5, pady=(0, 10))

        self.results_canvas = tk.Canvas(results_host, bg="white", highlightthickness=0)
        self.results_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(results_host, orient="vertical", command=self.results_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_container = tk.Frame(self.results_canvas, bg="white")
        self.cards_window = self.results_canvas.create_window((0, 0), window=self.cards_container, anchor="nw")
        
        # Chia 2 cột kết quả để thẻ ảnh to rõ ràng
        for c in range(2): 
            self.cards_container.grid_columnconfigure(c, weight=1, minsize=300)

        self.cards_container.bind("<Configure>", lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))
        self.results_canvas.bind("<Configure>", lambda e: self.results_canvas.itemconfigure(self.cards_window, width=e.width))
        self.results_canvas.bind_all("<MouseWheel>", lambda e: self.results_canvas.yview_scroll(int(-e.delta / 120), "units"))

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(title="Chọn ảnh khuôn mặt trẻ em", filetypes=IMAGE_FILE_TYPES)
        if not path: return
        self.selected_image_path = Path(path)
        self.file_var.set(self._format_selected_file_text(self.selected_image_path))
        self.status_var.set("Sẵn sàng tìm kiếm.")
        self.search_button.config(state="normal")
        self._set_query_preview(self.selected_image_path)
        self._clear_results()

    def start_search(self) -> None:
        if self.is_searching or self.selected_image_path is None: return
        self.is_searching = True
        self.status_var.set("Đang xử lý đặc trưng...")
        self.elapsed_var.set("Thời gian tìm kiếm: Đang xử lý...")
        self.search_button.config(state="disabled"); self.choose_button.config(state="disabled")
        self._clear_results()
        threading.Thread(target=self._search_worker, daemon=True).start()

    def _search_worker(self) -> None:
        try:
            results, elapsed_ms = search_similar_images(self.selected_image_path)
            self.after(0, lambda: self._render_results(results, elapsed_ms))
        except Exception as exc:
            self.after(0, lambda: self._handle_search_error(str(exc)))

    def _render_results(self, results: list[SearchResult], elapsed_ms: float) -> None:
        self.is_searching = False
        self.search_button.config(state="normal"); self.choose_button.config(state="normal")
        self.status_var.set("Tìm kiếm hoàn tất.")
        self.elapsed_var.set(f"Thời gian tìm kiếm: {elapsed_ms:.2f} ms")
        self.result_title_var.set(f"{len(results)} Ảnh tương đồng nhất")
        
        if not results:
            tk.Label(self.cards_container, text="Không có kết quả phù hợp.", bg="white", fg="#5b6473", font=("Segoe UI", 11)).grid(row=0, column=0)
            return

        for i, res in enumerate(results):
            card = self._create_result_card(self.cards_container, res, i + 1)
            # Dàn kết quả theo 2 cột trong vùng bên phải
            card.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")

    def _handle_search_error(self, msg: str) -> None:
        self.is_searching = False
        self.search_button.config(state="normal"); self.choose_button.config(state="normal")
        self.status_var.set("Lỗi tìm kiếm.")
        messagebox.showerror("Lỗi", msg)

    def _clear_results(self) -> None:
        self.result_photos.clear()
        for w in self.cards_container.winfo_children(): w.destroy()
        self.results_canvas.yview_moveto(0)

    def _set_query_preview(self, path: Path) -> None:
        self.query_photo = self._load_photo(path, (220, 220), background_color="#f4f7fb")
        self.query_image_label.config(image=self.query_photo, text="")

    def _load_photo(self, path: Path, size: tuple[int, int], background_color: str = "#f3f6fb") -> ImageTk.PhotoImage:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail(size, RESAMPLE)
            bg = Image.new("RGB", size, background_color)
            bg.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
        return ImageTk.PhotoImage(bg)

    def _create_result_card(self, parent: tk.Widget, res: SearchResult, rank: int) -> tk.Frame:
        card = tk.Frame(parent, bg="#fbfcfe", bd=1, relief="solid", highlightbackground="#e2e8f0", highlightthickness=1)
        path = resolve_image_path(res.metadata.image_path)
        photo = self._load_photo(path, (160, 160), background_color="#fbfcfe")
        self.result_photos.append(photo)
        
        tk.Label(card, image=photo, bg="#fbfcfe").pack(fill="x", pady=(10, 0))
        body = tk.Frame(card, bg="#fbfcfe"); body.pack(fill="both", expand=True, padx=12, pady=10)
        
        tk.Label(body, text=f"{res.metadata.age} Tuổi, {gender_to_text(res.metadata.gender)}", bg="#fbfcfe", fg="#111827", font=("Segoe UI", 11, "bold")).pack(fill="x")
        tk.Label(body, text=f"Độ tương đồng tổng thể: {res.total_similarity * 100:.2f}%", bg="#fbfcfe", fg="#2f5fd0", font=("Segoe UI", 9, "bold"), wraplength=250).pack(fill="x", pady=(6, 2))
        tk.Label(body, text=f"Hạng #{rank} | Chủng tộc: {race_to_text(res.metadata.race)}", bg="#fbfcfe", fg="#6b7280", font=("Segoe UI", 8)).pack(fill="x", pady=(0, 8))
        
        tk.Label(body, text="Chi tiết các đặc trưng:", bg="#fbfcfe", fg="#4b5563", font=("Segoe UI", 8, "italic"), anchor="w").pack(fill="x", pady=(0, 6))
        self._add_similarity_row(body, "Cấu trúc (HOG)", res.hog_similarity, "Hog")
        self._add_similarity_row(body, "Màu sắc (HSV)", res.color_similarity, "Color")
        self._add_similarity_row(body, "Hình học (Landmark)", res.landmark_similarity, "Landmark")
        return card

    def _add_similarity_row(self, parent: tk.Widget, label: str, score: float, style: str) -> None:
        frame = tk.Frame(parent, bg="#fbfcfe")
        frame.pack(fill="x", pady=(0, 7))
        frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            frame,
            text=label,
            bg="#fbfcfe",
            fg="#374151",
            font=("Segoe UI", 8),
            width=20,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        ttk.Progressbar(
            frame,
            style=f"{style}.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100,
            value=score * 100,
        ).grid(row=0, column=1, sticky="ew")

        tk.Label(
            frame,
            text=f"{score * 100:.2f}%",
            bg="#fbfcfe",
            fg="#374151",
            font=("Segoe UI", 8, "bold"),
            width=8,
            anchor="e",
        ).grid(row=0, column=2, sticky="e", padx=(8, 0))

    def _format_selected_file_text(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(BASE_DIR)
            return f"Tệp: {path.name}\nĐường dẫn: {rel.as_posix()}"
        except ValueError: return f"Tệp: {path.name}"

def shutdown_resources() -> None:
    _FEATURE_EXTRACTOR.close()
    if _DB_POOL is not None: _DB_POOL.closeall()

atexit.register(shutdown_resources)

if __name__ == "__main__":
    app = ChildFaceRetrievalApp()
    app.mainloop()

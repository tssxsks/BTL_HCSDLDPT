from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import execute_values

from features import FeatureExtractor
load_dotenv()
LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset" / "MainData"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

UNDERSCORE_FILENAME_PATTERN = re.compile(
    r"^(?P<age>\d{1,2})_(?P<gender>\d)_(?P<race>\d)_(?P<timestamp>\d+)"
)
COMPACT_FILENAME_PATTERN = re.compile(
    r"^(?P<age>\d{1,2})(?P<gender>\d)(?P<race>\d)_(?P<timestamp>\d+)"
)


@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str | None
    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass(frozen=True)
class ParsedMetadata:
    age: int
    gender: int
    race: int
    timestamp: str


@dataclass(frozen=True)
class ETLRecord:
    image_path: str
    age: int
    gender: int
    race: int
    hog: list[float]
    color_hist: list[float]
    landmark: list[float]


def parse_metadata_from_filename(filename: str) -> ParsedMetadata:
    for pattern in (UNDERSCORE_FILENAME_PATTERN, COMPACT_FILENAME_PATTERN):
        match = pattern.match(filename)
        if match:
            groups = match.groupdict()
            return ParsedMetadata(
                age=int(groups["age"]),
                gender=int(groups["gender"]),
                race=int(groups["race"]),
                timestamp=groups["timestamp"],
            )

    raise ValueError(f"Unsupported file name format: {filename}")


def iter_image_paths(dataset_dir: Path) -> Iterator[Path]:
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_database_config(args: argparse.Namespace) -> DatabaseConfig:
    return DatabaseConfig(
        dsn=args.dsn or os.getenv("DATABASE_URL"),
        host=args.host or os.getenv("PGHOST", "localhost"),
        port=int(args.port or os.getenv("PGPORT", "5432")),
        database=args.database or os.getenv("PGDATABASE", "postgres"),
        user=args.user or os.getenv("PGUSER", "postgres"),
        password=args.password or os.getenv("PGPASSWORD", ""),
    )


def create_connection(config: DatabaseConfig):
    if config.dsn:
        return psycopg2.connect(config.dsn)

    return psycopg2.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
    )


def create_tables(connection) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS face_images (
        id SERIAL PRIMARY KEY,
        image_path TEXT NOT NULL,
        age INT NOT NULL,
        gender INT NOT NULL,
        race INT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS image_features (
        image_id INT PRIMARY KEY REFERENCES face_images(id) ON DELETE CASCADE,
        hog FLOAT8[] NOT NULL,
        color_hist FLOAT8[] NOT NULL,
        landmark FLOAT8[] NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_face_images_image_path ON face_images(image_path);
    CREATE INDEX IF NOT EXISTS idx_image_features_image_id ON image_features(image_id);
    """

    with connection.cursor() as cursor:
        cursor.execute(create_sql)
    connection.commit()


def drop_tables(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DROP TABLE IF EXISTS image_features CASCADE;
            DROP TABLE IF EXISTS face_images CASCADE;
            """
        )
    connection.commit()


def reset_tables(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE image_features, face_images RESTART IDENTITY CASCADE;")
    connection.commit()


def to_storage_path(image_path: Path, project_root: Path) -> str:
    try:
        return image_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(image_path.resolve())


def build_record(image_path: Path, project_root: Path, extractor: FeatureExtractor) -> ETLRecord:
    metadata = parse_metadata_from_filename(image_path.name)
    features = extractor.extract_from_path(image_path).as_python_lists()
    return ETLRecord(
        image_path=to_storage_path(image_path, project_root),
        age=metadata.age,
        gender=metadata.gender,
        race=metadata.race,
        hog=features["hog"],
        color_hist=features["color_hist"],
        landmark=features["landmark"],
    )


def bulk_insert_records(connection, records: list[ETLRecord]) -> None:
    if not records:
        return

    face_rows = [(record.image_path, record.age, record.gender, record.race) for record in records]

    with connection.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO face_images (image_path, age, gender, race)
            VALUES %s
            RETURNING id, image_path;
            """,
            face_rows,
            page_size=len(face_rows),
        )
        inserted_rows = cursor.fetchall()

        image_id_map: dict[str, list[int]] = {}
        for image_id, image_path in inserted_rows:
            image_id_map.setdefault(image_path, []).append(image_id)

        feature_rows = []
        for record in records:
            available_ids = image_id_map.get(record.image_path)
            if not available_ids:
                raise RuntimeError(f"Missing generated image_id for path: {record.image_path}")

            feature_rows.append(
                (
                    available_ids.pop(0),
                    record.hog,
                    record.color_hist,
                    record.landmark,
                )
            )

        execute_values(
            cursor,
            """
            INSERT INTO image_features (image_id, hog, color_hist, landmark)
            VALUES %s;
            """,
            feature_rows,
            page_size=len(feature_rows),
        )


def flush_batch(connection, batch: list[ETLRecord]) -> int:
    if not batch:
        return 0

    try:
        bulk_insert_records(connection, batch)
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    inserted_count = len(batch)
    batch.clear()
    return inserted_count


def run_etl(connection, dataset_dir: Path, batch_size: int) -> tuple[int, int]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    project_root = Path(__file__).resolve().parent
    inserted_count = 0
    skipped_count = 0
    batch: list[ETLRecord] = []

    with FeatureExtractor() as extractor:
        for image_path in iter_image_paths(dataset_dir):
            try:
                batch.append(build_record(image_path, project_root, extractor))
            except Exception as exc:
                skipped_count += 1
                LOGGER.warning("Skipping %s: %s", image_path, exc)
                continue

            if len(batch) >= batch_size:
                inserted_count += flush_batch(connection, batch)
                LOGGER.info("Inserted %s records so far.", inserted_count)

        if batch:
            inserted_count += flush_batch(connection, batch)

    return inserted_count, skipped_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ETL pipeline for child face retrieval data.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to MainData directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of images processed per bulk insert batch.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing face_images and image_features tables before recreating them.",
    )
    parser.add_argument("--reset", action="store_true", help="Truncate tables before ETL.")
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN or DATABASE_URL override.")
    parser.add_argument("--host", default=None, help="PostgreSQL host.")
    parser.add_argument("--port", default=None, help="PostgreSQL port.")
    parser.add_argument("--database", default=None, help="PostgreSQL database name.")
    parser.add_argument("--user", default=None, help="PostgreSQL user.")
    parser.add_argument("--password", default=None, help="PostgreSQL password.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    connection = create_connection(build_database_config(args))

    try:
        if args.drop_existing:
            drop_tables(connection)
        create_tables(connection)
        if args.reset:
            reset_tables(connection)

        inserted_count, skipped_count = run_etl(
            connection=connection,
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
        )
        LOGGER.info("ETL completed. Inserted=%s, Skipped=%s", inserted_count, skipped_count)
    finally:
        connection.close()


if __name__ == "__main__":
    main()

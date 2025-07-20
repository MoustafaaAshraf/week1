from pathlib import Path
import pandas as pd


def generate_dummy_dataset(
    output_path: Path,
    num_records: int = 10_000,
) -> Path:
    """Create a Parquet file with synthetic camera & lidar columns.

    Args:
        output_path (Path): The path to save the dataset.
        num_records (int, optional): The number of records
            to generate. Defaults to 10_000.

    Returns:
        Path: The path to the generated dataset.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = pd.DataFrame(
        {
            "frame_id": range(num_records),
            "camera_jpeg": [b"\\xFF\\xD8\\xFF"] * num_records,  # dummy bytes
            "lidar_npz": [b"PK\\x03\\x04"] * num_records,
        }
    )
    file_path = output_path / "dummy_dataset.parquet"
    df.to_parquet(file_path)
    return file_path

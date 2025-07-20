from pathlib import Path
import pytest
from dataset_tools import generate_dummy_dataset
import pandas as pd


@pytest.fixture
def sample_sensor_data(
    num_records: int = 100_000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_id": range(num_records),
            "camera_jpeg": [b"\\xFF\\xD8\\xFF"] * num_records,
            "lidar_npz": [b"PK\\x03\\x04"] * num_records,
        }
    )


def test_sample_sensor_data(
    sample_sensor_data: pd.DataFrame,
) -> None:
    assert len(sample_sensor_data) == 100_000
    assert sample_sensor_data["frame_id"].nunique() == 100_000
    assert sample_sensor_data["camera_jpeg"].nunique() == 1
    assert sample_sensor_data["lidar_npz"].nunique() == 1


def test_generate_dummy_dataset_default(
    tmp_path: Path,
) -> None:
    """Test default behavior with 10,000 records."""
    output_path = generate_dummy_dataset(tmp_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.name == "dummy_dataset.parquet"


def test_generate_dummy_dataset_custom_records(
    tmp_path: Path,
) -> None:
    """Test with custom number of records."""
    output_path = generate_dummy_dataset(tmp_path, num_records=100)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_dummy_dataset_zero_records(
    tmp_path: Path,
) -> None:
    """Test with zero records (edge case)."""
    output_path = generate_dummy_dataset(tmp_path, num_records=0)
    assert output_path.exists()
    # File should exist but might be small with no data


def test_generate_dummy_dataset_single_record(
    tmp_path: Path,
) -> None:
    """Test with single record."""
    output_path = generate_dummy_dataset(tmp_path, num_records=1)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_dummy_dataset_nested_path(
    tmp_path: Path,
) -> None:
    """Test creating dataset in nested directory structure."""
    nested_path = tmp_path / "deeply" / "nested" / "directory"
    output_path = generate_dummy_dataset(nested_path)
    assert output_path.exists()
    assert nested_path.exists()


def test_generate_dummy_dataset_existing_directory(
    tmp_path: Path,
) -> None:
    """Test creating dataset in existing directory."""
    # Create directory first
    (tmp_path / "existing_dir").mkdir()
    output_path = generate_dummy_dataset(tmp_path / "existing_dir")
    assert output_path.exists()


def test_generate_dummy_dataset_large_records(
    tmp_path: Path,
) -> None:
    """Test with large number of records."""
    output_path = generate_dummy_dataset(tmp_path, num_records=50_000)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

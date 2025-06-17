import csv
import json
from typing import Dict, List, Optional  # noqa: F401


class Logger:
    """
    Simple logger for recording SCvx iteration metrics and saving to file.
    """

    def __init__(self):
        self.records: List[Dict] = []

    def log(self, record: Dict) -> None:
        """
        Append a record (e.g., per-iteration metrics).

        Args:
            record: Dictionary of metric names to values.
        """
        self.records.append(record)

    def save_csv(self, filepath: str) -> None:
        """
        Save logged records to a CSV file.

        Args:
            filepath: Path to output CSV file.
        """
        if not self.records:
            return
        keys = list(self.records[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)

    def save_json(self, filepath: str) -> None:
        """
        Save logged records to a JSON file.

        Args:
            filepath: Path to output JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=2)

    def clear(self) -> None:
        """
        Clear all logged records.
        """
        self.records = []

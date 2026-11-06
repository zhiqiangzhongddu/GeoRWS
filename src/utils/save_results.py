"""Append result rows while serializing writers on the shared filesystem."""
import csv
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from src.utils.paths import ensure_dir


@contextmanager
def exclusive_table_lock(table_path: Path):
    """Hold the persistent advisory lock used by concurrent table writers."""
    ensure_dir(table_path.parent)
    lock_path = table_path.with_suffix('.tsv.lock')
    with open(lock_path, 'a') as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def append_result_row(table_path, row: dict) -> None:
    path = Path(table_path)
    with exclusive_table_lock(path):
        if path.exists() and path.stat().st_size:
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                header = list(reader.fieldnames or [])
                new_columns = [key for key in row if key not in header]
                if new_columns:
                    rows = list(reader)
                else:
                    with path.open('a', newline='', encoding='utf-8') as out:
                        csv.DictWriter(out, fieldnames=header, delimiter='\t').writerow(row)
                    return
        else:
            header, new_columns, rows = [], list(row), []
        header += new_columns
        fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', newline='', encoding='utf-8') as out:
                writer = csv.DictWriter(out, fieldnames=header, delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow(row)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

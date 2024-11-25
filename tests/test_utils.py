import ast
import math
import re

import numpy as np
from syrupy.extensions.amber import AmberSnapshotExtension


class CustomFloatSnapshotExtension(AmberSnapshotExtension):
    def parse_snapshot_to_numpy_no_eval(self, snapshot: str) -> tuple[np.ndarray]:
        # Remove metadata lines starting with `#`
        snapshot = "\n".join(line for line in snapshot.splitlines() if not line.strip().startswith("#"))

        # Extract array strings using regex
        array_pattern = r"array\((\[.*?\])\)"
        matches = re.findall(array_pattern, snapshot, flags=re.S)

        # Parse each array string into a NumPy array
        arrays = []
        for match in matches:
            # Replace "..." with the repeating last row/column to avoid parsing errors
            cleaned_array = match.replace("...,", "")
            # Convert the array string into a NumPy array using `np.array` and `eval`
            arrays.append(
                np.array(ast.literal_eval(cleaned_array)),
            )  # Use `eval` only for literals, not the whole snapshot
        return tuple(arrays)

    def matches(self, *, serialized_data: str, snapshot_data: str) -> bool:
        try:
            # Convert serialized and snapshot data to floats and compare within tolerance
            a = float(serialized_data)
            b = float(snapshot_data)
            return math.isclose(a, b, abs_tol=1e-4)
        except ValueError:
            # If conversion to float fails, fallback to default comparison
            pass
        try:
            # Convert serialized and snapshot data to np arrays and compare within tolerance
            a = self.parse_snapshot_to_numpy_no_eval(serialized_data)
            b = self.parse_snapshot_to_numpy_no_eval(snapshot_data)
            for array_a, array_b in zip(a, b):
                if not all(
                    math.isclose(array_a[index], array_b[index], rel_tol=1e-4) for index in np.ndindex(array_a.shape)
                ):
                    return False
            return True
        except ValueError:
            return serialized_data == snapshot_data

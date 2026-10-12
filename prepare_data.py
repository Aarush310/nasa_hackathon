"""
Pre-compute the per-grid-cell table the app serves from.

    python prepare_data.py            # writes data/grid_latest.csv (~1 MB)

The app runs this automatically if the file is missing, but doing it ahead of
time (the Docker build does) makes startup instant and keeps the 97 MB zip out
of the runtime image.
"""

import sys
from pathlib import Path

from farmiq_core import build_grid

SOURCE = Path("combined_europe_data_Test1.csv.zip")
OUTPUT = Path("data/grid_latest.csv")


def main(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    grid = build_grid(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False)
    print(f"Wrote {len(grid)} grid cells ({grid['period_start'].iloc[0]} → "
          f"{grid['period_end'].iloc[0]}) to {output}")
    return output


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))

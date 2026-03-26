from __future__ import annotations

import argparse
import io
import pickle
import struct
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable
from typing import cast

DEFAULT_INPUT_PATH = Path("data/shared/fi_results.pkl")
TABLE_COLUMNS = ("circuit_type", "nqubit", "nlayer", "fi_value")


@dataclass(frozen=True)
class PrintableFiResult:
    circuit_type: str
    nqubit: int
    fi_value: float
    nlayer: int | None = None


class FakeDType:
    def __init__(self, code: str, *_: Any) -> None:
        self.code = code
        self.byteorder = "<"

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        if len(state) > 1 and state[1] in {"<", ">"}:
            self.byteorder = state[1]


def _numpy_scalar(dtype: FakeDType, raw_value: bytes) -> float:
    if dtype.code != "f8":
        raise TypeError(f"expected f8 dtype, got {dtype.code}")

    format_code = "<d" if dtype.byteorder != ">" else ">d"
    return cast(float, struct.unpack(format_code, raw_value)[0])


class FiResultUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "fisher_information_utils" and name == "FiResult":
            return PrintableFiResult
        if module == "numpy" and name == "dtype":
            return FakeDType
        if module == "numpy._core.multiarray" and name == "scalar":
            return _numpy_scalar
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print FI results from a pickle file.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        type=Path,
        help=f"Input pickle file. Default: {DEFAULT_INPUT_PATH}",
    )
    return parser.parse_args()


def _normalize_result(item: Any) -> PrintableFiResult:
    if isinstance(item, PrintableFiResult):
        return item

    field_names = {field.name for field in fields(PrintableFiResult)}
    if is_dataclass(item):
        item = {name: getattr(item, name) for name in field_names}

    if isinstance(item, dict):
        return PrintableFiResult(**{name: item[name] for name in field_names})

    raise TypeError(f"unsupported result type: {type(item)!r}")


def load_results(input_path: Path) -> list[PrintableFiResult]:
    if not input_path.exists():
        raise FileNotFoundError(f"missing input file: {input_path}")

    with input_path.open("rb") as file_obj:
        data = FiResultUnpickler(io.BytesIO(file_obj.read())).load()

    if not isinstance(data, list):
        raise TypeError("pickle payload must be a list")

    return [_normalize_result(item) for item in data]


def build_rows(results: Iterable[PrintableFiResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in sorted(
        results,
        key=lambda item: (item.nqubit, item.circuit_type, item.nlayer or -1),
    ):
        rows.append(
            {
                "circuit_type": result.circuit_type,
                "nqubit": str(result.nqubit),
                "nlayer": "-" if result.nlayer is None else str(result.nlayer),
                "fi_value": f"{result.fi_value:.10f}",
            }
        )
    return rows


def format_table(rows: list[dict[str, str]]) -> str:
    headers = {column: column for column in TABLE_COLUMNS}
    widths = {
        column: max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in TABLE_COLUMNS
    }

    def render_separator(fill: str = "-") -> str:
        return "+" + "+".join(fill * (widths[column] + 2) for column in TABLE_COLUMNS) + "+"

    def render_row(row: dict[str, str]) -> str:
        return "| " + " | ".join(row[column].ljust(widths[column]) for column in TABLE_COLUMNS) + " |"

    lines = [render_separator(), render_row(headers), render_separator("=")]
    lines.extend(render_row(row) for row in rows)
    lines.append(render_separator())
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = load_results(args.input_path)
    if not results:
        print(f"no FI results found in {args.input_path}")
        return

    rows = build_rows(results)
    print(format_table(rows))
    print(f"printed {len(rows)} rows from {args.input_path}")


if __name__ == "__main__":
    main()

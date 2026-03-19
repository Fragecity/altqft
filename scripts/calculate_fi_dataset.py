from __future__ import annotations

import pickle
from pathlib import Path

from fisher_information_utils import (
    FiExperimentConfig,
    FiResult,
    calculate_fi_results,
    random_hlayout,
)


def build_dataset() -> list[FiResult]:
    all_results: list[FiResult] = []

    all_results.extend(
        calculate_fi_results(
            FiExperimentConfig(
                circuit_type="qft",
                nqubit_values=list(range(3, 6)),
                repeat=1,
            )
        )
    )

    for nqubit in range(3, 6):
        canonical_hlayout = list(range(nqubit))
        all_results.extend(
            calculate_fi_results(
                FiExperimentConfig(
                    circuit_type="ph",
                    nqubit_values=[nqubit],
                    hlayout=canonical_hlayout,
                    repeat=1,
                )
            )
        )

        sampled_hlayout = random_hlayout(nqubit)
        all_results.extend(
            calculate_fi_results(
                FiExperimentConfig(
                    circuit_type="ph",
                    nqubit_values=[nqubit],
                    hlayout=sampled_hlayout,
                    repeat=1,
                )
            )
        )

    return all_results


def save_dataset(results: list[FiResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(results, file_obj)


def main() -> None:
    output_path = Path("data/shared/fi_results.pkl")
    results = build_dataset()
    for result in results:
        print(result)
    save_dataset(results, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from fisher_information_utils import FiExperimentConfig, calculate_fi_results


def main() -> None:
    config = FiExperimentConfig(
        circuit_type="qft",
        nqubit_values=list(range(3, 11)),
        repeat=1,
    )
    results = calculate_fi_results(config)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()

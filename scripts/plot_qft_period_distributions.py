from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from altqft.circuits.ph import qft
from altqft.nn.process_qc import make_prob


def calculate_distributions(nqubit: int, period: int, shift: int = 0) -> tuple[list[float], list[float]]:
    num_states = 2**nqubit
    circuit = qft(nqubit)

    prob_func1 = make_prob(circuit, period)
    prob_func2 = make_prob(circuit, period + 1)

    prob1_dist = [prob_func1(col, shift) for col in range(num_states)]
    prob2_dist = [prob_func2(col, shift) for col in range(num_states)]
    return prob1_dist, prob2_dist


def plot_distributions(dist1: list[float], dist2: list[float], nqubit: int, save_path: Path) -> None:
    x_axis = range(len(dist1))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    color1 = "#8ecae6"
    color2 = "#219ebc"

    plt.plot(x_axis, dist1, color=color1, linestyle="-", linewidth=1.5)
    plt.fill_between(x_axis, dist1, color=color1, alpha=0.5)
    plt.plot(x_axis, dist2, color=color2, linestyle="-", linewidth=1.5)
    plt.fill_between(x_axis, dist2, color=color2, alpha=0.5)

    plt.xlabel("computational basis")
    plt.ylabel("counts")
    plt.title(f"QFT Probability Distributions (qubit num={nqubit})")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    axis = plt.gca()
    for spine in axis.spines.values():
        spine.set_alpha(0.3)

    plt.tight_layout()
    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0)
    plt.show()


def main() -> None:
    nqubit = 10
    period = 17
    shift = 0
    save_path = Path("figs/fi_fig/qft_prob_dist.svg")
    prob1_dist, prob2_dist = calculate_distributions(nqubit, period, shift)
    plot_distributions(prob1_dist, prob2_dist, nqubit, save_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INPUT_SVG = Path("figs/fi_fig/fi_vs_nqubits.svg")
OUTPUT_PLOT = Path("figs/fi_fig/fi_log_regression_plot.pdf")
OUTPUT_SUMMARY = Path("figs/fi_fig/fi_log_regression_summary.json")
ASYMPTOTIC_QFT_SLOPE = 2.0 * math.log(2.0)
MATPLOTLIB_MARKERS = {
    "*": "o",
    "square*": "s",
    "triangle*": "^",
    "diamond*": "D",
    "pentagon*": "p",
}


@dataclass(frozen=True)
class DataPoint:
    nqubit: int
    fi_value: float


@dataclass(frozen=True)
class RegressionResult:
    label: str
    color: str
    marker: str
    npoints: int
    slope: float
    intercept: float
    r_squared: float
    slope_standard_error: float
    slope_ci_low: float
    slope_ci_high: float
    p_value: float
    rmse_log: float
    data: list[DataPoint]


SERIES_STYLE = {
    "#0081a7": ("ph1", "*"),
    "#1d3557": ("optimized 1ph", "square*"),
    "#00afb9": ("ph1_random", "triangle*"),
    "#fed9b7": ("HPrandom", "diamond*"),
    "#f07167": ("HPrandom_phase", "pentagon*"),
}
SERIES_ORDER = [
    "ph1",
    "optimized 1ph",
    "ph1_random",
    "HPrandom",
    "HPrandom_phase",
]

XTICK_PATTERN = re.compile(
    r'<use xlink:href="#m4be7968bac" x="([0-9.]+)" y="272\.12".*?<!--\s*([0-9]+)\s*-->',
    re.S,
)
AXIS_BOX_PATTERN = re.compile(
    r'<g id="patch_2">\s*<path d="M ([0-9.]+) ([0-9.]+)\s*L ([0-9.]+) ([0-9.]+)\s*L ([0-9.]+) ([0-9.]+)',
    re.S,
)
LINE_PATTERN = re.compile(
    r'<g id="line2d_(\d+)">\s*<path d="([^"]+)"[^>]*stroke: (#[0-9a-f]+);',
    re.S,
)
POINT_PATTERN = re.compile(r"[ML]\s*([0-9.]+)\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit log-linear models to the Fisher-information qubit-scaling figure.",
    )
    parser.add_argument("--input-svg", type=Path, default=INPUT_SVG)
    parser.add_argument("--output-plot", type=Path, default=OUTPUT_PLOT)
    parser.add_argument("--output-summary", type=Path, default=OUTPUT_SUMMARY)
    return parser.parse_args()


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    prefactor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    threshold = (a + 1.0) / (a + b + 2.0)
    if x < threshold:
        return prefactor * _betacf(a, b, x) / a
    return 1.0 - prefactor * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(value: float, df: int) -> float:
    x_value = df / (df + value * value)
    beta_value = _regularized_incomplete_beta(df / 2.0, 0.5, x_value)
    if value >= 0.0:
        return 1.0 - 0.5 * beta_value
    return 0.5 * beta_value


def student_t_ppf(probability: float, df: int) -> float:
    lower = -20.0
    upper = 20.0

    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if student_t_cdf(midpoint, df) < probability:
            lower = midpoint
        else:
            upper = midpoint

    return (lower + upper) / 2.0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_axis_map(svg_text: str) -> tuple[tuple[float, int], tuple[float, int], float, float]:
    xticks = sorted(
        (float(x_value), int(tick))
        for x_value, tick in XTICK_PATTERN.findall(svg_text)
    )
    if len(xticks) < 2:
        raise ValueError("failed to locate x-axis ticks in the SVG")

    axis_box_match = AXIS_BOX_PATTERN.search(svg_text)
    if axis_box_match is None:
        raise ValueError("failed to locate the plot bounds in the SVG")

    _, y_bottom, _, _, _, y_top = axis_box_match.groups()
    return xticks[0], xticks[-1], float(y_bottom), float(y_top)


def map_point(
    x_value: float,
    y_value: float,
    x_start: tuple[float, int],
    x_end: tuple[float, int],
    y_bottom: float,
    y_top: float,
) -> DataPoint:
    x_coord_start, n_start = x_start
    x_coord_end, n_end = x_end
    nqubit = round(
        n_start
        + (x_value - x_coord_start) * (n_end - n_start) / (x_coord_end - x_coord_start)
    )
    fi_value = (y_bottom - y_value) * 50.0 / (y_bottom - y_top)
    return DataPoint(nqubit=nqubit, fi_value=fi_value)


def extract_series(svg_text: str) -> dict[str, tuple[str, list[DataPoint]]]:
    x_start, x_end, y_bottom, y_top = extract_axis_map(svg_text)
    extracted: dict[str, tuple[str, list[DataPoint]]] = {}

    for _, path_body, color in LINE_PATTERN.findall(svg_text):
        if color not in SERIES_STYLE:
            continue
        point_pairs = [
            map_point(float(x_value), float(y_value), x_start, x_end, y_bottom, y_top)
            for x_value, y_value in POINT_PATTERN.findall(path_body)
        ]
        if len(point_pairs) < 4:
            continue
        label, _ = SERIES_STYLE[color]
        extracted[label] = (color, point_pairs)

    missing = {label for label, _ in SERIES_STYLE.values()} - set(extracted)
    if missing:
        raise ValueError(f"failed to extract SVG series: {sorted(missing)}")

    return extracted


def fit_log_linear(label: str, color: str, data: list[DataPoint]) -> RegressionResult:
    x_values = [point.nqubit for point in data]
    y_values = [math.log(point.fi_value) for point in data]
    npoints = len(x_values)
    x_mean = sum(x_values) / npoints
    y_mean = sum(y_values) / npoints
    sxx = sum((x_value - x_mean) ** 2 for x_value in x_values)
    sxy = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x_value for x_value in x_values]
    residuals = [y_value - fit_value for y_value, fit_value in zip(y_values, fitted)]
    sse = sum(residual * residual for residual in residuals)
    sst = sum((y_value - y_mean) ** 2 for y_value in y_values)
    degrees_of_freedom = npoints - 2
    sigma_squared = sse / degrees_of_freedom
    slope_standard_error = math.sqrt(sigma_squared / sxx)
    t_value = slope / slope_standard_error
    p_value = 2.0 * (1.0 - student_t_cdf(abs(t_value), degrees_of_freedom))
    t_critical = student_t_ppf(0.975, degrees_of_freedom)
    slope_ci_low = slope - t_critical * slope_standard_error
    slope_ci_high = slope + t_critical * slope_standard_error
    marker = SERIES_STYLE.get(color, ("", "*"))[1]
    rmse_log = math.sqrt(sse / npoints)

    return RegressionResult(
        label=label,
        color=color,
        marker=marker,
        npoints=npoints,
        slope=slope,
        intercept=intercept,
        r_squared=1.0 - sse / sst,
        slope_standard_error=slope_standard_error,
        slope_ci_low=slope_ci_low,
        slope_ci_high=slope_ci_high,
        p_value=p_value,
        rmse_log=rmse_log,
        data=data,
    )


def build_qft_window_series(n_values: list[int]) -> list[DataPoint]:
    return [
        DataPoint(
            nqubit=nqubit,
            fi_value=(4.0 * math.pi * math.pi / 9.0)
            * (((1 << nqubit) // (nqubit * nqubit)) ** 2 - 1),
        )
        for nqubit in n_values
    ]


def extract_plot_range(results: list[RegressionResult]) -> tuple[list[int], float, float]:
    x_values = sorted({point.nqubit for result in results for point in result.data})
    y_values = [math.log(point.fi_value) for result in results for point in result.data]
    return x_values, min(y_values) - 0.15, max(y_values) + 0.15


def try_import_matplotlib() -> Any | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    plt.rcParams["font.family"] = "DejaVu Sans"
    return plt


def save_plot_matplotlib(results: list[RegressionResult], output_path: Path) -> bool:
    plt = try_import_matplotlib()
    if plt is None:
        return False

    x_values, y_min, y_max = extract_plot_range(results)
    figure, axis = plt.subplots(figsize=(4.3, 3.0))
    axis.set_xlabel(r"Number of qubits $n$")
    axis.set_ylabel(r"$\ln I_{\min}$")
    axis.set_xlim(min(x_values) - 0.2, max(x_values) + 0.2)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(x_values)
    axis.grid(True, which="both", linewidth=0.35, color="#d9d9d9")

    for result in results:
        point_x = [point.nqubit for point in result.data]
        point_y = [math.log(point.fi_value) for point in result.data]
        fit_x = [min(point_x), max(point_x)]
        fit_y = [result.intercept + result.slope * x_value for x_value in fit_x]
        marker = MATPLOTLIB_MARKERS[result.marker]
        axis.plot(
            fit_x,
            fit_y,
            color=result.color,
            linewidth=1.5,
        )
        axis.scatter(
            point_x,
            point_y,
            color=result.color,
            marker=marker,
            s=24,
            label=result.label,
            zorder=3,
        )

    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return True


def pick_tick_step(y_min: float, y_max: float, target_ticks: int = 6) -> float:
    span = y_max - y_min
    raw_step = span / max(target_ticks - 1, 1)
    scale = 10 ** math.floor(math.log10(raw_step))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        step = multiplier * scale
        if step >= raw_step:
            return step
    return 10.0 * scale


def build_ticks(y_min: float, y_max: float) -> list[float]:
    step = pick_tick_step(y_min, y_max)
    tick_start = math.floor(y_min / step) * step
    tick_stop = math.ceil(y_max / step) * step
    ticks: list[float] = []
    tick = tick_start
    while tick <= tick_stop + 1e-9:
        ticks.append(round(tick, 10))
        tick += step
    return ticks


def format_tick(value: float) -> str:
    rounded = round(value, 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    text = f"{rounded:.3f}".rstrip("0")
    return text.rstrip(".")


def parse_hex_color(color: str) -> tuple[float, float, float]:
    cleaned = color.removeprefix("#")
    return tuple(int(cleaned[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def draw_marker(context: Any, marker: str, x_value: float, y_value: float, size: float) -> None:
    if marker == "*":
        context.arc(x_value, y_value, size, 0.0, 2.0 * math.pi)
        context.fill()
        return
    if marker == "square*":
        context.rectangle(x_value - size, y_value - size, 2.0 * size, 2.0 * size)
        context.fill()
        return
    if marker == "triangle*":
        context.move_to(x_value, y_value - size * 1.15)
        context.line_to(x_value + size * 1.1, y_value + size * 0.85)
        context.line_to(x_value - size * 1.1, y_value + size * 0.85)
        context.close_path()
        context.fill()
        return
    if marker == "diamond*":
        context.move_to(x_value, y_value - size * 1.2)
        context.line_to(x_value + size * 1.2, y_value)
        context.line_to(x_value, y_value + size * 1.2)
        context.line_to(x_value - size * 1.2, y_value)
        context.close_path()
        context.fill()
        return
    if marker == "pentagon*":
        for index in range(5):
            angle = -math.pi / 2.0 + 2.0 * math.pi * index / 5.0
            x_coord = x_value + size * math.cos(angle)
            y_coord = y_value + size * math.sin(angle)
            if index == 0:
                context.move_to(x_coord, y_coord)
            else:
                context.line_to(x_coord, y_coord)
        context.close_path()
        context.fill()
        return
    raise ValueError(f"unsupported marker: {marker}")


def draw_text(context: Any, text: str, x_value: float, y_value: float, align: str = "left") -> None:
    extents = context.text_extents(text)
    if align == "center":
        x_draw = x_value - extents.width / 2.0 - extents.x_bearing
    elif align == "right":
        x_draw = x_value - extents.width - extents.x_bearing
    else:
        x_draw = x_value
    y_draw = y_value - extents.y_bearing
    context.move_to(x_draw, y_draw)
    context.show_text(text)


def save_plot_cairo(results: list[RegressionResult], output_path: Path) -> None:
    try:
        import cairo
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "rendering the regression figure requires either matplotlib or pycairo"
        ) from exc

    width = 320.0
    height = 218.0
    left = 56.0
    right = 16.0
    top = 18.0
    bottom = 34.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values, y_min, y_max = extract_plot_range(results)
    xmin = min(x_values) - 0.2
    xmax = max(x_values) + 0.2
    y_ticks = build_ticks(y_min, y_max)

    def x_to_page(value: float) -> float:
        return left + (value - xmin) * plot_width / (xmax - xmin)

    def y_to_page(value: float) -> float:
        return height - bottom - (value - y_min) * plot_height / (y_max - y_min)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface = cairo.PDFSurface(str(output_path), width, height)
    context = cairo.Context(surface)

    context.set_source_rgb(1.0, 1.0, 1.0)
    context.paint()

    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(9.0)

    for tick in y_ticks:
        y_coord = y_to_page(tick)
        context.set_source_rgb(0.85, 0.85, 0.85)
        context.set_line_width(0.6)
        context.move_to(left, y_coord)
        context.line_to(width - right, y_coord)
        context.stroke()
        context.set_source_rgb(0.15, 0.15, 0.15)
        draw_text(context, format_tick(tick), left - 8.0, y_coord, align="right")

    for tick in x_values:
        x_coord = x_to_page(tick)
        context.set_source_rgb(0.85, 0.85, 0.85)
        context.set_line_width(0.6)
        context.move_to(x_coord, top)
        context.line_to(x_coord, height - bottom)
        context.stroke()
        context.set_source_rgb(0.15, 0.15, 0.15)
        draw_text(context, str(tick), x_coord, height - bottom + 8.0, align="center")

    context.set_source_rgb(0.05, 0.05, 0.05)
    context.set_line_width(0.9)
    context.rectangle(left, top, plot_width, plot_height)
    context.stroke()

    for result in results:
        red, green, blue = parse_hex_color(result.color)
        context.set_source_rgb(red, green, blue)
        point_x = [point.nqubit for point in result.data]
        point_y = [math.log(point.fi_value) for point in result.data]
        fit_x = [min(point_x), max(point_x)]
        fit_y = [result.intercept + result.slope * x_value for x_value in fit_x]
        context.set_line_width(1.4)
        context.move_to(x_to_page(fit_x[0]), y_to_page(fit_y[0]))
        context.line_to(x_to_page(fit_x[1]), y_to_page(fit_y[1]))
        context.stroke()
        for x_value, y_value in zip(point_x, point_y):
            draw_marker(context, result.marker, x_to_page(x_value), y_to_page(y_value), 3.2)

    context.set_source_rgb(0.1, 0.1, 0.1)
    draw_text(context, "Number of qubits n", left + plot_width / 2.0, height - 5.0, align="center")

    context.save()
    context.translate(12.0, top + plot_height / 2.0)
    context.rotate(-math.pi / 2.0)
    draw_text(context, "ln I_min", 0.0, 0.0, align="center")
    context.restore()

    legend_x = left + 8.0
    legend_y = top + 12.0
    line_length = 11.0
    context.set_font_size(8.2)
    for index, result in enumerate(results):
        y_coord = legend_y + index * 13.0
        red, green, blue = parse_hex_color(result.color)
        context.set_source_rgb(red, green, blue)
        context.set_line_width(1.2)
        context.move_to(legend_x, y_coord)
        context.line_to(legend_x + line_length, y_coord)
        context.stroke()
        draw_marker(context, result.marker, legend_x + line_length / 2.0, y_coord, 2.7)
        context.set_source_rgb(0.1, 0.1, 0.1)
        draw_text(context, result.label, legend_x + line_length + 6.0, y_coord, align="left")

    context.show_page()
    surface.finish()


def save_plot(results: list[RegressionResult], output_path: Path) -> None:
    if save_plot_matplotlib(results, output_path):
        return
    save_plot_cairo(results, output_path)


def summary_payload(
    results: list[RegressionResult],
    qft_window_result: RegressionResult,
) -> dict[str, object]:
    return {
        "model": r"ln(I_min) = k n + b",
        "qft_asymptotic_slope": ASYMPTOTIC_QFT_SLOPE,
        "qft_window_matched": asdict(qft_window_result),
        "series": [asdict(result) for result in results],
    }


def main() -> None:
    args = parse_args()
    svg_text = read_text(args.input_svg)
    extracted = extract_series(svg_text)
    results = [
        fit_log_linear(label, extracted[label][0], extracted[label][1])
        for label in SERIES_ORDER
    ]
    qft_window_data = build_qft_window_series(
        sorted({point.nqubit for result in results for point in result.data})
    )
    qft_window_result = fit_log_linear("qft_window", "#000000", qft_window_data)

    save_plot(results, args.output_plot)
    args.output_summary.write_text(
        json.dumps(summary_payload(results, qft_window_result), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

// CUDA Monte Carlo estimator for the C=2 hard-tail HP-1 DFI lower bound.
//
// For uniform x, p=2^n P_r(x), q=2^n P_{r+1}(x), this computes
//
//     L_2(n,r) = E[(q-p)^2 1{p<2} / 2] <= I_r(n).
//
// No 2^n statevector is allocated.  One CUDA warp evaluates one sampled x;
// roots-of-unity point queries cost O(n (odd(r) + odd(r+1))).  The code uses
// the source HP-1 little-endian convention and supports n <= 300.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=native \
//     scripts/experiments/hp1_tail_dfi_mc_cuda.cu \
//     -o hp1_tail_dfi_mc_cuda
//
// A large n does not by itself guarantee a reliable Monte Carlo estimate:
// rare resonance outputs can collapse ESS.  LOW_ESS rows are excluded from
// the printed log-linear fit.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr int kMaxN = 300;
constexpr int kMaxOddQubits = kMaxN / 2;
constexpr int kMaxWords = (kMaxN + 63) / 64;

struct Args {
    int n_min = 20;
    int n_max = 300;
    int n_step = 10;
    std::vector<uint64_t> periods = {12};
    uint64_t samples = 1ULL << 20;
    uint64_t samples_per_n2 = 0;
    int batch_samples = 1 << 17;
    int max_odd_part = 4096;
    double min_fit_ess = 100.0;
    uint64_t seed = 20260806ULL;
    int device = 0;
    std::filesystem::path output = "data/hp1_tail_dfi_cuda/hard_tail.csv";
};

struct DevicePeriod {
    int two_power = 0;
    int odd_part = 1;
    int width = 0;
    double log2_prefactor = 0.0;
    const double2* roots = nullptr;
};

struct DeviceStats {
    double sum_y = 0.0;
    double sum_y_squared = 0.0;
    double sum_w = 0.0;
    double sum_tail_w = 0.0;
    double max_y = 0.0;
    unsigned long long active_count = 0;
};

struct WarpStats {
    double y = 0.0;
    double y_squared = 0.0;
    double w = 0.0;
    double tail_w = 0.0;
    double max_y = 0.0;
    unsigned long long active = 0;
};

struct ResultRow {
    int n = 0;
    uint64_t period = 0;
    uint64_t period_odd_part = 0;
    uint64_t next_period_odd_part = 0;
    uint64_t samples = 0;
    double active_fraction = 0.0;
    double tail_dfi_lower = 0.0;
    double standard_error = 0.0;
    double effective_sample_size = 0.0;
    double maximum_weight_fraction = 0.0;
    double energy_fraction = 0.0;
    double c_effective = 0.0;
    int reliable = 0;
    double seconds = 0.0;
};

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << context << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

std::vector<uint64_t> parse_periods(const std::string& raw) {
    std::vector<uint64_t> periods;
    for (const std::string& value : split(raw, ',')) {
        periods.push_back(std::stoull(value));
    }
    if (periods.empty()) {
        throw std::runtime_error("--periods must not be empty");
    }
    return periods;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto require_value = [&](const std::string& option) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + option);
            }
            return argv[++index];
        };

        if (key == "--n-min") {
            args.n_min = std::stoi(require_value(key));
        } else if (key == "--n-max") {
            args.n_max = std::stoi(require_value(key));
        } else if (key == "--n-step") {
            args.n_step = std::stoi(require_value(key));
        } else if (key == "--periods") {
            args.periods = parse_periods(require_value(key));
        } else if (key == "--samples") {
            args.samples = std::stoull(require_value(key));
        } else if (key == "--samples-per-n2") {
            args.samples_per_n2 = std::stoull(require_value(key));
        } else if (key == "--batch-samples") {
            args.batch_samples = std::stoi(require_value(key));
        } else if (key == "--max-odd-part") {
            args.max_odd_part = std::stoi(require_value(key));
        } else if (key == "--min-fit-ess") {
            args.min_fit_ess = std::stod(require_value(key));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value(key));
        } else if (key == "--device") {
            args.device = std::stoi(require_value(key));
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: hp1_tail_dfi_mc_cuda [--n-min 20] [--n-max 300] "
                << "[--n-step 10] [--periods 12,20] [--samples 1048576] "
                << "[--samples-per-n2 0] [--batch-samples 131072] "
                << "[--max-odd-part 4096] "
                << "[--min-fit-ess 100] [--seed 20260806] [--device 0] "
                << "[--output result.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (args.n_min < 2 || args.n_max > kMaxN || args.n_min > args.n_max || args.n_step < 1) {
        throw std::runtime_error("need 2 <= n-min <= n-max <= 300 and n-step >= 1");
    }
    if (args.samples == 0 || args.batch_samples < 1 || args.max_odd_part < 1 || args.min_fit_ess < 1.0) {
        throw std::runtime_error("samples, batch-samples, max-odd-part, and min-fit-ess must be positive");
    }
    for (uint64_t period : args.periods) {
        if (period < 1 || period == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("periods must lie in [1, 2^64-2]");
        }
    }
    return args;
}

int two_power(uint64_t value) {
    return __builtin_ctzll(value);
}

uint64_t odd_part(uint64_t value) {
    return value >> two_power(value);
}

long double u128_to_long_double(unsigned __int128 value) {
    const uint64_t lower = static_cast<uint64_t>(value);
    const uint64_t upper = static_cast<uint64_t>(value >> 64);
    return std::ldexp(static_cast<long double>(upper), 64)
        + static_cast<long double>(lower);
}

double log2_support_count(int n, uint64_t period) {
    if (n < 128) {
        const unsigned __int128 state_count = static_cast<unsigned __int128>(1) << n;
        const unsigned __int128 support_count =
            ((state_count - 1) / static_cast<unsigned __int128>(period)) + 1;
        return static_cast<double>(std::log2(u128_to_long_double(support_count)));
    }

    // The ceiling correction is below double precision here because period is
    // at most 64 bit while 2^n has n >= 128.
    return static_cast<double>(n) - std::log2(static_cast<double>(period));
}

struct PeriodAllocation {
    DevicePeriod device_spec;
    double2* device_roots = nullptr;

    PeriodAllocation() = default;
    PeriodAllocation(const PeriodAllocation&) = delete;
    PeriodAllocation& operator=(const PeriodAllocation&) = delete;

    PeriodAllocation(PeriodAllocation&& other) noexcept
        : device_spec(other.device_spec), device_roots(other.device_roots) {
        other.device_roots = nullptr;
        other.device_spec.roots = nullptr;
    }

    ~PeriodAllocation() {
        if (device_roots != nullptr) {
            cudaFree(device_roots);
        }
    }
};

PeriodAllocation build_period(int n, uint64_t period, int max_odd_part) {
    const int a = two_power(period);
    if (a > n) {
        throw std::runtime_error("period must not contain a power of two larger than 2^n");
    }
    const uint64_t odd = odd_part(period);
    if (odd > static_cast<uint64_t>(max_odd_part) || odd > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        std::ostringstream message;
        message << "period " << period << " has odd part " << odd
                << ", above --max-odd-part=" << max_odd_part;
        throw std::runtime_error(message.str());
    }

    const int u = static_cast<int>(odd);
    const int width = n - a;
    std::vector<double2> roots(static_cast<size_t>(width) * static_cast<size_t>(u));
    uint64_t residue = u == 1 ? 0ULL : 1ULL % odd;
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (int digit = 0; digit < width; ++digit) {
        for (int frequency = 0; frequency < u; ++frequency) {
            const unsigned __int128 product =
                static_cast<unsigned __int128>(static_cast<uint64_t>(frequency)) * residue;
            const uint64_t phase_index = u == 1 ? 0ULL : static_cast<uint64_t>(product % odd);
            const double angle = two_pi * static_cast<double>(phase_index) / static_cast<double>(odd);
            roots[static_cast<size_t>(digit) * u + frequency] = make_double2(std::cos(angle), std::sin(angle));
        }
        residue = u == 1 ? 0ULL : static_cast<uint64_t>((static_cast<unsigned __int128>(residue) * 2U) % odd);
    }

    PeriodAllocation allocation;
    allocation.device_spec.two_power = a;
    allocation.device_spec.odd_part = u;
    allocation.device_spec.width = width;
    allocation.device_spec.log2_prefactor =
        2.0 * static_cast<double>(width) - log2_support_count(n, period);
    const size_t bytes = roots.size() * sizeof(double2);
    check_cuda(cudaMalloc(&allocation.device_roots, bytes), "cudaMalloc roots");
    check_cuda(cudaMemcpy(allocation.device_roots, roots.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy roots");
    allocation.device_spec.roots = allocation.device_roots;
    return allocation;
}

__device__ uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ int output_bit(const uint64_t* words, int bit) {
    return static_cast<int>((words[bit >> 6] >> (bit & 63)) & 1ULL);
}

__device__ double warp_sum(double value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ double atomic_max_nonnegative(double* address, double value) {
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (__longlong_as_double(static_cast<long long>(old)) < value) {
        const unsigned long long assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
    return __longlong_as_double(static_cast<long long>(old));
}

__device__ double scaled_probability(
    const DevicePeriod& period,
    const uint64_t* words,
    const double2* odd_weights,
    int lane
) {
    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int frequency = lane; frequency < period.odd_part; frequency += kWarpSize) {
        double product_real = 1.0;
        double product_imag = 0.0;
        for (int digit = 0; digit < period.width; ++digit) {
            const int qubit = period.two_power + digit;
            double z_real = 0.0;
            double z_imag = 0.0;
            if ((qubit & 1) == 0) {
                z_real = output_bit(words, qubit) == 0 ? 1.0 : -1.0;
            } else {
                const double2 z = odd_weights[qubit >> 1];
                z_real = z.x;
                z_imag = z.y;
            }

            const double2 root = period.roots[static_cast<size_t>(digit) * period.odd_part + frequency];
            const double rotated_real = z_real * root.x - z_imag * root.y;
            const double rotated_imag = z_real * root.y + z_imag * root.x;
            const double factor_real = 0.5 * (1.0 + rotated_real);
            const double factor_imag = 0.5 * rotated_imag;
            const double next_real = product_real * factor_real - product_imag * factor_imag;
            product_imag = product_real * factor_imag + product_imag * factor_real;
            product_real = next_real;
        }
        sum_real += product_real;
        sum_imag += product_imag;
    }

    sum_real = warp_sum(sum_real);
    sum_imag = warp_sum(sum_imag);
    double scaled = 0.0;
    if (lane == 0) {
        const double inverse_odd = 1.0 / static_cast<double>(period.odd_part);
        sum_real *= inverse_odd;
        sum_imag *= inverse_odd;
        const double magnitude_squared = sum_real * sum_real + sum_imag * sum_imag;
        if (magnitude_squared > 0.0) {
            const double log2_scaled = period.log2_prefactor + std::log2(magnitude_squared);
            scaled = std::exp2(log2_scaled);
        }
    }
    return __shfl_sync(0xffffffffU, scaled, 0);
}

__global__ void hard_tail_kernel(
    int n,
    uint64_t sample_offset,
    int sample_count,
    uint64_t seed,
    DevicePeriod period,
    DevicePeriod next_period,
    DeviceStats* totals
) {
    __shared__ uint64_t shared_words[kWarpsPerBlock][kMaxWords];
    __shared__ double2 shared_odd_weights[kWarpsPerBlock][kMaxOddQubits];
    __shared__ WarpStats shared_stats[kWarpsPerBlock];

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int sample_in_grid = blockIdx.x * kWarpsPerBlock + warp;
    const bool valid = sample_in_grid < sample_count;
    const uint64_t sample = sample_offset + static_cast<uint64_t>(sample_in_grid);
    const int word_count = (n + 63) / 64;

    if (valid && lane < word_count) {
        const uint64_t base = seed ^ (sample * 0xd1b54a32d192ed03ULL);
        shared_words[warp][lane] = splitmix64(base ^ (static_cast<uint64_t>(lane) * 0x9e3779b97f4a7c15ULL));
    }
    __syncwarp();

    if (valid && lane == 0) {
        const int odd_count = n / 2;
        const int even_count = (n + 1) / 2;
        double left = 0.0;
        for (int odd_index = 0; odd_index < odd_count; ++odd_index) {
            left = 0.25 * left + 0.5 * static_cast<double>(output_bit(shared_words[warp], 2 * odd_index));
            shared_odd_weights[warp][odd_index].x = left;
        }

        double right = 0.0;
        for (int odd_index = odd_count - 1; odd_index >= 0; --odd_index) {
            const int next_even = odd_index + 1;
            right *= 0.25;
            if (next_even < even_count) {
                right += 0.5 * static_cast<double>(output_bit(shared_words[warp], 2 * next_even));
            }
            const double phase_over_pi = shared_odd_weights[warp][odd_index].x + right;
            double sine = 0.0;
            double cosine = 0.0;
            sincospi(phase_over_pi, &sine, &cosine);
            const double sign = output_bit(shared_words[warp], 2 * odd_index + 1) == 0 ? 1.0 : -1.0;
            shared_odd_weights[warp][odd_index] = make_double2(sign * cosine, sign * sine);
        }
    }
    __syncwarp();

    WarpStats result;
    if (valid) {
        const double p = scaled_probability(period, shared_words[warp], shared_odd_weights[warp], lane);
        const double q = scaled_probability(next_period, shared_words[warp], shared_odd_weights[warp], lane);
        if (lane == 0) {
            const double difference = q - p;
            const double weight = difference * difference;
            const bool active = p < 2.0;
            const double y = active ? 0.5 * weight : 0.0;
            result.y = y;
            result.y_squared = y * y;
            result.w = weight;
            result.tail_w = active ? weight : 0.0;
            result.max_y = y;
            result.active = active ? 1ULL : 0ULL;
            shared_stats[warp] = result;
        }
    } else if (lane == 0) {
        shared_stats[warp] = result;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        WarpStats block_total;
        for (int warp_index = 0; warp_index < kWarpsPerBlock; ++warp_index) {
            const WarpStats value = shared_stats[warp_index];
            block_total.y += value.y;
            block_total.y_squared += value.y_squared;
            block_total.w += value.w;
            block_total.tail_w += value.tail_w;
            block_total.max_y = fmax(block_total.max_y, value.max_y);
            block_total.active += value.active;
        }
        atomicAdd(&totals->sum_y, block_total.y);
        atomicAdd(&totals->sum_y_squared, block_total.y_squared);
        atomicAdd(&totals->sum_w, block_total.w);
        atomicAdd(&totals->sum_tail_w, block_total.tail_w);
        atomic_max_nonnegative(&totals->max_y, block_total.max_y);
        atomicAdd(&totals->active_count, block_total.active);
    }
}

uint64_t samples_for_n(const Args& args, int n) {
    if (args.samples_per_n2 == 0) {
        return args.samples;
    }
    return args.samples_per_n2 * static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
}

ResultRow run_case(const Args& args, int n, uint64_t period) {
    const uint64_t sample_count = samples_for_n(args, n);
    PeriodAllocation first = build_period(n, period, args.max_odd_part);
    PeriodAllocation second = build_period(n, period + 1, args.max_odd_part);

    DeviceStats* device_stats = nullptr;
    check_cuda(cudaMalloc(&device_stats, sizeof(DeviceStats)), "cudaMalloc stats");
    check_cuda(cudaMemset(device_stats, 0, sizeof(DeviceStats)), "cudaMemset stats");

    const auto started = std::chrono::steady_clock::now();
    for (uint64_t offset = 0; offset < sample_count;) {
        const int batch = static_cast<int>(std::min<uint64_t>(args.batch_samples, sample_count - offset));
        const int blocks = (batch + kWarpsPerBlock - 1) / kWarpsPerBlock;
        hard_tail_kernel<<<blocks, kBlockSize>>>(
            n,
            offset,
            batch,
            args.seed + period * 1000003ULL + static_cast<uint64_t>(n) * 10007ULL,
            first.device_spec,
            second.device_spec,
            device_stats
        );
        check_cuda(cudaGetLastError(), "hard_tail_kernel launch");
        offset += static_cast<uint64_t>(batch);
    }
    check_cuda(cudaDeviceSynchronize(), "hard_tail_kernel synchronize");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    DeviceStats stats;
    check_cuda(cudaMemcpy(&stats, device_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost), "cudaMemcpy stats");
    check_cuda(cudaFree(device_stats), "cudaFree stats");

    const double count = static_cast<double>(sample_count);
    const double mean = stats.sum_y / count;
    const double second_moment = stats.sum_y_squared / count;
    const double variance = std::max(0.0, second_moment - mean * mean);
    const double active_fraction = static_cast<double>(stats.active_count) / count;
    const double energy_fraction = stats.sum_w > 0.0 ? stats.sum_tail_w / stats.sum_w : 0.0;
    const double effective_sample_size = stats.sum_y_squared > 0.0
        ? stats.sum_y * stats.sum_y / stats.sum_y_squared
        : 0.0;

    const double standard_error = std::sqrt(variance / count);
    const double maximum_weight_fraction = stats.sum_y > 0.0 ? stats.max_y / stats.sum_y : 0.0;
    const int reliable =
        effective_sample_size >= args.min_fit_ess
        && maximum_weight_fraction <= 0.1
        && mean > 0.0
        && standard_error / mean <= 0.25;

    return ResultRow{
        n,
        period,
        odd_part(period),
        odd_part(period + 1),
        sample_count,
        active_fraction,
        mean,
        standard_error,
        effective_sample_size,
        maximum_weight_fraction,
        energy_fraction,
        active_fraction > 0.0 ? energy_fraction / active_fraction : 0.0,
        reliable,
        seconds,
    };
}

void print_row(const ResultRow& row) {
    std::cout
        << std::setprecision(10)
        << "n=" << row.n
        << " r=" << row.period
        << " odd=(" << row.period_odd_part << ',' << row.next_period_odd_part << ')'
        << " M=" << row.samples
        << " L2=" << row.tail_dfi_lower
        << " se=" << row.standard_error
        << " ESS=" << row.effective_sample_size
        << " max_share=" << row.maximum_weight_fraction
        << " active=" << row.active_fraction
        << " c_eff=" << row.c_effective
        << " status=" << (row.reliable != 0 ? "OK" : "LOW_ESS")
        << " time=" << row.seconds << "s\n";
}

void write_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path.string());
    }
    output
        << "n,period,period_odd_part,next_period_odd_part,samples,active_fraction,"
        << "tail_dfi_lower,standard_error,effective_sample_size,maximum_weight_fraction,"
        << "energy_fraction,c_effective,reliable,seconds\n";
    output << std::setprecision(17);
    for (const ResultRow& row : rows) {
        output
            << row.n << ',' << row.period << ',' << row.period_odd_part << ','
            << row.next_period_odd_part << ',' << row.samples << ','
            << row.active_fraction << ',' << row.tail_dfi_lower << ','
            << row.standard_error << ',' << row.effective_sample_size << ','
            << row.maximum_weight_fraction << ',' << row.energy_fraction << ','
            << row.c_effective << ',' << row.reliable << ',' << row.seconds << '\n';
    }
}

void print_fits(const std::vector<ResultRow>& rows, const std::vector<uint64_t>& periods) {
    for (uint64_t period : periods) {
        std::vector<const ResultRow*> selected;
        for (const ResultRow& row : rows) {
            if (
                row.period == period
                && row.reliable != 0
                && row.tail_dfi_lower > 0.0
                && std::isfinite(row.tail_dfi_lower)
            ) {
                selected.push_back(&row);
            }
        }
        if (selected.size() < 2) {
            continue;
        }

        double mean_n = 0.0;
        double mean_log = 0.0;
        for (const ResultRow* row : selected) {
            mean_n += row->n;
            mean_log += std::log(row->tail_dfi_lower);
        }
        mean_n /= static_cast<double>(selected.size());
        mean_log /= static_cast<double>(selected.size());

        double covariance = 0.0;
        double variance_n = 0.0;
        double total_log_variance = 0.0;
        for (const ResultRow* row : selected) {
            const double centered_n = row->n - mean_n;
            const double centered_log = std::log(row->tail_dfi_lower) - mean_log;
            covariance += centered_n * centered_log;
            variance_n += centered_n * centered_n;
            total_log_variance += centered_log * centered_log;
        }
        const double slope = covariance / variance_n;
        const double intercept = mean_log - slope * mean_n;
        double residual = 0.0;
        for (const ResultRow* row : selected) {
            const double error = std::log(row->tail_dfi_lower) - (slope * row->n + intercept);
            residual += error * error;
        }
        const double r_squared = total_log_variance > 0.0 ? 1.0 - residual / total_log_variance : 1.0;
        std::cout
            << "fit r=" << period
            << " log(L2)=" << slope << "*n+" << intercept
            << " R2=" << r_squared << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        check_cuda(cudaSetDevice(args.device), "cudaSetDevice");
        check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, 8192), "cudaDeviceSetLimit stack");

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, args.device), "cudaGetDeviceProperties");
        std::cout
            << "device=" << properties.name
            << " n=" << args.n_min << ':' << args.n_max << ':' << args.n_step
            << " samples=";
        if (args.samples_per_n2 == 0) {
            std::cout << args.samples;
        } else {
            std::cout << args.samples_per_n2 << "*n^2";
        }
        std::cout << " periods=";
        for (size_t index = 0; index < args.periods.size(); ++index) {
            std::cout << (index == 0 ? "" : ",") << args.periods[index];
        }
        std::cout << '\n';

        std::vector<ResultRow> rows;
        for (int n = args.n_min; n <= args.n_max; n += args.n_step) {
            for (uint64_t period : args.periods) {
                const ResultRow row = run_case(args, n, period);
                rows.push_back(row);
                print_row(row);
            }
        }
        write_csv(args.output, rows);
        print_fits(rows, args.periods);
        std::cout << "output=" << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

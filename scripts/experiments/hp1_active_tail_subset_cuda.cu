// Four-GPU rare-event subset simulation for an active small-denominator HP-1 set.
//
// Write N=2^n, p=N P_r(x), q=N P_{r+1}(x), and fix tau>0.  This program
// estimates the uniform fraction of
//
//   A_tau(n,r) = {x : p(x)<2 and (q(x)-p(x))^2 r^2 >= tau N}.
//
// Every point in this set gives a deterministic DFI contribution
//
//   (P_{r+1}(x)-P_r(x))^2 / P_r(x) >= tau/(2 r^2),
//
// hence I_r >= tau |A_tau|/(2 r^2).  Direct uniform Monte Carlo cannot see
// A_tau at large n.  We instead use subset simulation: repeatedly retain an
// upper score quantile, resample, and apply a symmetric bit-flip Markov chain
// that leaves the uniform law conditioned on the current score level invariant.
// Independent replicates are required because the remaining approximation is
// MCMC mixing, not point-forward evaluation.
//
// Point probabilities use the exact roots-of-unity filter in complex128-like
// CUDA double arithmetic.  Its cost is O(n(odd(r)+odd(r+1))) per state, so this
// executable is intended for representative periods with bounded odd parts.
//
// Independent (n, period, replicate) jobs are dynamically scheduled across
// four GPUs by default.  Within each GPU, particles are processed by 16- or
// 32-thread cooperative tiles, GPU-side sorting/compaction keeps subset levels
// off the CPU, and all large buffers remain device resident.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=native \
//     scripts/experiments/hp1_active_tail_subset_cuda.cu \
//     -o hp1_active_tail_subset_cuda
//
// Figure 5(b), fixed-r series:
//   ./hp1_active_tail_subset_cuda --n-min 20 --n-max 200 --n-step 10 \
//     --periods 12 --replicates 1 --devices 0,1,2,3 \
//     --output data/hp1_active_tail_subset_cuda/r12_n20_200_step10_single.csv
//
// Figure 5(b), moving-window-edge series (one task per explicit n:r pair):
//   ./hp1_active_tail_subset_cuda \
//     --pairs 20:30,24:60,28:124,32:250,36:500,40:1000,44:2000,48:4000,52:8000,56:16000,60:32000 \
//     --max-odd-part 40000 --replicates 1 --devices 0,1,2,3 \
//     --output data/hp1_active_tail_subset_cuda/near_window_edge_n20_60_single.csv

#include <cuda_runtime.h>
#include <math_constants.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int kBlockSize = 128;
constexpr int kWarpSize = 32;
constexpr int kMaxN = 300;
constexpr int kMaxOddQubits = kMaxN / 2;
constexpr int kMaxWords = (kMaxN + 63) / 64;
constexpr int kMinimumTileSize = 16;
constexpr size_t kAllocatorSafetyBytes = 64ULL << 20;

struct Point {
    int n = 0;
    uint64_t period = 0;
};

struct Args {
    int n_min = 20;
    int n_max = 200;
    int n_step = 10;
    std::vector<uint64_t> periods = {12};
    double tau = 3.0e-4;
    double tau_decay = 0.0;
    int particles = 8192;
    double retain_fraction = 0.1;
    int mutation_steps = 64;
    int maximum_flips = 3;
    int maximum_levels = 100;
    int replicates = 4;
    int max_odd_part = 4096;
    uint64_t seed = 20260806ULL;
    std::vector<int> devices = {0, 1, 2, 3};
    std::vector<Point> points;
    double memory_fraction = 0.85;
    bool dry_run = false;
    std::filesystem::path output = "data/hp1_active_tail_subset_cuda/active_tail_4gpu.csv";
};

struct DevicePeriod {
    int two_power = 0;
    int odd_part = 1;
    int width = 0;
    double log2_prefactor = 0.0;
    const double2* roots = nullptr;
};

struct MutationStats {
    unsigned long long accepted = 0;
    unsigned long long proposed = 0;
};

struct ResultRow {
    int n = 0;
    uint64_t period = 0;
    int replicate = 0;
    double tau = 0.0;
    double tau_decay = 0.0;
    double effective_tau = 0.0;
    int particles = 0;
    double retain_fraction = 0.0;
    int mutation_steps = 0;
    int levels = 0;
    double log_active_fraction = 0.0;
    double active_fraction = 0.0;
    double final_conditional_fraction = 0.0;
    double minimum_acceptance = 0.0;
    double mean_acceptance = 0.0;
    int final_distinct_states = 0;
    double log_dfi_count_bound = 0.0;
    double seconds = 0.0;
    int device = -1;
    uint64_t estimated_device_bytes = 0;
};

struct Task {
    Point point;
    int replicate = 0;
    size_t ordinal = 0;
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

std::vector<int> parse_devices(const std::string& raw) {
    std::vector<int> devices;
    for (const std::string& value : split(raw, ',')) {
        devices.push_back(std::stoi(value));
    }
    if (devices.empty()) {
        throw std::runtime_error("--devices must not be empty");
    }
    return devices;
}

std::vector<Point> parse_points(const std::string& raw) {
    std::vector<Point> points;
    for (const std::string& item : split(raw, ',')) {
        const size_t separator = item.find(':');
        if (separator == std::string::npos || item.find(':', separator + 1) != std::string::npos) {
            throw std::runtime_error("--pairs entries must have the form n:period");
        }
        points.push_back(Point{
            std::stoi(item.substr(0, separator)),
            std::stoull(item.substr(separator + 1)),
        });
    }
    if (points.empty()) {
        throw std::runtime_error("--pairs must not be empty");
    }
    return points;
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
        } else if (key == "--tau") {
            args.tau = std::stod(require_value(key));
        } else if (key == "--tau-decay") {
            args.tau_decay = std::stod(require_value(key));
        } else if (key == "--particles") {
            args.particles = std::stoi(require_value(key));
        } else if (key == "--retain-fraction") {
            args.retain_fraction = std::stod(require_value(key));
        } else if (key == "--mutation-steps") {
            args.mutation_steps = std::stoi(require_value(key));
        } else if (key == "--maximum-flips") {
            args.maximum_flips = std::stoi(require_value(key));
        } else if (key == "--maximum-levels") {
            args.maximum_levels = std::stoi(require_value(key));
        } else if (key == "--replicates") {
            args.replicates = std::stoi(require_value(key));
        } else if (key == "--max-odd-part") {
            args.max_odd_part = std::stoi(require_value(key));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value(key));
        } else if (key == "--devices") {
            args.devices = parse_devices(require_value(key));
        } else if (key == "--device") {
            args.devices.clear();
            args.devices.push_back(std::stoi(require_value(key)));
        } else if (key == "--pairs") {
            args.points = parse_points(require_value(key));
        } else if (key == "--memory-fraction") {
            args.memory_fraction = std::stod(require_value(key));
        } else if (key == "--dry-run") {
            args.dry_run = true;
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: hp1_active_tail_subset_cuda [--n-min 20] [--n-max 200] "
                << "[--n-step 10] [--periods 12] [--tau 3e-4] [--tau-decay 0] "
                << "[--particles 8192] [--retain-fraction 0.1] "
                << "[--mutation-steps 64] [--maximum-flips 3] "
                << "[--maximum-levels 100] [--replicates 4] "
                << "[--max-odd-part 4096] [--seed 20260806] "
                << "[--devices 0,1,2,3] [--pairs 20:30,24:60] "
                << "[--memory-fraction 0.85] [--dry-run] [--output result.csv]\n"
                << "--pairs overrides the Cartesian n-grid/period list.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (args.n_min < 2 || args.n_max > kMaxN || args.n_min > args.n_max || args.n_step < 1) {
        throw std::runtime_error("need 2 <= n-min <= n-max <= 300 and n-step >= 1");
    }
    if (
        args.tau <= 0.0
        || args.tau_decay < 0.0
        || args.particles < 100
        || args.mutation_steps < 1
        || args.maximum_flips < 1
    ) {
        throw std::runtime_error("tau, particles, mutation-steps, and maximum-flips must be positive; tau-decay must be nonnegative");
    }
    if (!(args.retain_fraction > 0.0 && args.retain_fraction < 1.0)) {
        throw std::runtime_error("retain-fraction must lie in (0,1)");
    }
    if (args.maximum_levels < 1 || args.replicates < 1 || args.max_odd_part < 1) {
        throw std::runtime_error("maximum-levels, replicates, and max-odd-part must be positive");
    }
    for (uint64_t period : args.periods) {
        if (period < 2 || period == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("periods must lie in [2,2^64-2]");
        }
    }
    for (const Point& point : args.points) {
        if (point.n < 2 || point.n > kMaxN) {
            throw std::runtime_error("--pairs qubit counts must lie in [2,300]");
        }
        if (point.period < 2 || point.period == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("--pairs periods must lie in [2,2^64-2]");
        }
    }
    if (!(args.memory_fraction > 0.0 && args.memory_fraction <= 1.0)) {
        throw std::runtime_error("--memory-fraction must lie in (0,1]");
    }
    const std::set<int> unique_devices(args.devices.begin(), args.devices.end());
    if (unique_devices.size() != args.devices.size()) {
        throw std::runtime_error("--devices must not contain duplicates");
    }
    if (unique_devices.empty() || *unique_devices.begin() < 0) {
        throw std::runtime_error("GPU device indices must be nonnegative");
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
        throw std::runtime_error("period contains a power of two larger than 2^n");
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

template <int TileSize>
__device__ unsigned int tile_mask() {
    if constexpr (TileSize == kWarpSize) {
        return 0xffffffffU;
    } else {
        const int warp_lane = threadIdx.x & (kWarpSize - 1);
        const int tile_base = (warp_lane / TileSize) * TileSize;
        return ((1U << TileSize) - 1U) << tile_base;
    }
}

template <int TileSize>
__device__ double tile_sum(double value, unsigned int mask) {
    #pragma unroll
    for (int offset = TileSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset, TileSize);
    }
    return value;
}

template <int TileSize>
__device__ double tile_max(double value, unsigned int mask) {
    #pragma unroll
    for (int offset = TileSize / 2; offset > 0; offset >>= 1) {
        value = fmax(value, __shfl_down_sync(mask, value, offset, TileSize));
    }
    return value;
}

__device__ double quarter_power(int exponent) {
    return ldexp(1.0, -2 * exponent);
}

template <int TileSize>
__device__ void build_odd_weights(
    int n,
    const uint64_t* words,
    double2* odd_weights,
    int lane,
    unsigned int mask
) {
    // The original implementation constructed both recurrences and every
    // sincospi value on lane 0.  Here each TileSize-wide tile performs a
    // chunked affine prefix scan, then evaluates odd-qubit phases in parallel.
    const int odd_count = n / 2;
    const int even_count = (n + 1) / 2;

    double carry = 0.0;
    for (int base = 0; base < odd_count; base += TileSize) {
        const int odd_index = base + lane;
        double scan = odd_index < odd_count
            ? 0.5 * static_cast<double>(output_bit(words, 2 * odd_index))
            : 0.0;
        #pragma unroll
        for (int offset = 1; offset < TileSize; offset <<= 1) {
            const double previous = __shfl_up_sync(mask, scan, offset, TileSize);
            if (lane >= offset) {
                scan += quarter_power(offset) * previous;
            }
        }
        scan += quarter_power(lane + 1) * carry;
        if (odd_index < odd_count) {
            odd_weights[odd_index].x = scan;
        }
        const int valid_lanes = min(TileSize, odd_count - base);
        carry = __shfl_sync(mask, scan, valid_lanes - 1, TileSize);
    }

    carry = 0.0;
    for (int base = 0; base < odd_count; base += TileSize) {
        const int reverse_index = base + lane;
        const int odd_index = odd_count - 1 - reverse_index;
        const int next_even = odd_index + 1;
        double scan = reverse_index < odd_count && next_even < even_count
            ? 0.5 * static_cast<double>(output_bit(words, 2 * next_even))
            : 0.0;
        #pragma unroll
        for (int offset = 1; offset < TileSize; offset <<= 1) {
            const double previous = __shfl_up_sync(mask, scan, offset, TileSize);
            if (lane >= offset) {
                scan += quarter_power(offset) * previous;
            }
        }
        scan += quarter_power(lane + 1) * carry;
        if (reverse_index < odd_count) {
            odd_weights[odd_index].y = scan;
        }
        const int valid_lanes = min(TileSize, odd_count - base);
        carry = __shfl_sync(mask, scan, valid_lanes - 1, TileSize);
    }
    __syncwarp(mask);

    for (int odd_index = lane; odd_index < odd_count; odd_index += TileSize) {
        const double phase_over_pi = odd_weights[odd_index].x + odd_weights[odd_index].y;
        double sine = 0.0;
        double cosine = 0.0;
        sincospi(phase_over_pi, &sine, &cosine);
        const double sign = output_bit(words, 2 * odd_index + 1) == 0 ? 1.0 : -1.0;
        odd_weights[odd_index] = make_double2(sign * cosine, sign * sine);
    }
    __syncwarp(mask);
}

template <int TileSize>
__device__ double log2_scaled_probability(
    const DevicePeriod& period,
    const uint64_t* words,
    const double2* odd_weights,
    int lane,
    unsigned int mask
) {
    // Each frequency product can be exponentially small.  Periodic
    // renormalization preserves its phase and records its logarithmic scale;
    // a complex log-sum-exp then combines roots-of-unity frequencies without
    // the n≈100 underflow of a direct product.
    double local_scale = -CUDART_INF;
    double local_real = 0.0;
    double local_imag = 0.0;
    for (int frequency = lane; frequency < period.odd_part; frequency += TileSize) {
        double product_real = 1.0;
        double product_imag = 0.0;
        double log_scale = 0.0;
        bool nonzero = true;
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

            if ((digit & 15) == 15 || digit + 1 == period.width) {
                const double magnitude = hypot(product_real, product_imag);
                if (magnitude == 0.0) {
                    nonzero = false;
                    break;
                }
                product_real /= magnitude;
                product_imag /= magnitude;
                log_scale += log(magnitude);
            }
        }
        if (!nonzero) {
            continue;
        }
        if (log_scale > local_scale) {
            const double rescale = isfinite(local_scale) ? exp(local_scale - log_scale) : 0.0;
            local_real *= rescale;
            local_imag *= rescale;
            local_scale = log_scale;
        }
        const double relative = exp(log_scale - local_scale);
        local_real += relative * product_real;
        local_imag += relative * product_imag;
    }

    double global_scale = tile_max<TileSize>(local_scale, mask);
    global_scale = __shfl_sync(mask, global_scale, 0, TileSize);
    const double lane_rescale = isfinite(local_scale) ? exp(local_scale - global_scale) : 0.0;
    double sum_real = tile_sum<TileSize>(local_real * lane_rescale, mask);
    double sum_imag = tile_sum<TileSize>(local_imag * lane_rescale, mask);
    double result = -CUDART_INF;
    if (lane == 0 && isfinite(global_scale)) {
        const double sum_squared = sum_real * sum_real + sum_imag * sum_imag;
        if (sum_squared > 0.0) {
            constexpr double inverse_ln2 = 1.442695040888963407359924681002;
            result = period.log2_prefactor
                + (2.0 * global_scale + log(sum_squared)) * inverse_ln2
                - 2.0 * log2(static_cast<double>(period.odd_part));
        }
    }
    return __shfl_sync(mask, result, 0, TileSize);
}

template <int TileSize>
__device__ double active_score(
    int n,
    double log2_tau,
    double log2_period,
    const DevicePeriod& period,
    const DevicePeriod& next_period,
    const uint64_t* words,
    double2* odd_weights,
    int lane,
    unsigned int mask
) {
    build_odd_weights<TileSize>(n, words, odd_weights, lane, mask);
    const double log2_p = log2_scaled_probability<TileSize>(period, words, odd_weights, lane, mask);
    const double log2_q = log2_scaled_probability<TileSize>(next_period, words, odd_weights, lane, mask);
    double score = -CUDART_INF;
    if (lane == 0) {
        const double score_probability = 1.0 - log2_p;
        const double high = fmax(log2_p, log2_q);
        const double low = fmin(log2_p, log2_q);
        double log2_difference = -CUDART_INF;
        if (isfinite(high)) {
            const double ratio = isfinite(low) ? exp2(low - high) : 0.0;
            const double relative_difference = fabs(1.0 - ratio);
            if (relative_difference > 0.0) {
                log2_difference = high + log2(relative_difference);
            }
        }
        const double score_numerator = isfinite(log2_difference)
            ? 2.0 * log2_difference + 2.0 * log2_period - log2_tau - static_cast<double>(n)
            : -CUDART_INF;
        // Once p<2 is satisfied, rank states only by numerator strength.  A
        // literal min(score_probability, score_numerator) creates large score
        // plateaus near exact/symmetry-induced p values and can stall adaptive
        // subset levels, while this piecewise score has exactly the same
        // nonnegative event set.
        score = log2_p < 1.0 ? score_numerator : fmin(score_probability, score_numerator);
        if (score < 0.0 && isfinite(score)) {
            // Exact phase cancellations create atoms in the physical score.
            // A deterministic infinitesimal ordering keeps its sign (and thus
            // the active event) unchanged while letting subset simulation
            // traverse those atoms without repeatedly selecting one level.
            uint64_t hash = 0x243f6a8885a308d3ULL;
            const int word_count = (n + 63) / 64;
            for (int word = 0; word < word_count; ++word) {
                hash = splitmix64(hash ^ words[word]);
            }
            const double unit = static_cast<double>(hash >> 11) * 0x1.0p-53;
            score *= 1.0 + 1.0e-9 * unit;
        }
    }
    return __shfl_sync(mask, score, 0, TileSize);
}

template <int TileSize>
__global__ void initialize_kernel(
    int n,
    int particles,
    uint64_t seed,
    double log2_tau,
    double log2_period,
    DevicePeriod period,
    DevicePeriod next_period,
    uint64_t* states,
    double* scores
) {
    constexpr int tiles_per_block = kBlockSize / TileSize;
    __shared__ uint64_t shared_words[tiles_per_block][kMaxWords];
    __shared__ double2 shared_odd_weights[tiles_per_block][kMaxOddQubits];
    const int tile = threadIdx.x / TileSize;
    const int lane = threadIdx.x & (TileSize - 1);
    const unsigned int mask = tile_mask<TileSize>();
    const int particle = blockIdx.x * tiles_per_block + tile;
    if (particle >= particles) {
        return;
    }
    const int word_count = (n + 63) / 64;
    if (lane < word_count) {
        const uint64_t key = seed
            ^ (static_cast<uint64_t>(particle) * 0xd1b54a32d192ed03ULL)
            ^ (static_cast<uint64_t>(lane) * 0x9e3779b97f4a7c15ULL);
        uint64_t word = splitmix64(key);
        if (lane == word_count - 1 && (n & 63) != 0) {
            word &= (1ULL << (n & 63)) - 1ULL;
        }
        shared_words[tile][lane] = word;
        states[static_cast<size_t>(particle) * word_count + lane] = word;
    }
    __syncwarp(mask);
    const double score = active_score<TileSize>(
        n,
        log2_tau,
        log2_period,
        period,
        next_period,
        shared_words[tile],
        shared_odd_weights[tile],
        lane,
        mask
    );
    if (lane == 0) {
        scores[particle] = score;
    }
}

__global__ void resample_kernel(
    int word_count,
    int particles,
    int survivor_count,
    int level,
    uint64_t seed,
    const int* survivor_indices,
    const uint64_t* source_states,
    const double* source_scores,
    uint64_t* target_states,
    double* target_scores
) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particles) {
        return;
    }
    const uint64_t random = splitmix64(
        seed
        ^ (static_cast<uint64_t>(level) * 0x8cb92baa3f3d8dd7ULL)
        ^ (static_cast<uint64_t>(particle) * 0xd1b54a32d192ed03ULL)
    );
    const int parent = survivor_indices[random % static_cast<uint64_t>(survivor_count)];
    #pragma unroll
    for (int word = 0; word < word_count; ++word) {
        target_states[static_cast<size_t>(particle) * word_count + word] =
            source_states[static_cast<size_t>(parent) * word_count + word];
    }
    target_scores[particle] = source_scores[parent];
}

template <int TileSize>
__global__ void mutate_kernel(
    int n,
    int particles,
    int mutation_steps,
    int maximum_flips,
    int level,
    uint64_t seed,
    double threshold,
    double log2_tau,
    double log2_period,
    DevicePeriod period,
    DevicePeriod next_period,
    uint64_t* states,
    double* scores,
    MutationStats* stats
) {
    constexpr int tiles_per_block = kBlockSize / TileSize;
    __shared__ uint64_t shared_words[tiles_per_block][kMaxWords];
    __shared__ double2 shared_odd_weights[tiles_per_block][kMaxOddQubits];
    __shared__ int shared_bits[tiles_per_block][8];
    const int tile = threadIdx.x / TileSize;
    const int lane = threadIdx.x & (TileSize - 1);
    const unsigned int mask = tile_mask<TileSize>();
    const int particle = blockIdx.x * tiles_per_block + tile;
    if (particle >= particles) {
        return;
    }
    const int word_count = (n + 63) / 64;
    if (lane < word_count) {
        shared_words[tile][lane] = states[static_cast<size_t>(particle) * word_count + lane];
    }
    __syncwarp(mask);

    double current_score = scores[particle];
    unsigned long long accepted = 0ULL;
    for (int step = 0; step < mutation_steps; ++step) {
        int flip_count = 1;
        if (lane == 0) {
            const uint64_t base = seed
                ^ (static_cast<uint64_t>(level) * 0x8cb92baa3f3d8dd7ULL)
                ^ (static_cast<uint64_t>(particle) * 0xd1b54a32d192ed03ULL)
                ^ (static_cast<uint64_t>(step) * 0x9e3779b97f4a7c15ULL);
            flip_count = 1 + static_cast<int>(splitmix64(base) % static_cast<uint64_t>(maximum_flips));
            for (int index = 0; index < flip_count; ++index) {
                const int bit = static_cast<int>(splitmix64(base + static_cast<uint64_t>(index + 1)) % static_cast<uint64_t>(n));
                shared_bits[tile][index] = bit;
                shared_words[tile][bit >> 6] ^= 1ULL << (bit & 63);
            }
            shared_bits[tile][7] = flip_count;
        }
        __syncwarp(mask);
        flip_count = shared_bits[tile][7];
        const double candidate_score = active_score<TileSize>(
            n,
            log2_tau,
            log2_period,
            period,
            next_period,
            shared_words[tile],
            shared_odd_weights[tile],
            lane,
            mask
        );
        if (lane == 0) {
            if (candidate_score >= threshold) {
                current_score = candidate_score;
                ++accepted;
            } else {
                for (int index = 0; index < flip_count; ++index) {
                    const int bit = shared_bits[tile][index];
                    shared_words[tile][bit >> 6] ^= 1ULL << (bit & 63);
                }
            }
        }
        __syncwarp(mask);
    }

    if (lane < word_count) {
        states[static_cast<size_t>(particle) * word_count + lane] = shared_words[tile][lane];
    }
    if (lane == 0) {
        scores[particle] = current_score;
        atomicAdd(&stats->accepted, accepted);
        atomicAdd(&stats->proposed, static_cast<unsigned long long>(mutation_steps));
    }
}

__global__ void hash_states_kernel(
    const uint64_t* states,
    int word_count,
    int particles,
    uint64_t* hashes
) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particles) {
        return;
    }
    uint64_t hash = 0x243f6a8885a308d3ULL;
    #pragma unroll
    for (int word = 0; word < word_count; ++word) {
        const uint64_t value = splitmix64(
            states[static_cast<size_t>(particle) * word_count + word]
        );
        hash ^= value + (hash << 6) + (hash >> 2);
    }
    hashes[particle] = hash;
}

struct NonnegativeScore {
    __host__ __device__ bool operator()(double score) const {
        return score >= 0.0;
    }
};

struct ScoreAtLeast {
    double threshold;

    __host__ __device__ bool operator()(double score) const {
        return score >= threshold;
    }
};

int distinct_state_count_gpu(
    const uint64_t* states,
    int particles,
    int word_count,
    uint64_t* hashes,
    cudaStream_t stream
) {
    const int blocks = (particles + kBlockSize - 1) / kBlockSize;
    hash_states_kernel<<<blocks, kBlockSize, 0, stream>>>(
        states,
        word_count,
        particles,
        hashes
    );
    check_cuda(cudaGetLastError(), "hash_states_kernel launch");
    auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<uint64_t> begin(hashes);
    thrust::sort(policy, begin, begin + particles);
    const auto unique_end = thrust::unique(policy, begin, begin + particles);
    check_cuda(cudaStreamSynchronize(stream), "distinct-state synchronize");
    return static_cast<int>(unique_end - begin);
}

size_t root_bytes_for(int n, uint64_t period) {
    return static_cast<size_t>(n - two_power(period))
        * static_cast<size_t>(odd_part(period))
        * sizeof(double2);
}

size_t estimated_task_bytes(const Args& args, int n, uint64_t period) {
    const size_t particles = static_cast<size_t>(args.particles);
    const size_t word_count = static_cast<size_t>((n + 63) / 64);
    const size_t states = 2 * particles * word_count * sizeof(uint64_t);
    const size_t scores = 2 * particles * sizeof(double);
    const size_t survivors = particles * sizeof(int);
    const size_t hashes = particles * sizeof(uint64_t);
    // Thrust/CUB temporary storage varies by toolkit.  This conservative term
    // covers sorting, compaction and unique buffers without claiming it is an
    // exact allocator trace.
    const size_t thrust_temporary = 8 * particles * (
        sizeof(double) + sizeof(int) + sizeof(uint64_t)
    );
    const size_t roots = root_bytes_for(n, period) + root_bytes_for(n, period + 1);
    return states + scores + survivors + hashes + sizeof(MutationStats)
        + thrust_temporary + roots + kAllocatorSafetyBytes;
}

struct DeviceWorkspace {
    uint64_t* states_a = nullptr;
    uint64_t* states_b = nullptr;
    double* scores_a = nullptr;
    double* scores_b = nullptr;
    int* survivor_indices = nullptr;
    uint64_t* hashes = nullptr;
    MutationStats* mutation_stats = nullptr;
    cudaStream_t stream = nullptr;

    DeviceWorkspace() = default;
    DeviceWorkspace(const DeviceWorkspace&) = delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;

    ~DeviceWorkspace() {
        cudaFree(states_a);
        cudaFree(states_b);
        cudaFree(scores_a);
        cudaFree(scores_b);
        cudaFree(survivor_indices);
        cudaFree(hashes);
        cudaFree(mutation_stats);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

void launch_initialize(
    int tile_size,
    int n,
    int particles,
    uint64_t seed,
    double log2_tau,
    double log2_period,
    DevicePeriod period,
    DevicePeriod next_period,
    uint64_t* states,
    double* scores,
    cudaStream_t stream
) {
    const int tiles_per_block = kBlockSize / tile_size;
    const int blocks = (particles + tiles_per_block - 1) / tiles_per_block;
    if (tile_size == kMinimumTileSize) {
        initialize_kernel<kMinimumTileSize><<<blocks, kBlockSize, 0, stream>>>(
            n, particles, seed, log2_tau, log2_period,
            period, next_period, states, scores
        );
    } else {
        initialize_kernel<kWarpSize><<<blocks, kBlockSize, 0, stream>>>(
            n, particles, seed, log2_tau, log2_period,
            period, next_period, states, scores
        );
    }
    check_cuda(cudaGetLastError(), "initialize_kernel launch");
}

void launch_mutation(
    int tile_size,
    int n,
    int particles,
    int mutation_steps,
    int maximum_flips,
    int level,
    uint64_t seed,
    double threshold,
    double log2_tau,
    double log2_period,
    DevicePeriod period,
    DevicePeriod next_period,
    uint64_t* states,
    double* scores,
    MutationStats* stats,
    cudaStream_t stream
) {
    const int tiles_per_block = kBlockSize / tile_size;
    const int blocks = (particles + tiles_per_block - 1) / tiles_per_block;
    if (tile_size == kMinimumTileSize) {
        mutate_kernel<kMinimumTileSize><<<blocks, kBlockSize, 0, stream>>>(
            n, particles, mutation_steps, maximum_flips, level, seed,
            threshold, log2_tau, log2_period, period, next_period,
            states, scores, stats
        );
    } else {
        mutate_kernel<kWarpSize><<<blocks, kBlockSize, 0, stream>>>(
            n, particles, mutation_steps, maximum_flips, level, seed,
            threshold, log2_tau, log2_period, period, next_period,
            states, scores, stats
        );
    }
    check_cuda(cudaGetLastError(), "mutate_kernel launch");
}

ResultRow run_replicate(
    const Args& args,
    int n,
    uint64_t period_value,
    int replicate,
    int device,
    const DevicePeriod& period,
    const DevicePeriod& next_period,
    size_t task_bytes
) {
    const int particles = args.particles;
    const int word_count = (n + 63) / 64;
    const size_t state_bytes = static_cast<size_t>(particles) * word_count * sizeof(uint64_t);
    const size_t score_bytes = static_cast<size_t>(particles) * sizeof(double);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t roots_bytes = root_bytes_for(n, period_value) + root_bytes_for(n, period_value + 1);
    const size_t remaining_estimate = task_bytes > roots_bytes ? task_bytes - roots_bytes : task_bytes;
    if (static_cast<double>(remaining_estimate) > args.memory_fraction * static_cast<double>(free_bytes)) {
        std::ostringstream message;
        message << "GPU " << device << " memory plan rejected: estimated remaining bytes="
                << remaining_estimate << " free=" << free_bytes
                << " fraction=" << args.memory_fraction;
        throw std::runtime_error(message.str());
    }

    DeviceWorkspace workspace;
    check_cuda(cudaStreamCreateWithFlags(&workspace.stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check_cuda(cudaMalloc(&workspace.states_a, state_bytes), "cudaMalloc states_a");
    check_cuda(cudaMalloc(&workspace.states_b, state_bytes), "cudaMalloc states_b");
    check_cuda(cudaMalloc(&workspace.scores_a, score_bytes), "cudaMalloc scores_a");
    check_cuda(cudaMalloc(&workspace.scores_b, score_bytes), "cudaMalloc scores_b/sort workspace");
    check_cuda(cudaMalloc(&workspace.survivor_indices, static_cast<size_t>(particles) * sizeof(int)), "cudaMalloc survivors");
    check_cuda(cudaMalloc(&workspace.hashes, static_cast<size_t>(particles) * sizeof(uint64_t)), "cudaMalloc hashes");
    check_cuda(cudaMalloc(&workspace.mutation_stats, sizeof(MutationStats)), "cudaMalloc mutation stats");

    uint64_t* states_a = workspace.states_a;
    uint64_t* states_b = workspace.states_b;
    double* scores_a = workspace.scores_a;
    double* scores_b = workspace.scores_b;
    const uint64_t replicate_seed = args.seed
        ^ (period_value * 1000003ULL)
        ^ (static_cast<uint64_t>(n) * 10007ULL)
        ^ (static_cast<uint64_t>(replicate) * 0xd1b54a32d192ed03ULL);
    const double effective_tau = args.tau * std::exp(-args.tau_decay * static_cast<double>(n));
    const double log2_tau = std::log2(effective_tau);
    const double log2_period = std::log2(static_cast<double>(period_value));
    const int maximum_odd_part = std::max(period.odd_part, next_period.odd_part);
    const int tile_size = maximum_odd_part <= kMinimumTileSize ? kMinimumTileSize : kWarpSize;
    const auto started = std::chrono::steady_clock::now();
    launch_initialize(
        tile_size, n, particles, replicate_seed, log2_tau, log2_period,
        period, next_period, states_a, scores_a, workspace.stream
    );

    std::vector<double> acceptances;
    double log_probability = 0.0;
    double final_fraction = 0.0;
    double previous_threshold = -std::numeric_limits<double>::infinity();
    int completed_levels = 0;
    auto policy = thrust::cuda::par.on(workspace.stream);
    const auto index_begin = thrust::make_counting_iterator<int>(0);

    for (int level = 0; level < args.maximum_levels; ++level) {
        thrust::device_ptr<double> scores_begin(scores_a);
        const int event_count = static_cast<int>(thrust::count_if(
            policy,
            scores_begin,
            scores_begin + particles,
            NonnegativeScore{}
        ));
        check_cuda(cudaMemcpyAsync(
            scores_b,
            scores_a,
            score_bytes,
            cudaMemcpyDeviceToDevice,
            workspace.stream
        ), "cudaMemcpyAsync score workspace");
        thrust::device_ptr<double> ordered_begin(scores_b);
        thrust::sort(
            policy,
            ordered_begin,
            ordered_begin + particles,
            thrust::greater<double>()
        );
        const int requested = std::max(
            1,
            static_cast<int>(std::ceil(args.retain_fraction * particles))
        );
        double next_threshold = 0.0;
        check_cuda(cudaMemcpyAsync(
            &next_threshold,
            scores_b + requested - 1,
            sizeof(double),
            cudaMemcpyDeviceToHost,
            workspace.stream
        ), "cudaMemcpyAsync threshold");
        check_cuda(cudaStreamSynchronize(workspace.stream), "threshold synchronize");
        if (next_threshold >= 0.0) {
            final_fraction = static_cast<double>(event_count) / static_cast<double>(particles);
            if (final_fraction <= 0.0) {
                throw std::runtime_error("nonpositive final conditional fraction");
            }
            log_probability += std::log(final_fraction);
            break;
        }
        if (!(next_threshold > previous_threshold)) {
            next_threshold = std::nextafter(
                previous_threshold,
                std::numeric_limits<double>::infinity()
            );
        }

        thrust::device_ptr<int> survivor_begin(workspace.survivor_indices);
        const auto survivor_end = thrust::copy_if(
            policy,
            index_begin,
            index_begin + particles,
            scores_begin,
            survivor_begin,
            ScoreAtLeast{next_threshold}
        );
        const int survivor_count = static_cast<int>(survivor_end - survivor_begin);
        check_cuda(cudaStreamSynchronize(workspace.stream), "survivor compaction synchronize");
        if (survivor_count < 1) {
            throw std::runtime_error("subset score level has no survivors");
        }
        const double conditional_fraction =
            static_cast<double>(survivor_count) / static_cast<double>(particles);
        log_probability += std::log(conditional_fraction);

        const int copy_blocks = (particles + kBlockSize - 1) / kBlockSize;
        resample_kernel<<<copy_blocks, kBlockSize, 0, workspace.stream>>>(
            word_count,
            particles,
            survivor_count,
            level,
            replicate_seed ^ 0xa0761d6478bd642fULL,
            workspace.survivor_indices,
            states_a,
            scores_a,
            states_b,
            scores_b
        );
        check_cuda(cudaGetLastError(), "resample_kernel launch");
        std::swap(states_a, states_b);
        std::swap(scores_a, scores_b);

        check_cuda(cudaMemsetAsync(
            workspace.mutation_stats,
            0,
            sizeof(MutationStats),
            workspace.stream
        ), "cudaMemsetAsync mutation stats");
        launch_mutation(
            tile_size,
            n,
            particles,
            args.mutation_steps,
            std::min(args.maximum_flips, 7),
            level,
            replicate_seed,
            next_threshold,
            log2_tau,
            log2_period,
            period,
            next_period,
            states_a,
            scores_a,
            workspace.mutation_stats,
            workspace.stream
        );
        MutationStats mutation_stats{};
        check_cuda(cudaMemcpyAsync(
            &mutation_stats,
            workspace.mutation_stats,
            sizeof(MutationStats),
            cudaMemcpyDeviceToHost,
            workspace.stream
        ), "cudaMemcpyAsync mutation stats");
        check_cuda(cudaStreamSynchronize(workspace.stream), "mutation synchronize");
        const double acceptance = mutation_stats.proposed > 0
            ? static_cast<double>(mutation_stats.accepted) / static_cast<double>(mutation_stats.proposed)
            : 0.0;
        acceptances.push_back(acceptance);
        previous_threshold = next_threshold;
        completed_levels = level + 1;

        if (level + 1 == args.maximum_levels) {
            throw std::runtime_error("maximum subset levels reached before the active event");
        }
    }

    const int distinct = distinct_state_count_gpu(
        states_a,
        particles,
        word_count,
        workspace.hashes,
        workspace.stream
    );
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    const double minimum_acceptance = acceptances.empty()
        ? 1.0
        : *std::min_element(acceptances.begin(), acceptances.end());
    const double mean_acceptance = acceptances.empty()
        ? 1.0
        : std::accumulate(acceptances.begin(), acceptances.end(), 0.0)
            / static_cast<double>(acceptances.size());
    const double active_fraction = log_probability > std::log(std::numeric_limits<double>::min())
        ? std::exp(log_probability)
        : 0.0;
    const double log_dfi_count_bound =
        std::log(effective_tau / 2.0)
        - 2.0 * std::log(static_cast<double>(period_value))
        + static_cast<double>(n) * std::log(2.0)
        + log_probability;

    ResultRow row;
    row.n = n;
    row.period = period_value;
    row.replicate = replicate;
    row.tau = args.tau;
    row.tau_decay = args.tau_decay;
    row.effective_tau = effective_tau;
    row.particles = particles;
    row.retain_fraction = args.retain_fraction;
    row.mutation_steps = args.mutation_steps;
    row.levels = completed_levels;
    row.log_active_fraction = log_probability;
    row.active_fraction = active_fraction;
    row.final_conditional_fraction = final_fraction;
    row.minimum_acceptance = minimum_acceptance;
    row.mean_acceptance = mean_acceptance;
    row.final_distinct_states = distinct;
    row.log_dfi_count_bound = log_dfi_count_bound;
    row.seconds = seconds;
    row.device = device;
    row.estimated_device_bytes = static_cast<uint64_t>(task_bytes);
    return row;
}

void print_row(const ResultRow& row) {
    std::cout
        << std::setprecision(10)
        << "gpu=" << row.device
        << " n=" << row.n
        << " r=" << row.period
        << " rep=" << row.replicate
        << " tau_n=" << row.effective_tau
        << " levels=" << row.levels
        << " log_a=" << row.log_active_fraction
        << " a=" << row.active_fraction
        << " final=" << row.final_conditional_fraction
        << " accept=" << row.mean_acceptance
        << " min_accept=" << row.minimum_acceptance
        << " distinct=" << row.final_distinct_states << '/' << row.particles
        << " log_DFI_lb=" << row.log_dfi_count_bound
        << " estimated_device_MiB="
        << static_cast<double>(row.estimated_device_bytes) / static_cast<double>(1ULL << 20)
        << " time=" << row.seconds << "s\n";
}

void write_csv_header(std::ostream& output) {
    output
        << "n,period,replicate,tau,tau_decay,effective_tau,particles,retain_fraction,mutation_steps,levels,"
        << "log_active_fraction,active_fraction,final_conditional_fraction,"
        << "minimum_acceptance,mean_acceptance,final_distinct_states,log_dfi_count_bound,seconds,"
        << "device,estimated_device_bytes\n";
}

void write_csv_row(std::ostream& output, const ResultRow& row) {
    output << std::setprecision(17);
    output
        << row.n << ',' << row.period << ',' << row.replicate << ',' << row.tau << ','
        << row.tau_decay << ',' << row.effective_tau << ',' << row.particles << ','
        << row.retain_fraction << ',' << row.mutation_steps << ','
        << row.levels << ',' << row.log_active_fraction << ',' << row.active_fraction << ','
        << row.final_conditional_fraction << ',' << row.minimum_acceptance << ','
        << row.mean_acceptance << ',' << row.final_distinct_states << ','
        << row.log_dfi_count_bound << ',' << row.seconds << ','
        << row.device << ',' << row.estimated_device_bytes << '\n';
}

void write_csv(const std::filesystem::path& path, const std::vector<ResultRow>& rows) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path.string());
    }
    write_csv_header(output);
    for (const ResultRow& row : rows) {
        write_csv_row(output, row);
    }
}

void append_csv_row(const std::filesystem::path& path, const ResultRow& row) {
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("failed to append output: " + path.string());
    }
    write_csv_row(output, row);
}

void print_fits(
    const std::vector<ResultRow>& rows,
    const std::vector<uint64_t>& periods,
    double tau_decay
) {
    constexpr double ln2 = 0.693147180559945309417232121458;
    for (uint64_t period : periods) {
        std::vector<std::pair<double, double>> points;
        for (int n = 0; n <= kMaxN; ++n) {
            double sum = 0.0;
            int count = 0;
            for (const ResultRow& row : rows) {
                if (row.period == period && row.n == n && std::isfinite(row.log_active_fraction)) {
                    sum += row.log_active_fraction;
                    ++count;
                }
            }
            if (count > 0) {
                points.emplace_back(static_cast<double>(n), sum / static_cast<double>(count));
            }
        }
        if (points.size() < 2) {
            continue;
        }
        double mean_n = 0.0;
        double mean_log = 0.0;
        for (const auto& point : points) {
            mean_n += point.first;
            mean_log += point.second;
        }
        mean_n /= static_cast<double>(points.size());
        mean_log /= static_cast<double>(points.size());
        double covariance = 0.0;
        double variance_n = 0.0;
        double total_variance = 0.0;
        for (const auto& point : points) {
            covariance += (point.first - mean_n) * (point.second - mean_log);
            variance_n += (point.first - mean_n) * (point.first - mean_n);
            total_variance += (point.second - mean_log) * (point.second - mean_log);
        }
        const double slope = covariance / variance_n;
        const double intercept = mean_log - slope * mean_n;
        double residual = 0.0;
        for (const auto& point : points) {
            const double error = point.second - (slope * point.first + intercept);
            residual += error * error;
        }
        const double r_squared = total_variance > 0.0 ? 1.0 - residual / total_variance : 1.0;
        std::cout
            << "fit r=" << period
            << " log(active)=" << slope << "*n+" << intercept
            << " R2=" << r_squared
            << " beta=" << -slope
            << " beta<ln2=" << ((-slope) < ln2 ? "yes" : "no")
            << " fixed_r_DFI_exponent=" << ln2 + slope - tau_decay
            << '\n';
    }
}

void validate_point(const Args& args, const Point& point) {
    for (uint64_t period : {point.period, point.period + 1}) {
        if (two_power(period) > point.n) {
            throw std::runtime_error("period contains a power of two larger than 2^n");
        }
        if (odd_part(period) > static_cast<uint64_t>(args.max_odd_part)) {
            std::ostringstream message;
            message << "(n,r)=(" << point.n << ',' << point.period << ") requires odd part "
                    << odd_part(period) << " for period " << period
                    << ", above --max-odd-part=" << args.max_odd_part;
            throw std::runtime_error(message.str());
        }
    }
}

long double estimated_task_work(const Task& task) {
    const long double odd_work = static_cast<long double>(odd_part(task.point.period))
        + static_cast<long double>(odd_part(task.point.period + 1));
    const long double n = static_cast<long double>(task.point.n);
    // Mutation work grows with point-query width and typically with the number
    // of rare-event levels.  n^2 * odd_work is a useful longest-job-first key;
    // actual jobs are still claimed dynamically by workers.
    return n * n * odd_work;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        int detected_devices = 0;
        check_cuda(cudaGetDeviceCount(&detected_devices), "cudaGetDeviceCount");
        for (int device : args.devices) {
            if (device >= detected_devices) {
                std::ostringstream message;
                message << "requested GPU " << device << " but only "
                        << detected_devices << " CUDA devices are visible";
                throw std::runtime_error(message.str());
            }
        }

        std::vector<Point> points = args.points;
        if (points.empty()) {
            for (int n = args.n_min; n <= args.n_max; n += args.n_step) {
                for (uint64_t period : args.periods) {
                    points.push_back(Point{n, period});
                }
            }
        }
        for (const Point& point : points) {
            validate_point(args, point);
        }

        std::vector<Task> tasks;
        tasks.reserve(points.size() * static_cast<size_t>(args.replicates));
        size_t ordinal = 0;
        for (const Point& point : points) {
            for (int replicate = 0; replicate < args.replicates; ++replicate) {
                tasks.push_back(Task{point, replicate, ordinal++});
            }
        }
        std::stable_sort(
            tasks.begin(),
            tasks.end(),
            [](const Task& left, const Task& right) {
                return estimated_task_work(left) > estimated_task_work(right);
            }
        );

        size_t maximum_task_bytes = 0;
        for (const Point& point : points) {
            maximum_task_bytes = std::max(
                maximum_task_bytes,
                estimated_task_bytes(args, point.n, point.period)
            );
        }
        std::cout
            << "multi_gpu devices=";
        for (size_t index = 0; index < args.devices.size(); ++index) {
            std::cout << (index == 0 ? "" : ",") << args.devices[index];
        }
        std::cout
            << " tasks=" << tasks.size()
            << " points=" << points.size()
            << " particles=" << args.particles
            << " mutation_steps=" << args.mutation_steps
            << " replicates=" << args.replicates
            << " max_estimated_task_MiB="
            << static_cast<double>(maximum_task_bytes) / static_cast<double>(1ULL << 20)
            << '\n';
        for (int device : args.devices) {
            check_cuda(cudaSetDevice(device), "planner cudaSetDevice");
            cudaDeviceProp properties{};
            check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "planner cudaMemGetInfo");
            std::cout
                << "worker gpu=" << device
                << " name=" << properties.name
                << " SMs=" << properties.multiProcessorCount
                << " memory_MiB=" << properties.totalGlobalMem / (1ULL << 20)
                << " free_MiB=" << free_bytes / (1ULL << 20)
                << '\n';
        }
        if (tasks.size() < args.devices.size()) {
            std::cout << "warning: fewer independent tasks than GPUs; some GPUs will be idle\n";
        }
        if (args.dry_run) {
            for (const Point& point : points) {
                std::cout
                    << "memory_plan n=" << point.n
                    << " r=" << point.period
                    << " root_MiB="
                    << static_cast<double>(
                        root_bytes_for(point.n, point.period)
                        + root_bytes_for(point.n, point.period + 1)
                    ) / static_cast<double>(1ULL << 20)
                    << " conservative_total_MiB="
                    << static_cast<double>(
                        estimated_task_bytes(args, point.n, point.period)
                    ) / static_cast<double>(1ULL << 20)
                    << '\n';
            }
            return 0;
        }
        // Start a valid checkpoint file before workers launch.  Completed rows
        // are appended in O(1); the final file is rewritten in logical order.
        write_csv(args.output, {});

        std::vector<ResultRow> result_slots(ordinal);
        std::vector<unsigned char> completed(ordinal, 0);
        std::atomic<size_t> next_task{0};
        std::atomic<bool> cancel{false};
        std::mutex output_mutex;
        std::mutex error_mutex;
        std::exception_ptr first_error;
        std::vector<std::thread> workers;
        workers.reserve(args.devices.size());

        for (int device : args.devices) {
            workers.emplace_back([&, device]() {
                try {
                    check_cuda(cudaSetDevice(device), "worker cudaSetDevice");
                    check_cuda(
                        cudaDeviceSetLimit(cudaLimitStackSize, 8192),
                        "worker cudaDeviceSetLimit stack"
                    );
                    while (!cancel.load(std::memory_order_relaxed)) {
                        const size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                        if (task_index >= tasks.size()) {
                            break;
                        }
                        const Task task = tasks[task_index];
                        const size_t task_bytes = estimated_task_bytes(
                            args,
                            task.point.n,
                            task.point.period
                        );
                        PeriodAllocation period = build_period(
                            task.point.n,
                            task.point.period,
                            args.max_odd_part
                        );
                        PeriodAllocation next_period = build_period(
                            task.point.n,
                            task.point.period + 1,
                            args.max_odd_part
                        );
                        ResultRow row = run_replicate(
                            args,
                            task.point.n,
                            task.point.period,
                            task.replicate,
                            device,
                            period.device_spec,
                            next_period.device_spec,
                            task_bytes
                        );
                        {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            result_slots[task.ordinal] = row;
                            completed[task.ordinal] = 1;
                            print_row(row);
                            append_csv_row(args.output, row);
                        }
                    }
                    check_cuda(cudaDeviceSynchronize(), "worker final synchronize");
                } catch (...) {
                    cancel.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
        if (first_error) {
            std::rethrow_exception(first_error);
        }

        std::vector<ResultRow> rows;
        rows.reserve(result_slots.size());
        for (size_t index = 0; index < result_slots.size(); ++index) {
            if (completed[index] == 0) {
                throw std::runtime_error("multi-GPU scheduler ended with incomplete tasks");
            }
            rows.push_back(result_slots[index]);
        }
        write_csv(args.output, rows);
        std::set<uint64_t> unique_periods;
        for (const Point& point : points) {
            unique_periods.insert(point.period);
        }
        print_fits(
            rows,
            std::vector<uint64_t>(unique_periods.begin(), unique_periods.end()),
            args.tau_decay
        );
        std::cout << "output=" << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

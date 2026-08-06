// Rare-event subset simulation for an active small-denominator HP-1 set.
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
// Build:
//   nvcc -O3 -std=c++17 -arch=native \
//     scripts/experiments/hp1_active_tail_subset_cuda.cu \
//     -o hp1_active_tail_subset_cuda

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
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
    int device = 0;
    std::filesystem::path output = "data/hp1_active_tail_subset_cuda/active_tail.csv";
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
        } else if (key == "--device") {
            args.device = std::stoi(require_value(key));
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: hp1_active_tail_subset_cuda [--n-min 20] [--n-max 200] "
                << "[--n-step 10] [--periods 12] [--tau 3e-4] [--tau-decay 0] "
                << "[--particles 8192] [--retain-fraction 0.1] "
                << "[--mutation-steps 64] [--maximum-flips 3] "
                << "[--maximum-levels 100] [--replicates 4] "
                << "[--max-odd-part 4096] [--seed 20260806] [--device 0] "
                << "[--output result.csv]\n";
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

__device__ double warp_sum(double value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ double warp_max(double value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmax(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ void build_odd_weights(int n, const uint64_t* words, double2* odd_weights) {
    const int odd_count = n / 2;
    const int even_count = (n + 1) / 2;
    double left = 0.0;
    for (int odd_index = 0; odd_index < odd_count; ++odd_index) {
        left = 0.25 * left + 0.5 * static_cast<double>(output_bit(words, 2 * odd_index));
        odd_weights[odd_index].x = left;
    }

    double right = 0.0;
    for (int odd_index = odd_count - 1; odd_index >= 0; --odd_index) {
        const int next_even = odd_index + 1;
        right *= 0.25;
        if (next_even < even_count) {
            right += 0.5 * static_cast<double>(output_bit(words, 2 * next_even));
        }
        const double phase_over_pi = odd_weights[odd_index].x + right;
        double sine = 0.0;
        double cosine = 0.0;
        sincospi(phase_over_pi, &sine, &cosine);
        const double sign = output_bit(words, 2 * odd_index + 1) == 0 ? 1.0 : -1.0;
        odd_weights[odd_index] = make_double2(sign * cosine, sign * sine);
    }
}

__device__ double log2_scaled_probability(
    const DevicePeriod& period,
    const uint64_t* words,
    const double2* odd_weights,
    int lane
) {
    // Each frequency product can be exponentially small.  Periodic
    // renormalization preserves its phase and records its logarithmic scale;
    // a complex log-sum-exp then combines roots-of-unity frequencies without
    // the n≈100 underflow of a direct product.
    double local_scale = -CUDART_INF;
    double local_real = 0.0;
    double local_imag = 0.0;
    for (int frequency = lane; frequency < period.odd_part; frequency += kWarpSize) {
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

    double global_scale = warp_max(local_scale);
    global_scale = __shfl_sync(0xffffffffU, global_scale, 0);
    const double lane_rescale = isfinite(local_scale) ? exp(local_scale - global_scale) : 0.0;
    double sum_real = warp_sum(local_real * lane_rescale);
    double sum_imag = warp_sum(local_imag * lane_rescale);
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
    return __shfl_sync(0xffffffffU, result, 0);
}

__device__ double active_score(
    int n,
    double log2_tau,
    double log2_period,
    const DevicePeriod& period,
    const DevicePeriod& next_period,
    const uint64_t* words,
    double2* odd_weights,
    int lane
) {
    if (lane == 0) {
        build_odd_weights(n, words, odd_weights);
    }
    __syncwarp();
    const double log2_p = log2_scaled_probability(period, words, odd_weights, lane);
    const double log2_q = log2_scaled_probability(next_period, words, odd_weights, lane);
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
    return __shfl_sync(0xffffffffU, score, 0);
}

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
    __shared__ uint64_t shared_words[kWarpsPerBlock][kMaxWords];
    __shared__ double2 shared_odd_weights[kWarpsPerBlock][kMaxOddQubits];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int particle = blockIdx.x * kWarpsPerBlock + warp;
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
        shared_words[warp][lane] = word;
        states[static_cast<size_t>(particle) * word_count + lane] = word;
    }
    __syncwarp();
    const double score = active_score(
        n,
        log2_tau,
        log2_period,
        period,
        next_period,
        shared_words[warp],
        shared_odd_weights[warp],
        lane
    );
    if (lane == 0) {
        scores[particle] = score;
    }
}

__global__ void resample_kernel(
    int word_count,
    int particles,
    const int* parent_indices,
    const uint64_t* source_states,
    const double* source_scores,
    uint64_t* target_states,
    double* target_scores
) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particles) {
        return;
    }
    const int parent = parent_indices[particle];
    for (int word = 0; word < word_count; ++word) {
        target_states[static_cast<size_t>(particle) * word_count + word] =
            source_states[static_cast<size_t>(parent) * word_count + word];
    }
    target_scores[particle] = source_scores[parent];
}

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
    __shared__ uint64_t shared_words[kWarpsPerBlock][kMaxWords];
    __shared__ double2 shared_odd_weights[kWarpsPerBlock][kMaxOddQubits];
    __shared__ int shared_bits[kWarpsPerBlock][8];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int particle = blockIdx.x * kWarpsPerBlock + warp;
    if (particle >= particles) {
        return;
    }
    const int word_count = (n + 63) / 64;
    if (lane < word_count) {
        shared_words[warp][lane] = states[static_cast<size_t>(particle) * word_count + lane];
    }
    __syncwarp();

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
                shared_bits[warp][index] = bit;
                shared_words[warp][bit >> 6] ^= 1ULL << (bit & 63);
            }
            shared_bits[warp][7] = flip_count;
        }
        __syncwarp();
        flip_count = shared_bits[warp][7];
        const double candidate_score = active_score(
            n,
            log2_tau,
            log2_period,
            period,
            next_period,
            shared_words[warp],
            shared_odd_weights[warp],
            lane
        );
        if (lane == 0) {
            if (candidate_score >= threshold) {
                current_score = candidate_score;
                ++accepted;
            } else {
                for (int index = 0; index < flip_count; ++index) {
                    const int bit = shared_bits[warp][index];
                    shared_words[warp][bit >> 6] ^= 1ULL << (bit & 63);
                }
            }
        }
        __syncwarp();
    }

    if (lane < word_count) {
        states[static_cast<size_t>(particle) * word_count + lane] = shared_words[warp][lane];
    }
    if (lane == 0) {
        scores[particle] = current_score;
        atomicAdd(&stats->accepted, accepted);
        atomicAdd(&stats->proposed, static_cast<unsigned long long>(mutation_steps));
    }
}

uint64_t hash_state(const uint64_t* words, int word_count) {
    uint64_t hash = 0x243f6a8885a308d3ULL;
    for (int word = 0; word < word_count; ++word) {
        uint64_t value = words[word] + 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        value ^= value >> 31;
        hash ^= value + (hash << 6) + (hash >> 2);
    }
    return hash;
}

int distinct_state_count(const std::vector<uint64_t>& states, int particles, int word_count) {
    std::vector<uint64_t> hashes(particles);
    for (int particle = 0; particle < particles; ++particle) {
        hashes[particle] = hash_state(&states[static_cast<size_t>(particle) * word_count], word_count);
    }
    std::sort(hashes.begin(), hashes.end());
    return static_cast<int>(std::unique(hashes.begin(), hashes.end()) - hashes.begin());
}

ResultRow run_replicate(
    const Args& args,
    int n,
    uint64_t period_value,
    int replicate,
    const DevicePeriod& period,
    const DevicePeriod& next_period
) {
    const int particles = args.particles;
    const int word_count = (n + 63) / 64;
    const size_t state_bytes = static_cast<size_t>(particles) * word_count * sizeof(uint64_t);
    const size_t score_bytes = static_cast<size_t>(particles) * sizeof(double);
    uint64_t* states_a = nullptr;
    uint64_t* states_b = nullptr;
    double* scores_a = nullptr;
    double* scores_b = nullptr;
    int* parent_indices_device = nullptr;
    MutationStats* mutation_stats_device = nullptr;
    check_cuda(cudaMalloc(&states_a, state_bytes), "cudaMalloc states_a");
    check_cuda(cudaMalloc(&states_b, state_bytes), "cudaMalloc states_b");
    check_cuda(cudaMalloc(&scores_a, score_bytes), "cudaMalloc scores_a");
    check_cuda(cudaMalloc(&scores_b, score_bytes), "cudaMalloc scores_b");
    check_cuda(cudaMalloc(&parent_indices_device, particles * sizeof(int)), "cudaMalloc parents");
    check_cuda(cudaMalloc(&mutation_stats_device, sizeof(MutationStats)), "cudaMalloc mutation stats");

    const uint64_t replicate_seed = args.seed
        ^ (period_value * 1000003ULL)
        ^ (static_cast<uint64_t>(n) * 10007ULL)
        ^ (static_cast<uint64_t>(replicate) * 0xd1b54a32d192ed03ULL);
    const int warp_blocks = (particles + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const double effective_tau = args.tau * std::exp(-args.tau_decay * static_cast<double>(n));
    const double log2_tau = std::log2(effective_tau);
    const double log2_period = std::log2(static_cast<double>(period_value));
    const auto started = std::chrono::steady_clock::now();
    initialize_kernel<<<warp_blocks, kBlockSize>>>(
        n,
        particles,
        replicate_seed,
        log2_tau,
        log2_period,
        period,
        next_period,
        states_a,
        scores_a
    );
    check_cuda(cudaGetLastError(), "initialize_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "initialize_kernel synchronize");

    std::vector<double> scores(particles);
    std::vector<int> parents(particles);
    std::vector<double> acceptances;
    std::mt19937_64 rng(replicate_seed ^ 0xa0761d6478bd642fULL);
    double log_probability = 0.0;
    double final_fraction = 0.0;
    double previous_threshold = -std::numeric_limits<double>::infinity();
    int completed_levels = 0;

    for (int level = 0; level < args.maximum_levels; ++level) {
        check_cuda(cudaMemcpy(scores.data(), scores_a, score_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy scores");
        const int event_count = static_cast<int>(std::count_if(
            scores.begin(),
            scores.end(),
            [](double score) { return score >= 0.0; }
        ));

        std::vector<double> ordered = scores;
        std::sort(ordered.begin(), ordered.end(), std::greater<double>());
        const int requested = std::max(1, static_cast<int>(std::ceil(args.retain_fraction * particles)));
        double next_threshold = ordered[requested - 1];
        if (next_threshold >= 0.0) {
            final_fraction = static_cast<double>(event_count) / static_cast<double>(particles);
            if (final_fraction <= 0.0) {
                throw std::runtime_error("nonpositive final conditional fraction");
            }
            log_probability += std::log(final_fraction);
            break;
        }
        if (!(next_threshold > previous_threshold)) {
            // The score distribution can have genuine atoms from exact HP-1
            // cancellations.  Skip the negative atom rather than multiplying
            // by the same conditional level again.  The target event lies at
            // score >= 0, so excluding an atom below zero is lossless.
            next_threshold = std::nextafter(
                previous_threshold,
                std::numeric_limits<double>::infinity()
            );
        }

        std::vector<int> survivors;
        survivors.reserve(requested + 16);
        for (int particle = 0; particle < particles; ++particle) {
            if (scores[particle] >= next_threshold) {
                survivors.push_back(particle);
            }
        }
        const double conditional_fraction =
            static_cast<double>(survivors.size()) / static_cast<double>(particles);
        if (survivors.empty()) {
            throw std::runtime_error("subset score atom contains the whole population; increase mutation-steps");
        }
        log_probability += std::log(conditional_fraction);
        std::uniform_int_distribution<size_t> choose_parent(0, survivors.size() - 1);
        for (int particle = 0; particle < particles; ++particle) {
            parents[particle] = survivors[choose_parent(rng)];
        }
        check_cuda(cudaMemcpy(
            parent_indices_device,
            parents.data(),
            particles * sizeof(int),
            cudaMemcpyHostToDevice
        ), "cudaMemcpy parents");
        const int copy_blocks = (particles + kBlockSize - 1) / kBlockSize;
        resample_kernel<<<copy_blocks, kBlockSize>>>(
            word_count,
            particles,
            parent_indices_device,
            states_a,
            scores_a,
            states_b,
            scores_b
        );
        check_cuda(cudaGetLastError(), "resample_kernel launch");
        std::swap(states_a, states_b);
        std::swap(scores_a, scores_b);

        check_cuda(cudaMemset(mutation_stats_device, 0, sizeof(MutationStats)), "cudaMemset mutation stats");
        mutate_kernel<<<warp_blocks, kBlockSize>>>(
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
            mutation_stats_device
        );
        check_cuda(cudaGetLastError(), "mutate_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "mutate_kernel synchronize");
        MutationStats mutation_stats;
        check_cuda(cudaMemcpy(
            &mutation_stats,
            mutation_stats_device,
            sizeof(MutationStats),
            cudaMemcpyDeviceToHost
        ), "cudaMemcpy mutation stats");
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

    std::vector<uint64_t> final_states(static_cast<size_t>(particles) * word_count);
    check_cuda(cudaMemcpy(final_states.data(), states_a, state_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy final states");
    const int distinct = distinct_state_count(final_states, particles, word_count);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const double minimum_acceptance = acceptances.empty()
        ? 1.0
        : *std::min_element(acceptances.begin(), acceptances.end());
    const double mean_acceptance = acceptances.empty()
        ? 1.0
        : std::accumulate(acceptances.begin(), acceptances.end(), 0.0) / static_cast<double>(acceptances.size());
    const double active_fraction = log_probability > std::log(std::numeric_limits<double>::min())
        ? std::exp(log_probability)
        : 0.0;
    const double log_dfi_count_bound =
        std::log(effective_tau / 2.0)
        - 2.0 * std::log(static_cast<double>(period_value))
        + static_cast<double>(n) * std::log(2.0)
        + log_probability;

    check_cuda(cudaFree(states_a), "cudaFree states_a");
    check_cuda(cudaFree(states_b), "cudaFree states_b");
    check_cuda(cudaFree(scores_a), "cudaFree scores_a");
    check_cuda(cudaFree(scores_b), "cudaFree scores_b");
    check_cuda(cudaFree(parent_indices_device), "cudaFree parents");
    check_cuda(cudaFree(mutation_stats_device), "cudaFree mutation stats");

    return ResultRow{
        n,
        period_value,
        replicate,
        args.tau,
        args.tau_decay,
        effective_tau,
        particles,
        args.retain_fraction,
        args.mutation_steps,
        completed_levels,
        log_probability,
        active_fraction,
        final_fraction,
        minimum_acceptance,
        mean_acceptance,
        distinct,
        log_dfi_count_bound,
        seconds,
    };
}

void print_row(const ResultRow& row) {
    std::cout
        << std::setprecision(10)
        << "n=" << row.n
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
        << "n,period,replicate,tau,tau_decay,effective_tau,particles,retain_fraction,mutation_steps,levels,"
        << "log_active_fraction,active_fraction,final_conditional_fraction,"
        << "minimum_acceptance,mean_acceptance,final_distinct_states,log_dfi_count_bound,seconds\n";
    output << std::setprecision(17);
    for (const ResultRow& row : rows) {
        output
            << row.n << ',' << row.period << ',' << row.replicate << ',' << row.tau << ','
            << row.tau_decay << ',' << row.effective_tau << ',' << row.particles << ','
            << row.retain_fraction << ',' << row.mutation_steps << ','
            << row.levels << ',' << row.log_active_fraction << ',' << row.active_fraction << ','
            << row.final_conditional_fraction << ',' << row.minimum_acceptance << ','
            << row.mean_acceptance << ',' << row.final_distinct_states << ','
            << row.log_dfi_count_bound << ',' << row.seconds << '\n';
    }
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
            << " tau=" << args.tau
            << " tau_decay=" << args.tau_decay
            << " particles=" << args.particles
            << " retain=" << args.retain_fraction
            << " mutation_steps=" << args.mutation_steps
            << " replicates=" << args.replicates
            << " periods=";
        for (size_t index = 0; index < args.periods.size(); ++index) {
            std::cout << (index == 0 ? "" : ",") << args.periods[index];
        }
        std::cout << '\n';

        std::vector<ResultRow> rows;
        for (int n = args.n_min; n <= args.n_max; n += args.n_step) {
            for (uint64_t period_value : args.periods) {
                PeriodAllocation period = build_period(n, period_value, args.max_odd_part);
                PeriodAllocation next_period = build_period(n, period_value + 1, args.max_odd_part);
                for (int replicate = 0; replicate < args.replicates; ++replicate) {
                    const ResultRow row = run_replicate(
                        args,
                        n,
                        period_value,
                        replicate,
                        period.device_spec,
                        next_period.device_spec
                    );
                    rows.push_back(row);
                    print_row(row);
                    write_csv(args.output, rows);
                }
            }
        }
        print_fits(rows, args.periods, args.tau_decay);
        std::cout << "output=" << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

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
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxN = 30;

struct Args {
    int n = 30;
    int pair_count = 1000;
    uint64_t samples_per_pair = 1ULL << 20;
    int chunks_per_pair = 256;
    uint64_t seed = 20260710;
    uint32_t s_min = 0;
    uint32_t s_max = 0;
    std::string devices = "all";
    std::filesystem::path output = "data/hp1_overlap_a_cuda/n30_random_pairs.csv";
};

struct PairInput {
    uint32_t pair_id = 0;
    uint32_t s = 0;
    uint32_t t = 0;
    uint64_t r_s = 0;
    uint64_t r_t = 0;
    uint64_t draws = 0;
    int a_common = 0;
    int exact = 0;
};

struct Partial {
    double sum_re = 0.0;
    double sum_im = 0.0;
    double sum_re2 = 0.0;
    uint64_t count = 0;
};

struct ResultRow {
    uint32_t pair_id = 0;
    int n = 0;
    uint32_t s = 0;
    uint32_t t = 0;
    uint64_t r_s = 0;
    uint64_t r_t = 0;
    int a_common = 0;
    uint64_t draws = 0;
    int exact = 0;
    double mean_re = 0.0;
    double mean_im = 0.0;
    double a_value = 0.0;
    double a_imag = 0.0;
    double stderr_a = 0.0;
};

std::mutex log_mutex;

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

std::vector<int> parse_devices(const std::string& raw_devices) {
    std::vector<int> devices;
    if (raw_devices == "all") {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        for (int device = 0; device < device_count; ++device) {
            devices.push_back(device);
        }
        return devices;
    }

    for (const std::string& item : split(raw_devices, ',')) {
        devices.push_back(std::stoi(item));
    }
    if (devices.empty()) {
        throw std::runtime_error("no CUDA devices selected");
    }
    return devices;
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

        if (key == "--n") {
            args.n = std::stoi(require_value(key));
        } else if (key == "--pair-count") {
            args.pair_count = std::stoi(require_value(key));
        } else if (key == "--samples-per-pair") {
            args.samples_per_pair = std::stoull(require_value(key));
        } else if (key == "--chunks-per-pair") {
            args.chunks_per_pair = std::stoi(require_value(key));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value(key));
        } else if (key == "--s-min") {
            args.s_min = static_cast<uint32_t>(std::stoul(require_value(key)));
        } else if (key == "--s-max") {
            args.s_max = static_cast<uint32_t>(std::stoul(require_value(key)));
        } else if (key == "--devices") {
            args.devices = require_value(key);
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: hp1_overlap_a_cuda [--n 30] [--pair-count 1000] "
                << "[--samples-per-pair 1048576] [--chunks-per-pair 256] "
                << "[--s-min ceil(2^(n/4))] [--s-max floor(2^(n/2))] [--seed 20260710] "
                << "[--devices all|6,7] "
                << "[--output data/hp1_overlap_a_cuda/n30_random_pairs.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (args.n < 1 || args.n > kMaxN) {
        throw std::runtime_error("n must satisfy 1 <= n <= 30");
    }
    const uint64_t n_states = 1ULL << args.n;
    if (args.s_min == 0) {
        args.s_min = static_cast<uint32_t>(std::ceil(std::pow(2.0, static_cast<double>(args.n) / 4.0)));
    }
    if (args.s_max == 0) {
        args.s_max = static_cast<uint32_t>(std::floor(std::pow(2.0, static_cast<double>(args.n) / 2.0)));
    }
    if (args.s_min < 1 || args.s_min > args.s_max || args.s_max >= n_states) {
        throw std::runtime_error("s range must satisfy 1 <= s_min <= s_max < 2^n");
    }
    if (args.pair_count <= 0 || args.samples_per_pair == 0 || args.chunks_per_pair <= 0) {
        throw std::runtime_error("pair-count, samples-per-pair, and chunks-per-pair must be positive");
    }
    return args;
}

uint64_t period_count(int n, uint32_t period) {
    const uint64_t n_states = 1ULL << n;
    return ((n_states - 1ULL) / static_cast<uint64_t>(period)) + 1ULL;
}

int common_power_of_two(uint32_t s, uint32_t t) {
    return __builtin_ctz(s | t);
}

uint64_t capped_combination_count(uint64_t r_s, uint64_t r_t, uint64_t cap) {
    __uint128_t combinations = static_cast<__uint128_t>(r_s) * r_s * r_t * r_t;
    if (combinations > cap) {
        return cap + 1ULL;
    }
    return static_cast<uint64_t>(combinations);
}

std::vector<PairInput> sample_pairs(const Args& args) {
    std::mt19937_64 rng(args.seed);
    std::uniform_int_distribution<uint32_t> distribution(args.s_min, args.s_max);
    std::unordered_set<uint64_t> seen;
    std::vector<PairInput> pairs;
    pairs.reserve(args.pair_count);

    while (static_cast<int>(pairs.size()) < args.pair_count) {
        uint32_t s = distribution(rng);
        uint32_t t = distribution(rng);
        if (s == t) {
            continue;
        }
        if (s > t) {
            std::swap(s, t);
        }
        const uint64_t key = (static_cast<uint64_t>(s) << 32) | static_cast<uint64_t>(t);
        if (!seen.insert(key).second) {
            continue;
        }

        const uint64_t r_s = period_count(args.n, s);
        const uint64_t r_t = period_count(args.n, t);
        const uint64_t exact_combinations =
            capped_combination_count(r_s, r_t, args.samples_per_pair);
        const bool exact = exact_combinations <= args.samples_per_pair;
        pairs.push_back(
            PairInput{
                static_cast<uint32_t>(pairs.size()),
                s,
                t,
                r_s,
                r_t,
                exact ? exact_combinations : args.samples_per_pair,
                common_power_of_two(s, t),
                exact ? 1 : 0,
            }
        );
    }
    return pairs;
}

__device__ uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ uint64_t bounded_random(uint64_t value, uint64_t bound) {
    return bound == 0ULL ? 0ULL : value % bound;
}

__device__ void exact_indices(
    uint64_t sample,
    uint64_t r_s,
    uint64_t r_t,
    uint64_t* q,
    uint64_t* q_prime,
    uint64_t* ell,
    uint64_t* ell_prime
) {
    *q = sample % r_s;
    sample /= r_s;
    *q_prime = sample % r_s;
    sample /= r_s;
    *ell = sample % r_t;
    sample /= r_t;
    *ell_prime = sample;
}

__device__ void random_indices(
    uint64_t seed,
    uint32_t pair_id,
    uint64_t sample,
    uint64_t r_s,
    uint64_t r_t,
    uint64_t* q,
    uint64_t* q_prime,
    uint64_t* ell,
    uint64_t* ell_prime
) {
    const uint64_t base =
        seed
        ^ (static_cast<uint64_t>(pair_id) * 0xd1b54a32d192ed03ULL)
        ^ (sample * 0xabc98388fb8fac03ULL);
    *q = bounded_random(splitmix64(base + 0ULL), r_s);
    *q_prime = bounded_random(splitmix64(base + 1ULL), r_s);
    *ell = bounded_random(splitmix64(base + 2ULL), r_t);
    *ell_prime = bounded_random(splitmix64(base + 3ULL), r_t);
}

__device__ double2 hp1_product_term(
    int n,
    uint64_t qs,
    uint64_t qps,
    uint64_t lt,
    uint64_t lpt
) {
    int delta[kMaxN];
    for (int bit = 0; bit < n; ++bit) {
        delta[bit] =
            static_cast<int>((qs >> bit) & 1ULL)
            - static_cast<int>((qps >> bit) & 1ULL)
            + static_cast<int>((lt >> bit) & 1ULL)
            - static_cast<int>((lpt >> bit) & 1ULL);
    }

    for (int row = 1; row < n; row += 2) {
        if ((delta[row] & 1) != 0) {
            return make_double2(0.0, 0.0);
        }
    }

    double real = ldexp(1.0, n / 2);
    double imag = 0.0;
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (int row = 0; row < n; row += 2) {
        double row_value = static_cast<double>(delta[row]);
        for (int column = 1; column < n; column += 2) {
            const int distance = abs(row - column);
            row_value += static_cast<double>(delta[column]) / static_cast<double>(1U << distance);
        }

        double sin_value = 0.0;
        double cos_value = 0.0;
        sincos(pi * row_value, &sin_value, &cos_value);
        const double factor_real = 1.0 + cos_value;
        const double factor_imag = sin_value;
        const double next_real = real * factor_real - imag * factor_imag;
        const double next_imag = real * factor_imag + imag * factor_real;
        real = next_real;
        imag = next_imag;
    }
    return make_double2(real, imag);
}

__global__ void estimate_a_kernel(
    const PairInput* pairs,
    int pair_count,
    int n,
    int chunks_per_pair,
    uint64_t seed,
    Partial* partials
) {
    const int block_index = blockIdx.x;
    const int pair_index = block_index / chunks_per_pair;
    const int chunk_index = block_index - pair_index * chunks_per_pair;
    if (pair_index >= pair_count) {
        return;
    }

    __shared__ double re_shared[kBlockSize];
    __shared__ double im_shared[kBlockSize];
    __shared__ double re2_shared[kBlockSize];
    __shared__ uint64_t count_shared[kBlockSize];

    const PairInput pair = pairs[pair_index];
    const uint64_t begin = (pair.draws * static_cast<uint64_t>(chunk_index)) / chunks_per_pair;
    const uint64_t end = (pair.draws * static_cast<uint64_t>(chunk_index + 1)) / chunks_per_pair;

    double local_re = 0.0;
    double local_im = 0.0;
    double local_re2 = 0.0;
    uint64_t local_count = 0ULL;
    for (uint64_t sample = begin + threadIdx.x; sample < end; sample += blockDim.x) {
        uint64_t q = 0ULL;
        uint64_t q_prime = 0ULL;
        uint64_t ell = 0ULL;
        uint64_t ell_prime = 0ULL;
        if (pair.exact != 0) {
            exact_indices(sample, pair.r_s, pair.r_t, &q, &q_prime, &ell, &ell_prime);
        } else {
            random_indices(
                seed,
                pair.pair_id,
                sample,
                pair.r_s,
                pair.r_t,
                &q,
                &q_prime,
                &ell,
                &ell_prime
            );
        }

        const uint64_t qs = q * static_cast<uint64_t>(pair.s);
        const uint64_t qps = q_prime * static_cast<uint64_t>(pair.s);
        const uint64_t lt = ell * static_cast<uint64_t>(pair.t);
        const uint64_t lpt = ell_prime * static_cast<uint64_t>(pair.t);
        const double2 term = hp1_product_term(n, qs, qps, lt, lpt);
        local_re += term.x;
        local_im += term.y;
        local_re2 += term.x * term.x;
        ++local_count;
    }

    re_shared[threadIdx.x] = local_re;
    im_shared[threadIdx.x] = local_im;
    re2_shared[threadIdx.x] = local_re2;
    count_shared[threadIdx.x] = local_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            re_shared[threadIdx.x] += re_shared[threadIdx.x + stride];
            im_shared[threadIdx.x] += im_shared[threadIdx.x + stride];
            re2_shared[threadIdx.x] += re2_shared[threadIdx.x + stride];
            count_shared[threadIdx.x] += count_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int partial_index = pair_index * chunks_per_pair + chunk_index;
        partials[partial_index] = Partial{
            re_shared[0],
            im_shared[0],
            re2_shared[0],
            count_shared[0],
        };
    }
}

std::vector<ResultRow> run_pairs_on_device(
    int device,
    const std::vector<PairInput>& pairs,
    const Args& args
) {
    if (pairs.empty()) {
        return {};
    }

    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cout << "device=" << device << " name=" << properties.name
                  << " pairs=" << pairs.size() << std::endl;
    }

    PairInput* d_pairs = nullptr;
    Partial* d_partials = nullptr;
    const size_t pair_bytes = sizeof(PairInput) * pairs.size();
    const size_t partial_count = pairs.size() * static_cast<size_t>(args.chunks_per_pair);
    const size_t partial_bytes = sizeof(Partial) * partial_count;

    check_cuda(cudaMalloc(&d_pairs, pair_bytes), "cudaMalloc pairs");
    check_cuda(cudaMalloc(&d_partials, partial_bytes), "cudaMalloc partials");
    check_cuda(cudaMemcpy(d_pairs, pairs.data(), pair_bytes, cudaMemcpyHostToDevice), "cudaMemcpy pairs");

    const int grid_size = static_cast<int>(partial_count);
    estimate_a_kernel<<<grid_size, kBlockSize>>>(
        d_pairs,
        static_cast<int>(pairs.size()),
        args.n,
        args.chunks_per_pair,
        args.seed,
        d_partials
    );
    check_cuda(cudaGetLastError(), "estimate_a_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "estimate_a_kernel sync");

    std::vector<Partial> partials(partial_count);
    check_cuda(
        cudaMemcpy(partials.data(), d_partials, partial_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy partials"
    );
    check_cuda(cudaFree(d_pairs), "cudaFree pairs");
    check_cuda(cudaFree(d_partials), "cudaFree partials");

    std::vector<ResultRow> rows;
    rows.reserve(pairs.size());
    for (size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
        double sum_re = 0.0;
        double sum_im = 0.0;
        double sum_re2 = 0.0;
        uint64_t count = 0ULL;
        for (int chunk = 0; chunk < args.chunks_per_pair; ++chunk) {
            const Partial& partial = partials[pair_index * args.chunks_per_pair + chunk];
            sum_re += partial.sum_re;
            sum_im += partial.sum_im;
            sum_re2 += partial.sum_re2;
            count += partial.count;
        }
        if (count == 0ULL) {
            throw std::runtime_error("internal error: zero samples collected");
        }

        const PairInput& pair = pairs[pair_index];
        const double mean_re = sum_re / static_cast<double>(count);
        const double mean_im = sum_im / static_cast<double>(count);
        const double normalizer = ldexp(1.0, pair.a_common);
        const double second_moment = sum_re2 / static_cast<double>(count);
        const double variance = std::max(0.0, second_moment - mean_re * mean_re);
        const double stderr_a =
            pair.exact != 0 ? 0.0 : std::sqrt(variance / static_cast<double>(count)) / normalizer;
        rows.push_back(
            ResultRow{
                pair.pair_id,
                args.n,
                pair.s,
                pair.t,
                pair.r_s,
                pair.r_t,
                pair.a_common,
                count,
                pair.exact,
                mean_re,
                mean_im,
                mean_re / normalizer,
                mean_im / normalizer,
                stderr_a,
            }
        );
    }
    return rows;
}

void write_csv(const std::filesystem::path& path, std::vector<ResultRow> rows) {
    std::sort(rows.begin(), rows.end(), [](const ResultRow& left, const ResultRow& right) {
        return left.pair_id < right.pair_id;
    });

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path.string());
    }

    output
        << "pair_id,n,s,t,R_s,R_t,a,draws,exact,mean_product_re,"
        << "mean_product_im,A,A_imag,stderr_A\n";
    output << std::setprecision(17);
    for (const ResultRow& row : rows) {
        output << row.pair_id << ','
               << row.n << ','
               << row.s << ','
               << row.t << ','
               << row.r_s << ','
               << row.r_t << ','
               << row.a_common << ','
               << row.draws << ','
               << row.exact << ','
               << row.mean_re << ','
               << row.mean_im << ','
               << row.a_value << ','
               << row.a_imag << ','
               << row.stderr_a << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<int> devices = parse_devices(args.devices);
        const std::vector<PairInput> pairs = sample_pairs(args);

        std::vector<std::vector<PairInput>> pairs_by_device(devices.size());
        for (size_t index = 0; index < pairs.size(); ++index) {
            pairs_by_device[index % devices.size()].push_back(pairs[index]);
        }

        const auto started = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        std::vector<std::vector<ResultRow>> worker_rows(devices.size());
        for (size_t index = 0; index < devices.size(); ++index) {
            threads.emplace_back([&, index]() {
                worker_rows[index] = run_pairs_on_device(devices[index], pairs_by_device[index], args);
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }

        std::vector<ResultRow> rows;
        for (const auto& item : worker_rows) {
            rows.insert(rows.end(), item.begin(), item.end());
        }
        write_csv(args.output, std::move(rows));

        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        const auto exact_count = std::count_if(
            pairs.begin(),
            pairs.end(),
            [](const PairInput& pair) { return pair.exact != 0; }
        );
        std::cout << "output=" << args.output << std::endl;
        std::cout << "n=" << args.n
                  << " pair_count=" << args.pair_count
                  << " s_min=" << args.s_min
                  << " s_max=" << args.s_max
                  << " samples_per_pair=" << args.samples_per_pair
                  << " exact_pairs=" << exact_count
                  << std::endl;
        std::cout << "elapsed_seconds=" << std::fixed << std::setprecision(3)
                  << seconds << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

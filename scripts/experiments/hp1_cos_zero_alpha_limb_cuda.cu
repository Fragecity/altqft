#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 128;
constexpr int kMaxN = 512;
constexpr int kLimbs = 8;

struct Args {
    int n = 500;
    int alpha_min = 0;
    int alpha_max = 100;
    int samples_per_alpha = 500;
    int s_min_exp = 25;
    int s_max_exp = 50;
    uint64_t seed = 20260710ULL;
    int device = 0;
    std::filesystem::path sample_output =
        "data/hp1_cos_cdelta_hist/n500_alpha0_100_500per_samples.csv";
};

struct SampleSummary {
    uint64_t s = 0;
    int alpha = 0;
    int v2_s = 0;
    int beta = 0;
    int count_one = 0;
    int count_zero = 0;
};

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << context << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
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
        } else if (key == "--alpha-min") {
            args.alpha_min = std::stoi(require_value(key));
        } else if (key == "--alpha-max") {
            args.alpha_max = std::stoi(require_value(key));
        } else if (key == "--samples-per-alpha") {
            args.samples_per_alpha = std::stoi(require_value(key));
        } else if (key == "--s-min-exp") {
            args.s_min_exp = std::stoi(require_value(key));
        } else if (key == "--s-max-exp") {
            args.s_max_exp = std::stoi(require_value(key));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value(key));
        } else if (key == "--device") {
            args.device = std::stoi(require_value(key));
        } else if (key == "--sample-output") {
            args.sample_output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: hp1_cos_zero_alpha_limb_cuda [--n 500] "
                << "[--alpha-min 0] [--alpha-max 100] [--samples-per-alpha 500] "
                << "[--s-min-exp 25] [--s-max-exp 50] [--device 0] "
                << "[--sample-output samples.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    if (args.n < 1 || args.n > kMaxN) {
        throw std::runtime_error("n must satisfy 1 <= n <= 512");
    }
    if (args.alpha_min < 0 || args.alpha_max < args.alpha_min || args.alpha_max >= args.n) {
        throw std::runtime_error("alpha range must satisfy 0 <= alpha_min <= alpha_max < n");
    }
    if (args.samples_per_alpha <= 0) {
        throw std::runtime_error("samples-per-alpha must be positive");
    }
    if (args.s_min_exp < 0 || args.s_max_exp >= 63 || args.s_min_exp > args.s_max_exp) {
        throw std::runtime_error("s exponent range must satisfy 0 <= min <= max < 63");
    }
    return args;
}

__device__ uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ uint64_t bounded_u64(uint64_t value, uint64_t bound) {
    return bound == 0ULL ? 0ULL : value % bound;
}

__device__ int ceil_log2_u64(uint64_t value) {
    if (value <= 1ULL) {
        return 0;
    }
    return 64 - __clzll(value - 1ULL);
}

__device__ uint64_t sample_period_s_with_v2_limit(
    uint64_t seed,
    uint64_t sample,
    int s_min_exp,
    int s_max_exp,
    int max_v2
) {
    const int v2_limit = max_v2 < s_max_exp ? max_v2 : s_max_exp - 1;
    const int v2 = static_cast<int>(
        bounded_u64(splitmix64(seed ^ (sample * 0x8cb92ba72f3d8dd7ULL)), v2_limit + 1)
    );
    const uint64_t lower = v2 < s_min_exp ? (1ULL << (s_min_exp - v2)) : 1ULL;
    const uint64_t upper = 1ULL << (s_max_exp - v2);
    uint64_t odd =
        lower + bounded_u64(splitmix64(seed ^ (sample * 0x7a7c159e3779b97fULL)), upper - lower);
    odd |= 1ULL;
    if (odd >= upper) {
        odd = upper - 1ULL;
    }
    return odd << v2;
}

__device__ void clear_limbs(uint64_t* limbs) {
    for (int index = 0; index < kLimbs; ++index) {
        limbs[index] = 0ULL;
    }
}

__device__ void random_q_limbs(uint64_t seed, int sample, int q_bits, uint64_t* q) {
    for (int limb = 0; limb < kLimbs; ++limb) {
        q[limb] = splitmix64(seed ^ (static_cast<uint64_t>(sample) * 0xd1b54a32d192ed03ULL) ^ limb);
    }
    if (q_bits < kLimbs * 64) {
        const int full_limbs = q_bits / 64;
        const int rem_bits = q_bits % 64;
        for (int limb = full_limbs + (rem_bits > 0 ? 1 : 0); limb < kLimbs; ++limb) {
            q[limb] = 0ULL;
        }
        if (rem_bits == 0) {
            if (full_limbs < kLimbs) {
                q[full_limbs] = 0ULL;
            }
        } else {
            q[full_limbs] &= (1ULL << rem_bits) - 1ULL;
        }
    }
}

__device__ void clear_low_bits(uint64_t* limbs, int bit_count) {
    const int full_limbs = bit_count / 64;
    const int rem_bits = bit_count % 64;
    for (int limb = 0; limb < full_limbs && limb < kLimbs; ++limb) {
        limbs[limb] = 0ULL;
    }
    if (rem_bits != 0 && full_limbs < kLimbs) {
        limbs[full_limbs] &= ~((1ULL << rem_bits) - 1ULL);
    }
}

__device__ void set_bit(uint64_t* limbs, int bit) {
    limbs[bit / 64] |= 1ULL << (bit % 64);
}

__device__ void random_qdiff_limbs(
    uint64_t seed,
    int sample,
    int q_bits,
    int beta,
    uint64_t* qdiff
) {
    const int limit_bits = q_bits - 1;
    for (int limb = 0; limb < kLimbs; ++limb) {
        qdiff[limb] =
            splitmix64(seed ^ (static_cast<uint64_t>(sample) * 0xabc98388fb8fac03ULL) ^ (17ULL + limb));
    }
    if (limit_bits < kLimbs * 64) {
        const int full_limbs = limit_bits / 64;
        const int rem_bits = limit_bits % 64;
        for (int limb = full_limbs + (rem_bits > 0 ? 1 : 0); limb < kLimbs; ++limb) {
            qdiff[limb] = 0ULL;
        }
        if (rem_bits == 0) {
            if (full_limbs < kLimbs) {
                qdiff[full_limbs] = 0ULL;
            }
        } else {
            qdiff[full_limbs] &= (1ULL << rem_bits) - 1ULL;
        }
    }
    clear_low_bits(qdiff, beta);
    set_bit(qdiff, beta);
}

__device__ void multiply_limbs_by_u64(const uint64_t* q, uint64_t factor, uint64_t* out) {
    unsigned __int128 carry = 0;
    for (int limb = 0; limb < kLimbs; ++limb) {
        const unsigned __int128 product =
            static_cast<unsigned __int128>(q[limb]) * static_cast<unsigned __int128>(factor) + carry;
        out[limb] = static_cast<uint64_t>(product);
        carry = product >> 64;
    }
}

__device__ void make_shifted_128(uint64_t lo, uint64_t hi, int shift, uint64_t* out) {
    clear_limbs(out);
    const int limb_shift = shift / 64;
    const int bit_shift = shift % 64;
    const uint64_t values[2] = {lo, hi};
    for (int source = 0; source < 2; ++source) {
        const uint64_t value = values[source];
        if (value == 0ULL) {
            continue;
        }
        const int target = limb_shift + source;
        if (target < kLimbs) {
            out[target] |= value << bit_shift;
        }
        if (bit_shift != 0 && target + 1 < kLimbs) {
            out[target + 1] |= value >> (64 - bit_shift);
        }
    }
}

__device__ void add_limbs(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    unsigned __int128 carry = 0;
    for (int limb = 0; limb < kLimbs; ++limb) {
        const unsigned __int128 sum =
            static_cast<unsigned __int128>(a[limb]) + static_cast<unsigned __int128>(b[limb]) + carry;
        out[limb] = static_cast<uint64_t>(sum);
        carry = sum >> 64;
    }
}

__device__ int get_bit(const uint64_t* limbs, int bit) {
    return static_cast<int>((limbs[bit / 64] >> (bit % 64)) & 1ULL);
}

__global__ void sample_kernel(
    int n,
    int alpha_min,
    int samples_per_alpha,
    int total_samples,
    int s_min_exp,
    int s_max_exp,
    uint64_t seed,
    SampleSummary* summaries
) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= total_samples) {
        return;
    }

    const int alpha = alpha_min + sample / samples_per_alpha;
    const uint64_t s =
        sample_period_s_with_v2_limit(seed, sample, s_min_exp, s_max_exp, alpha);
    const int v2_s = __ffsll(static_cast<long long>(s)) - 1;
    const int beta = alpha - v2_s;

    uint64_t q[kLimbs];
    uint64_t qdiff[kLimbs];
    uint64_t qs[kLimbs];
    uint64_t d[kLimbs];
    uint64_t qps[kLimbs];

    const int q_bits = n - ceil_log2_u64(s);
    random_q_limbs(seed, sample, q_bits - 1, q);
    random_qdiff_limbs(seed, sample, q_bits, beta, qdiff);
    multiply_limbs_by_u64(q, s, qs);
    multiply_limbs_by_u64(qdiff, s, d);
    add_limbs(qs, d, qps);

    int delta[kMaxN];
    for (int bit = 0; bit < n; ++bit) {
        delta[bit] = get_bit(qs, bit) - get_bit(qps, bit);
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    int count_one = 0;
    int count_zero = 0;
    for (int row = 0; row < n; ++row) {
        double row_value = static_cast<double>(delta[row]);
        if ((row & 1) == 0) {
            for (int column = 1; column < n; column += 2) {
                const int distance = row > column ? row - column : column - row;
                row_value += ldexp(static_cast<double>(delta[column]), -distance);
            }
        }
        const double cos_value = cos(0.5 * pi * row_value);
        if (cos_value > 1.0 - 1.0e-12) {
            ++count_one;
        }
        if (fabs(cos_value) < 1.0e-12) {
            ++count_zero;
        }
    }

    summaries[sample] = SampleSummary{s, alpha, v2_s, beta, count_one, count_zero};
}

void write_sample_summary(const Args& args, const std::vector<SampleSummary>& summaries) {
    std::filesystem::create_directories(args.sample_output.parent_path());
    std::ofstream file(args.sample_output);
    if (!file) {
        throw std::runtime_error("failed to open sample output");
    }
    file << "sample_id,s,alpha,v2_s,beta,count_one,count_zero,p_one,p_zero\n";
    file << std::setprecision(17);
    for (std::size_t index = 0; index < summaries.size(); ++index) {
        const SampleSummary& row = summaries[index];
        file << index << ','
             << row.s << ','
             << row.alpha << ','
             << row.v2_s << ','
             << row.beta << ','
             << row.count_one << ','
             << row.count_zero << ','
             << static_cast<double>(row.count_one) / static_cast<double>(args.n) << ','
             << static_cast<double>(row.count_zero) / static_cast<double>(args.n)
             << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        check_cuda(cudaSetDevice(args.device), "cudaSetDevice");

        const int total_samples =
            (args.alpha_max - args.alpha_min + 1) * args.samples_per_alpha;
        SampleSummary* d_summaries = nullptr;
        check_cuda(
            cudaMalloc(&d_summaries, sizeof(SampleSummary) * total_samples),
            "cudaMalloc summaries"
        );

        const int block_count = (total_samples + kBlockSize - 1) / kBlockSize;
        sample_kernel<<<block_count, kBlockSize>>>(
            args.n,
            args.alpha_min,
            args.samples_per_alpha,
            total_samples,
            args.s_min_exp,
            args.s_max_exp,
            args.seed,
            d_summaries
        );
        check_cuda(cudaGetLastError(), "sample_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "sample_kernel sync");

        std::vector<SampleSummary> summaries(total_samples);
        check_cuda(
            cudaMemcpy(
                summaries.data(),
                d_summaries,
                sizeof(SampleSummary) * total_samples,
                cudaMemcpyDeviceToHost
            ),
            "copy summaries"
        );
        write_sample_summary(args, summaries);
        cudaFree(d_summaries);
        std::cout << "wrote " << args.sample_output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

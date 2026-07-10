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

constexpr int kBlockSize = 256;
constexpr int kMaxN = 120;
constexpr int kOffsetRadius = 24;
constexpr int kOffsetBins = 2 * kOffsetRadius + 1;

struct Args {
    int n = 100;
    int samples = 10000;
    int bins = 401;
    int s_min_exp = 25;
    int s_max_exp = 50;
    int alpha_min = -1;
    int alpha_max = -1;
    int samples_per_alpha = 0;
    uint64_t seed = 20260710ULL;
    int device = 0;
    std::filesystem::path output =
        "data/hp1_cos_cdelta_hist/n100_samples10000_hist.csv";
    std::filesystem::path alpha_output =
        "data/hp1_cos_cdelta_hist/n100_samples10000_alpha.csv";
    std::filesystem::path offset_output =
        "data/hp1_cos_cdelta_hist/n100_samples10000_offset.csv";
    std::filesystem::path value_output =
        "data/hp1_cos_cdelta_hist/n100_samples10000_values.csv";
    std::filesystem::path sample_output =
        "data/hp1_cos_cdelta_hist/n100_samples10000_samples.csv";
};

struct SampleSummary {
    uint64_t s = 0;
    uint64_t q_lo = 0;
    uint64_t q_hi = 0;
    uint64_t qp_lo = 0;
    uint64_t qp_hi = 0;
    uint64_t qdiff_lo = 0;
    uint64_t qdiff_hi = 0;
    int v2_s = 0;
    int v2_qdiff = 0;
    int alpha = 0;
    int count_one = 0;
    int count_zero = 0;
    int count_minus_one = 0;
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
        } else if (key == "--samples") {
            args.samples = std::stoi(require_value(key));
        } else if (key == "--bins") {
            args.bins = std::stoi(require_value(key));
        } else if (key == "--s-min-exp") {
            args.s_min_exp = std::stoi(require_value(key));
        } else if (key == "--s-max-exp") {
            args.s_max_exp = std::stoi(require_value(key));
        } else if (key == "--alpha-min") {
            args.alpha_min = std::stoi(require_value(key));
        } else if (key == "--alpha-max") {
            args.alpha_max = std::stoi(require_value(key));
        } else if (key == "--samples-per-alpha") {
            args.samples_per_alpha = std::stoi(require_value(key));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value(key));
        } else if (key == "--device") {
            args.device = std::stoi(require_value(key));
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--alpha-output") {
            args.alpha_output = require_value(key);
        } else if (key == "--offset-output") {
            args.offset_output = require_value(key);
        } else if (key == "--value-output") {
            args.value_output = require_value(key);
        } else if (key == "--sample-output") {
            args.sample_output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: hp1_cos_cdelta_hist_cuda [--n 100] [--samples 10000] "
                << "[--bins 401] [--s-min-exp 25] [--s-max-exp 50] "
                << "[--alpha-min 0 --alpha-max 40 --samples-per-alpha 1000] "
                << "[--seed 20260710] [--device 0] [--output hist.csv] "
                << "[--alpha-output alpha.csv] [--offset-output offset.csv] "
                << "[--value-output values.csv] [--sample-output samples.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (args.n < 1 || args.n > kMaxN) {
        throw std::runtime_error("n must satisfy 1 <= n <= 120");
    }
    if (args.samples <= 0 || args.bins <= 1) {
        throw std::runtime_error("samples and bins must be positive");
    }
    if (args.s_min_exp < 0 || args.s_max_exp >= 63 || args.s_min_exp > args.s_max_exp) {
        throw std::runtime_error("s exponent range must satisfy 0 <= min <= max < 63");
    }
    if (args.s_max_exp >= args.n) {
        throw std::runtime_error("need s_max_exp < n");
    }
    const bool stratified = args.alpha_min >= 0 || args.alpha_max >= 0 || args.samples_per_alpha > 0;
    if (stratified) {
        if (args.alpha_min < 0 || args.alpha_max < args.alpha_min || args.alpha_max >= args.n) {
            throw std::runtime_error("stratified alpha range must satisfy 0 <= alpha_min <= alpha_max < n");
        }
        if (args.samples_per_alpha <= 0) {
            throw std::runtime_error("samples-per-alpha must be positive in stratified mode");
        }
        args.samples = (args.alpha_max - args.alpha_min + 1) * args.samples_per_alpha;
    }
    return args;
}

__host__ __device__ unsigned __int128 one_u128() {
    return static_cast<unsigned __int128>(1);
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

__device__ unsigned __int128 random_u128(uint64_t seed, uint64_t sample, uint64_t salt) {
    const uint64_t base =
        seed
        ^ (sample * 0x9e3779b97f4a7c15ULL)
        ^ (salt * 0xd1b54a32d192ed03ULL);
    const uint64_t lo = splitmix64(base);
    const uint64_t hi = splitmix64(base + 0xabc98388fb8fac03ULL);
    return (static_cast<unsigned __int128>(hi) << 64) | lo;
}

__device__ unsigned __int128 bounded_u128(
    uint64_t seed,
    uint64_t sample,
    uint64_t salt,
    unsigned __int128 bound
) {
    if (bound == 0) {
        return 0;
    }
    return random_u128(seed, sample, salt) % bound;
}

__device__ uint64_t sample_period_s(
    uint64_t seed,
    uint64_t sample,
    int s_min_exp,
    int s_max_exp
) {
    const uint64_t s_min = 1ULL << s_min_exp;
    const uint64_t s_max_exclusive = 1ULL << s_max_exp;
    const uint64_t width = s_max_exclusive - s_min;
    return s_min + bounded_u64(splitmix64(seed ^ (sample * 0x94d049bb133111ebULL)), width);
}

__device__ uint64_t sample_period_s_with_v2_limit(
    uint64_t seed,
    uint64_t sample,
    int s_min_exp,
    int s_max_exp,
    int max_v2
) {
    const int v2_limit = max_v2 < s_max_exp ? max_v2 : s_max_exp - 1;
    const int v2 = static_cast<int>(bounded_u64(splitmix64(seed ^ (sample * 0x8cb92ba72f3d8dd7ULL)), v2_limit + 1));
    const uint64_t lower = v2 < s_min_exp ? (1ULL << (s_min_exp - v2)) : 1ULL;
    const uint64_t upper = 1ULL << (s_max_exp - v2);
    uint64_t odd = lower + bounded_u64(splitmix64(seed ^ (sample * 0x7a7c159e3779b97fULL)), upper - lower);
    odd |= 1ULL;
    if (odd >= upper) {
        odd = upper - 1ULL;
    }
    return odd << v2;
}

__device__ unsigned __int128 period_count(int n, uint64_t s) {
    const unsigned __int128 n_states = one_u128() << n;
    return ((n_states - 1) / static_cast<unsigned __int128>(s)) + 1;
}

__device__ int ctz_u128(unsigned __int128 value) {
    if (value == 0) {
        return 128;
    }
    const uint64_t lo = static_cast<uint64_t>(value);
    if (lo != 0ULL) {
        return __ffsll(static_cast<long long>(lo)) - 1;
    }
    const uint64_t hi = static_cast<uint64_t>(value >> 64);
    return 64 + __ffsll(static_cast<long long>(hi)) - 1;
}

__device__ int bit_u128(unsigned __int128 value, int bit) {
    return static_cast<int>((value >> bit) & 1);
}

__global__ void hist_kernel(
    int n,
    int sample_count,
    int bins,
    int s_min_exp,
    int s_max_exp,
    int alpha_min,
    int samples_per_alpha,
    uint64_t seed,
    unsigned long long* hist,
    unsigned long long* alpha_sample_counts,
    unsigned long long* offset_counts,
    double* offset_sum_cos,
    unsigned long long* offset_near_one,
    unsigned long long* offset_near_zero,
    unsigned long long* value_counts,
    SampleSummary* sample_summaries
) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) {
        return;
    }

    const bool stratified = alpha_min >= 0;
    const int target_alpha =
        stratified ? alpha_min + (sample / samples_per_alpha) : -1;
    const uint64_t s =
        stratified
            ? sample_period_s_with_v2_limit(seed, sample, s_min_exp, s_max_exp, target_alpha)
            : sample_period_s(seed, sample, s_min_exp, s_max_exp);
    const unsigned __int128 r_s = period_count(n, s);
    unsigned __int128 q = 0;
    unsigned __int128 q_prime = 0;
    if (stratified) {
        const int v2_s = __ffsll(static_cast<long long>(s)) - 1;
        const int beta = target_alpha - v2_s;
        const unsigned __int128 max_odd = (r_s - 2) >> beta;
        const unsigned __int128 odd_count = (max_odd + 1) >> 1;
        const unsigned __int128 odd = 1 + 2 * bounded_u128(seed, sample, 3ULL, odd_count);
        const unsigned __int128 qdiff_target = odd << beta;
        q = 1 + bounded_u128(seed, sample, 4ULL, r_s - qdiff_target - 1);
        q_prime = q + qdiff_target;
    } else {
        q = 1 + bounded_u128(seed, sample, 1ULL, r_s - 1);
        q_prime = 1 + bounded_u128(seed, sample, 2ULL, r_s - 1);
        if (q_prime == q) {
            q_prime = q + 1;
            if (q_prime >= r_s) {
                q_prime = q - 1;
            }
        }
        if (q_prime < q) {
            const unsigned __int128 tmp = q;
            q = q_prime;
            q_prime = tmp;
        }
    }
    const unsigned __int128 qdiff = q_prime - q;
    const unsigned __int128 qs = q * static_cast<unsigned __int128>(s);
    const unsigned __int128 qps = q_prime * static_cast<unsigned __int128>(s);

    unsigned __int128 difference = 0;
    if (qs >= qps) {
        difference = qs - qps;
    } else {
        difference = qps - qs;
    }
    int alpha = ctz_u128(difference);
    if (alpha > n) {
        alpha = n;
    }
    atomicAdd(&alpha_sample_counts[alpha], 1ULL);
    const int v2_s = __ffsll(static_cast<long long>(s)) - 1;
    int v2_qdiff = ctz_u128(qdiff);
    if (v2_qdiff > n) {
        v2_qdiff = n;
    }

    int delta[kMaxN];
    for (int bit = 0; bit < n; ++bit) {
        delta[bit] = bit_u128(qs, bit) - bit_u128(qps, bit);
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    int sample_count_one = 0;
    int sample_count_zero = 0;
    int sample_count_minus_one = 0;
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
            atomicAdd(&value_counts[0], 1ULL);
            ++sample_count_one;
        }
        if (fabs(cos_value) < 1.0e-12) {
            atomicAdd(&value_counts[1], 1ULL);
            ++sample_count_zero;
        }
        if (cos_value < -1.0 + 1.0e-12) {
            atomicAdd(&value_counts[2], 1ULL);
            ++sample_count_minus_one;
        }

        int bin = static_cast<int>(floor(0.5 * (cos_value + 1.0) * bins));
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= bins) {
            bin = bins - 1;
        }
        atomicAdd(&hist[bin], 1ULL);

        const int offset = row - alpha;
        if (offset >= -kOffsetRadius && offset <= kOffsetRadius) {
            const int offset_index = offset + kOffsetRadius;
            atomicAdd(&offset_counts[offset_index], 1ULL);
            atomicAdd(&offset_sum_cos[offset_index], cos_value);
            if (cos_value > 0.999999) {
                atomicAdd(&offset_near_one[offset_index], 1ULL);
            }
            if (fabs(cos_value) < 1.0e-6) {
                atomicAdd(&offset_near_zero[offset_index], 1ULL);
            }
        }
    }

    sample_summaries[sample] = SampleSummary{
        s,
        static_cast<uint64_t>(q),
        static_cast<uint64_t>(q >> 64),
        static_cast<uint64_t>(q_prime),
        static_cast<uint64_t>(q_prime >> 64),
        static_cast<uint64_t>(qdiff),
        static_cast<uint64_t>(qdiff >> 64),
        v2_s,
        v2_qdiff,
        alpha,
        sample_count_one,
        sample_count_zero,
        sample_count_minus_one,
    };
}

void write_histogram(
    const Args& args,
    const std::vector<unsigned long long>& hist
) {
    std::filesystem::create_directories(args.output.parent_path());
    std::ofstream file(args.output);
    if (!file) {
        throw std::runtime_error("failed to open histogram output");
    }
    const double total = static_cast<double>(args.samples) * static_cast<double>(args.n);
    file << "bin_left,bin_right,bin_center,count,probability\n";
    file << std::setprecision(17);
    for (int bin = 0; bin < args.bins; ++bin) {
        const double left = -1.0 + 2.0 * static_cast<double>(bin) / args.bins;
        const double right = -1.0 + 2.0 * static_cast<double>(bin + 1) / args.bins;
        const double center = 0.5 * (left + right);
        const double probability = static_cast<double>(hist[bin]) / total;
        file << left << ',' << right << ',' << center << ','
             << hist[bin] << ',' << probability << '\n';
    }
}

void write_alpha_summary(
    const Args& args,
    const std::vector<unsigned long long>& alpha_counts
) {
    std::filesystem::create_directories(args.alpha_output.parent_path());
    std::ofstream file(args.alpha_output);
    if (!file) {
        throw std::runtime_error("failed to open alpha output");
    }
    file << "alpha,sample_count,probability\n";
    file << std::setprecision(17);
    for (int alpha = 0; alpha <= args.n; ++alpha) {
        const double probability =
            static_cast<double>(alpha_counts[alpha]) / static_cast<double>(args.samples);
        file << alpha << ',' << alpha_counts[alpha] << ',' << probability << '\n';
    }
}

void write_offset_summary(
    const Args& args,
    const std::vector<unsigned long long>& counts,
    const std::vector<double>& sum_cos,
    const std::vector<unsigned long long>& near_one,
    const std::vector<unsigned long long>& near_zero
) {
    std::filesystem::create_directories(args.offset_output.parent_path());
    std::ofstream file(args.offset_output);
    if (!file) {
        throw std::runtime_error("failed to open offset output");
    }
    file << "row_minus_alpha,count,mean_cos,prob_near_one,prob_near_zero\n";
    file << std::setprecision(17);
    for (int index = 0; index < kOffsetBins; ++index) {
        const int offset = index - kOffsetRadius;
        const double count = static_cast<double>(counts[index]);
        const double mean = count == 0.0 ? 0.0 : sum_cos[index] / count;
        const double p_one = count == 0.0 ? 0.0 : static_cast<double>(near_one[index]) / count;
        const double p_zero = count == 0.0 ? 0.0 : static_cast<double>(near_zero[index]) / count;
        file << offset << ',' << counts[index] << ',' << mean << ','
             << p_one << ',' << p_zero << '\n';
    }
}

void write_value_summary(
    const Args& args,
    const std::vector<unsigned long long>& counts
) {
    std::filesystem::create_directories(args.value_output.parent_path());
    std::ofstream file(args.value_output);
    if (!file) {
        throw std::runtime_error("failed to open value output");
    }
    const double total = static_cast<double>(args.samples) * static_cast<double>(args.n);
    file << "value,count,probability\n";
    file << std::setprecision(17);
    file << "1," << counts[0] << ',' << static_cast<double>(counts[0]) / total << '\n';
    file << "0," << counts[1] << ',' << static_cast<double>(counts[1]) / total << '\n';
    file << "-1," << counts[2] << ',' << static_cast<double>(counts[2]) / total << '\n';
}

std::string u128_hex(uint64_t hi, uint64_t lo) {
    std::ostringstream stream;
    stream << "0x";
    if (hi != 0ULL) {
        stream << std::hex << hi << std::setw(16) << std::setfill('0') << lo;
    } else {
        stream << std::hex << lo;
    }
    return stream.str();
}

void write_sample_summary(
    const Args& args,
    const std::vector<SampleSummary>& samples
) {
    std::filesystem::create_directories(args.sample_output.parent_path());
    std::ofstream file(args.sample_output);
    if (!file) {
        throw std::runtime_error("failed to open sample output");
    }
    file << "sample_id,s,q_hex,qprime_hex,qdiff_hex,v2_s,v2_qdiff,alpha,"
         << "count_one,count_zero,count_minus_one,p_one,p_zero,p_minus_one\n";
    file << std::setprecision(17);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const SampleSummary& row = samples[index];
        file << index << ','
             << row.s << ','
             << u128_hex(row.q_hi, row.q_lo) << ','
             << u128_hex(row.qp_hi, row.qp_lo) << ','
             << u128_hex(row.qdiff_hi, row.qdiff_lo) << ','
             << row.v2_s << ','
             << row.v2_qdiff << ','
             << row.alpha << ','
             << row.count_one << ','
             << row.count_zero << ','
             << row.count_minus_one << ','
             << static_cast<double>(row.count_one) / static_cast<double>(args.n) << ','
             << static_cast<double>(row.count_zero) / static_cast<double>(args.n) << ','
             << static_cast<double>(row.count_minus_one) / static_cast<double>(args.n)
             << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        check_cuda(cudaSetDevice(args.device), "cudaSetDevice");

        unsigned long long* d_hist = nullptr;
        unsigned long long* d_alpha_counts = nullptr;
        unsigned long long* d_offset_counts = nullptr;
        double* d_offset_sum_cos = nullptr;
        unsigned long long* d_offset_near_one = nullptr;
        unsigned long long* d_offset_near_zero = nullptr;
        unsigned long long* d_value_counts = nullptr;
        SampleSummary* d_sample_summaries = nullptr;

        check_cuda(cudaMalloc(&d_hist, sizeof(unsigned long long) * args.bins), "cudaMalloc hist");
        check_cuda(cudaMalloc(&d_alpha_counts, sizeof(unsigned long long) * (args.n + 1)), "cudaMalloc alpha");
        check_cuda(cudaMalloc(&d_offset_counts, sizeof(unsigned long long) * kOffsetBins), "cudaMalloc offset counts");
        check_cuda(cudaMalloc(&d_offset_sum_cos, sizeof(double) * kOffsetBins), "cudaMalloc offset sum");
        check_cuda(cudaMalloc(&d_offset_near_one, sizeof(unsigned long long) * kOffsetBins), "cudaMalloc near one");
        check_cuda(cudaMalloc(&d_offset_near_zero, sizeof(unsigned long long) * kOffsetBins), "cudaMalloc near zero");
        check_cuda(cudaMalloc(&d_value_counts, sizeof(unsigned long long) * 3), "cudaMalloc value counts");
        check_cuda(cudaMalloc(&d_sample_summaries, sizeof(SampleSummary) * args.samples), "cudaMalloc sample summaries");

        check_cuda(cudaMemset(d_hist, 0, sizeof(unsigned long long) * args.bins), "cudaMemset hist");
        check_cuda(cudaMemset(d_alpha_counts, 0, sizeof(unsigned long long) * (args.n + 1)), "cudaMemset alpha");
        check_cuda(cudaMemset(d_offset_counts, 0, sizeof(unsigned long long) * kOffsetBins), "cudaMemset offset counts");
        check_cuda(cudaMemset(d_offset_sum_cos, 0, sizeof(double) * kOffsetBins), "cudaMemset offset sum");
        check_cuda(cudaMemset(d_offset_near_one, 0, sizeof(unsigned long long) * kOffsetBins), "cudaMemset near one");
        check_cuda(cudaMemset(d_offset_near_zero, 0, sizeof(unsigned long long) * kOffsetBins), "cudaMemset near zero");
        check_cuda(cudaMemset(d_value_counts, 0, sizeof(unsigned long long) * 3), "cudaMemset value counts");

        const int block_count = (args.samples + kBlockSize - 1) / kBlockSize;
        hist_kernel<<<block_count, kBlockSize>>>(
            args.n,
            args.samples,
            args.bins,
            args.s_min_exp,
            args.s_max_exp,
            args.alpha_min,
            args.samples_per_alpha,
            args.seed,
            d_hist,
            d_alpha_counts,
            d_offset_counts,
            d_offset_sum_cos,
            d_offset_near_one,
            d_offset_near_zero,
            d_value_counts,
            d_sample_summaries
        );
        check_cuda(cudaGetLastError(), "hist_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "hist_kernel sync");

        std::vector<unsigned long long> hist(args.bins);
        std::vector<unsigned long long> alpha_counts(args.n + 1);
        std::vector<unsigned long long> offset_counts(kOffsetBins);
        std::vector<double> offset_sum_cos(kOffsetBins);
        std::vector<unsigned long long> offset_near_one(kOffsetBins);
        std::vector<unsigned long long> offset_near_zero(kOffsetBins);
        std::vector<unsigned long long> value_counts(3);
        std::vector<SampleSummary> sample_summaries(args.samples);

        check_cuda(cudaMemcpy(hist.data(), d_hist, sizeof(unsigned long long) * args.bins, cudaMemcpyDeviceToHost), "copy hist");
        check_cuda(cudaMemcpy(alpha_counts.data(), d_alpha_counts, sizeof(unsigned long long) * (args.n + 1), cudaMemcpyDeviceToHost), "copy alpha");
        check_cuda(cudaMemcpy(offset_counts.data(), d_offset_counts, sizeof(unsigned long long) * kOffsetBins, cudaMemcpyDeviceToHost), "copy offset counts");
        check_cuda(cudaMemcpy(offset_sum_cos.data(), d_offset_sum_cos, sizeof(double) * kOffsetBins, cudaMemcpyDeviceToHost), "copy offset sum");
        check_cuda(cudaMemcpy(offset_near_one.data(), d_offset_near_one, sizeof(unsigned long long) * kOffsetBins, cudaMemcpyDeviceToHost), "copy near one");
        check_cuda(cudaMemcpy(offset_near_zero.data(), d_offset_near_zero, sizeof(unsigned long long) * kOffsetBins, cudaMemcpyDeviceToHost), "copy near zero");
        check_cuda(cudaMemcpy(value_counts.data(), d_value_counts, sizeof(unsigned long long) * 3, cudaMemcpyDeviceToHost), "copy value counts");
        check_cuda(cudaMemcpy(sample_summaries.data(), d_sample_summaries, sizeof(SampleSummary) * args.samples, cudaMemcpyDeviceToHost), "copy sample summaries");

        write_histogram(args, hist);
        write_alpha_summary(args, alpha_counts);
        write_offset_summary(args, offset_counts, offset_sum_cos, offset_near_one, offset_near_zero);
        write_value_summary(args, value_counts);
        write_sample_summary(args, sample_summaries);

        cudaFree(d_hist);
        cudaFree(d_alpha_counts);
        cudaFree(d_offset_counts);
        cudaFree(d_offset_sum_cos);
        cudaFree(d_offset_near_one);
        cudaFree(d_offset_near_zero);
        cudaFree(d_value_counts);
        cudaFree(d_sample_summaries);

        std::cout << "wrote " << args.output << '\n'
                  << "wrote " << args.alpha_output << '\n'
                  << "wrote " << args.offset_output << '\n'
                  << "wrote " << args.value_output << '\n'
                  << "wrote " << args.sample_output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

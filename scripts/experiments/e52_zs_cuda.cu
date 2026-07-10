#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    int n_min = 12;
    int n_max = 30;
    int x_count = 100;
    int s_count = 30;
    uint64_t seed = 20260710;
    std::string devices = "6,7";
    std::filesystem::path output = "data/e52_zs_cuda/e52_zs_y.csv";
};

struct ResultRow {
    int n = 0;
    int x_index = 0;
    uint32_t x = 0;
    int s_index = 0;
    uint32_t s = 0;
    uint64_t r_count = 0;
    double y = 0.0;
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

        if (key == "--n-min") {
            args.n_min = std::stoi(require_value(key));
        } else if (key == "--n-max") {
            args.n_max = std::stoi(require_value(key));
        } else if (key == "--x-count") {
            args.x_count = std::stoi(require_value(key));
        } else if (key == "--s-count") {
            args.s_count = std::stoi(require_value(key));
        } else if (key == "--seed") {
            args.seed = std::stoull(require_value(key));
        } else if (key == "--devices") {
            args.devices = require_value(key);
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--help") {
            std::cout
                << "Usage: e52_zs_cuda [--n-min 12] [--n-max 30] "
                << "[--x-count 100] [--s-count 30] [--seed 20260710] "
                << "[--devices 6,7|all] [--output data/e52_zs_cuda/e52_zs_y.csv]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (args.n_min < 1 || args.n_max > kMaxN || args.n_min > args.n_max) {
        throw std::runtime_error("n range must satisfy 1 <= n_min <= n_max <= 30");
    }
    if (args.x_count <= 0 || args.s_count <= 0) {
        throw std::runtime_error("x-count and s-count must be positive");
    }
    return args;
}

std::vector<uint32_t> sample_x_values(int n, int count, uint64_t seed) {
    const uint32_t upper = (uint32_t(1) << n) - 1U;
    if (count > static_cast<int>(upper)) {
        throw std::runtime_error("x-count exceeds nonzero output strings");
    }

    std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(n)));
    std::uniform_int_distribution<uint32_t> distribution(1U, upper);
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> values;
    values.reserve(count);
    while (static_cast<int>(values.size()) < count) {
        const uint32_t value = distribution(rng);
        if (seen.insert(value).second) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<uint32_t> sample_s_values(int n, int count, uint64_t seed) {
    const uint64_t n_states = uint64_t(1) << n;
    const uint32_t root = static_cast<uint32_t>(std::sqrt(static_cast<double>(n_states)));
    const uint32_t lower = std::max<uint32_t>(3U, root > 4U * uint32_t(count) ? root - 4U * uint32_t(count) : 3U);
    std::vector<uint32_t> candidates;
    for (uint32_t value = lower; value <= root; ++value) {
        if ((value & 1U) != 0U) {
            candidates.push_back(value);
        }
    }
    if (static_cast<int>(candidates.size()) < count) {
        throw std::runtime_error("not enough odd s candidates near sqrt(2^n)");
    }

    std::mt19937_64 rng(seed ^ (0x517cc1b727220a95ULL + static_cast<uint64_t>(n)));
    std::shuffle(candidates.begin(), candidates.end(), rng);
    candidates.resize(count);
    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

uint64_t period_count(int n, uint32_t s) {
    const uint64_t n_states = uint64_t(1) << n;
    return ((n_states - 1ULL) / static_cast<uint64_t>(s)) + 1ULL;
}

__global__ void compute_y_kernel(
    const uint32_t* x_values,
    int x_count,
    int n,
    uint32_t s,
    uint64_t r_count,
    double* y_values
) {
    const int x_index = blockIdx.x;
    if (x_index >= x_count) {
        return;
    }

    __shared__ double weights[kMaxN];
    __shared__ double re_shared[kBlockSize];
    __shared__ double im_shared[kBlockSize];

    const uint32_t x = x_values[x_index];
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        double weight = static_cast<double>((x >> j) & 1U);
        if ((j & 1) == 1) {
            for (int i = 0; i < n; i += 2) {
                const int distance = abs(i - j);
                if ((distance & 1) == 1) {
                    const uint32_t x_bit = (x >> i) & 1U;
                    weight += static_cast<double>(x_bit) / static_cast<double>(uint32_t(1) << distance);
                }
            }
        }
        weights[j] = weight;
    }
    __syncthreads();

    double local_re = 0.0;
    double local_im = 0.0;
    for (uint64_t q = threadIdx.x; q < r_count; q += blockDim.x) {
        uint32_t bits = static_cast<uint32_t>(q * static_cast<uint64_t>(s));
        double phase = 0.0;
        while (bits != 0U) {
            const int bit = __ffs(bits) - 1;
            phase += weights[bit];
            bits &= bits - 1U;
        }
        double sin_value = 0.0;
        double cos_value = 0.0;
        sincos(M_PI * phase, &sin_value, &cos_value);
        local_re += cos_value;
        local_im += sin_value;
    }

    re_shared[threadIdx.x] = local_re;
    im_shared[threadIdx.x] = local_im;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            re_shared[threadIdx.x] += re_shared[threadIdx.x + stride];
            im_shared[threadIdx.x] += im_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const double re = re_shared[0];
        const double im = im_shared[0];
        y_values[x_index] = (re * re + im * im) / static_cast<double>(r_count);
    }
}

std::vector<ResultRow> run_n_on_device(
    int device,
    const std::vector<int>& n_values,
    const Args& args
) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cout << "device=" << device << " name=" << properties.name << " n_values=";
        for (int n : n_values) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    }

    std::vector<ResultRow> rows;
    for (int n : n_values) {
        const auto started = std::chrono::steady_clock::now();
        const std::vector<uint32_t> x_values = sample_x_values(n, args.x_count, args.seed);
        const std::vector<uint32_t> s_values = sample_s_values(n, args.s_count, args.seed);

        uint32_t* d_x_values = nullptr;
        double* d_y_values = nullptr;
        check_cuda(cudaMalloc(&d_x_values, sizeof(uint32_t) * x_values.size()), "cudaMalloc x");
        check_cuda(cudaMalloc(&d_y_values, sizeof(double) * x_values.size()), "cudaMalloc y");
        check_cuda(
            cudaMemcpy(
                d_x_values,
                x_values.data(),
                sizeof(uint32_t) * x_values.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy x"
        );

        std::vector<double> y_values(x_values.size(), 0.0);
        for (int s_index = 0; s_index < static_cast<int>(s_values.size()); ++s_index) {
            const uint32_t s = s_values[s_index];
            const uint64_t r_count = period_count(n, s);
            compute_y_kernel<<<args.x_count, kBlockSize>>>(
                d_x_values,
                args.x_count,
                n,
                s,
                r_count,
                d_y_values
            );
            check_cuda(cudaGetLastError(), "compute_y_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "compute_y_kernel sync");
            check_cuda(
                cudaMemcpy(
                    y_values.data(),
                    d_y_values,
                    sizeof(double) * y_values.size(),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy y"
            );

            for (int x_index = 0; x_index < static_cast<int>(x_values.size()); ++x_index) {
                rows.push_back(
                    ResultRow{
                        n,
                        x_index,
                        x_values[x_index],
                        s_index,
                        s,
                        r_count,
                        y_values[x_index],
                    }
                );
            }
        }

        check_cuda(cudaFree(d_x_values), "cudaFree x");
        check_cuda(cudaFree(d_y_values), "cudaFree y");

        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "done n=" << n << " device=" << device
                      << " rows=" << args.x_count * args.s_count
                      << " seconds=" << std::fixed << std::setprecision(3)
                      << seconds << std::endl;
        }
    }

    return rows;
}

void write_csv(const std::filesystem::path& path, std::vector<ResultRow> rows) {
    std::sort(rows.begin(), rows.end(), [](const ResultRow& left, const ResultRow& right) {
        if (left.n != right.n) {
            return left.n < right.n;
        }
        if (left.s_index != right.s_index) {
            return left.s_index < right.s_index;
        }
        return left.x_index < right.x_index;
    });

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path.string());
    }

    output << "n,x_index,x,s_index,s,R,y\n";
    output << std::setprecision(17);
    for (const ResultRow& row : rows) {
        output << row.n << ','
               << row.x_index << ','
               << row.x << ','
               << row.s_index << ','
               << row.s << ','
               << row.r_count << ','
               << row.y << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<int> devices = parse_devices(args.devices);
        std::vector<std::vector<int>> n_by_device(devices.size());
        for (int n = args.n_min; n <= args.n_max; ++n) {
            n_by_device[static_cast<size_t>(n - args.n_min) % devices.size()].push_back(n);
        }

        const auto started = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        std::vector<std::vector<ResultRow>> worker_rows(devices.size());
        for (size_t index = 0; index < devices.size(); ++index) {
            threads.emplace_back([&, index]() {
                worker_rows[index] = run_n_on_device(devices[index], n_by_device[index], args);
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
        std::cout << "output=" << args.output << std::endl;
        std::cout << "elapsed_seconds=" << std::fixed << std::setprecision(3)
                  << seconds << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

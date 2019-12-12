#include <iostream>
#include <ctime>
#include <chrono>
#include <functional>
#include <cmath>
#include "CppFlow.hpp"
#include "TfLite.hpp"
#include "OpenCV.hpp"

template<typename T>
auto generateData(std::size_t len) -> std::vector<T> {
    std::vector<T> ret;

    for (std::size_t c = 0; c < len; ++c) {
        ret.emplace_back(static_cast<T>(rand()));
    }

    return ret;
}

auto generateMat(const cv::Size& size, int type) -> cv::Mat {
    cv::Mat ret{size, type};
    auto maxIndex = static_cast<std::size_t>(size.area() * ret.channels());
    for (std::size_t c = 0; c < maxIndex; ++c) {
        ret.data[c] = static_cast<uchar>(rand());
    }
    return ret;
}

template<typename T, typename F>
void randomInputBenchmark(T &t, F f, const std::string &name) {
    using namespace std::chrono;
    constexpr auto TRIES = 1000;

    for (std::size_t c = 0; c < 10; ++c) {
        t.forward(f());
    }

    std::vector<double> times;
    times.resize(TRIES);
    for (auto &time : times) {
        auto data = f();
        auto start = high_resolution_clock::now();

        t.forward(data);

        auto end = high_resolution_clock::now();
        auto dur = duration_cast<duration<double>>(end - start);

        time = dur.count() * 1000;
    }

    auto mean = 0.0;
    auto min = std::numeric_limits<double>::infinity();
    auto max = -std::numeric_limits<double>::infinity();
    for (const auto &time : times) {
        mean += time;
        min = std::min(min, time);
        max = std::max(max, time);
    }
    mean /= times.size();

    auto stdDev = 0.0;
    for (const auto &time : times) {
        stdDev += std::pow(time - mean, 2);
    }
    stdDev /= static_cast<double>(times.size() - 1);
    stdDev = std::sqrt(stdDev);

    std::cout << "| " << name << " | " << mean << " | " << stdDev <<
              " | " << min << " | " << max << " |" << std::endl;
}

void testCppFlowSemSeg() {
    CppFlow cppFlow{"semseg.pb", "conv2d_input", "output/Softmax", 128, 1};
    randomInputBenchmark(cppFlow, std::bind(generateData<float>, 128 * 128),
                         "CppFlow");
}

void testTfliteSemSeg(int numOfThreads) {
    TfLite tfLite1{"semseg.tflite", numOfThreads, 128, 1};
    randomInputBenchmark(tfLite1, std::bind(generateData<float>, 128 * 128),
                         "TfLite (" + std::to_string(numOfThreads) + " Threads)");
}

void testOpenCVSemSeg() {
    OpenCV openCv{"semseg.pb"};
    randomInputBenchmark(openCv,
                         std::bind(generateMat, cv::Size{128, 128}, CV_8UC1),
                         "OpenCV");
}

void testCppFlowClass() {
    CppFlow cppFlow{"class.pb", "input", "output", 80, 3};
    randomInputBenchmark(cppFlow, std::bind(generateData<float>, 80 * 80 * 3),
                         "CppFlow");
}

void testTfliteClass(int numOfThreads) {
    TfLite tfLite1{"class.tflite", numOfThreads, 80, 3};
    randomInputBenchmark(tfLite1, std::bind(generateData<float>, 80 * 80 * 3),
                         "TfLite (" + std::to_string(numOfThreads) + " Threads)");
}

void testOpenCVClass() {
    OpenCV openCv{"class.pb"};
    randomInputBenchmark(openCv,
                         std::bind(generateMat, cv::Size{80, 80}, CV_8UC3),
                         "OpenCV");
}

int main() {
    std::srand(std::time(nullptr));

    std::cout << "SemSeg" << std::endl;

    testOpenCVSemSeg();
    testCppFlowSemSeg();
    for (auto c=1; c<=8; ++c) {
        testTfliteSemSeg(c);
    }

    std::cout << std::endl << "Class" << std::endl;
    testOpenCVClass();
    testCppFlowClass();
    for (auto c=1; c<=8; ++c) {
        testTfliteClass(c);
    }

    return 0;
}

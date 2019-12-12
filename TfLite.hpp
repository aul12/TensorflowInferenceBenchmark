/**
 * @file TfLite.hpp
 * @author paul
 * @date 01.12.19
 * @brief TfLite @TODO
 */

#ifndef TFBENCHMARK_TFLITE_HPP
#define TFBENCHMARK_TFLITE_HPP


#include <string>
#include <memory>
#include <vector>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>

class TfLite {
public:
    explicit TfLite(const std::string &fname, int numOfThreads, int size, int channels);

    auto forward(const std::vector<float> &data) -> const float *;
private:
    std::vector<int64_t> tensorDims;

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};


#endif //TFBENCHMARK_TFLITE_HPP

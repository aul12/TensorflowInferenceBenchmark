/**
 * @file TfLite.cpp
 * @author paul
 * @date 01.12.19
 * @brief TfLite @TODO
 */

#include <iostream>
#include "TfLite.hpp"

TfLite::TfLite(const std::string &fname, int numOfThreads, int size, int channels) : tensorDims{1,size, size,channels},
    model{tflite::FlatBufferModel::BuildFromFile(fname.c_str())} {

    tflite::ops::builtin::BuiltinOpResolver resolver;

    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
    }

    interpreter->SetNumThreads(numOfThreads);

    interpreter->AllocateTensors();

    if (model == nullptr) {
        throw std::runtime_error{"Failed to build FlatBufferModel from file"};
    }

    interpreter->Invoke();
}

auto TfLite::forward(const std::vector<float> &) -> const float * {
    interpreter->Invoke();

    return nullptr;
}

/**
 * @file CppFlow.hpp
 * @author paul
 * @date 01.12.19
 * @brief CppFlow @TODO
 */

#ifndef TFBENCHMARK_CPPFLOW_HPP
#define TFBENCHMARK_CPPFLOW_HPP

#include <string>
#include <CppFlow/Model.h>

class CppFlow {
public:
    CppFlow(const std::string &fname,
            const std::string &inputName, const std::string &outputName,
            int size, int channels);

    auto forward(const std::vector<float> &data) -> std::vector<float>;

private:
    Model model;
    Tensor input, output;
    std::vector<int64_t> tensorDims;
    int64_t inputSize;
};


#endif //TFBENCHMARK_CPPFLOW_HPP

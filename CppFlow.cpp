/**
 * @file CppFlow.cpp
 * @author paul
 * @date 01.12.19
 * @brief CppFlow @TODO
 */

#include "CppFlow.hpp"

CppFlow::CppFlow(const std::string &fname, const std::string &inputName,
                 const std::string &outputName, int size, int channels) :
        model{fname}, input{model, inputName}, output{model, outputName},
        tensorDims{1, size, size, channels},inputSize{tensorDims[0] * tensorDims[1] * tensorDims[2] * tensorDims[3]} {

}

auto CppFlow::forward(const std::vector<float> &data) -> std::vector<float> {
    input.set_data(data, tensorDims);
    model.run(input, output);
    return output.get_data<float>();
}

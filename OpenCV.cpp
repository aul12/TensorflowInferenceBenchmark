/**
 * @file OpenCV.cpp
 * @author paul
 * @date 01.12.19
 * @brief OpenCV @TODO
 */

#include "OpenCV.hpp"

OpenCV::OpenCV(const std::string &fname) : net{cv::dnn::readNetFromTensorflow(fname)} {}

auto OpenCV::forward(const cv::Mat &data) -> cv::Mat {
    auto input = cv::dnn::blobFromImage(data, 1.0, data.size(), 0.0, true, CV_32F);
    net.setInput(input);

    return net.forward();
}

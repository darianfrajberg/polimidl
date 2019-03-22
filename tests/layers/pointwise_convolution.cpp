#include <polimidl/network.hpp>
#include <polimidl/layers/pointwise_convolution.hpp>
#include <polimidl/layers/alignment.hpp>
#include <iostream>
#include "../utility/comparator.hpp"
#include "../utility/copy.hpp"

int test(unsigned int number_of_workers){
  using polimidl::build_network;
  using polimidl::layers::components;
  using polimidl::layers::pointwise_convolution;
  using polimidl::testing::copy;
  using polimidl::testing::is_close;
  using polimidl::layers::buffer_alignment;
  const std::size_t h = 2;
  const std::size_t w = 2;
  constexpr std::size_t channels = 3;
  const float input[] = {0,4,8,1,5,9,2,6,10,3,7,11}; //2x2x3

  {
    alignas(buffer_alignment::byte_alignment) constexpr float weights[] = {0,1,2};
    constexpr float expected_output[] = {20,23,26,29};
    auto net = build_network<float>(h, w, number_of_workers,
        pointwise_convolution<float, components<channels, 1>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for pointwise convolution with 1 output channel: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for pointwise convolution with 1 output channel: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != 1) {
      std::cerr <<
          "Invalid output shape for pointwise convolution with 1 output channel: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for pointwise convolution with 1 output channel: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr <<
          "Invalid output value for pointwise convolution with 1 output channel" <<
          std::endl;
      return -1;
    }
  }

  {
    alignas(buffer_alignment::byte_alignment) constexpr float weights[] = {0,1,2};
    constexpr float expected_output[] = {20,23,26,29};
    auto net = build_network<float>(h, w, number_of_workers,
      copy<float, components<channels>>(),
      pointwise_convolution<float, components<channels, 1>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for inverted pointwise convolution with 1 output channel: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for inverted pointwise convolution with 1 output channel: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2  || net.output_components() != 1) {
      std::cerr <<
          "Invalid output shape for inverted pointwise convolution with 1 output channel" <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for inverted pointwise convolution with 1 output channel: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr <<
          "Invalid output value for inverted pointwise convolution with 1 output channel" <<
          std::endl;
      return -1;
    }
  }

  {
    alignas(buffer_alignment::byte_alignment) constexpr float weights[] = {0,1,2,3,4,5};
    constexpr float expected_output[] = {20,56,23,68,26,80,29,92};
    auto net = build_network<float>(h, w, number_of_workers,
        pointwise_convolution<float, components<3, 2>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for pointwise convolution with 2 output channels: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for pointwise convolution with 2 output channels: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != 2) {
      std::cerr <<
          "Invalid output shape for pointwise convolution with 2 output channels: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for pointwise convolution with 2 output channels: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr <<
          "Invalid output value for pointwise convolution with 2 output channels" <<
          std::endl;
      return -1;
    }
  }

  {
    alignas(buffer_alignment::byte_alignment) constexpr float weights[] = {0,1,2,3,4,5};
    constexpr float expected_output[] = {20,56,23,68,26,80,29,92};
    auto net = build_network<float>(h, w, number_of_workers,
      copy<float, components<3>>(),
      pointwise_convolution<float, components<3, 2>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for inverted pointwise convolution with 2 output channels: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for inverted pointwise convolution with 2 output channels: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != 2) {
      std::cerr <<
          "Invalid output shape for inverted pointwise convolution with 2 output channels: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for inverted pointwise convolution with 2 output channels: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr <<
          "Invalid output value for inverted pointwise convolution with 2 output channels" <<
          std::endl;
      return -1;
    }
  }

  {
    alignas(buffer_alignment::byte_alignment) constexpr float weights[] = {0,1,2,3,4,5,6,7,8};
    constexpr float expected_output[] = {20,56,92,23,68,113,26,80,134,29,92,155};
    auto net = build_network<float>(h, w, number_of_workers,
        pointwise_convolution<float, components<channels, 3>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for pointwise convolution with 3 output channels: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for pointwise convolution with 3 output channels: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != 3) {
      std::cerr <<
          "Invalid output shape for pointwise convolution with 3 output channels:" <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for pointwise convolution with 3 output channels: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr <<
          "Invalid output value for pointwise convolution with 3 output channels" <<
          std::endl;
      return -1;
    }
  }

  {
    alignas(buffer_alignment::byte_alignment) constexpr float weights[] = {0,1,2,3,4,5,6,7,8};
    constexpr float expected_output[] = {20,56,92,23,68,113,26,80,134,29,92,155};
    auto net = build_network<float>(h, w, number_of_workers,
      copy<float, components<channels>>(),
      pointwise_convolution<float, components<channels, 3>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for inverted pointwise convolution with 3 output channels: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for inverted pointwise convolution with 3 output channels: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != 3) {
      std::cerr <<
          "Invalid output shape for inverted pointwise convolution with 3 output channels: " <<
          net.output_rows() << "x" << net.output_columns() <<  "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for inverted pointwise convolution with 3 output channels: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr <<
          "Invalid output value for inverted pointwise convolution with 3 output channels" <<
          std::endl;
      return -1;
    }
  }
  return 0;
}

int main() {
    using polimidl::suggested_number_of_workers;
    int value = test(1);
    if (value) return value;
    value = test(suggested_number_of_workers());
    if (value) return value;
    std::cout << "Passed test" << std::endl;
    return 0;
}

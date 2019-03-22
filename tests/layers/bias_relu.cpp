#include <polimidl/network.hpp>
#include <polimidl/layers/bias_relu.hpp>
#include <polimidl/layers/alignment.hpp>
#include <iostream>
#include "../utility/comparator.hpp"

int test(unsigned int number_of_workers){
    using polimidl::build_network;
    using polimidl::layers::bias_relu;
    using polimidl::layers::components;
    using polimidl::testing::is_close;
    using polimidl::layers::buffer_alignment;
    constexpr std::size_t h = 2;
    constexpr std::size_t w = 2;
    constexpr std::size_t channels = 3;
    const float input[] = {0,4,8,1,5,9,2,6,10,3,7,11}; //2x2x3
    alignas(buffer_alignment::byte_alignment) const float biases[] = {-2,3,-4}; //1x1x3

    {
        constexpr float expected_output[] = {0,7,4,0,8,5,0,9,6,1,10,7};

        auto net = build_network<float>(h, w, number_of_workers,
            bias_relu<float, components<channels>>(biases)
        );
        if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
            std::cerr << "Invalid input shape for bias + relu: " <<
            net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
            return -1;
        }
        if (net.input().size() != std::size(input)) {
            std::cerr << "Invalid input size for bias + relu: " <<
            net.input().size() << std::endl;
            return -1;
        }
        if (net.output_rows() != h || net.output_columns() != w || net.output_components() != channels) {
            std::cerr << "Invalid output shape for bias + relu: " <<
            net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
            return -1;
        }
        std::copy(std::begin(input), std::end(input), net.input().begin());
        auto output = net();
        if (output.size() != std::size(expected_output)) {
            std::cerr << "Invalid output size for bias + relu: " <<
            output.size() << std::endl;
            return -1;
        }
        if (!std::equal(output.begin(), output.end(),
                        std::begin(expected_output), std::end(expected_output),
                        is_close)){
            std::cerr << "Invalid output value for bias + relu" <<
            std::endl;
            return -1;
        }
    }

    {
        constexpr float expected_output[] = {0,6,4,0,6,5,0,6,6,1,6,6};

        auto net = build_network<float>(h, w, number_of_workers,
            bias_relu<float, components<channels>>(biases, /* min */ 0, /* max */ 6)
        );
        if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
            std::cerr << "Invalid input shape for bias + relu6: " <<
            net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
            return -1;
        }
        if (net.input().size() != std::size(input)) {
            std::cerr << "Invalid input size for bias + relu6: " <<
            net.input().size() << std::endl;
            return -1;
        }
        if (net.output_rows() != h || net.output_columns() != w || net.output_components() != channels) {
            std::cerr << "Invalid output shape for bias + relu6: " <<
            net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
            return -1;
        }
        std::copy(std::begin(input), std::end(input), net.input().begin());
        auto output = net();
        if (output.size() != std::size(expected_output)) {
            std::cerr << "Invalid output size for bias + relu6: " <<
            output.size() << std::endl;
            return -1;
        }
        if (!std::equal(output.begin(), output.end(),
                        std::begin(expected_output), std::end(expected_output),
                        is_close)){
            std::cerr << "Invalid output value for bias + relu6" <<
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

#ifndef POLIMIDL_LAYERS_INTERNAL_POINTWISE_CONVOLUTION_HPP
#define POLIMIDL_LAYERS_INTERNAL_POINTWISE_CONVOLUTION_HPP

#include <algorithm>
#include <chrono>
#include <iterator>
#include <iostream>

#include <Eigen/Dense>

#include "../alignment.hpp"
#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class pointwise_convolution : public layer<type_t, components> {
 public:
  pointwise_convolution(const type_t* coeff) :
      coeff_(coeff),
      batch_size_(1) {}

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const std::size_t cells = rows * columns;
    for (std::size_t start_cell = 0; start_cell < cells;
         start_cell += batch_size_) {
      const std::size_t end_cell = std::min(cells, start_cell + batch_size_);
      scheduler.schedule([=] (std::size_t worker) {
        const std::size_t cells_in_this_batch = end_cell - start_cell;
        auto i_input = &input[components::input * start_cell];
        auto i_output = &output[components::output * start_cell];
        matrix_input_t val(i_input, components::input, cells_in_this_batch);
        matrix_output_t out(i_output, components::output, cells_in_this_batch);
        out.noalias() = coeff_ * val;
      });
    }
    scheduler.wait();
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temporary, output_t output,
                    std::size_t rows, std::size_t columns,
                    const scheduler_t& scheduler) {
    const std::size_t cells = rows * columns;
    if (scheduler.number_of_workers() == 1) {
      batch_size_ = cells;
      return;
    }

    auto best_duration = std::chrono::high_resolution_clock::duration::max();

    std::size_t best_batch_size = 1;
    auto test_and_maybe_set = [&]() {
      for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        operator()(input, temporary, output, rows, columns, scheduler);
        const auto current_duration =
            std::chrono::high_resolution_clock::now() - start;
        if (current_duration < best_duration) {
          best_duration = current_duration;
          best_batch_size = batch_size_;
        }
      }
    };
    batch_size_ = 1;
    while (batch_size_ < cells) {
      // Ensure the alignment of all the Eigen Matrixes.
      if ((batch_size_ * components::input) % alignment_t::type_t_alignment ||
          (batch_size_ * components::output) % alignment_t::type_t_alignment) {
        batch_size_ *= 2;
        continue;
      }
      test_and_maybe_set();
      batch_size_ *= 2;
    }
    batch_size_ = cells;
    test_and_maybe_set();
    batch_size_ = best_batch_size;
  }
 private:
  using alignment_t = alignment<type_t>;
  using matrix_input_t = Eigen::Map<
      Eigen::Matrix<type_t, components::input, Eigen::Dynamic>,
      alignment_t::eigen_alignment>;
  using matrix_output_t = Eigen::Map<
      Eigen::Matrix<type_t, components::output, Eigen::Dynamic>,
      alignment_t::eigen_alignment>;
  using matrix_coefficients_t = Eigen::Map<
      const Eigen::Matrix<type_t, components::output, components::input,
                          Eigen::RowMajor>,
      alignment_t::eigen_alignment>;

  matrix_coefficients_t coeff_;
  std::size_t batch_size_;
};
}
}
}

#endif  // POLIMIDL_LAYERS_INTERNAL_POINTWISE_CONVOLUTION_HPP

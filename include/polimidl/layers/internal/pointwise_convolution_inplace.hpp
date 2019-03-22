#ifndef POLIMIDL_LAYERS_INTERNAL_POINTWISE_CONVOLUTION_INPLACE_HPP
#define POLIMIDL_LAYERS_INTERNAL_POINTWISE_CONVOLUTION_INPLACE_HPP

#include <algorithm>
#include <chrono>
#include <iterator>

#include <Eigen/Dense>

#include "../alignment.hpp"
#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class pointwise_convolution_inplace : public layer<type_t, components, true> {
 public:
  pointwise_convolution_inplace(const type_t* coeff) :
      coeff_(coeff),
      batch_size_(1) {}

  static std::size_t temporary_size(std::size_t rows, std::size_t columns,
                                    std::size_t number_of_workers) {
    const std::size_t cells = rows * columns;
    std::size_t batch_size = 1;
    // Ensure the alignment of the input Eigen Matrix.
    while ((batch_size * components::input) % alignment_t::type_t_alignment ||
           batch_size < std::max(rows, columns) / 2) {
      batch_size *= 2;
    }
    if (batch_size > cells) {
      batch_size = cells;
    }
    return batch_size * components::output * number_of_workers;
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const std::size_t cells = rows * columns;
    const unsigned int number_of_workers = scheduler.number_of_workers();
    for (std::size_t start_cell = 0; start_cell < cells;
         start_cell += batch_size_) {
      const std::size_t end_cell = std::min(cells, start_cell + batch_size_);
      scheduler.schedule([=] (std::size_t worker) {
        const std::size_t cells_in_this_batch = end_cell - start_cell;
        auto temporary_worker = temporary.slice(worker, number_of_workers);
        auto i_input = &input[components::input * start_cell];
        matrix_input_t val(i_input, components::input, cells_in_this_batch);
        auto i_temp = &temporary_worker[0];
        matrix_output_t out(i_temp, components::output, cells_in_this_batch);
        out.noalias() = coeff_ * val;
        std::copy_n(i_temp, components::output * cells_in_this_batch, i_input);
      });
    }
    scheduler.wait();
    if constexpr (components::are_reducing) {
      if (input.begin() == output.begin() ) { // beginning of the buffer
        auto i_input = input.begin();
        auto i_output = output.begin();
        for (std::size_t start_cell = 0; start_cell < cells;
             start_cell += batch_size_) {
          const std::size_t end_cell =
              std::min(cells, start_cell + batch_size_);
          const std::size_t cells_in_this_batch = end_cell - start_cell;
          const std::size_t input_batch_size =
              components::input * cells_in_this_batch;
          const std::size_t output_batch_size =
              components::output * cells_in_this_batch;
          i_output = std::copy_n(i_input, output_batch_size, i_output);
          i_input += input_batch_size;
        }
      } else { // end of the buffer
        auto i_input = std::reverse_iterator(input.end());
        auto i_output = std::reverse_iterator(output.end());
        const std::size_t n_batches = cells / batch_size_ + 1;
        std::size_t cells_in_this_batch = cells % batch_size_;
        for (std::size_t batch = 0; batch < n_batches; ++batch) {
          const std::size_t input_batch_size =
              components::input * cells_in_this_batch;
          const std::size_t output_batch_size =
              components::output * cells_in_this_batch;
          i_input += input_batch_size - output_batch_size;
          i_output = std::copy_n(i_input, output_batch_size, i_output);
          i_input += output_batch_size;
          cells_in_this_batch = batch_size_;
        }
      }
    }
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temporary, output_t output,
                    std::size_t rows, std::size_t columns,
                    const scheduler_t& scheduler) {
    const std::size_t cells = rows * columns;
    const std::size_t worker_temp_size =
        temporary.slice_size(scheduler.number_of_workers());
    const std::size_t max_batch_size =
        std::min(cells, worker_temp_size / components::output);

    auto best_duration = std::chrono::high_resolution_clock::duration::max();

    std::size_t best_batch_size = 1;
    auto test_and_maybe_set = [&]() {
      for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        operator()(input, temporary, output, rows, columns,
                   scheduler);
        const auto current_duration =
            std::chrono::high_resolution_clock::now() - start;
        if (current_duration < best_duration) {
          best_duration = current_duration;
          best_batch_size = batch_size_;
        }
      }
    };
    batch_size_ = 1;
    // Ensure the alignment of the output Eigen Matrix.
    while (batch_size_ <= max_batch_size) {
      if ((batch_size_ * components::input) % alignment_t::type_t_alignment) {
        batch_size_ *= 2;
        continue;
      }
      test_and_maybe_set();
      batch_size_ *= 2;
    }
    if (cells == max_batch_size) {
      batch_size_ = cells;
      test_and_maybe_set();
    }
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

#endif  // POLIMIDL_LAYERS_INTERNAL_POINTWISE_CONVOLUTION_INPLACE_HPP

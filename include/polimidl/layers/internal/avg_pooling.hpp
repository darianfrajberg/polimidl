#ifndef POLIMIDL_LAYERS_INTERNAL_AVG_POOLING_HPP
#define POLIMIDL_LAYERS_INTERNAL_AVG_POOLING_HPP

#include <chrono>

#include <Eigen/Dense>

#include "../alignment.hpp"
#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components, typename kernel,
          typename stride>
class avg_pooling : public layer<type_t, components> {
 public:
  avg_pooling() : batch_size_(1) {}

  static std::size_t output_rows(std::size_t input_rows) {
    return ((input_rows - kernel::rows) / stride::rows) + 1;
  }
  static std::size_t output_columns(std::size_t input_columns) {
    return ((input_columns - kernel::columns) / stride::columns) + 1;
  }
  static std::size_t temporary_size(std::size_t input_rows,
                                    std::size_t input_columns,
                                    unsigned int number_of_workers) {
     return cell_size * number_of_workers;
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t input_rows, std::size_t input_columns,
                  const scheduler_t& scheduler) const {
    if (input_rows == kernel::rows && input_columns == kernel::columns) {
      matrix_input_t val(&input[0]);
      matrix_output_t out(&output[0]);
      out = val.rowwise().mean();
      return;
    }
    const std::size_t output_rows = avg_pooling::output_rows(input_rows);
    const std::size_t output_columns =
        avg_pooling::output_columns(input_columns);
    const std::size_t cells = output_rows * output_columns;
    const unsigned int number_of_workers = scheduler.number_of_workers();

    for (std::size_t start_cell = 0; start_cell < cells;
        start_cell += batch_size_) {
      const std::size_t end_cell = std::min(cells, start_cell + batch_size_);
      scheduler.schedule([=] (std::size_t worker) {
        auto worker_temporary = temporary.slice(worker, number_of_workers);
        for (std::size_t cell = start_cell; cell < end_cell; ++cell) {
          const std::size_t row = cell / output_columns;
          const std::size_t column = cell % output_columns;
          for (std::size_t kernel_row = 0; kernel_row < kernel::rows;
               ++kernel_row) {
            auto i_input = &input[components::value * (
                input_columns * (stride::rows * row  + kernel_row) +
                stride::columns * column)];
            auto i_temporary = &worker_temporary[
                components::value * kernel::columns * kernel_row];
            std::copy_n(i_input, components::value * kernel::columns,
                        i_temporary);
          }
          matrix_input_t val(&worker_temporary[0]);
          auto i_output = &output[components::value * cell];
          matrix_output_t out(i_output);
          out = val.rowwise().mean();
        }
      });
    }
    scheduler.wait();
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temporary, output_t output,
                  std::size_t input_rows, std::size_t input_columns,
                  const scheduler_t& scheduler) {
    const std::size_t output_rows = avg_pooling::output_rows(input_rows);
    const std::size_t output_columns =
        avg_pooling::output_columns(input_columns);
    const std::size_t cells = output_rows * output_columns;

    if (scheduler.number_of_workers() == 1) {
      batch_size_ = cells;
      return;
    }

    auto best_duration = std::chrono::high_resolution_clock::duration::max();

    std::size_t best_batch_size = 1;
    auto test_and_maybe_set = [&]() {
      for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        operator()(input, temporary, output, input_rows, input_columns,
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
    while (batch_size_ < cells) {
        test_and_maybe_set();
        batch_size_ = batch_size_ * 2;
    }
    batch_size_ = cells;
    test_and_maybe_set();
    batch_size_ = cells / scheduler.number_of_workers() +
                  cells % scheduler.number_of_workers() ? 1 : 0;
    while (batch_size_ > 1) {
        test_and_maybe_set();
        batch_size_ = batch_size_ / 2;
    }
    batch_size_ = best_batch_size;
  }

 private:
  static constexpr std::size_t cell_size =
      components::value * kernel::rows * kernel::columns;
  using alignment_t = alignment<type_t>;
  using matrix_input_t = Eigen::Map<
      Eigen::Matrix<type_t, components::value, kernel::columns * kernel::rows>,
      alignment_t::eigen_alignment>;
  using matrix_output_t = Eigen::Map<
      Eigen::Matrix<type_t, components::value, 1>>;
  std::size_t batch_size_;
};
}
}
}

#endif // POLIMIDL_LAYERS_INTERNAL_AVG_POOLING_HPP

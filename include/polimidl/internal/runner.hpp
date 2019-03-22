#ifndef POLIMIDL_INTERNAL_RUNNER_HPP
#define POLIMIDL_INTERNAL_RUNNER_HPP

#include <chrono>
#include <sstream>
#include <type_traits>

#include "./memory_layout.hpp"

namespace polimidl {
namespace internal {
template <typename layer_t>
std::size_t input_size_of(std::size_t input_rows, std::size_t input_columns) {
  return layer_t::input_components * input_rows * input_columns;
}

template <typename layer_t>
std::size_t output_size_of(std::size_t ouput_rows, std::size_t output_columns) {
  return layer_t::output_components * ouput_rows * output_columns;
}

template <std::size_t alignment, typename type_t, bool is_inverted,
          typename... Layers>
class runner {};

template <std::size_t alignment, typename type_t, bool is_inverted,
          typename layer_t>
class runner<alignment, type_t, is_inverted, layer_t> {
public:
  runner(span<type_t> buffer, std::size_t input_rows, std::size_t input_columns,
         unsigned int number_of_workers, layer_t&& layer) :
    layer_(std::forward<layer_t>(layer)),
    input_rows_(input_rows),
    input_columns_(input_columns),
    output_rows_(layer_t::output_rows(input_rows_)),
    output_columns_(layer_t::output_columns(input_columns_)),
    input_(
        input_of<alignment, layer_t, is_inverted>(buffer, this->input_size())),
    temporary_(temporary_of<alignment, layer_t, is_inverted>(
        buffer, this->input_size(), this->output_size(), number_of_workers)),
    output_(
        output_of<alignment, layer_t, is_inverted>(buffer,
                                                   this->output_size())),
    accumulated_execution_duration_(0) {}

  std::size_t input_rows() const { return input_rows_; }
  std::size_t input_columns() const { return input_columns_; }
  std::size_t input_components() const {
    return layer_t::input_components;
  }
  std::size_t input_size() const {
    return input_size_of<layer_t>(input_rows(), input_columns());
  }
  std::size_t output_rows() const { return output_rows_; }
  std::size_t output_columns() const { return output_columns_; }
  std::size_t output_components() const {
    return layer_t::output_components;
  }
  std::size_t output_size() const {
    return output_size_of<layer_t>(output_rows(), output_columns());
  }
  span<type_t> input() const { return input_; }
  span<type_t> output() const { return output_; }

  template <typename scheduler_t>
  void run(const scheduler_t& scheduler) const {
    layer_(input(), temporary_, output(), input_rows(), input_columns(),
           scheduler);
  }

  template <typename scheduler_t>
  void run_with_statistics(const scheduler_t& scheduler) {
    using std::chrono::high_resolution_clock;
    const auto start = high_resolution_clock::now();
    layer_(input(), temporary_, output(), input_rows(), input_columns(),
           scheduler);
    accumulated_execution_duration_ += high_resolution_clock::now() - start;
  }

  template <typename scheduler_t>
  void optimize_for(const scheduler_t& scheduler) {
    layer_.optimize_for(input(), temporary_, output(), input_rows(),
                        input_columns(), scheduler);
  }

  void print_statistics(std::stringstream& stream, std::size_t position,
                        std::size_t accumulated_executions) {
    stream << "Layer " << position << " Avg Time = "
        << std::chrono::duration_cast<std::chrono::microseconds>(
            accumulated_execution_duration_ / accumulated_executions).count()
        << " microseconds" << std::endl;
  }
private:
 static_assert(std::is_same<type_t, typename layer_t::type_t>::value,
               "The layer is not of the same data type of the network");
 layer_t layer_;
 std::size_t input_rows_;
 std::size_t input_columns_;
 std::size_t output_rows_;
 std::size_t output_columns_;
 span<type_t> input_;
 span<type_t> temporary_;
 span<type_t> output_;
 std::chrono::high_resolution_clock::duration accumulated_execution_duration_;
};

template <std::size_t alignment, typename type_t, bool is_inverted,
          typename layer_t, typename next_layer_t, typename... Layers>
class runner<alignment, type_t, is_inverted, layer_t, next_layer_t, Layers...> {
 public:
  runner(span<type_t> buffer, std::size_t input_rows, std::size_t input_columns,
         unsigned int number_of_workers, layer_t&& layer,
         next_layer_t&& next_layer, Layers&&... layers) :
    layer_(std::forward<layer_t>(layer)),
    input_rows_(input_rows),
    input_columns_(input_columns),
    next_runner_(buffer, layer_t::output_rows(input_rows_),
                 layer_t::output_columns(input_columns_),
                 number_of_workers,
                 std::forward<next_layer_t>(next_layer),
                 std::forward<Layers>(layers)...),
    input_(
        input_of<alignment, layer_t, is_inverted>(buffer, this->input_size())),
    temporary_(temporary_of<alignment, layer_t, is_inverted>(
        buffer, this->input_size(), next_runner_.input_size(),
        number_of_workers)),
    accumulated_execution_duration_(0) {}

  std::size_t input_rows() const { return input_rows_; }
  std::size_t input_columns() const { return input_columns_; }
  std::size_t input_components() const {
    return layer_t::input_components;
  }
  std::size_t input_size() const {
    return input_size_of<layer_t>(input_rows(), input_columns());
  }
  std::size_t output_rows() const { return next_runner_.output_rows(); }
  std::size_t output_columns() const { return next_runner_.output_columns(); }
  std::size_t output_components() const {
    return next_runner_.output_components();
  }
  std::size_t output_size() const { return next_runner_.output_size(); }
  span<type_t> input() const { return input_; }
  span<type_t> output() const { return next_runner_.output(); }

  template <typename scheduler_t>
  void run(const scheduler_t& scheduler) const {
    layer_(input(), temporary_, next_runner_.input(), input_rows(),
           input_columns(), scheduler);
    next_runner_.run(scheduler);
  }

  template <typename scheduler_t>
  void run_with_statistics(const scheduler_t& scheduler) {
    using std::chrono::high_resolution_clock;
    const auto start = high_resolution_clock::now();
    layer_(input(), temporary_, next_runner_.input(), input_rows(),
           input_columns(), scheduler);
    accumulated_execution_duration_ += high_resolution_clock::now() - start;
    next_runner_.run_with_statistics(scheduler);
  }

  template <typename scheduler_t>
  void optimize_for(const scheduler_t& scheduler) {
    layer_.optimize_for(input(), temporary_, next_runner_.input(), input_rows(),
                        input_columns(), scheduler);
    next_runner_.optimize_for(scheduler);
  }

  void print_statistics(std::stringstream& stream, std::size_t position,
                        std::size_t accumulated_executions) {
    stream << "Layer " << ++position << " Avg Time = "
        << std::chrono::duration_cast<std::chrono::microseconds>(
            accumulated_execution_duration_ / accumulated_executions).count()
        << " microseconds" << std::endl;
    next_runner_.print_statistics(stream, position, accumulated_executions);
  }
 private:
  static_assert(std::is_same<type_t, typename layer_t::type_t>::value,
                "The layer is not of the same data type of the network");
  static_assert(layer_t::output_components == next_layer_t::input_components,
                "The number of output components of this layer is different "
                "from the number of input components in the next one");
  layer_t layer_;
  std::size_t input_rows_;
  std::size_t input_columns_;
  static constexpr bool is_next_layer_inverted =
      (is_inverted && layer_t::is_in_place) ||
      (!is_inverted && !layer_t::is_in_place);
  runner<alignment, type_t, is_next_layer_inverted, next_layer_t, Layers...>
      next_runner_;
  span<type_t> input_;
  span<type_t> temporary_;
  std::chrono::high_resolution_clock::duration accumulated_execution_duration_;
};
}
}

#endif // POLIMIDL_INTERNAL_RUNNER_HPP

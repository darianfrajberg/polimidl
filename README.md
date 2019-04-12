# PolimiDL

__PolimiDL__ is a performance oriented, mobile first, Deep Learning inference framework.

## What is PolimiDL?
__PolimiDL__ is a framework for the execution of small Deep Neural Networks.

It is designed with __Fully Convolution Neural Networks__ in mind, but it can also execute networks with __Fully Connected layers__.

### Mobile First
__PolimiDL__ is mainly targeted to mobile devices and embedded systems, where resources (expecially RAM and CPU) are limited.

### Designed for the Future
__PolimiDL__ is developed in __C++17__ and exploits all the features of the language that can improve performance or usability.

## What is not PolimiDL?
__PolimiDL__ is not a training framework.
The focus of the project is execution, enabling use-cases and optimization which are not possible (or at least not easily impletable) in all-in-one solutions.
__PolimiDL__ assumes that the model has already been trained in the framework of your choice and later converted into a compatible format.


## X to PolimiDL Converters
* [PolimiDL Converter](https://github.com/darianfrajberg/polimidl_converter) currently supports TensorFlow.
* TODO add your converter if you write one.

## Creating a Network
__PolimiDL__ follows a _Code as Configuration_ pattern.
This enables the compiler to perform various optimizations during compilation, instead of runtime.

Here is an example of how to instantiate a network:
```c++
#include <algorithm>

// Include the Network library
#include <polimidl/network.hpp>
// Include the layers you need
#include <polimidl/layers/relu.hpp>

int main() {
    // Just simplify the code a little
    using ::polimidl::build_network;
    using ::polimidl::suggested_number_of_workers;
    using ::polimidl::layers::components;
    using ::polimidl::layers::relu;

    // Define the input dimensions
    constexpr std::size_t rows = 18;
    constexpr std::size_t columns = 18;

    // Create the Network with the provided input dimentions, the suggested
    // number of workers and a relu layer which works on 3 float components.
    auto network = build_network(rows, columns, suggested_number_of_workers(),
                                 relu<float, components<3>>());
    
    // You can query the network for input and output dimensions.
    // network.input_rows();
    // network.input_columns();
    // network.input_components();
    // network.output_rows();
    // network.output_columns();
    // network.output_components();
    
    // Obtain the input buffer of the Network
    auto input = network.input();
    
    // Copy the input data on the buffer
    std::copy(std::begin(my_input_buffer), std::end(my_input, input.begin());
    
    // Run the Network and obtain the output buffer
    auto output = network();
    
    // Copy data out of the output buffer
    std::copy(output.begin(), output.end(), std::begin(my_output_buffer));
}
```

### Use a Network as an Class member
Network types are not meant to be explicitly typed, but thanks to C++ type deduction you don't have to, just use auto.

If you are using a network as a member of a class, you can use `decltype` to deduce the proper type:
```c++
// Include the Network library
#include <polimidl/network.hpp>
// Include the layers you need
#include <polimidl/layers/relu.hpp>

class my_amazing_wita_a_network_inside {
 public:
  my_amazing_wita_a_network_inside() :
     network_(build_my_amazing_network()) {}
  
  void use_network() {
    // Do something with network_
  }

 private:
  decltype(build_my_amazing_network()) network_;
  static auto build_my_amazing_network() {
    // Define the input dimensions
    constexpr std::size_t rows = 18;
    constexpr std::size_t columns = 18;

    // Create the Network with the provided input dimentions, the suggested
    // number of workers and a relu layer which works on 3 float components.
    return build_network(rows, columns, suggested_number_of_workers(),
                         relu<float, components<3>>());
  }
}
```

#### What about lazy initialization?
If yout want to initialize your network lazily you use `make_network` instead of `build_network`.
The network will be created in a std::unique_ptr with the proper template argument.
```c++
// Include the Network library
#include <polimidl/network.hpp>
// Include the layers you need
#include <polimidl/layers/relu.hpp>

class my_amazing_wita_a_network_inside {
 public:
  void lazy_initialize() {
    if (!network_) {
      network_ = make_my_amazing_network();
    }
  }

 private:
  decltype(make_my_amazing_network()) network_;
  static auto make_my_amazing_network() {
    // Define the input dimensions
    constexpr std::size_t rows = 18;
    constexpr std::size_t columns = 18;

    // Create the Network with the provided input dimentions, the suggested
    // number of workers and a relu layer which works on 3 float components.
    return make_network(rows, columns, suggested_number_of_workers(),
                        relu<float, components<3>>());
  }
}
```

## What are workers?
__PolimiDL__ accelerates execution using a worker pool which divides the work among different threads.
You can set the number of threads as the third argument of `build_network` and `make_network`.
You can set any value greater or equal to `1`.

__PolimiDL__ provides two helpers `::polimidl::max_number_of_workers()` and `::polimidl::suggested_number_of_workers()` which tell you, rispectively, the theoretical maximum and suggested number of workers.

## Available Layers
The current supported __layers__ in __PolimiDL__ are:
* AVG Pooling
* Batch Normalization
* Bias
* Convolution
* Depthwise Convolution
* Max Pooling
* Normalize
* Pointwise Convolution
* ReLu/ReLu6
* Softmax

## Instantiating an available layer
You can instantiate an available layer by using the signatures defined in  `polimidl/layers`.

E.g. the Bias layer applies a per-component bias to all the elements in the input.
```c++
#include <polimidl/layers/bias.hpp>
::polimidl::layers::bias<data_type, ::polimidl::layers::components>(const data_type* bias_values);
```
`bias_values` must be aligned to `::polimidl::layers::buffer_alignment::byte_alignment`.
```c++
#include <polimidl/layers/alignment.hpp>
alignas(polimidl::layers::buffer_alignment::byte_alignment) data_type[] = { ... };
```

## Writing a custom layer
The best way to learn how to write a custom layer is to look at one of the existing implementations in `polimidl/layers`.

A new layer type is a class that inherits from `::polimidl::layers::layer<data_type, ::polimidl::layers::components, bool is_in_place>` and implements the `()` operator.

```c++
#include <algorithm>

#include <polimidl/layers/layer.hpp"

template <typename type_t, typename components>
class plus_one : public polimidl::layers::layer<type_t, components> {
 public:
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    std::transform(input.begin(), input.end(), output.begin(), [](type_t value) {
      return value + type_t(1);
    });
  }
}
```

### InPlace layers
An inplace layer guarantees the execution framework that `input` and `output` can share the same memory.
The `plus_one` layer defined above is a perfect example of this type of layers.
Just set the third template parameter of `polimidl::layers::layer` to `true` and the framework will do the rest.
```c++
#include <algorithm>

#include <polimidl/layers/layer.hpp"

template <typename type_t, typename components>
class plus_one : public polimidl::layers::layer<type_t, components, true> {
 public:
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    std::transform(input.begin(), input.end(), output.begin(), [](type_t value) {
      return value + type_t(1);
    });
  }
}
```

### Using the Scheduler
If your task can perform better if parallelized, you can easily split the work among the available workers.
```c++
#include <algorithm>

#include <polimidl/layers/layer.hpp"

template <typename type_t, typename components>
class plus_one : public polimidl::layers::layer<type_t, components, true> {
 public:
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const std::size_t cells = rows * columns * components::value;
    const std::size_t number_of_workers = scheduler.number_of_workers();
    const std::size_t batch_size = cells / number_of_workers;
    for (std::size_t start_cell = 0; start_cell < cells; start_cell += batch_size) {
      scheduler.schedule([=](unsigned int worker) {
        const std::size_t cells_in_this_batch = std::min(batch_size, cells - start_cell);
        auto input_begin = &input[start_cell];
        auto input_end = begin + cells_in_this_batch;
        auto output_begin = &output[start_cell];
        std::transform(input_begin, input_end, output_begin, [](type_t value) {
          return value + type_t(1);
        }); 
      });
    }
    scheduler.wait();
  }
}
```

### Changing the number of Components
If two template parameters are passed to the `components` configuration the number of input and output components can be set differently.
__PolimiDL__ will ensure that subsequent layers share the same number of components.

### Resizing the data
By default, the layer implementation supposes that the number of `rows` and `columns` remains the same.
You can override this behavior by implementing the `output_rows` and `output_columns` static functions.
```c++
#include <algorithm>

#include <polimidl/layers/layer.hpp"

template <typename type_t, typename components, typename pooling>
class pull_first : public polimidl::layers::layer<type_t, components> {
 public:
  static std::size_t output_rows(std::size_t input_rows) {
    return input_rows / pooling::rows;
  }
  
  static std::size_t output_columns(std::size_t input_columns) {
    return input_columns / pooling::columns;
  }
  
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t input_rows, std::size_t input_columns,
                  const scheduler_t& scheduler) const {
    const std::size_t rows = output_rows(input_rows);
    const std::size_t columns = output_rows(input_columns);
    for (std::size_t row = 0; row < rows, ++row) {
      for (std::size_t column = 0; column < columns, ++column) {
        auto input_begin = &input[components::value * (column * pooling::column + row * pooling::rows * input_columns];
        auto output_begin = &output[components::value * (column * + row * columns];
        std::copy_n(input_begin, components::value, output_begin);
      }
    }
  }
}
```

### Using temporary buffers
You can request the framework a bit of extra memory that can be used as a temporary buffer during computation.
You can request a minimum amount of memory by implementing the `temporary_memory` static function.
```c++
#include <algorithm>

#include <polimidl/layers/layer.hpp"

template <typename type_t, typename components>
class plus_one : public polimidl::layers::layer<type_t, components> {
 public:
  static std::size_t temporary_size(std::size_t input_rows,
                                    std::size_t input_columns,
                                    std::size_t number_of_workers) {
    return input_columns * components::value;
  }
 
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    for (std::size_t row; row < rows, ++row) {
      auto input_begin = &input[components::value * row * columns];
      auto temporary_begin = &temporary[components::value * row * columns];
      auto input_end = input_begin + components::value * columns;
      std::transform(input_begin, input_end, temporary_begin, [](type_t value) {
        return value + type_t(1);
      });
      std::copy_n(temporary_begin, components::value * columns, output_begin);
    }
  }
}
```
The temporary buffer can be split in worker specific buffers invoking `temporary.slice(worker, number_of_workers)`.

### Initialiation-time Optimization
__PolimiDL__ supports initialization-time optimization by allowing the layers to optimize some parameters for the data at hand.
Just implement the `optimize_for` method and compute the parameters that can speedup the execution.
```c++
#include <algorithm>

#include <polimidl/layers/layer.hpp"

template <typename type_t, typename components>
class plus_one : public polimidl::layers::layer<type_t, components, true> {
 public:
  plus_one() :
      batch_size_(1) {}
      
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    // Compite the best batch size and save it.
    batch_size_ = ...;
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const std::size_t cells = rows * columns * components::value;
    const std::size_t number_of_workers = scheduler.number_of_workers();
    for (std::size_t start_cell = 0; start_cell < cells; start_cell += batch_size_) {
      scheduler.schedule([=](unsigned int worker) {
        const std::size_t cells_in_this_batch = std::min(batch_size_, cells - start_cell);
        auto input_begin = &input[start_cell];
        auto input_end = begin + cells_in_this_batch;
        auto output_begin = &output[start_cell];
        std::transform(input_begin, input_end, output_begin, [](type_t value) {
          return value + type_t(1);
        }); 
      });
    }
    scheduler.wait();
  }
}
```

## Available examples

### MobileNet
In `polimidl/examples/mobilenet.hpp` there is an available pre-converted version of MobileNet.


## Citing PolimiDL

If you use PolimiDL or wish to refer it, please use the following BibTex entry.

```
__Coming Soon__
```


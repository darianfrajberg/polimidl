#ifndef POLIMIDL_INTERNAL_SPAN_HPP
#define POLIMIDL_INTERNAL_SPAN_HPP

#include <cstddef>

namespace polimidl {
namespace internal {
template <typename type_t>
class span {
 public:
  span(type_t* begin, type_t* end) :
    begin_(begin), end_(end) {}
  span(type_t* begin, std::size_t size) :
    begin_(begin), end_(begin + size) {}
  span(std::size_t size, type_t* end) :
    begin_(end - size), end_(end) {}

  auto begin() const { return begin_; }
  auto end() const { return end_; }
  std::size_t size() const { return end() - begin(); }

  type_t& operator[](std::size_t index) const {
    return *(begin() + index);
  }

  auto slice_size(std::size_t slices) const {
    return size() / slices;
  }

  auto slice(std::size_t index, std::size_t slices) const {
    auto size = slice_size(slices);
    return span<type_t>(begin() + size * index, size);
  }

 private:
  type_t* begin_;
  type_t* end_;
};
}
}

#endif // POLIMIDL_INTERNAL_SPAN_HPP

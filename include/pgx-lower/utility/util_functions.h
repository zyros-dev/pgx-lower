#pragma once

#include <algorithm>
#include <initializer_list>

namespace pgx {

template<typename T>
inline bool is_in(const T& value, std::initializer_list<T> list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
} // namespace pgx
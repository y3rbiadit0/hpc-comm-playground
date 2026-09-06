#pragma once

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_bench {

inline bool has_leading_minus(const char* value) {
  while (std::isspace(static_cast<unsigned char>(*value)) != 0) {
    ++value;
  }
  return *value == '-';
}

inline std::size_t checked_size_multiply(std::size_t lhs, std::size_t rhs, const char* description) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error(std::string(description) + " size overflow");
  }
  return lhs * rhs;
}

inline std::size_t parse_size_arg(int argc, char** argv, std::size_t default_value) {
  if (argc < 2) {
    return default_value;
  }

  char* end = nullptr;
  errno = 0;
  const auto value = std::strtoull(argv[1], &end, 10);
  if (has_leading_minus(argv[1]) || errno == ERANGE || end == argv[1] || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("expected a positive vector size");
  }

  return static_cast<std::size_t>(value);
}

inline std::vector<std::size_t> default_power_of_two_sizes(std::size_t max_value) {
  std::vector<std::size_t> sizes;
  for (std::size_t size = 1; size <= max_value; size *= 2U) {
    sizes.push_back(size);
    if (size > max_value / 2U) {
      break;
    }
  }
  return sizes;
}

inline std::size_t parse_size_token(const std::string& token, std::size_t max_value) {
  if (token.empty()) {
    throw std::invalid_argument("expected a comma-separated list of positive message sizes");
  }

  char* end = nullptr;
  errno = 0;
  const auto value = std::strtoull(token.c_str(), &end, 10);
  if (has_leading_minus(token.c_str()) || errno == ERANGE || end == token.c_str() || *end != '\0' ||
      value == 0 || value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("expected a comma-separated list of positive message sizes");
  }
  if (value > max_value) {
    throw std::invalid_argument("message size exceeds maximum message size");
  }

  return static_cast<std::size_t>(value);
}

inline std::vector<std::size_t> parse_size_list_arg(int argc, char** argv, int index,
                                                    std::size_t max_value) {
  if (argc <= index) {
    return default_power_of_two_sizes(max_value);
  }

  const std::string input(argv[index]);
  std::vector<std::size_t> sizes;
  std::size_t start = 0;
  while (start <= input.size()) {
    const auto comma = input.find(',', start);
    const auto end = comma == std::string::npos ? input.size() : comma;
    sizes.push_back(parse_size_token(input.substr(start, end - start), max_value));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1U;
  }
  if (sizes.empty()) {
    throw std::invalid_argument("expected a comma-separated list of positive message sizes");
  }
  return sizes;
}

/* Like parse_size_list_arg, but a missing argument means "just `value`" rather
 * than a power-of-two sweep.
 *
 * For benchmarks whose size changes the decomposition rather than only the
 * message, so that a bare invocation keeps measuring the one configuration it
 * always measured. cg_step's grid side sets the rank's column count; sweeping it
 * by default would silently turn one measurement into twenty. */
inline std::vector<std::size_t> parse_size_list_or_single(int argc, char** argv, int index,
                                                          std::size_t value) {
  if (argc <= index) {
    return {value};
  }
  return parse_size_list_arg(argc, argv, index, value);
}

inline int parse_positive_int_arg(int argc, char** argv, int index, int default_value) {
  if (argc <= index) {
    return default_value;
  }

  char* end = nullptr;
  errno = 0;
  const auto value = std::strtol(argv[index], &end, 10);
  if (has_leading_minus(argv[index]) || errno == ERANGE || end == argv[index] || *end != '\0' || value <= 0 ||
      value > INT_MAX) {
    throw std::invalid_argument("expected a positive integer argument");
  }

  return static_cast<int>(value);
}

inline int parse_positive_int_env(const char* name, int default_value) {
  const char* input = std::getenv(name);
  if (input == nullptr || *input == '\0') {
    return default_value;
  }

  char* end = nullptr;
  errno = 0;
  const auto value = std::strtol(input, &end, 10);
  if (has_leading_minus(input) || errno == ERANGE || end == input || *end != '\0' || value <= 0 ||
      value > INT_MAX) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }

  return static_cast<int>(value);
}

}  // namespace gpu_bench

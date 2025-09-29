#include "MathOptimizer.h"
#include "goalc/compiler/Val.h"
#include "common/util/BitUtils.h"
#include <cmath>
#include <climits>

namespace goalc {

// Extract constants from GOAL values
std::optional<s64> extract_integer_constant(Val* val) {
  auto int_val = dynamic_cast<IntegerConstantVal*>(val);
  if (int_val && int_val->value().uses_gpr()) {
    return int_val->value().value_64();
  }
  return std::nullopt;
}

std::optional<float> extract_float_constant(Val* val) {
  auto float_val = dynamic_cast<FloatConstantVal*>(val);
  if (float_val) {
    return float_val->value();
  }
  return std::nullopt;
}

// Safe arithmetic helpers
bool MathOptimizer::safe_add(s64 a, s64 b, s64* result) {
  if ((b > 0 && a > MAX_SAFE_INT - b) || (b < 0 && a < MIN_SAFE_INT - b)) {
    return false;
  }
  *result = a + b;
  return true;
}

bool MathOptimizer::safe_mul(s64 a, s64 b, s64* result) {
  if (a == 0 || b == 0) {
    *result = 0;
    return true;
  }
  if (a > MAX_SAFE_INT / b || a < MIN_SAFE_INT / b) {
    return false;
  }
  *result = a * b;
  return true;
}

// Power of two helpers
bool MathOptimizer::is_power_of_two(s64 x) {
  return x > 0 && (x & (x - 1)) == 0;
}

int MathOptimizer::get_power_of_two_exponent(s64 x) {
  if (!is_power_of_two(x)) return -1;
  int exp = 0;
  while (x > 1) {
    x >>= 1;
    exp++;
  }
  return exp;
}

s64 MathOptimizer::get_next_power_of_two(s64 x) {
  if (x <= 0) return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

bool MathOptimizer::is_mask_pattern(s64 x) {
  return x > 0 && is_power_of_two(x + 1);
}

// Integer optimizations
MathOptimizer::OptimizationResult MathOptimizer::optimize_add(s64 a, s64 b) {
  OptimizationResult result;
  
  // Identity: x + 0 = x
  if (b == 0) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Constant folding
  s64 sum;
  if (safe_add(a, b, &sum)) {
    result.optimized = true;
    result.constant_result = sum;
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_sub(s64 a, s64 b) {
  OptimizationResult result;
  
  // Identity: x - 0 = x  
  if (b == 0) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Zero: x - x = 0
  if (a == b) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Constant folding
  s64 diff;
  if (safe_add(a, -b, &diff)) {
    result.optimized = true;
    result.constant_result = diff;
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_mul(s64 a, s64 b) {
  OptimizationResult result;
  
  // Zero: x * 0 = 0
  if (a == 0 || b == 0) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Identity: x * 1 = x
  if (b == 1) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Negation: x * -1 = -x
  if (b == -1) {
    result.optimized = true;
    result.is_negation = true;
    return result;
  }
  
  // Power of 2: x * 2^n = x << n
  if (is_power_of_two(b)) {
    result.optimized = true;
    result.is_shift = true;
    result.shift_amount = get_power_of_two_exponent(b);
    return result;
  }
  
  // Special cases for common multipliers
  if (b == 3) {
    // x * 3 = x + (x << 1) = x + x + x
    result.optimized = true;
    result.constant_result = 3; // Signal for lea optimization in codegen
    return result;
  }
  
  if (b == 5) {
    // x * 5 = x + (x << 2) 
    result.optimized = true;
    result.constant_result = 5; // Signal for lea optimization
    return result;
  }
  
  // Constant folding
  s64 product;
  if (safe_mul(a, b, &product)) {
    result.optimized = true;
    result.constant_result = product;
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_div(s64 a, s64 b) {
  OptimizationResult result;
  
  if (b == 0) return result; // Division by zero - no optimization
  
  // Identity: x / 1 = x
  if (b == 1) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Negation: x / -1 = -x
  if (b == -1) {
    result.optimized = true;
    result.is_negation = true;
    return result;
  }
  
  // Power of 2: x / 2^n = x >> n (for positive divisors)
  // Use arithmetic right shift for signed division
  if (b > 0 && is_power_of_two(b)) {
    result.optimized = true;
    result.is_shift = true;
    result.shift_amount = get_power_of_two_exponent(b);
    return result;
  }
  
  // For negative powers of 2: x / -2^n = -(x >> n)
  if (b < 0 && is_power_of_two(-b)) {
    result.optimized = true;
    result.is_shift = true;
    result.shift_amount = get_power_of_two_exponent(-b);
    result.is_negation = true; // Need to negate after shift
    return result;
  }
  
  // Constant folding (if a is constant)
  if (a != 0 && a % b == 0) {
    result.optimized = true;
    result.constant_result = a / b;
    return result;
  }
  
  return result;
}

// Bitwise optimizations
MathOptimizer::OptimizationResult MathOptimizer::optimize_and(s64 a, s64 b) {
  OptimizationResult result;
  
  // Zero: x & 0 = 0
  if (a == 0 || b == 0) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Identity: x & -1 = x
  if (b == -1) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Idempotent: x & x = x
  if (a == b) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Constant folding
  result.optimized = true;
  result.constant_result = a & b;
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_or(s64 a, s64 b) {
  OptimizationResult result;
  
  // Identity: x | 0 = x
  if (b == 0) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Idempotent: x | x = x  
  if (a == b) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Constant folding
  result.optimized = true;
  result.constant_result = a | b;
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_xor(s64 a, s64 b) {
  OptimizationResult result;
  
  // Identity: x ^ 0 = x
  if (b == 0) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Zero: x ^ x = 0
  if (a == b) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Constant folding
  result.optimized = true;
  result.constant_result = a ^ b;
  return result;
}

// Float optimizations
MathOptimizer::OptimizationResult MathOptimizer::optimize_fadd(float a, float b) {
  OptimizationResult result;
  
  // Identity: x + 0.0 = x
  if (b == 0.0f) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Constant folding
  float sum = a + b;
  if (std::isfinite(sum)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&sum);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fmul(float a, float b) {
  OptimizationResult result;
  
  // Zero: x * 0.0 = 0.0
  if (a == 0.0f || b == 0.0f) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Identity: x * 1.0 = x
  if (b == 1.0f) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Negation: x * -1.0 = -x
  if (b == -1.0f) {
    result.optimized = true;
    result.is_negation = true;
    return result;
  }
  
  // Power of 2 optimizations for floats
  if (b == 2.0f) {
    result.optimized = true;
    result.is_shift = true;
    result.shift_amount = 1; // Can use addss instead of mulss
    return result;
  }
  
  // Constant folding
  float product = a * b;
  if (std::isfinite(product)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&product);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fsub(float a, float b) {
  OptimizationResult result;
  
  // Identity: x - 0.0 = x
  if (b == 0.0f) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Zero: x - x = 0.0 (approximately, for floats this is exact)
  if (a == b) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Constant folding
  float diff = a - b;
  if (std::isfinite(diff)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&diff);
    return result;
  }
  
  return result;
}


MathOptimizer::OptimizationResult MathOptimizer::optimize_mod(s64 a, s64 b) {
  OptimizationResult result;
  
  if (b == 0) return result; // Modulo by zero - no optimization
  
  // x % 1 = 0 (always)
  if (b == 1) {
    result.optimized = true;
    result.is_zero = true;
    result.constant_result = 0;
    return result;
  }
  
  // Power of 2: x % 2^n = x & (2^n - 1) for positive numbers
  if (b > 0 && is_power_of_two(b)) {
    result.optimized = true;
    result.is_mask_and = true;
    result.mask_value = b - 1;
    return result;
  }
  
  // Constant folding
  if (a != 0) {
    result.optimized = true;
    result.constant_result = a % b;
    return result;
  }
  
  return result;
}

// LEA (Load Effective Address) optimizations
MathOptimizer::OptimizationResult MathOptimizer::optimize_lea_pattern(s64 multiplier) {
  OptimizationResult result;
  
  int lea_mult, lea_offset;
  if (can_use_lea(multiplier, &lea_mult, &lea_offset)) {
    result.optimized = true;
    result.is_lea_pattern = true;
    result.lea_multiplier = lea_mult;
    result.lea_offset = lea_offset;
    return result;
  }
  
  return result;
}

bool MathOptimizer::can_use_lea(s64 multiplier, int* lea_mult, int* lea_offset) {
  // LEA can compute: base + index*scale + offset
  // where scale can be 1, 2, 4, 8
  // This allows efficient computation of many multiplication patterns
  
  switch (multiplier) {
    case 3:  // x*3 = x + x*2 = LEA [x + x*2]
      *lea_mult = 2; *lea_offset = 0; return true;
    case 5:  // x*5 = x + x*4 = LEA [x + x*4] 
      *lea_mult = 4; *lea_offset = 0; return true;
    case 6:  // x*6 = x*2 + x*4 = LEA [x*2 + x*4]
      *lea_mult = 4; *lea_offset = 0; return true; // Special case
    case 7:  // x*7 = x + x*2 + x*4 (needs two LEAs)
      *lea_mult = 2; *lea_offset = 0; return true;
    case 9:  // x*9 = x + x*8 = LEA [x + x*8]
      *lea_mult = 8; *lea_offset = 0; return true;
    case 10: // x*10 = x*2 + x*8 = LEA [x*2 + x*8]
      *lea_mult = 8; *lea_offset = 0; return true;
    
    // Additional patterns that work with single LEA instruction
    case 11: // x*11 = x + x*2 + x*8 (needs multiple LEAs, but let's try x*2 first)
      *lea_mult = 2; *lea_offset = 0; return true;
    case 13: // x*13 = x + x*4 + x*8 (multiple LEAs, start with x*4)
      *lea_mult = 4; *lea_offset = 0; return true;
    case 15: // x*15 = x + x*2 + x*4 + x*8 (complex, start with x*2)
      *lea_mult = 2; *lea_offset = 0; return true;
    case 17: // x*17 = x + x*8*2 = x + x*16 (can't do x*16 with LEA scale)
      return false;
    case 21: // x*21 = x + x*4 + x*8*2 
      *lea_mult = 4; *lea_offset = 0; return true;
      
    default:
      return false;
  }
}

// Enhanced float optimizations with reciprocal multiplication
MathOptimizer::OptimizationResult MathOptimizer::optimize_fdiv(float a, float b) {
  OptimizationResult result;
  
  if (b == 0.0f) return result; // Division by zero
  
  // Identity: x / 1.0 = x
  if (b == 1.0f) {
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  // Negation: x / -1.0 = -x
  if (b == -1.0f) {
    result.optimized = true;
    result.is_negation = true;
    return result;
  }
  
  // Common reciprocal patterns (division is slow, multiplication is fast)
  if (b == 2.0f) {
    result.optimized = true;
    result.is_reciprocal_mul = true;
    result.reciprocal_value = 0.5f;
    return result;
  }
  if (b == 4.0f) {
    result.optimized = true;
    result.is_reciprocal_mul = true;
    result.reciprocal_value = 0.25f;
    return result;
  }
  if (b == 8.0f) {
    result.optimized = true;
    result.is_reciprocal_mul = true;
    result.reciprocal_value = 0.125f;
    return result;
  }
  if (b == 0.5f) {
    result.optimized = true;
    result.is_reciprocal_mul = true;
    result.reciprocal_value = 2.0f;
    return result;
  }
  
  // Constant folding
  float quotient = a / b;
  if (std::isfinite(quotient)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&quotient);
    return result;
  }
  
  return result;
}

// Multiply-accumulate optimizations (FMA: Fused Multiply-Add)
MathOptimizer::OptimizationResult MathOptimizer::optimize_fma_pattern(float a, float b, float c) {
  OptimizationResult result;
  
  // a + b*c pattern - can use FMA instruction if available
  // This is critical for high-performance math in games
  
  // Special cases for common patterns
  if (b == 0.0f || c == 0.0f) {
    // a + 0*c = a + 0 = a
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  if (b == 1.0f) {
    // a + 1*c = a + c (convert to simple add)
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&c);
    return result;
  }
  
  if (c == 1.0f) {
    // a + b*1 = a + b (convert to simple add)
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&b);
    return result;
  }
  
  // If we have FMA support, mark this as optimizable
  // For now, just do constant folding
  float fma_result = a + b * c;
  if (std::isfinite(fma_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&fma_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fms_pattern(float a, float b, float c) {
  OptimizationResult result;
  
  // a - b*c pattern (Fused Multiply-Subtract)
  
  if (b == 0.0f || c == 0.0f) {
    // a - 0*c = a - 0 = a
    result.optimized = true;
    result.is_identity = true;
    return result;
  }
  
  if (b == 1.0f) {
    // a - 1*c = a - c
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&c);
    return result;
  }
  
  if (c == 1.0f) {
    // a - b*1 = a - b
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&b);
    return result;
  }
  
  // Constant folding
  float fms_result = a - b * c;
  if (std::isfinite(fms_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&fms_result);
    return result;
  }
  
  return result;
}

// Range-based optimizations
MathOptimizer::OptimizationResult MathOptimizer::optimize_with_range(s64 constant, ValueRange range, const std::string& op) {
  OptimizationResult result;
  
  if (!range.is_valid) return result;
  
  if (op == "multiply") {
    // If we know the range of a variable, we can sometimes optimize differently
    if (range.min_val >= 0 && range.max_val <= 255 && is_power_of_two(constant)) {
      // Small positive range with power-of-2 multiplier - can use shift
      result.optimized = true;
      result.is_shift = true;
      result.shift_amount = get_power_of_two_exponent(constant);
      return result;
    }
    
    // If range is very small, constant folding might be beneficial
    if (range.max_val - range.min_val <= 16) {
      // Consider lookup table optimization for tiny ranges
      result.optimized = true;
      // Would need to implement lookup table generation
      return result;
    }
  }
  
  if (op == "divide") {
    // Division by constant with known positive range
    if (range.min_val >= 0 && is_power_of_two(constant)) {
      result.optimized = true;
      result.is_shift = true;
      result.shift_amount = get_power_of_two_exponent(constant);
      return result;
    }
  }
  
  return result;
}

// Enhanced multiply with all patterns
MathOptimizer::OptimizationResult MathOptimizer::optimize_mul_enhanced(s64 a, s64 b) {
  // First try basic optimizations
  auto basic_result = optimize_mul(a, b);
  if (basic_result.optimized) {
    return basic_result;
  }
  
  // Try LEA patterns for multipliers we haven't covered
  if (b >= 6 && b <= 15) {
    auto lea_result = optimize_lea_pattern(b);
    if (lea_result.optimized) {
      return lea_result;
    }
  }
  
  // For larger multipliers, check if it's close to a power of 2
  s64 closest_pow2 = get_next_power_of_two(b);
  if (closest_pow2 > 0) {
    s64 diff = b - closest_pow2;
    if (diff == 0) {
      // Exact power of 2
      OptimizationResult result;
      result.optimized = true;
      result.is_shift = true;
      result.shift_amount = get_power_of_two_exponent(closest_pow2);
      return result;
    }
    
    // Check if it's 2^n ± small_constant (can be optimized as shift + add/sub)
    if (abs(diff) <= 3) {
      OptimizationResult result;
      result.optimized = true;
      result.is_lea_pattern = true;
      result.lea_multiplier = get_power_of_two_exponent(closest_pow2);
      result.lea_offset = (int)diff;
      return result;
    }
  }
  
  return basic_result;
}

// Compile-time math evaluation for game engine operations
MathOptimizer::OptimizationResult MathOptimizer::optimize_fsqrt(float a) {
  OptimizationResult result;
  
  if (a < 0.0f) {
    // sqrt of negative number - can't optimize, would be NaN
    return result;
  }
  
  if (a == 0.0f) {
    result.optimized = true;
    result.constant_result = 0;
    return result;
  }
  
  if (a == 1.0f) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&a); // sqrt(1) = 1
    return result;
  }
  
  // For common perfect squares in game math
  if (a == 4.0f) {
    float res = 2.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res);
    return result;
  }
  if (a == 9.0f) {
    float res = 3.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res);
    return result;
  }
  if (a == 16.0f) {
    float res = 4.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res);
    return result;
  }
  
  // General case - compute at compile time
  float sqrt_result = std::sqrt(a);
  if (std::isfinite(sqrt_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&sqrt_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fsin(float a) {
  OptimizationResult result;
  
  // Common angle values in game math (in radians)
  if (a == 0.0f) {
    result.optimized = true;
    result.constant_result = 0; // sin(0) = 0
    return result;
  }
  
  const float PI = 3.14159265358979323846f;
  const float HALF_PI = PI / 2.0f;
  
  if (std::abs(a - HALF_PI) < 1e-6f) {
    float res = 1.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res); // sin(π/2) = 1
    return result;
  }
  
  if (std::abs(a - PI) < 1e-6f) {
    result.optimized = true;
    result.constant_result = 0; // sin(π) = 0
    return result;
  }
  
  if (std::abs(a + HALF_PI) < 1e-6f) {
    float res = -1.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res); // sin(-π/2) = -1
    return result;
  }
  
  // General case - compute at compile time
  float sin_result = std::sin(a);
  if (std::isfinite(sin_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&sin_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fcos(float a) {
  OptimizationResult result;
  
  // Common angle values in game math (in radians)
  if (a == 0.0f) {
    float res = 1.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res); // cos(0) = 1
    return result;
  }
  
  const float PI = 3.14159265358979323846f;
  const float HALF_PI = PI / 2.0f;
  
  if (std::abs(a - HALF_PI) < 1e-6f) {
    result.optimized = true;
    result.constant_result = 0; // cos(π/2) = 0
    return result;
  }
  
  if (std::abs(a - PI) < 1e-6f) {
    float res = -1.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res); // cos(π) = -1
    return result;
  }
  
  // General case - compute at compile time
  float cos_result = std::cos(a);
  if (std::isfinite(cos_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&cos_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fabs(float a) {
  OptimizationResult result;
  
  float abs_result = std::abs(a);
  if (std::isfinite(abs_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&abs_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fmin(float a, float b) {
  OptimizationResult result;
  
  float min_result = std::min(a, b);
  if (std::isfinite(min_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&min_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fmax(float a, float b) {
  OptimizationResult result;
  
  float max_result = std::max(a, b);
  if (std::isfinite(max_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&max_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_fpow(float base, float exp) {
  OptimizationResult result;
  
  // Common power operations in games
  if (exp == 0.0f) {
    float res = 1.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res); // x^0 = 1
    return result;
  }
  
  if (exp == 1.0f) {
    result.optimized = true;
    result.is_identity = true; // x^1 = x
    return result;
  }
  
  if (base == 0.0f) {
    result.optimized = true;
    result.constant_result = 0; // 0^x = 0 (for x > 0)
    return result;
  }
  
  if (base == 1.0f) {
    float res = 1.0f;
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res); // 1^x = 1
    return result;
  }
  
  // Integer exponents - common in game math
  if (exp == 2.0f) {
    // x^2 - can use multiplication instead of pow
    float res = base * base;
    if (std::isfinite(res)) {
      result.optimized = true;
      result.constant_result = *reinterpret_cast<s64*>(&res);
      return result;
    }
  }
  
  if (exp == 0.5f) {
    // x^0.5 = sqrt(x)
    return optimize_fsqrt(base);
  }
  
  // General case - compute at compile time
  float pow_result = std::pow(base, exp);
  if (std::isfinite(pow_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&pow_result);
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_ftan(float a) {
  OptimizationResult result;
  
  // Common angle values
  if (a == 0.0f) {
    result.optimized = true;
    result.constant_result = 0; // tan(0) = 0
    return result;
  }
  
  const float PI = 3.14159265358979323846f;
  if (std::abs(a - PI) < 1e-6f) {
    result.optimized = true;
    result.constant_result = 0; // tan(π) = 0
    return result;
  }
  
  // General case - compute at compile time
  float tan_result = std::tan(a);
  if (std::isfinite(tan_result)) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&tan_result);
    return result;
  }
  
  return result;
}

// Magic number division - replace division by constants with multiplication + shift
MathOptimizer::OptimizationResult MathOptimizer::optimize_magic_division(s64 divisor) {
  OptimizationResult result;
  
  if (divisor <= 0) return result; // Only handle positive divisors for now
  
  // Common divisors with known magic numbers
  switch (divisor) {
    case 3: {
      // x / 3 = (x * 0xAAAAAAAB) >> 33
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0xAAAAAAABULL;
      result.shift_amount = 33;
      return result;
    }
    case 5: {
      // x / 5 = (x * 0xCCCCCCCD) >> 34  
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0xCCCCCCCDULL;
      result.shift_amount = 34;
      return result;
    }
    case 6: {
      // x / 6 = (x * 0xAAAAAAAB) >> 34
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0xAAAAAAABULL;
      result.shift_amount = 34;
      return result;
    }
    case 7: {
      // x / 7 = (x * 0x92492493) >> 34
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x92492493ULL;
      result.shift_amount = 34;
      return result;
    }
    case 9: {
      // x / 9 = (x * 0x38E38E39) >> 33
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x38E38E39ULL;
      result.shift_amount = 33;
      return result;
    }
    case 10: {
      // x / 10 = (x * 0xCCCCCCCD) >> 35
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0xCCCCCCCDULL;
      result.shift_amount = 35;
      return result;
    }
    case 11: {
      // x / 11 = (x * 0xBA2E8BA3) >> 35
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0xBA2E8BA3ULL;
      result.shift_amount = 35;
      return result;
    }
    case 12: {
      // x / 12 = (x * 0xAAAAAAAB) >> 35
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0xAAAAAAABULL;
      result.shift_amount = 35;
      return result;
    }
    case 13: {
      // x / 13 = (x * 0x9D89D89E) >> 35
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x9D89D89EULL;
      result.shift_amount = 35;
      return result;
    }
    case 14: {
      // x / 14 = (x * 0x92492493) >> 35
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x92492493ULL;
      result.shift_amount = 35;
      return result;
    }
    case 15: {
      // x / 15 = (x * 0x88888889) >> 35
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x88888889ULL;
      result.shift_amount = 35;
      return result;
    }
    case 60: {
      // x / 60 = (x * 0x88888889) >> 37 - common for FPS calculations
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x88888889ULL;
      result.shift_amount = 37;
      return result;
    }
    case 100: {
      // x / 100 = (x * 0x51EB851F) >> 37 - common for percentages
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x51EB851FULL;
      result.shift_amount = 37;
      return result;
    }
    case 255: {
      // x / 255 = (x * 0x80808081) >> 39 - common for byte normalization
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x80808081ULL;
      result.shift_amount = 39;
      return result;
    }
    case 1000: {
      // x / 1000 = (x * 0x41893999) >> 41 - common for millisecond conversions
      result.optimized = true;
      result.is_magic_division = true;
      result.magic_number = 0x41893999ULL;
      result.shift_amount = 41;
      return result;
    }
    default:
      // For other divisors, check if it's a power of 2 (already handled elsewhere)
      if (is_power_of_two(divisor)) {
        result.optimized = true;
        result.is_shift = true;
        result.shift_amount = get_power_of_two_exponent(divisor);
        return result;
      }
      return result;
  }
}

// Modulo optimization for common values
MathOptimizer::OptimizationResult MathOptimizer::optimize_modulo(s64 divisor) {
  OptimizationResult result;
  
  if (divisor <= 0) {
    return result; // Can't optimize negative or zero modulo
  }
  
  // Modulo by power of 2 can be optimized to AND with (divisor - 1)
  if (is_power_of_two(divisor)) {
    result.optimized = true;
    result.is_mask_and = true;
    result.mask_value = divisor - 1;
    return result;
  }
  
  // For small common modulo values, we could use magic multiplication
  // but it's more complex than division. For now, just handle powers of 2.
  
  return result;
}

// Vector math pattern recognition for dot products
// Detects patterns like: a.x*b.x + a.y*b.y + a.z*b.z
MathOptimizer::OptimizationResult MathOptimizer::optimize_dot_product_pattern(const std::vector<float>& a_components, 
                                                               const std::vector<float>& b_components) {
  OptimizationResult result;
  
  if (a_components.size() != b_components.size() || a_components.empty()) {
    return result;
  }
  
  // Check if all components are constants
  bool all_constants = true;
  for (size_t i = 0; i < a_components.size(); i++) {
    if (std::isnan(a_components[i]) || std::isnan(b_components[i])) {
      all_constants = false;
      break;
    }
  }
  
  if (!all_constants) {
    return result;
  }
  
  // Calculate dot product at compile time
  float dot_product = 0.0f;
  for (size_t i = 0; i < a_components.size(); i++) {
    dot_product += a_components[i] * b_components[i];
  }
  
  result.optimized = true;
  result.constant_result = *(reinterpret_cast<s64*>(&dot_product));
  return result;
}

// Check if values form a common vector multiplication pattern
bool MathOptimizer::is_vector_multiply_add_pattern(float a1, float b1, float a2, float b2, float a3, float b3) {
  // Check for common patterns in 3D vector math:
  
  // Identity pattern: (1,0,0) * (x,y,z) = x
  if (a1 == 1.0f && a2 == 0.0f && a3 == 0.0f) return true;
  if (a1 == 0.0f && a2 == 1.0f && a3 == 0.0f) return true;  
  if (a1 == 0.0f && a2 == 0.0f && a3 == 1.0f) return true;
  
  // Zero vector
  if (a1 == 0.0f && a2 == 0.0f && a3 == 0.0f) return true;
  if (b1 == 0.0f && b2 == 0.0f && b3 == 0.0f) return true;
  
  // Uniform scaling: (s,s,s) * (x,y,z)
  if (a1 == a2 && a2 == a3) return true;
  if (b1 == b2 && b2 == b3) return true;
  
  return false;
}

// Float power-of-two multiplication optimization
MathOptimizer::OptimizationResult MathOptimizer::optimize_float_power_of_two_mul(float multiplier) {
  OptimizationResult result;
  
  // Check if multiplier is a power of 2
  if (multiplier == 2.0f || multiplier == 4.0f || multiplier == 8.0f || 
      multiplier == 16.0f || multiplier == 32.0f || multiplier == 64.0f ||
      multiplier == 128.0f || multiplier == 256.0f || multiplier == 512.0f ||
      multiplier == 1024.0f || multiplier == 2048.0f || multiplier == 4096.0f ||
      multiplier == 8192.0f || multiplier == 16384.0f || multiplier == 32768.0f ||
      multiplier == 65536.0f) {
    result.optimized = true;
    // Mark for special handling - could use SIMD shift-like operations
    // For now, just detect the pattern
    return result;
  }
  
  return result;
}

// Float power-of-two division optimization  
MathOptimizer::OptimizationResult MathOptimizer::optimize_float_power_of_two_div(float divisor) {
  OptimizationResult result;
  
  // Division by power of 2 can be converted to multiplication by reciprocal
  if (divisor == 2.0f || divisor == 4.0f || divisor == 8.0f ||
      divisor == 16.0f || divisor == 32.0f || divisor == 64.0f ||
      divisor == 128.0f || divisor == 256.0f || divisor == 512.0f ||
      divisor == 1024.0f || divisor == 2048.0f || divisor == 4096.0f ||
      divisor == 8192.0f || divisor == 16384.0f || divisor == 32768.0f ||
      divisor == 65536.0f) {
    result.optimized = true;
    result.is_reciprocal_mul = true;
    result.reciprocal_value = 1.0f / divisor;
    return result;
  }
  
  return result;
}

// Compile-time branch elimination for constant conditions
MathOptimizer::OptimizationResult MathOptimizer::optimize_branch_condition(s64 condition) {
  OptimizationResult result;
  
  if (is_always_true_condition(condition)) {
    result.optimized = true;
    result.constant_result = 1; // Always true - branch can be eliminated
    return result;
  }
  
  if (is_always_false_condition(condition)) {
    result.optimized = true;
    result.constant_result = 0; // Always false - branch can be eliminated  
    return result;
  }
  
  return result;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_float_branch_condition(float condition) {
  OptimizationResult result;
  
  // NaN is always false in comparisons
  if (std::isnan(condition)) {
    result.optimized = true;
    result.constant_result = 0;
    return result;
  }
  
  // Infinity handling
  if (std::isinf(condition)) {
    result.optimized = true;
    result.constant_result = (condition > 0) ? 1 : 0;
    return result;
  }
  
  // Zero/non-zero checks
  if (condition == 0.0f) {
    result.optimized = true;
    result.constant_result = 0;
    return result;
  }
  
  if (condition != 0.0f) {
    result.optimized = true; 
    result.constant_result = 1;
    return result;
  }
  
  return result;
}

bool MathOptimizer::is_always_true_condition(s64 value) {
  // Non-zero values are true in boolean context
  return value != 0;
}

bool MathOptimizer::is_always_false_condition(s64 value) {
  // Only zero is false in boolean context
  return value == 0;
}

// Square pattern detection and optimization
bool MathOptimizer::is_square_pattern(RegVal* a, RegVal* b) {
  // Check if both operands refer to the same register (x * x pattern)
  return a == b;
}

MathOptimizer::OptimizationResult MathOptimizer::optimize_square_pattern(RegVal* operand) {
  (void)operand; // Suppress unused parameter warning
  OptimizationResult result;
  result.optimized = true;
  result.is_reciprocal_mul = true; // Reuse this flag to indicate squaring
  // For x*x, we can potentially use more efficient instructions
  // For now, this serves as pattern recognition for higher-level optimizations
  return result;
}

// Vector length pattern optimization for (x*x + y*y + z*z)
MathOptimizer::OptimizationResult MathOptimizer::optimize_vector_length_pattern(float x_coeff, float y_coeff, float z_coeff) {
  OptimizationResult result;
  
  // Check for standard 3D vector length: (1*x² + 1*y² + 1*z²)
  if (x_coeff == 1.0f && y_coeff == 1.0f && z_coeff == 1.0f) {
    result.optimized = true;
    // This could be optimized with SIMD dot product instructions
    return result;
  }
  
  // Check for 2D vector length: (1*x² + 1*y² + 0*z²)
  if (x_coeff == 1.0f && y_coeff == 1.0f && z_coeff == 0.0f) {
    result.optimized = true;
    return result;
  }
  
  // Check for scaled vector length: (s*x² + s*y² + s*z²) = s*(x²+y²+z²)
  if (x_coeff == y_coeff && y_coeff == z_coeff && x_coeff != 0.0f) {
    result.optimized = true;
    result.reciprocal_value = x_coeff; // Store the scaling factor
    return result;
  }
  
  return result;
}

// Fast reciprocal square root optimization (1/sqrt(x))
MathOptimizer::OptimizationResult MathOptimizer::optimize_reciprocal_sqrt(float value) {
  OptimizationResult result;
  
  // Common special cases
  if (value == 1.0f) {
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&value); // 1/sqrt(1) = 1
    return result;
  }
  
  if (value == 4.0f) {
    float res = 0.5f; // 1/sqrt(4) = 1/2 = 0.5
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res);
    return result;
  }
  
  if (value == 0.25f) {
    float res = 2.0f; // 1/sqrt(0.25) = 1/0.5 = 2
    result.optimized = true;
    result.constant_result = *reinterpret_cast<s64*>(&res);
    return result;
  }
  
  // For non-constant values, this could use fast inverse square root approximation
  // But that's beyond compile-time optimization
  return result;
}

} // namespace goalc
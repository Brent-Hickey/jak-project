#pragma once

/*!
 * @file MathOptimizer.h
 * Enhanced mathematical optimizations for GOAL compiler
 * Provides constant folding and strength reduction optimizations
 */

#include <optional>
#include "common/common_types.h"
#include "goalc/compiler/Val.h"

namespace goalc {

class MathOptimizer {
public:
  // Enhanced strength reduction patterns
  struct OptimizationResult {
    bool optimized = false;
    s64 constant_result = 0;
    bool is_shift = false;
    int shift_amount = 0;
    bool is_zero = false;
    bool is_identity = false;
    bool is_negation = false;
    bool is_lea_pattern = false;
    int lea_multiplier = 0;
    int lea_offset = 0;
    bool is_mask_and = false;
    s64 mask_value = 0;
    bool is_reciprocal_mul = false;
    float reciprocal_value = 0.0f;
    bool is_magic_division = false;
    u64 magic_number = 0;
  };

  // Integer optimizations
  static OptimizationResult optimize_add(s64 a, s64 b);
  static OptimizationResult optimize_sub(s64 a, s64 b);
  static OptimizationResult optimize_mul(s64 a, s64 b);
  static OptimizationResult optimize_div(s64 a, s64 b);
  static OptimizationResult optimize_mod(s64 a, s64 b);
  static OptimizationResult optimize_mul_enhanced(s64 a, s64 b);
  
  // Bitwise optimizations
  static OptimizationResult optimize_and(s64 a, s64 b);
  static OptimizationResult optimize_or(s64 a, s64 b);
  static OptimizationResult optimize_xor(s64 a, s64 b);
  
  // Float optimizations
  static OptimizationResult optimize_fadd(float a, float b);
  static OptimizationResult optimize_fsub(float a, float b);
  static OptimizationResult optimize_fmul(float a, float b);
  static OptimizationResult optimize_fdiv(float a, float b);
  static OptimizationResult optimize_float_power_of_two_mul(float multiplier);
  static OptimizationResult optimize_float_power_of_two_div(float divisor);
  
  // Pattern recognition helpers
  static bool is_power_of_two(s64 x);
  static int get_power_of_two_exponent(s64 x);
  static bool is_mask_pattern(s64 x);
  static s64 get_next_power_of_two(s64 x);
  
  // Advanced patterns
  static bool is_multiply_by_constant_add_pattern(s64 multiplier, s64 addend);
  static OptimizationResult optimize_multiply_add(s64 base, s64 multiplier, s64 addend);
  
  // LEA (Load Effective Address) patterns
  static OptimizationResult optimize_lea_pattern(s64 multiplier);
  static bool can_use_lea(s64 multiplier, int* lea_mult, int* lea_offset);
  
  // Enhanced division/modulo
  static OptimizationResult optimize_div_power_of_two(s64 divisor);
  static OptimizationResult optimize_mod_power_of_two(s64 divisor);
  
  // Range-based optimizations
  struct ValueRange {
    s64 min_val = 0;
    s64 max_val = 0;
    bool is_valid = false;
  };
  static OptimizationResult optimize_with_range(s64 constant, ValueRange range, const std::string& op);
  
  // Multiply-accumulate patterns
  static OptimizationResult optimize_fma_pattern(float a, float b, float c); // a + b*c
  static OptimizationResult optimize_fms_pattern(float a, float b, float c); // a - b*c
  
  // Compile-time math evaluation for game engine operations
  static OptimizationResult optimize_fsqrt(float a);
  static OptimizationResult optimize_fsin(float a);
  static OptimizationResult optimize_fcos(float a);
  static OptimizationResult optimize_ftan(float a);
  static OptimizationResult optimize_fabs(float a);
  static OptimizationResult optimize_fpow(float base, float exp);
  static OptimizationResult optimize_fmin(float a, float b);
  static OptimizationResult optimize_fmax(float a, float b);
  
  // Magic number division optimization
  static OptimizationResult optimize_magic_division(s64 divisor);
  
  // Modulo optimization
  static OptimizationResult optimize_modulo(s64 divisor);
  
  // Vector math pattern recognition
  static OptimizationResult optimize_dot_product_pattern(const std::vector<float>& a_components, 
                                                         const std::vector<float>& b_components);
  static bool is_vector_multiply_add_pattern(float a1, float b1, float a2, float b2, float a3, float b3);
  
  // Compile-time branch elimination
  static OptimizationResult optimize_branch_condition(s64 condition);
  static OptimizationResult optimize_float_branch_condition(float condition);
  static bool is_always_true_condition(s64 value);
  static bool is_always_false_condition(s64 value);
  
  // Square and vector length optimizations
  static bool is_square_pattern(RegVal* a, RegVal* b);
  static OptimizationResult optimize_square_pattern(RegVal* operand);
  static OptimizationResult optimize_vector_length_pattern(float x_coeff, float y_coeff, float z_coeff);
  static OptimizationResult optimize_reciprocal_sqrt(float value);
  
private:
  static constexpr s64 MAX_SAFE_INT = (1LL << 62) - 1;
  static constexpr s64 MIN_SAFE_INT = -(1LL << 62);
  static bool safe_add(s64 a, s64 b, s64* result);
  static bool safe_mul(s64 a, s64 b, s64* result);
};

// Helper to extract constant value from Val if possible  
std::optional<s64> extract_integer_constant(Val* val);
std::optional<float> extract_float_constant(Val* val);

}
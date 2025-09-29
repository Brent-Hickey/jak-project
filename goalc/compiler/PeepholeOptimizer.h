#pragma once

#include <vector>
#include <memory>
#include "IR.h"

class FunctionEnv;

/*!
 * IR Peephole Optimizer - performs local pattern-based optimizations on IR sequences
 * This is an optional optimization pass that can improve code quality by recognizing
 * and transforming common inefficient patterns in the IR.
 */
class PeepholeOptimizer {
public:
  struct OptimizationStats {
    int dead_code_eliminated = 0;
    int redundant_moves_eliminated = 0;
    int identity_operations_eliminated = 0;
    int constant_folding_applied = 0;
    int lea_patterns_optimized = 0;
    int magic_divisions_optimized = 0;
    int fma_patterns_optimized = 0;
    int algebraic_patterns_optimized = 0;
    int memory_patterns_optimized = 0;
    int bitmasking_patterns_optimized = 0;
    int goal_patterns_optimized = 0;
    int comparison_patterns_optimized = 0;
    int vector_patterns_optimized = 0;
    int simd_patterns_optimized = 0;
    int vector_normalization_optimized = 0;
    int control_flow_patterns_optimized = 0;
    int symbol_cache_optimized = 0;
    int object_field_access_optimized = 0;
    int stack_frame_optimized = 0;
    int small_constant_immediates_optimized = 0;
    int boolean_comparison_optimized = 0;
    int common_subexpressions_eliminated = 0;
    int total_optimizations() const {
      return dead_code_eliminated + redundant_moves_eliminated + 
             identity_operations_eliminated + constant_folding_applied +
             lea_patterns_optimized + magic_divisions_optimized + fma_patterns_optimized +
             algebraic_patterns_optimized + memory_patterns_optimized + bitmasking_patterns_optimized +
             goal_patterns_optimized + comparison_patterns_optimized + vector_patterns_optimized + simd_patterns_optimized + vector_normalization_optimized +
             control_flow_patterns_optimized + symbol_cache_optimized + 
             object_field_access_optimized + stack_frame_optimized + small_constant_immediates_optimized + boolean_comparison_optimized + common_subexpressions_eliminated;
    }
  };

  /*!
   * Run peephole optimizations on the given IR sequence
   */
  static OptimizationStats optimize(std::vector<std::unique_ptr<IR>>& ir_sequence, FunctionEnv* env);

private:
  /*!
   * Remove dead code (unused register assignments, no-op instructions)
   */
  static int eliminate_dead_code(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Remove redundant register moves (mov r1, r2; mov r2, r1 -> mov r1, r2)
   */
  static int eliminate_redundant_moves(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Eliminate identity operations that were missed by earlier passes
   */
  static int eliminate_identity_operations(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Fold constants in IR that weren't caught by higher-level passes
   */
  static int apply_constant_folding(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Check if an IR instruction has any side effects
   */
  static bool has_side_effects(const IR* instr);

  /*!
   * Check if a register value is used after the given position
   */
  static bool is_reg_used_after(const std::vector<std::unique_ptr<IR>>& ir_sequence, 
                               size_t pos, const RegVal* reg);

  /*!
   * Get the destination register of an IR instruction (if any)
   */
  static const RegVal* get_dest_reg(const IR* instr);

  /*!
   * Get the source registers of an IR instruction
   */
  static std::vector<const RegVal*> get_source_regs(const IR* instr);

  /*!
   * Optimize complex address calculations to use LEA instructions
   */
  static int optimize_lea_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Replace division by constants with multiplication by magic numbers
   */
  static int optimize_magic_division(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Detect and optimize fused multiply-add patterns
   */
  static int optimize_fma_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize bitmasking patterns - combine consecutive masks, etc.
   */
  static int optimize_bitmasking_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize GOAL-specific patterns like type checking, array indexing, etc.
   */
  static int optimize_goal_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize comparison and branching patterns
   */
  static int optimize_comparison_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Check if a value is a power of 2
   */
  static bool is_power_of_2(s64 value);

  /*!
   * Get the log2 of a power of 2 value
   */
  static int get_log2(s64 value);

  /*!
   * Generate magic number for division by constant
   */
  struct MagicDivisionConstants {
    u64 magic;
    int shift;
    bool add_one;
  };
  static MagicDivisionConstants compute_magic_division(s32 divisor);

  /*!
   * Advanced algebraic simplifications
   */
  static int optimize_algebraic_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize load/store patterns and memory access
   */
  static int optimize_memory_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence, FunctionEnv* env);

  /*!
   * Detect and optimize vector math patterns (dot product, cross product, etc.)
   */
  static int optimize_vector_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Detect SIMD promotion opportunities
   */
  static int optimize_simd_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Detect vector normalization patterns
   */
  static int optimize_vector_normalization(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize general control flow patterns (function calls, branching, etc.)
   */
  static int optimize_control_flow_patterns(std::vector<std::unique_ptr<IR>>& ir_sequence, FunctionEnv* env);

  /*!
   * Optimize symbol lookup patterns by caching frequently accessed symbols
   */
  static int optimize_symbol_cache(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize object field access patterns by coalescing consecutive field loads
   */
  static int optimize_object_field_access(std::vector<std::unique_ptr<IR>>& ir_sequence);

  /*!
   * Optimize stack frame operations by reducing redundant push/pop sequences
   */
  static int optimize_stack_frame_operations(std::vector<std::unique_ptr<IR>>& ir_sequence);
  
  /*!
   * Optimize small constant loads followed by arithmetic
   */
  static int optimize_small_constant_immediates(std::vector<std::unique_ptr<IR>>& ir_sequence);
  
  /*!
   * Optimize boolean comparisons with #f (which is 0)
   */
  static int optimize_boolean_comparisons(std::vector<std::unique_ptr<IR>>& ir_sequence);
  
  /*!
   * Optimize redundant constant loads by reusing registers
   */
  static int optimize_redundant_constant_loads(std::vector<std::unique_ptr<IR>>& ir_sequence);
  
  /*!
   * Common subexpression elimination - detect and reuse repeated calculations
   */
  static int optimize_common_subexpressions(std::vector<std::unique_ptr<IR>>& ir_sequence);
};
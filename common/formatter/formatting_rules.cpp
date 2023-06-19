#include "formatting_rules.h"

#include <set>

#include "common/util/string_util.h"

void formatter_rules::blank_lines::separate_by_newline(std::string& curr_text,
                                                       const FormatterTreeNode& containing_node,
                                                       const FormatterTreeNode& node,
                                                       const int index) {
  // We only are concerned with top level forms or elements
  // Skip the last element, no trailing new-lines (let the editors handle this!)
  // Also peek ahead to see if there was a comment on this line, if so don't separate things!
  if (!containing_node.metadata.is_top_level || index >= containing_node.refs.size() - 1 ||
      (containing_node.refs.at(index + 1).metadata.is_comment &&
       containing_node.refs.at(index + 1).metadata.is_inline)) {
    return;
  }
  curr_text += "\n";
  // If it's a comment, but has no following blank lines, dont insert a blank line
  if (node.metadata.is_comment && node.metadata.num_blank_lines_following == 0) {
    return;
  }
  // Otherwise, add only 1 blank line
  curr_text += "\n";
}

// TODO - probably need to include quoted literals as well, though the grammar currently does not
// differentiate between a quoted symbol and a quoted form
const std::set<std::string> constant_pair_types = {"kwd_lit",  "num_lit",  "str_lit", "char_lit",
                                                   "null_lit", "bool_lit", "sym_lit"};

bool formatter_rules::constant_pairs::is_element_second_in_constant_pair(
    const FormatterTreeNode& containing_node,
    const FormatterTreeNode& node,
    const int index) {
  if (containing_node.refs.empty() || index == 0) {
    return false;
  }
  // Ensure that a keyword came before hand
  if (containing_node.refs.at(index - 1).metadata.node_type != "kwd_lit") {
    return false;
  }
  // Check the type of the element
  if (constant_pair_types.find(node.metadata.node_type) != constant_pair_types.end()) {
    return true;
  }
  return false;
}

void IndentationRule::append_newline(std::string& curr_text,
                                     const FormatterTreeNode& node,
                                     const FormatterTreeNode& containing_node,
                                     const int depth,
                                     const int index) {
  if (index <= 0 && !containing_node.metadata.multiple_elements_first_line ||
      index <= 1 && containing_node.metadata.multiple_elements_first_line ||
      containing_node.metadata.is_top_level ||
      (node.metadata.is_comment && node.metadata.is_inline)) {
    return;
  }
  // Check if it's a constant pair
  if (formatter_rules::constant_pairs::is_element_second_in_constant_pair(containing_node, node,
                                                                          index)) {
    return;
  }
  curr_text = str_util::rtrim(curr_text) + "\n";
}

void IndentationRule::indent_token(std::string& curr_text,
                                   const FormatterTreeNode& node,
                                   const FormatterTreeNode& containing_node,
                                   const int depth,
                                   const int index) {
  if (node.metadata.is_top_level) {
    return;
  }
  // If the element is the second element in a constant pair, that means we did not append a
  // new-line before hand so we require no indentation (it's inline with the previous element)
  if (formatter_rules::constant_pairs::is_element_second_in_constant_pair(containing_node, node,
                                                                          index)) {
    return;
  }
  if (containing_node.metadata.multiple_elements_first_line) {
    if (index > 1) {
      // Only apply indentation if we are about to print a normal text token
      // TODO - unsafe
      if (node.token.has_value()) {
        curr_text += str_util::repeat(containing_node.refs.at(0).token.value().length() + 2, " ");
      }
    }
  } else {
    if (index > 0) {
      curr_text += str_util::repeat(depth, " ");
    }
  }
}

void IndentationRule::align_form_lines(std::string& text,
                                       const FormatterTreeNode& node,
                                       const FormatterTreeNode& containing_node) {
  const auto lines = str_util::split(text);
  // TODO - unsafe (breaks on a list of lists)
  int alignment_width = 1;
  if (containing_node.metadata.multiple_elements_first_line) {
    alignment_width = containing_node.refs.at(0).token.value().length() + 2;
  }
  std::string aligned_form = "";
  for (int i = 0; i < lines.size(); i++) {
    aligned_form += str_util::repeat(alignment_width, " ") + lines.at(i);
    if (i != lines.size() - 1) {
      aligned_form += "\n";
    }
  }
  text = aligned_form;
}

void InnerIndentationRule::append_newline(std::string& curr_text,
                                          const FormatterTreeNode& node,
                                          const FormatterTreeNode& containing_node,
                                          const int depth,
                                          const int index) {
  if (index < 1 || (m_depth != depth || m_index && m_index.value() != index)) {
    return;
  }
  // Check if it's a constant pair
  if (formatter_rules::constant_pairs::is_element_second_in_constant_pair(containing_node, node,
                                                                          index)) {
    return;
  }
  if (!node.metadata.was_on_first_line_of_form) {
    curr_text = str_util::rtrim(curr_text) + "\n";
  }
}

void InnerIndentationRule::indent_token(std::string& curr_text,
                                        const FormatterTreeNode& node,
                                        const FormatterTreeNode& containing_node,
                                        const int depth,
                                        const int index) {
  if (index < 1 || (m_depth != depth || m_index && m_index.value() != index)) {
    return;
  }
  // If the element is the second element in a constant pair, that means we did not append a
  // new-line before hand so we require no indentation (it's inline with the previous element)
  if (formatter_rules::constant_pairs::is_element_second_in_constant_pair(containing_node, node,
                                                                          index)) {
    return;
  }
  // We only new-line elements if they were not originally on the first line
  if (!node.metadata.was_on_first_line_of_form) {
    curr_text += str_util::repeat(depth * 2, " ");
  }
}

void InnerIndentationRule::align_form_lines(std::string& text,
                                            const FormatterTreeNode& node,
                                            const FormatterTreeNode& containing_node) {
  if (node.metadata.was_on_first_line_of_form) {
    return;
  }
  const auto lines = str_util::split(text);
  // TODO - unsafe (breaks on a list of lists)
  int alignment_width = 2;
  std::string aligned_form = "";
  for (int i = 0; i < lines.size(); i++) {
    aligned_form += str_util::repeat(alignment_width, " ") + lines.at(i);
    if (i != lines.size() - 1) {
      aligned_form += "\n";
    }
  }
  text = aligned_form;
  return;
}
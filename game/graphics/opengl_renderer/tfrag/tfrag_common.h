#pragma once

#include "common/math/Vector.h"
#include "game/graphics/opengl_renderer/BucketRenderer.h"

struct TfragRenderSettings {
  math::Matrix4f math_camera;
  math::Vector4f hvdf_offset;
  float fog_x;
  int tree_idx;
  float time_of_day_weights[8] = {0};
  math::Vector4f planes[4];
  bool do_culling = false;
  bool debug_culling = false;
  // todo occlusion culling string.
};

enum class DoubleDrawKind { NONE, AFAIL_NO_DEPTH_WRITE };

struct DoubleDraw {
  DoubleDrawKind kind = DoubleDrawKind::NONE;
  float aref = 0.;
};

DoubleDraw setup_tfrag_shader(const TfragRenderSettings& /*settings*/,
                              SharedRenderState* render_state,
                              DrawMode mode);
void first_tfrag_draw_setup(const TfragRenderSettings& settings, SharedRenderState* render_state);
void interp_time_of_day_slow(const float weights[8],
                             const std::vector<tfrag3::TimeOfDayColor>& in,
                             math::Vector<u8, 4>* out);

struct SwizzledTimeOfDay {
  std::vector<u8> data;
  u32 color_count = 0;
};

SwizzledTimeOfDay swizzle_time_of_day(const std::vector<tfrag3::TimeOfDayColor>& in);

void interp_time_of_day_fast(const float weights[8],
                             const SwizzledTimeOfDay& in,
                             math::Vector<u8, 4>* out);

void cull_check_all_slow(const math::Vector4f* planes,
                         const std::vector<tfrag3::VisNode>& nodes,
                         u8* out);
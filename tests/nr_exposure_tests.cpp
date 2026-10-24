// Run through CheekyTests --nr-exposure. Pure header logic; no GPU.
#include "../src/nr_exposure.hpp"
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {
using namespace cheeky::foveated_dlss;
void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}
bool close_to(float actual, float expected, float tolerance = 1e-6F) {
    return std::fabs(actual - expected) <= tolerance * (std::fabs(expected) > 1.0F ? std::fabs(expected) : 1.0F);
}
constexpr std::uint32_t r32g32b32a32_float = 2U, r16g16b16a16_float = 10U, r32g32_float = 16U,
    r11g11b10_float = 26U, r16g16_float = 34U, r32_float = 41U, r16_float = 54U;
constexpr float nan = std::numeric_limits<float>::quiet_NaN();
constexpr float infinity = std::numeric_limits<float>::infinity();

void test_decode_mode_follows_the_marker_exposure_formats() {
    for (const auto format : {r32g32b32a32_float, r32g32_float, r32_float})
        require(nr_exposure_decode_mode(format) == 1U, "32-bit float exposure formats decode as float32");
    for (const auto format : {r16g16b16a16_float, r16g16_float, r16_float})
        require(nr_exposure_decode_mode(format) == 2U, "16-bit float exposure formats decode as float16");
    require(nr_exposure_decode_mode(r11g11b10_float) == 0U, "Packed color formats are not exposure textures");
    require(nr_exposure_decode_mode(0U) == 0U, "Unknown format has no decode mode");
}

void test_multiplier_is_pre_exposure_over_scale_or_zero() {
    require(close_to(nr_exposure_multiplier(1.0F, 1.0F), 1.0F), "Default pre-exposure and scale give multiplier one");
    require(close_to(nr_exposure_multiplier(0.0625F, 2.0F), 0.03125F), "Multiplier divides pre-exposure by scale");
    require(nr_exposure_multiplier(0.0F, 1.0F) == 0.0F, "Zero pre-exposure is invalid");
    require(nr_exposure_multiplier(1.0F, 0.0F) == 0.0F, "Zero scale is invalid");
    require(nr_exposure_multiplier(-1.0F, 1.0F) == 0.0F, "Negative pre-exposure is invalid");
    require(nr_exposure_multiplier(nan, 1.0F) == 0.0F, "NaN pre-exposure is invalid");
    require(nr_exposure_multiplier(1.0F, infinity) == 0.0F, "Infinite scale is invalid");
}

void test_binding_requires_hdr_setting_and_a_supported_1x1_texture() {
    const auto bound = resolve_nr_exposure(true, true, true, r32g32_float, 1U, 1U, 1.0F, 1.0F);
    require(bound.mode == 1U && close_to(bound.multiplier, 1.0F), "GTA's R32G32 1x1 exposure binds as float32");
    const auto half = resolve_nr_exposure(true, true, true, r16g16b16a16_float, 1U, 1U, 0.5F, 2.0F);
    require(half.mode == 2U && close_to(half.multiplier, 0.25F), "Half-float exposure binds as float16 with its multiplier");
    require(resolve_nr_exposure(false, true, true, r32g32_float, 1U, 1U, 1.0F, 1.0F).mode == 0U,
        "SDR input never binds the exposure");
    require(resolve_nr_exposure(true, false, true, r32g32_float, 1U, 1U, 1.0F, 1.0F).mode == 0U,
        "The setting can disable the game exposure");
    require(resolve_nr_exposure(true, true, false, r32g32_float, 1U, 1U, 1.0F, 1.0F).mode == 0U,
        "A missing texture leaves the fixed paper white");
    require(resolve_nr_exposure(true, true, true, r11g11b10_float, 1U, 1U, 1.0F, 1.0F).mode == 0U,
        "Unsupported formats leave the fixed paper white");
    require(resolve_nr_exposure(true, true, true, r32g32_float, 2U, 1U, 1.0F, 1.0F).mode == 0U,
        "Only 1x1 textures are exposure values");
    require(resolve_nr_exposure(true, true, true, r32g32_float, 1U, 1U, 0.0F, 1.0F).mode == 0U,
        "An invalid multiplier leaves the fixed paper white");
}

void test_white_reference_matches_the_shader_formula() {
    require(close_to(nr_exposure_white(1.0F, 1.0F, 0.125F), 8.0F), "Exposure 1/8 makes scene value 8 the paper white");
    require(close_to(nr_exposure_white(2.0F, 1.0F, 0.125F), 16.0F), "The paper white slider scales the exposure white");
    require(close_to(nr_exposure_white(1.0F, 0.5F, 0.125F), 4.0F), "Pre-exposure over scale multiplies the white");
    require(close_to(nr_exposure_white(1.0F, 0.0F, 0.125F), 1.0F), "No binding keeps the fixed paper white");
    require(close_to(nr_exposure_white(1.0F, 1.0F, 0.0F), 1.0F), "Zero exposure keeps the fixed paper white");
    require(close_to(nr_exposure_white(1.0F, 1.0F, nan), 1.0F), "NaN exposure keeps the fixed paper white");
    require(close_to(nr_exposure_white(1.0F, 1.0F, -1.0F), 1.0F), "Negative exposure keeps the fixed paper white");
    require(close_to(nr_exposure_white(1.0F, 1.0F, 1e-9F), 65504.0F), "The white is bounded to the half-float range");
    require(close_to(nr_exposure_white(1.0F, 1.0F, 1e9F), 1e-6F), "The white never reaches zero");
    require(close_to(nr_exposure_white(0.0F, 1.0F, 0.125F), 0.0001F * 8.0F), "Paper white keeps the shader's lower bound");
}
void test_highlight_rolloff_is_identity_below_the_knee_and_never_clips() {
    const float knee = 0.8F;
    require(close_to(nr_highlight_curve(0.5F, knee), 0.5F), "Below the knee the curve is the identity");
    require(close_to(nr_highlight_curve(knee, knee), knee), "The curve is continuous at the knee");
    const float step = 1e-3F;
    require(close_to((nr_highlight_curve(knee + step, knee) - knee) / step, 1.0F, 1e-2F),
        "The slope stays one at the knee, so the roll-off starts invisibly");
    float previous = 0.0F;
    for (float value = 0.0F; value <= 64.0F; value += 0.25F) {
        const float curve = nr_highlight_curve(value, knee);
        require(curve >= previous && curve <= 1.0F, "The curve is monotonic and never exceeds white");
        previous = curve;
    }
    require(nr_highlight_curve(4.0F, knee) < 1.0F, "Moderate highlights stay below white");
    require(nr_highlight_curve(1000.0F, knee) > 0.9999F, "Very bright values approach white");
    require(nr_highlight_curve(4.0F, 0.0F) == 4.0F, "A zero knee disables the roll-off");
    require(nr_highlight_curve(4.0F, 1.0F) == 4.0F, "A knee at white or above disables the roll-off");
    require(nr_highlight_curve(4.0F, nan) == 4.0F, "A NaN knee disables the roll-off");
}

void test_highlight_scale_keeps_channel_ratios() {
    const float knee = 0.8F;
    const float scale = nr_highlight_scale(4.0F, knee);
    require(close_to(4.0F * scale, nr_highlight_curve(4.0F, knee)), "The scale maps the maximum channel onto the curve");
    require(scale > 0.0F && scale < 1.0F, "Over-range pixels are scaled down, never inverted");
    require(close_to(nr_highlight_scale(0.5F, knee), 1.0F), "In-range pixels are unscaled");
    require(nr_highlight_scale(0.0F, knee) == 1.0F, "Black is unscaled rather than divided by zero");
    require(nr_highlight_scale(4.0F, 0.0F) == 1.0F, "A disabled roll-off leaves the clamp to the encoder");
    // A sunlit (2, 1.5, 1) pixel seen from shade keeps its hue: the ratios survive.
    const float sky = nr_highlight_scale(2.0F, knee);
    require(close_to((2.0F * sky) / (1.0F * sky), 2.0F) && close_to((1.5F * sky) / (1.0F * sky), 1.5F),
        "Channel ratios are unchanged by the roll-off");
}
}  // namespace

int run_nr_exposure_tests() {
    try {
        test_decode_mode_follows_the_marker_exposure_formats();
        test_multiplier_is_pre_exposure_over_scale_or_zero();
        test_binding_requires_hdr_setting_and_a_supported_1x1_texture();
        test_white_reference_matches_the_shader_formula();
        test_highlight_rolloff_is_identity_below_the_knee_and_never_clips();
        test_highlight_scale_keeps_channel_ratios();
        std::puts("PASS NR game exposure binding, white reference and highlight roll-off");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

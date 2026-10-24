#pragma once
// Game exposure for the DLSS-NR codec. HDR DLSS input is scene-referred: GTA V
// Enhanced, for example, passes linear color with pre-exposure 1 and a 1x1
// exposure texture. Dividing by a fixed paper white clips daylight to white
// before the model sees it, and the luminance restore then returns grey. The
// codec instead divides by paper_white * pre_exposure / (exposure * scale),
// the same normalization the SR marker path uses (see debug_marker12.hpp).
// This header is free of Windows includes so its logic runs in plain tests.
#include <cmath>
#include <cstdint>

namespace cheeky::foveated_dlss {

// How the codec shader decodes the first channel of the private exposure copy.
// Written to CodecConstants::exposure_mode.
enum class NrExposureDecode : std::uint32_t { none = 0U, float32 = 1U, float16 = 2U };

struct NrExposureBinding {
    std::uint32_t mode{};   // NrExposureDecode; zero keeps the fixed paper white.
    float multiplier{};     // pre_exposure / exposure_scale, zero when unbound.
};

// DXGI_FORMAT values of the 1x1 exposure textures the marker path accepts
// (debug_exposure_supported). Red is the first channel for every one of them.
[[nodiscard]] constexpr std::uint32_t nr_exposure_decode_mode(const std::uint32_t dxgi_format) noexcept {
    switch (dxgi_format) {
    case 2U:   // DXGI_FORMAT_R32G32B32A32_FLOAT
    case 16U:  // DXGI_FORMAT_R32G32_FLOAT
    case 41U:  // DXGI_FORMAT_R32_FLOAT
        return static_cast<std::uint32_t>(NrExposureDecode::float32);
    case 10U:  // DXGI_FORMAT_R16G16B16A16_FLOAT
    case 34U:  // DXGI_FORMAT_R16G16_FLOAT
    case 54U:  // DXGI_FORMAT_R16_FLOAT
        return static_cast<std::uint32_t>(NrExposureDecode::float16);
    default:
        return static_cast<std::uint32_t>(NrExposureDecode::none);
    }
}

// pre_exposure / exposure_scale, or zero when either value cannot be used.
[[nodiscard]] inline float nr_exposure_multiplier(const float pre_exposure, const float exposure_scale) noexcept {
    if (!std::isfinite(pre_exposure) || !std::isfinite(exposure_scale) ||
        pre_exposure <= 0.0F || exposure_scale <= 0.0F) return 0.0F;
    const float multiplier = pre_exposure / exposure_scale;
    return std::isfinite(multiplier) && multiplier > 0.0F ? multiplier : 0.0F;
}

[[nodiscard]] inline NrExposureBinding resolve_nr_exposure(
    const bool hdr_input,
    const bool use_game_exposure,
    const bool texture_present,
    const std::uint32_t dxgi_format,
    const std::uint64_t width,
    const std::uint32_t height,
    const float pre_exposure,
    const float exposure_scale
) noexcept {
    if (!hdr_input || !use_game_exposure || !texture_present || width != 1U || height != 1U) return {};
    const auto mode = nr_exposure_decode_mode(dxgi_format);
    const auto multiplier = nr_exposure_multiplier(pre_exposure, exposure_scale);
    if (mode == 0U || multiplier <= 0.0F) return {};
    return {mode, multiplier};
}

// Highlight roll-off above the white (CPU reference of the shader's
// CompressHighlights). Values up to the knee are unchanged. Above it the
// rational shoulder knee + span * t / (t + span), with t = value - knee and
// span = 1 - knee, continues with slope one and approaches white without
// reaching it, keeping gradation far above white, so a sunlit exterior seen
// from shade keeps its hue instead of clamping to white in every channel. A
// knee at zero, at white or above, or non-finite disables the roll-off.
[[nodiscard]] inline float nr_highlight_curve(const float value, const float knee) noexcept {
    if (!std::isfinite(knee) || knee <= 0.0F || knee >= 1.0F || !(value > knee)) return value;
    const float span = 1.0F - knee;
    const float excess = value - knee;
    return knee + span * excess / (excess + span);
}

// Per-pixel factor that maps the maximum channel onto the curve; applying it to
// all three channels preserves their ratios. One for black and in-range pixels.
[[nodiscard]] inline float nr_highlight_scale(const float max_channel, const float knee) noexcept {
    if (!(max_channel > 0.0F)) return 1.0F;
    return nr_highlight_curve(max_channel, knee) / max_channel;
}

// CPU reference of the shader's per-frame white, for tests and diagnostics:
// max(paper_white, 0.0001) * clamp(multiplier / exposure, 1e-6, 65504), or the
// paper white alone when the exposure is unbound or unusable.
[[nodiscard]] inline float nr_exposure_white(const float paper_white, const float multiplier, const float exposure) noexcept {
    const float base = paper_white > 0.0001F ? paper_white : 0.0001F;
    if (multiplier <= 0.0F || !std::isfinite(exposure) || exposure <= 0.0F) return base;
    const float white = multiplier / exposure;
    return base * (white < 1e-6F ? 1e-6F : white > 65504.0F ? 65504.0F : white);
}

}  // namespace cheeky::foveated_dlss

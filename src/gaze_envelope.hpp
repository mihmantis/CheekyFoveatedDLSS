#pragma once

#include "gaze_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace cheeky::foveated_dlss {

// One submitted OpenXR view: its field of view as signed view-space tangents
// and the projection rectangle the application released for it.
struct GazeEnvelopeView {
    GazeProjection fov{};
    std::int32_t rect_x{};
    std::int32_t rect_y{};
    std::uint32_t rect_width{};
    std::uint32_t rect_height{};
    std::uint32_t array_index{};
    std::uint64_t resource_identity{};
    std::uint64_t swapchain_identity{};
    bool resource_valid{};
};

struct GazeEnvelopeInput {
    std::array<GazeEnvelopeView, 2U> views{};
    std::uint32_t view_count{};
    // The game's projection for this DLSS view when Streamline supplied one.
    GazeProjection camera{};
    std::uint32_t output_origin_x{};
    std::uint32_t output_origin_y{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
};

enum class GazeEnvelopeProof : std::uint32_t { none, camera, dimensions };

struct GazeEnvelopeResult {
    // Symmetric frustum covering both eyes; set whenever both FOVs are usable.
    GazeProjection frustum{};
    // Output size an envelope render of the eye rectangles would have.
    std::uint32_t expected_width{};
    std::uint32_t expected_height{};
    GazeEnvelopeProof proof{GazeEnvelopeProof::none};
    bool valid{};
};

namespace gaze_envelope_detail {
inline bool usable_fov(const GazeProjection& p) noexcept {
    float u{}, v{};
    // Reject degenerate and near-180-degree frusta (tangent 60 is 89.05 degrees).
    return projection_forward_center(p, u, v) && -p.left < 60.F && p.right < 60.F &&
        p.up < 60.F && -p.down < 60.F;
}
}  // namespace gaze_envelope_detail

// Alternating-eye hosts evaluate one DLSS feature once per XR frame and show
// each eye a window of a symmetric render that envelopes both eyes' asymmetric
// fields of view. Inside that envelope a gaze direction has the same pixel
// from either eye, so no eye identity is needed; only the envelope must be
// proven. Proof is the game's own symmetric camera projection or, failing
// that, the pixel-exact size fingerprint: an envelope render of the eye
// rectangle is wider by (full envelope width) / (widest eye span), and taller
// likewise. Both eyes must be whole, separate images of one size, so packed
// and sliced layouts keep their own routes.
[[nodiscard]] inline GazeEnvelopeResult resolve_gaze_envelope(
    const GazeEnvelopeInput& input
) noexcept {
    using gaze_envelope_detail::usable_fov;
    GazeEnvelopeResult result{};
    if (input.view_count != 2U) return result;
    const auto& first = input.views[0];
    const auto& second = input.views[1];
    if (!usable_fov(first.fov) || !usable_fov(second.fov)) return result;
    const float half_width = (std::max)({-first.fov.left, first.fov.right,
        -second.fov.left, second.fov.right});
    const float half_height = (std::max)({first.fov.up, -first.fov.down,
        second.fov.up, -second.fov.down});
    result.frustum = {-half_width, half_width, half_height, -half_height, true};
    if (!first.resource_valid || !second.resource_valid ||
        first.array_index != 0U || second.array_index != 0U ||
        first.rect_x < 0 || first.rect_y < 0 || second.rect_x < 0 || second.rect_y < 0 ||
        first.rect_width == 0U || first.rect_height == 0U ||
        first.rect_width != second.rect_width || first.rect_height != second.rect_height ||
        first.resource_identity == 0U || second.resource_identity == 0U ||
        first.resource_identity == second.resource_identity ||
        first.swapchain_identity == 0U || second.swapchain_identity == 0U ||
        first.swapchain_identity == second.swapchain_identity ||
        input.output_origin_x != 0U || input.output_origin_y != 0U ||
        input.output_width == 0U || input.output_height == 0U) {
        return result;
    }
    const double span_width = (std::max)(
        double(first.fov.right) - double(first.fov.left),
        double(second.fov.right) - double(second.fov.left));
    const double span_height = (std::max)(
        double(first.fov.up) - double(first.fov.down),
        double(second.fov.up) - double(second.fov.down));
    const double expected_width = first.rect_width * 2.0 * half_width / span_width;
    const double expected_height = first.rect_height * 2.0 * half_height / span_height;
    if (!std::isfinite(expected_width) || !std::isfinite(expected_height) ||
        expected_width < 1.0 || expected_height < 1.0 ||
        expected_width > 65535.0 || expected_height > 65535.0) {
        return result;
    }
    result.expected_width = static_cast<std::uint32_t>(std::lround(expected_width));
    result.expected_height = static_cast<std::uint32_t>(std::lround(expected_height));
    if (gaze_projection_matches(input.camera, result.frustum)) {
        result.proof = GazeEnvelopeProof::camera;
        result.valid = true;
        return result;
    }
    // Hosts may round either way when sizing the render target.
    if (std::abs(expected_width - double(input.output_width)) <= 1.5 &&
        std::abs(expected_height - double(input.output_height)) <= 1.5) {
        result.proof = GazeEnvelopeProof::dimensions;
        result.valid = true;
    }
    return result;
}

// Re-express a view-local UV (0..1 across that view's field of view) as a UV
// inside the envelope. Gaze slightly outside an eye image can still lie inside
// the larger envelope; the result is clamped to the envelope.
[[nodiscard]] inline bool project_into_gaze_envelope(
    const GazeProjection& view,
    const GazeProjection& envelope,
    float& u,
    float& v
) noexcept {
    using gaze_envelope_detail::usable_fov;
    if (!usable_fov(view) || !usable_fov(envelope) || !std::isfinite(u) || !std::isfinite(v))
        return false;
    const float ray_x = view.left + u * (view.right - view.left);
    const float ray_y = view.up - v * (view.up - view.down);
    const float envelope_u = (ray_x - envelope.left) / (envelope.right - envelope.left);
    const float envelope_v = (envelope.up - ray_y) / (envelope.up - envelope.down);
    if (!std::isfinite(envelope_u) || !std::isfinite(envelope_v)) return false;
    u = std::clamp(envelope_u, 0.F, 1.F);
    v = std::clamp(envelope_v, 0.F, 1.F);
    return true;
}

}  // namespace cheeky::foveated_dlss

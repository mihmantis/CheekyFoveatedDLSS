#pragma once

#include "ngx_abi.hpp"
#include "ngx_evaluation_extent.hpp"
#include "frame_contract.hpp"
#include "settings.hpp"
#include "dlss_nr.hpp"

#include <cstddef>
#include <cstdint>

namespace cheeky::foveated_dlss {

struct D3D11Evaluation;
struct D3D12Evaluation;

using D3D12CreateFeatureFn = NgxResult (*)(
    ID3D12GraphicsCommandList*,
    std::uint32_t,
    NgxParameters*,
    NgxHandle**
);
using D3D12EvaluateFeatureFn = NgxResult (*)(
    ID3D12GraphicsCommandList*,
    const NgxHandle*,
    const NgxParameters*,
    NgxProgressCallback
);
using D3D12ReleaseFeatureFn = NgxResult (*)(NgxHandle*);

struct D3D12BackendCallbacks {
    D3D12CreateFeatureFn create_feature{};
    D3D12EvaluateFeatureFn evaluate_feature{};
    D3D12ReleaseFeatureFn release_feature{};
};

struct D3D12BackendTiming {
    ID3D12QueryHeap* query_heap{};
    std::uint32_t begin_query_index{};
    std::uint32_t end_query_index{1U};
    bool write_begin_timestamp{};
    bool sr_timestamp_written{};
};

[[nodiscard]] NgxResult evaluate_d3d12_backend(
    ID3D12GraphicsCommandList* command_list,
    const DlssFrameContract& contract,
    const D3D12DlssInputs& inputs,
    NgxParameters* parameters,
    const CropGeometry& crop,
    const D3D12BackendCallbacks& callbacks,
    D3D12BackendTiming* timing = nullptr,
    const CropGeometry* display_crop = nullptr
) noexcept;

void release_d3d12_view(DlssViewId view_id) noexcept;
void skip_d3d12_history(DlssViewId view_id) noexcept;

[[nodiscard]] bool composite_d3d11_crop(
    ID3D11DeviceContext* context,
    ID3D11Resource* game_color,
    ID3D11Resource* game_output,
    ID3D11Resource* packed_dlss_output,
    const DlssFrameContract& contract,
    const CropGeometry& crop,
    const Settings& settings
) noexcept;

[[nodiscard]] D3D11Evaluation* prepare_d3d11(
    ID3D11DeviceContext* context,
    const NgxParameters* parameters,
    const Settings& settings
) noexcept;

void finish_d3d11(
    ID3D11DeviceContext* context,
    const NgxParameters* parameters,
    D3D11Evaluation* evaluation,
    NgxResult result
) noexcept;

void d3d11_set_composite_base(
    D3D11Evaluation* evaluation,
    ID3D11ShaderResourceView* base_srv,
    std::uint32_t width,
    std::uint32_t height
) noexcept;

// Output extent observed when the game created this DX11 feature; empty for
// features adopted after attachment.
[[nodiscard]] NgxOutputExtent d3d11_game_output_extent(
    const NgxHandle* game_handle
) noexcept;

[[nodiscard]] D3D12Evaluation* prepare_d3d12(
    ID3D12GraphicsCommandList* command_list,
    const NgxParameters* parameters,
    DlssViewId view_id,
    const Settings& settings
) noexcept;

void finish_d3d12(
    ID3D12GraphicsCommandList* command_list,
    const NgxParameters* parameters,
    D3D12Evaluation* evaluation,
    NgxResult result
) noexcept;

[[nodiscard]] D3D12Evaluation* prepare_d3d12_streamline(
    ID3D12GraphicsCommandList* command_list,
    ID3D12Resource* color,
    ID3D12Resource* output,
    std::uint32_t render_width,
    std::uint32_t render_height,
    std::uint32_t output_width,
    std::uint32_t output_height,
    std::uint32_t color_x,
    std::uint32_t color_y,
    std::uint32_t output_x,
    std::uint32_t output_y,
    DlssViewId view_id,
    const Settings& settings,
    bool diagnostic_trace,
    std::uint64_t diagnostic_sequence,
    D3D12_RESOURCE_STATES color_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATES output_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS
) noexcept;

[[nodiscard]] ID3D12Resource* d3d12_private_output(
    const D3D12Evaluation* evaluation
) noexcept;

void d3d12_align_reconstruction_grid(D3D12Evaluation*) noexcept;
[[nodiscard]] bool d3d12_set_composite_base(
    D3D12Evaluation* evaluation,
    ID3D12Resource* low_resolution_color,
    std::uint32_t input_base_x = 0U,
    std::uint32_t input_base_y = 0U
) noexcept;

[[nodiscard]] CropGeometry d3d12_reconstruction_crop(const D3D12Evaluation*) noexcept;

[[nodiscard]] CropGeometry d3d12_evaluation_crop(
    const D3D12Evaluation* evaluation
) noexcept;

[[nodiscard]] FoveationCenter d3d12_evaluation_center(const D3D12Evaluation* evaluation) noexcept;

[[nodiscard]] bool d3d12_evaluation_gaze_reset(
    const D3D12Evaluation* evaluation
) noexcept;

[[nodiscard]] bool d3d12_evaluation_low_res_motion(const D3D12Evaluation*) noexcept;

void finish_d3d12_streamline(
    ID3D12GraphicsCommandList* command_list,
    D3D12Evaluation* evaluation,
    bool succeeded
) noexcept;

void release_d3d11_resources() noexcept;
void collect_d3d12_resources() noexcept;
void release_d3d12_resources() noexcept;

// Reasons prepare_d3d12/prepare_d3d12_streamline leave an evaluation to the
// game's DLSS. Counts accumulate since startup; the first two of each reason
// are also traced.
enum class D3D12PrepareProbe : std::size_t {
    disabled,
    invalid_arguments,
    missing_resources,
    invalid_dimensions,
    coordinated_crop,
    get_device,
    resource_arguments,
    output_contract,
    resource_allocation,
    private_output,
    descriptor_heap,
    serialize_root_signature,
    create_root_signature,
    create_pipeline,
    resources_unavailable,
    evaluation_allocation,
    count,
};
[[nodiscard]] const char* d3d12_prepare_probe_name(D3D12PrepareProbe probe) noexcept;
[[nodiscard]] std::uint32_t d3d12_prepare_rejections(D3D12PrepareProbe probe) noexcept;

// Private D3D12 resource sets hold one crop-sized scratch DLSS output each and
// stay reserved by the command-list recording that used them until that list
// is reset and GPU-complete. When every set is reserved, the evaluation falls
// back to the game's DLSS and counts as an exhaustion.
struct D3D12ResourceCacheStatus {
    std::uint32_t capacity{};
    std::uint32_t entries{};
    std::uint32_t reserved{};
    // Highest reserved count observed: the demand a capacity has to cover.
    std::uint32_t peak_reserved{};
    std::uint64_t exhaustions{};
};
[[nodiscard]] D3D12ResourceCacheStatus d3d12_resource_cache_status() noexcept;
// True once after a prepare on the calling thread was rejected by a full cache.
[[nodiscard]] bool d3d12_take_resource_cache_exhaustion() noexcept;

}  // namespace cheeky::foveated_dlss

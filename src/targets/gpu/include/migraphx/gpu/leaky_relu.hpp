#ifndef MIGRAPHX_GUARD_RTGLIB_LEAKY_RELU_HPP
#define MIGRAPHX_GUARD_RTGLIB_LEAKY_RELU_HPP

#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_leaky_relu
{
    shared<activation_descriptor> ad;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return gpu::reflect(self.ad.get(), f);
    }

    std::string name() const { return "gpu::leaky_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

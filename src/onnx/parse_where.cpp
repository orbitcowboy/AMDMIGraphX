#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_where : op_parser<parse_where>
{
    std::vector<op_desc> operators() const { return {{"Where"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto lens = compute_broadcasted_lens(args[0]->get_shape().lens(), args[1]->get_shape().lens());
        lens      = compute_broadcasted_lens(lens, args[2]->get_shape().lens());
        if(args[1]->get_shape().lens() != lens)
        {
            args[1] =
                info.add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), args[1]);
        }

        if(args[2]->get_shape().lens() != lens)
        {
            args[2] =
                info.add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), args[2]);
        }
        if (args[0]->get_shape().elements() == 1)
        {
            auto cond =
                info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), args[0]);
            return info.add_instruction(make_op("if"), cond, args.at(1), args.at(2));
        }

        auto cond =
            info.add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), args[0]);
        if(cond->get_shape().lens() != lens)
        {
            cond = info.add_instruction(make_op("multibroadcast", {{"output_lens", lens}}), cond);
        }

        // compute index
        auto elem_num = args[1]->get_shape().elements();

        // concatenation of input data
        auto concat_data = info.add_instruction(make_op("concat", {{"axis", 0}}), args[2], args[1]);
        std::vector<int64_t> dims = {static_cast<int64_t>(2 * elem_num)};
        auto rsp_data = info.add_instruction(make_op("reshape", {{"dims", dims}}), concat_data);

        std::vector<int> ind(elem_num);
        std::iota(ind.begin(), ind.end(), 0);
        shape ind_s{shape::int32_type, lens};
        auto l_ind = info.add_literal(literal(ind_s, ind));
        std::vector<int> offset(elem_num, elem_num);
        auto l_offset   = info.add_literal(literal({shape::int32_type, lens}, offset));
        auto ins_offset = info.add_instruction(make_op("mul"), l_offset, cond);
        auto ins_ind    = info.add_instruction(make_op("add"), ins_offset, l_ind);

        return info.add_instruction(make_op("gather", {{"axis", 0}}), rsp_data, ins_ind);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

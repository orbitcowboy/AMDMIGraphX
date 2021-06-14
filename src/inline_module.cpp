#include <migraphx/inline_module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void inline_submodule(module& m, instruction_ref ins)
{

    auto arg_cond = ins->inputs().front()->eval();
    assert(not arg_cond.empty());
    const auto& mod_inputs = ins->module_inputs();
    const auto* smod       = (arg_cond.at<bool>()) ? mod_inputs.at(0) : mod_inputs.at(1);

    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::vector<instruction_ref> mod_outputs;
    for(auto sins : iterator_for(*smod))
    {
        instruction_ref copy_ins{};
        if(sins->name() == "@literal")
        {
            auto l   = sins->get_literal();
            copy_ins = m.add_literal(l);
        }
        else if(sins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(sins->get_operator()).parameter;
            auto s      = sins->get_shape();
            copy_ins    = m.add_parameter(name, s);
        }
        else if(sins->name() == "@outline")
        {
            auto s   = sins->get_shape();
            copy_ins = m.add_outline(s);
        }
        else
        {
            auto mod_args = sins->module_inputs();
            auto inputs   = sins->inputs();
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {
                return contains(map_ins, i) ? map_ins[i] : i;
            });

            if(sins->name() == "@return")
            {
                mod_outputs = copy_inputs;
                break;
            }

            copy_ins = m.insert_instruction(ins, sins->get_operator(), copy_inputs, mod_args);
        }
        map_ins[sins] = copy_ins;
        mod_outputs   = {copy_ins};
    }

    auto ins_outputs = ins->outputs();
    assert(mod_outputs.size() >= ins_outputs.size());
    for(const auto& out : ins_outputs)
    {
        auto val = out->get_operator().to_value();
        assert(val.contains("index"));
        auto index = val.at("index").to<std::size_t>();
        m.replace_instruction(out, mod_outputs.at(index));
    }
}

void inline_module::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "if")
            continue;

        auto arg_cond          = ins->inputs().front()->eval();
        const auto& mod_inputs = ins->module_inputs();
        // condition is not constant, but both subgraph outputs
        // are constant, so we can replace each subgraph with
        // a literal
        if(arg_cond.empty())
        {
            std::vector<argument> arg_outs;
            for(const auto& mod : mod_inputs)
            {
                auto last = std::prev(mod->end());
                std::vector<instruction_ref> mod_outputs;
                if(last->name() == "@return")
                {
                    mod_outputs = last->inputs();
                }
                else
                {
                    // no return instruction, last instruction is output
                    mod_outputs.push_back(last);
                }

                for(const auto& out : mod_outputs)
                {
                    auto mod_out = out->eval();
                    if(mod_out.empty())
                        return;
                    arg_outs.push_back(mod_out);
                }
            }

            auto out_num            = arg_outs.size() / 2;
            auto cond               = ins->inputs().front();
            const auto& ins_outputs = ins->outputs();
            for(std::size_t i = 0; i < out_num; ++i)
            {
                auto l0 = m.add_literal(literal(arg_outs.at(i).get_shape(), arg_outs.at(i).data()));
                auto l1 = m.add_literal(
                    literal(arg_outs.at(i + out_num).get_shape(), arg_outs.at(i + out_num).data()));
std::cout << "loc1" << std::endl;
                auto r = m.insert_instruction(ins, make_op("if"), {cond, l0, l1}, {});
std::cout << "loc2" << std::endl;
                m.replace_instruction(ins_outputs.at(i), r);
std::cout << "loc3" << std::endl;
            }
        }
        // cond is constant, inline the corresponding subgraph and discard the other one
        else
        {
            inline_submodule(m, ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

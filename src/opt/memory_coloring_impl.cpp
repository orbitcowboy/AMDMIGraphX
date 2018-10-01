#include "memory_coloring_impl.hpp"

namespace migraph {

void memory_coloring_impl::run()
{
    if (enabled(MIGRAPH_UNIFY_MEMORY_COLORING{}))
        unify_literals = true;
    MIGRAPH_DEBUG(dump("---Before memory coloring---"));
    MIGRAPH_DEBUG(dump_program());
    register_operand_alias();
    build();
    if(num_of_lives != 0)
    {
        MIGRAPH_DEBUG(dump_intervals());
        // Coloring
        while(!alloc_queue.empty())
        {
            interval_ptr interval = alloc_queue.top();
            allocate(interval);
            alloc_queue.pop();
        }
        rewrite();
        MIGRAPH_DEBUG(verify());
    }
}

bool memory_coloring_impl::allocate(interval_ptr interval)
{
    shape s          = interval->result;
    std::size_t size = s.bytes();
    if(size == 0)
        return false;
    std::size_t element_size = size / s.elements();
    live_range& segment      = interval->segment;
    int vn                   = segment.vn;
    std::priority_queue<live_range*, std::vector<live_range*>, ordering> conflict_queue;
    std::unordered_map<long long, live_range*> offset2_live;
    offset2_live.clear();

    if(conflict_table.find(vn) != conflict_table.end())
    {
        std::set<int>& vn_set = conflict_table[vn];
        for(auto& iter : vn_set)
        {
            live_range* range = live_ranges[iter];
            long long offset  = range->offset;
            if(offset != invalid_offset)
            {
                conflict_queue.push(range);
                if(offset2_live.find(offset) == offset2_live.end())
                {
                    offset2_live[offset] = range;
                }
                else
                {
                    live_range* prev = offset2_live[offset];
                    assert(prev->offset == offset);
                    if(prev->size < range->size)
                        offset2_live[offset] = range;
                }
            }
        }
    }

    long long offset = 0;
    while(!conflict_queue.empty())
    {
        live_range* range     = conflict_queue.top();
        long long iter_offset = range->offset;
        if(offset > iter_offset)
        {
            offset = std::max(offset, iter_offset + range->size);
        }
        else if(offset2_live[iter_offset] == range)
        {
            if((iter_offset > offset) && (iter_offset - offset) >= size)
            {
                break;
            }
            offset = iter_offset + range->size;
        }
        // alignment
        if((offset % element_size) != 0)
            offset += (element_size - (offset % element_size));
        conflict_queue.pop();
    }
    segment.offset = offset;
    MIGRAPH_DEBUG(segment.dump());
    required_bytes = std::max(required_bytes, offset + segment.size);
    return true;
}

void memory_coloring_impl::build()
{
    std::size_t num_of_instrs = p_program->size();
    if(num_of_instrs == 0)
        return;

    int cur_points        = num_of_instrs * 2;
    instruction_ref iter  = p_program->end();
    instruction_ref begin = p_program->begin();
    std::vector<instruction_ref> dead_instrs;
    std::set<int> live_set;
    // Build live intervals.
    live_intervals.resize(num_of_instrs);
    do
    {
        iter                      = std::prev(iter);
        const instruction* p_iter = &(*iter);
        interval_ptr def_interval = nullptr;
        bool is_dead              = false;
        if(instr2_live.find(p_iter) != instr2_live.end())
        {
            def_interval = instr2_live[p_iter];
            bool is_lit  = is_literal(iter);
            if(is_allocate(iter) || is_lit)
            {
                live_range& range        = def_interval->segment;
                def_interval->result     = iter->get_shape();
                def_interval->is_literal = is_lit;
                if(!is_lit || unify_literals)
                    alloc_queue.push(def_interval);
                range.begin             = cur_points;
                def_interval->def_point = cur_points;
                range.size              = (iter->get_shape()).bytes();
                live_set.erase(range.vn);
            }
        }
        else if(!is_param(iter) && !is_outline(iter) && !is_check_context(iter))
        {
            is_dead = true;
        }
        int tie_ndx = get_input_tie_ndx(iter);
        int cnt     = -1;
        for(auto&& arg : iter->inputs())
        {
            cnt++;
            if(is_param(arg) || is_outline(arg))
            {
                if(is_output_param(arg))
                    is_dead = false;
                if(def_interval != nullptr)
                {
                    def_interval->is_live_on_entry = true;
                }
                continue;
            }
            const instruction* p_arg = &(*arg);
            if(cnt == tie_ndx && (def_interval != nullptr))
            {
                // input memory is used as this instruction's output.
                // def is considered as use. Coalesce the live intervals.
                def_interval->add_use(cur_points);
                instr2_live[p_arg] = def_interval;
            }
            else if(instr2_live.find(p_arg) == instr2_live.end())
            {
                // First time see a use, create a live interval.
                int id                = num_of_lives++;
                interval_ptr interval = &(live_intervals[id]);
                interval->id          = id;
                interval->segment.end = cur_points;
                interval->segment.vn  = ++max_value_number;
                interval->add_use(cur_points);
                instr2_live[p_arg] = interval;
                add_conflicts(live_set, max_value_number);
                live_set.insert(max_value_number);
                live_ranges[max_value_number] = &(interval->segment);
                earliest_end_point            = cur_points;
                if(latest_end_point == -1)
                    latest_end_point = cur_points;
            }
            else
            {
                interval_ptr interval = instr2_live[p_arg];
                interval->add_use(cur_points);
                assert(live_set.find(interval->id) != live_set.end());
            }
        }
        if(is_dead)
            dead_instrs.push_back(iter);
        cur_points -= 2;
    } while(iter != begin);
}

void memory_coloring_impl::register_operand_alias()
{
    operand_alias["hip::allocate"]     = -1;
    operand_alias["hip::load_literal"] = -1;
    operand_alias["@outline"]          = -1;
    operand_alias["check_context"]     = -1;
    operand_alias["@literal"]          = -1;
    operand_alias["@param"]            = -1;
    operand_alias["transpose"]         = 0;
    operand_alias["flatten"]           = 0;
    operand_alias["broadcast"]         = 1;
    operand_alias["reshape"]           = 0;
    operand_alias["pass"]              = 0;
}

void memory_coloring_impl::rewrite()
{
    instruction_ref end           = p_program->end();
    instruction_ref scratch_param = end;
    std::vector<std::size_t> dims;
    dims.push_back(required_bytes / sizeof(float));
    shape s       = {shape::float_type, dims};
    scratch_param = p_program->add_parameter("scratch", s);
    for(auto ins : iterator_for(*p_program))
    {
        const instruction* p_iter = &(*ins);
        if(instr2_live.find(p_iter) != instr2_live.end())
        {
            interval_ptr interval = instr2_live[p_iter];
            if(interval->get_begin() == invalid_offset)
                continue;

            if(!unify_literals && interval->is_literal)
                continue;

            std::size_t offset = 0;
            if(interval->get_offset() == invalid_offset)
            {
                assert(interval->result.bytes() == 0);
            }
            else
            {
                offset = interval->get_offset();
            }

            if(is_allocate(ins))
            {
                assert(!ins->inputs().empty());
                p_program->replace_instruction(
                    ins, op::load{ins->inputs().at(0)->get_shape(), offset}, scratch_param);
            }
            else if(is_literal(ins) && unify_literals)
            {
                auto pre      = p_program->add_literal(ins->get_literal());
                bool pre_copy = (interval->get_begin() < earliest_end_point);
                p_program->replace_instruction(
                    ins, write_literal{offset, pre_copy}, scratch_param, pre);
            }
        }
    }
    MIGRAPH_DEBUG(dump("---After rewrite---"));
    MIGRAPH_DEBUG(dump_program());
}

#ifdef MIGRAPH_DEBUG_OPT

void memory_coloring_impl::dump(const std::string& str) { std::cout << str << std::endl; }

void memory_coloring_impl::dump_program() { std::cout << *p_program << std::endl; }

void memory_coloring_impl::dump_intervals()
{
    if(num_of_lives > 0)
    {
        std::cout << "---live intervals ---" << std::endl;
        for(int i = 0; i < num_of_lives; ++i)
        {
            live_interval& interval = live_intervals[i];
            interval.dump();
        }
        std::cout << "---conflict table---" << std::endl;
        for(int i = 0; i <= max_value_number; ++i)
        {
            std::cout << " segment:" << i;
            std::cout << " =>";
            std::set<int>& table = conflict_table[i];
            for(auto& iter : table)
            {
                std::cout << (iter) << ",";
            }
        }
        std::cout << std::endl;
    }
}

void memory_coloring_impl::verify()
{
    if(num_of_lives > 0)
    {
        for(int i = 0; i < num_of_lives; ++i)
        {
            live_interval& interval = live_intervals[i];
            live_range& segment     = interval.segment;

            if(segment.begin == invalid_offset)
            {
                assert(interval.is_live_on_entry);
                continue;
            }

            if(segment.offset == invalid_offset)
            {
                continue;
            }
            int vn = segment.vn;
            if(conflict_table.find(vn) != conflict_table.end())
            {
                std::set<int>& vn_set = conflict_table[vn];
                for(auto& iter : vn_set)
                {
                    live_range* range = live_ranges[iter];
                    if(range->offset == invalid_offset)
                        continue;
                    if(!is_disjoin(*range, segment))
                        assert(false);
                }
            }
        }
    }
}

// map liveness tracking point to instruction enum.
static int get_ins_enum(int x)
{
    if(x > 0)
    {
        return (x / 2) - 1;
    }
    else
        return invalid_offset;
}

void live_range::dump()
{
    std::cout << " segment:" << vn;
    std::cout << " [" << get_ins_enum(begin) << ", " << get_ins_enum(end) << "]";
    if(offset != invalid_offset)
    {
        std::cout << " mem:";
        std::cout << " [" << offset << "," << offset + size - 1 << "]";
    }
    std::cout << std::endl;
}

void live_interval::dump()
{
    std::cout << "id:" << id;
    segment.dump();
    std::cout << " uses:";
    for(auto& iter : use_points)
    {
        std::cout << " " << get_ins_enum(iter) << ",";
    }
    std::cout << " def:";
    std::cout << " " << get_ins_enum(def_point);

    if(is_literal)
        std::cout << " literal";
    std::cout << " " << result;
    std::cout << std::endl;
}

#endif
} // namespace migraph

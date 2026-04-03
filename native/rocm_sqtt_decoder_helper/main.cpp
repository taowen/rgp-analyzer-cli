#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct CodeObjectInput {
    uint64_t load_id = 0;
    uint64_t load_addr = 0;
    uint64_t load_size = 0;
    std::string path;
};

struct SqttInput {
    int index = -1;
    int shader_engine_index = -1;
    int compute_unit_index = -1;
    std::string path;
};

struct Hotspot {
    uint64_t code_object_id = 0;
    uint64_t address = 0;
    uint64_t hitcount = 0;
    int64_t total_duration = 0;
    int64_t total_stall = 0;
};

struct StreamSummary {
    int index = -1;
    int shader_engine_index = -1;
    int compute_unit_index = -1;
    uint64_t bytes = 0;
    uint64_t gfxip = 0;
    uint64_t rt_frequency = 0;
    uint64_t occupancy_records = 0;
    uint64_t occupancy_starts = 0;
    uint64_t occupancy_ends = 0;
    uint64_t wave_records = 0;
    uint64_t timeline_events = 0;
    uint64_t instructions = 0;
    uint64_t perf_events = 0;
    uint64_t shaderdata_records = 0;
    uint64_t realtime_records = 0;
    uint64_t occupancy_max_active = 0;
    int64_t occupancy_begin_time = 0;
    int64_t occupancy_end_time = 0;
    int64_t occupancy_weighted_time = 0;
    uint64_t occupancy_current_active = 0;
    bool occupancy_initialized = false;
    uint64_t instructions_with_stall = 0;
    int64_t total_instruction_duration = 0;
    int64_t total_instruction_stall = 0;
    int64_t total_wave_lifetime = 0;
    int64_t max_wave_lifetime = 0;
    std::map<std::string, uint64_t> category_counts;
    std::map<std::string, int64_t> category_duration_totals;
    std::map<std::string, int64_t> category_stall_totals;
    std::map<std::string, int64_t> wave_state_durations;
    std::map<std::string, uint64_t> info_counts;
    std::vector<Hotspot> hotspots;
};

struct DecodeContext {
    rocprofiler_thread_trace_decoder_id_t decoder{};
    StreamSummary* summary = nullptr;
    struct HotspotKeyHash {
        size_t operator()(const std::pair<uint64_t, uint64_t>& value) const noexcept {
            return std::hash<uint64_t>{}(value.first) ^ (std::hash<uint64_t>{}(value.second) << 1);
        }
    };
    std::unordered_map<std::pair<uint64_t, uint64_t>, Hotspot, HotspotKeyHash> hotspot_map;
};

std::string trim(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if(start == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string item;
    while(std::getline(ss, item, '\t')) parts.push_back(item);
    return parts;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if(!file) throw std::runtime_error("failed to open file: " + path);
    file.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    if(size && !file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("failed to read file: " + path);
    }
    return data;
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for(unsigned char c : input) {
        switch(c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if(c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

const char* category_name(uint32_t category) {
    switch(category) {
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE: return "NONE";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM: return "SMEM";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU: return "SALU";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM: return "VMEM";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT: return "FLAT";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS: return "LDS";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU: return "VALU";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP: return "JUMP";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT: return "NEXT";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED: return "IMMED";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT: return "CONTEXT";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE: return "MESSAGE";
        case ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH: return "BVH";
        default: return "UNKNOWN";
    }
}

const char* wave_state_name(int32_t type) {
    switch(type) {
        case ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY: return "EMPTY";
        case ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE: return "IDLE";
        case ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC: return "EXEC";
        case ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT: return "WAIT";
        case ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL: return "STALL";
        default: return "UNKNOWN";
    }
}

void flush_hotspots(DecodeContext& ctx) {
    ctx.summary->hotspots.clear();
    for(const auto& entry : ctx.hotspot_map) ctx.summary->hotspots.push_back(entry.second);
    std::sort(ctx.summary->hotspots.begin(),
              ctx.summary->hotspots.end(),
              [](const Hotspot& lhs, const Hotspot& rhs) {
                  if(lhs.total_duration != rhs.total_duration) return lhs.total_duration > rhs.total_duration;
                  return lhs.hitcount > rhs.hitcount;
              });
    if(ctx.summary->hotspots.size() > 16) ctx.summary->hotspots.resize(16);
}

void decode_callback(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                     void*                                          trace_events,
                     uint64_t                                       trace_size,
                     void*                                          userdata) {
    auto* ctx = static_cast<DecodeContext*>(userdata);
    auto& summary = *ctx->summary;
    switch(record_type_id) {
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP: {
            if(trace_size > 0) summary.gfxip = *static_cast<uint64_t*>(trace_events);
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY: {
            auto* events = static_cast<rocprofiler_thread_trace_decoder_occupancy_t*>(trace_events);
            summary.occupancy_records += trace_size;
            for(uint64_t i = 0; i < trace_size; ++i) {
                if(summary.occupancy_initialized) {
                    const auto delta = static_cast<int64_t>(events[i].time) - summary.occupancy_end_time;
                    if(delta > 0) summary.occupancy_weighted_time += delta * static_cast<int64_t>(summary.occupancy_current_active);
                } else {
                    summary.occupancy_begin_time = static_cast<int64_t>(events[i].time);
                    summary.occupancy_initialized = true;
                }
                summary.occupancy_end_time = static_cast<int64_t>(events[i].time);
                if(events[i].start) ++summary.occupancy_starts;
                else ++summary.occupancy_ends;
                if(events[i].start) ++summary.occupancy_current_active;
                else if(summary.occupancy_current_active > 0) --summary.occupancy_current_active;
                summary.occupancy_max_active = std::max(summary.occupancy_max_active, summary.occupancy_current_active);
            }
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT: {
            summary.perf_events += trace_size;
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE: {
            auto* waves = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(trace_events);
            summary.wave_records += trace_size;
            for(uint64_t w = 0; w < trace_size; ++w) {
                const auto& wave = waves[w];
                summary.timeline_events += wave.timeline_size;
                summary.instructions += wave.instructions_size;
                const auto wave_lifetime = wave.end_time - wave.begin_time;
                if(wave_lifetime > 0) {
                    summary.total_wave_lifetime += wave_lifetime;
                    summary.max_wave_lifetime = std::max(summary.max_wave_lifetime, wave_lifetime);
                }
                for(uint64_t t = 0; t < wave.timeline_size; ++t) {
                    const auto& state = wave.timeline_array[t];
                    summary.wave_state_durations[wave_state_name(state.type)] += state.duration;
                }
                for(uint64_t i = 0; i < wave.instructions_size; ++i) {
                    const auto& inst = wave.instructions_array[i];
                    const auto* cat_name = category_name(inst.category);
                    ++summary.category_counts[cat_name];
                    summary.category_duration_totals[cat_name] += inst.duration;
                    summary.category_stall_totals[cat_name] += inst.stall;
                    summary.total_instruction_duration += inst.duration;
                    summary.total_instruction_stall += inst.stall;
                    if(inst.stall > 0) ++summary.instructions_with_stall;
                    auto& hotspot = ctx->hotspot_map[{inst.pc.code_object_id, inst.pc.address}];
                    hotspot.code_object_id = inst.pc.code_object_id;
                    hotspot.address = inst.pc.address;
                    hotspot.hitcount += 1;
                    hotspot.total_duration += inst.duration;
                    hotspot.total_stall += inst.stall;
                }
            }
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO: {
            auto* infos = static_cast<rocprofiler_thread_trace_decoder_info_t*>(trace_events);
            for(uint64_t i = 0; i < trace_size; ++i) {
                const char* name = rocprofiler_thread_trace_decoder_info_string(ctx->decoder, infos[i]);
                ++summary.info_counts[name ? name : "UNKNOWN_INFO"];
            }
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA: {
            summary.shaderdata_records += trace_size;
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME: {
            summary.realtime_records += trace_size;
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY: {
            if(trace_size > 0) summary.rt_frequency = *static_cast<uint64_t*>(trace_events);
            break;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG:
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST:
        default:
            break;
    }
}

std::string format_summary_text(const std::vector<CodeObjectInput>& code_objects,
                                const std::vector<SqttInput>& sqtt_streams,
                                const std::vector<StreamSummary>& summaries,
                                const std::vector<std::string>& warnings,
                                const std::string& decoder_lib_dir,
                                const std::string& status,
                                uint64_t code_object_load_failures) {
    std::ostringstream out;
    out << "decode_sqtt:\n";
    out << "  status: " << status << "\n";
    out << "  decoder_lib_dir: " << decoder_lib_dir << "\n";
    out << "  code_objects: " << code_objects.size() << "\n";
    out << "  code_object_load_failures: " << code_object_load_failures << "\n";
    out << "  streams: " << sqtt_streams.size() << "\n";
    for(const auto& warning : warnings) out << "  warning: " << warning << "\n";
    for(const auto& summary : summaries) {
        out << "  stream[" << summary.index << "] se=" << summary.shader_engine_index
            << " cu=" << summary.compute_unit_index << " bytes=" << summary.bytes
            << " gfxip=" << summary.gfxip << " waves=" << summary.wave_records
            << " instructions=" << summary.instructions << "\n";
        out << "    occupancy start=" << summary.occupancy_starts << " end=" << summary.occupancy_ends
            << " max_active=" << summary.occupancy_max_active
            << " perf_events=" << summary.perf_events << " shaderdata=" << summary.shaderdata_records
            << " realtime=" << summary.realtime_records << "\n";
        out << "    categories=";
        bool first = true;
        for(const auto& item : summary.category_counts) {
            if(!first) out << ", ";
            out << item.first << ":" << item.second;
            first = false;
        }
        out << "\n";
        out << "    instruction_cycles duration=" << summary.total_instruction_duration
            << " stall=" << summary.total_instruction_stall
            << " stalled_insts=" << summary.instructions_with_stall << "\n";
        out << "    wave_lifetime total=" << summary.total_wave_lifetime
            << " max=" << summary.max_wave_lifetime << "\n";
        if(!summary.wave_state_durations.empty()) {
            out << "    wave_states=";
            first = true;
            for(const auto& item : summary.wave_state_durations) {
                if(!first) out << ", ";
                out << item.first << ":" << item.second;
                first = false;
            }
            out << "\n";
        }
        if(!summary.info_counts.empty()) {
            out << "    info=";
            first = true;
            for(const auto& item : summary.info_counts) {
                if(!first) out << ", ";
                out << item.first << ":" << item.second;
                first = false;
            }
            out << "\n";
        }
        for(size_t i = 0; i < std::min<size_t>(summary.hotspots.size(), 8); ++i) {
            const auto& hotspot = summary.hotspots[i];
            out << "    hotspot[" << i << "] code_object_id=" << hotspot.code_object_id
                << " address=0x" << std::hex << hotspot.address << std::dec
                << " hitcount=" << hotspot.hitcount
                << " total_duration=" << hotspot.total_duration
                << " total_stall=" << hotspot.total_stall << "\n";
        }
    }
    return out.str();
}

std::string format_summary_json(const std::vector<CodeObjectInput>& code_objects,
                                const std::vector<SqttInput>& sqtt_streams,
                                const std::vector<StreamSummary>& summaries,
                                const std::vector<std::string>& warnings,
                                const std::string& decoder_lib_dir,
                                const std::string& status,
                                uint64_t code_object_load_failures) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"status\": \"" << json_escape(status) << "\",\n";
    out << "  \"decoder_lib_dir\": \"" << json_escape(decoder_lib_dir) << "\",\n";
    out << "  \"code_object_count\": " << code_objects.size() << ",\n";
    out << "  \"code_object_load_failures\": " << code_object_load_failures << ",\n";
    out << "  \"stream_count\": " << sqtt_streams.size() << ",\n";
    out << "  \"warnings\": [";
    for(size_t i = 0; i < warnings.size(); ++i) {
        if(i) out << ", ";
        out << "\"" << json_escape(warnings[i]) << "\"";
    }
    out << "],\n";
    out << "  \"streams\": [\n";
    for(size_t i = 0; i < summaries.size(); ++i) {
        const auto& summary = summaries[i];
        out << "    {\n";
        out << "      \"index\": " << summary.index << ",\n";
        out << "      \"shader_engine_index\": " << summary.shader_engine_index << ",\n";
        out << "      \"compute_unit_index\": " << summary.compute_unit_index << ",\n";
        out << "      \"bytes\": " << summary.bytes << ",\n";
        out << "      \"gfxip\": " << summary.gfxip << ",\n";
        out << "      \"rt_frequency\": " << summary.rt_frequency << ",\n";
        out << "      \"occupancy_records\": " << summary.occupancy_records << ",\n";
        out << "      \"occupancy_starts\": " << summary.occupancy_starts << ",\n";
        out << "      \"occupancy_ends\": " << summary.occupancy_ends << ",\n";
        out << "      \"occupancy_max_active\": " << summary.occupancy_max_active << ",\n";
        out << "      \"occupancy_begin_time\": " << summary.occupancy_begin_time << ",\n";
        out << "      \"occupancy_end_time\": " << summary.occupancy_end_time << ",\n";
        out << "      \"occupancy_weighted_time\": " << summary.occupancy_weighted_time << ",\n";
        out << "      \"wave_records\": " << summary.wave_records << ",\n";
        out << "      \"timeline_events\": " << summary.timeline_events << ",\n";
        out << "      \"instructions\": " << summary.instructions << ",\n";
        out << "      \"instructions_with_stall\": " << summary.instructions_with_stall << ",\n";
        out << "      \"total_instruction_duration\": " << summary.total_instruction_duration << ",\n";
        out << "      \"total_instruction_stall\": " << summary.total_instruction_stall << ",\n";
        out << "      \"total_wave_lifetime\": " << summary.total_wave_lifetime << ",\n";
        out << "      \"max_wave_lifetime\": " << summary.max_wave_lifetime << ",\n";
        out << "      \"perf_events\": " << summary.perf_events << ",\n";
        out << "      \"shaderdata_records\": " << summary.shaderdata_records << ",\n";
        out << "      \"realtime_records\": " << summary.realtime_records << ",\n";
        out << "      \"category_counts\": {";
        bool first = true;
        for(const auto& item : summary.category_counts) {
            if(!first) out << ", ";
            out << "\"" << json_escape(item.first) << "\": " << item.second;
            first = false;
        }
        out << "},\n";
        out << "      \"category_duration_totals\": {";
        first = true;
        for(const auto& item : summary.category_duration_totals) {
            if(!first) out << ", ";
            out << "\"" << json_escape(item.first) << "\": " << item.second;
            first = false;
        }
        out << "},\n";
        out << "      \"category_stall_totals\": {";
        first = true;
        for(const auto& item : summary.category_stall_totals) {
            if(!first) out << ", ";
            out << "\"" << json_escape(item.first) << "\": " << item.second;
            first = false;
        }
        out << "},\n";
        out << "      \"wave_state_durations\": {";
        first = true;
        for(const auto& item : summary.wave_state_durations) {
            if(!first) out << ", ";
            out << "\"" << json_escape(item.first) << "\": " << item.second;
            first = false;
        }
        out << "},\n";
        out << "      \"info_counts\": {";
        first = true;
        for(const auto& item : summary.info_counts) {
            if(!first) out << ", ";
            out << "\"" << json_escape(item.first) << "\": " << item.second;
            first = false;
        }
        out << "},\n";
        out << "      \"hotspots\": [";
        for(size_t h = 0; h < summary.hotspots.size(); ++h) {
            const auto& hotspot = summary.hotspots[h];
            if(h) out << ", ";
            out << "{"
                << "\"code_object_id\": " << hotspot.code_object_id << ", "
                << "\"address\": " << hotspot.address << ", "
                << "\"hitcount\": " << hotspot.hitcount << ", "
                << "\"total_duration\": " << hotspot.total_duration << ", "
                << "\"total_stall\": " << hotspot.total_stall << "}";
        }
        out << "]\n";
        out << "    }";
        if(i + 1 < summaries.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

int run(int argc, char** argv) {
    std::string manifest_path;
    std::string decoder_lib_dir;
    bool as_json = false;
    bool strict = false;
    size_t hotspot_limit = 8;

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if(arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if(arg == "--decoder-lib-dir" && i + 1 < argc) decoder_lib_dir = argv[++i];
        else if(arg == "--json") as_json = true;
        else if(arg == "--strict") strict = true;
        else if(arg == "--hotspot-limit" && i + 1 < argc) hotspot_limit = static_cast<size_t>(std::stoull(argv[++i]));
        else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }

    if(manifest_path.empty()) {
        std::cerr << "missing --manifest\n";
        return 2;
    }
    if(decoder_lib_dir.empty()) {
        const char* env = std::getenv("ROCPROFILER_TRACE_DECODER_LIB_PATH");
        decoder_lib_dir = env ? env : "";
    }
    if(decoder_lib_dir.empty()) {
        std::cerr << "missing --decoder-lib-dir and ROCPROFILER_TRACE_DECODER_LIB_PATH is not set\n";
        return 2;
    }

    std::ifstream manifest(manifest_path);
    if(!manifest) {
        std::cerr << "failed to open manifest: " << manifest_path << "\n";
        return 2;
    }

    std::vector<CodeObjectInput> code_objects;
    std::vector<SqttInput> sqtt_streams;
    std::vector<std::string> warnings;
    for(std::string line; std::getline(manifest, line);) {
        line = trim(line);
        if(line.empty() || line[0] == '#') continue;
        const auto parts = split_tab(line);
        if(parts.empty()) continue;
        if(parts[0] == "CO" && parts.size() >= 5) {
            code_objects.push_back(
                {static_cast<uint64_t>(std::stoull(parts[1])),
                 static_cast<uint64_t>(std::stoull(parts[2])),
                 static_cast<uint64_t>(std::stoull(parts[3])),
                 parts[4]});
        } else if(parts[0] == "SQTT" && parts.size() >= 5) {
            sqtt_streams.push_back({std::stoi(parts[1]), std::stoi(parts[2]), std::stoi(parts[3]), parts[4]});
        }
    }

    rocprofiler_thread_trace_decoder_id_t decoder{};
    const auto create_status = rocprofiler_thread_trace_decoder_create(&decoder, decoder_lib_dir.c_str());
    if(create_status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "rocprofiler_thread_trace_decoder_create failed: "
                  << rocprofiler_get_status_name(create_status) << " - "
                  << rocprofiler_get_status_string(create_status) << "\n";
        return 1;
    }

    uint64_t code_object_load_failures = 0;
    for(const auto& code_object : code_objects) {
        const auto payload = read_file(code_object.path);
        const auto status = rocprofiler_thread_trace_decoder_codeobj_load(
            decoder,
            code_object.load_id,
            code_object.load_addr,
            code_object.load_size,
            payload.data(),
            payload.size());
        if(status != ROCPROFILER_STATUS_SUCCESS) {
            ++code_object_load_failures;
            warnings.push_back(
                "code object load failed for " + code_object.path + ": " + rocprofiler_get_status_name(status));
        }
    }

    std::vector<StreamSummary> summaries;
    for(const auto& sqtt : sqtt_streams) {
        auto payload = read_file(sqtt.path);
        StreamSummary summary{};
        summary.index = sqtt.index;
        summary.shader_engine_index = sqtt.shader_engine_index;
        summary.compute_unit_index = sqtt.compute_unit_index;
        summary.bytes = payload.size();

        DecodeContext ctx{};
        ctx.decoder = decoder;
        ctx.summary = &summary;

        const auto status = rocprofiler_trace_decode(decoder, decode_callback, payload.data(), payload.size(), &ctx);
        flush_hotspots(ctx);
        if(summary.hotspots.size() > hotspot_limit) summary.hotspots.resize(hotspot_limit);
        if(status != ROCPROFILER_STATUS_SUCCESS) {
            warnings.push_back(
                "decode failed for stream " + std::to_string(sqtt.index) + ": " + rocprofiler_get_status_name(status));
        }
        summaries.push_back(summary);
    }

    std::string decode_status = "full_decode";
    if(summaries.empty()) decode_status = "decode_failed";
    else if(!warnings.empty() || code_object_load_failures > 0) decode_status = "partial_decode";

    if(as_json) {
        std::cout << format_summary_json(
            code_objects, sqtt_streams, summaries, warnings, decoder_lib_dir, decode_status, code_object_load_failures);
    } else {
        std::cout << format_summary_text(
            code_objects, sqtt_streams, summaries, warnings, decoder_lib_dir, decode_status, code_object_load_failures);
    }

    rocprofiler_thread_trace_decoder_destroy(decoder);
    if(strict && decode_status != "full_decode") return 3;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch(const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

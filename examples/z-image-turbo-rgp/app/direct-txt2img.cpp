#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_adapter.h"

std::string executable_path = "";

namespace {

const char * k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

bool is_base64_char(unsigned char c) {
    return std::isalnum(c) || c == '+' || c == '/';
}

std::vector<std::uint8_t> base64_decode(const std::string & encoded) {
    int i = 0;
    int j = 0;
    int in = 0;
    int in_len = static_cast<int>(encoded.size());
    unsigned char char_array_4[4];
    unsigned char char_array_3[3];
    std::vector<std::uint8_t> decoded;

    while (in_len-- && encoded[in] != '=' && is_base64_char(static_cast<unsigned char>(encoded[in]))) {
        char_array_4[i++] = static_cast<unsigned char>(encoded[in++]);
        if (i == 4) {
            for (i = 0; i < 4; ++i) {
                const char * pos = std::strchr(k_base64_chars, char_array_4[i]);
                char_array_4[i] = static_cast<unsigned char>(pos ? pos - k_base64_chars : 0);
            }

            char_array_3[0] = static_cast<unsigned char>((char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4));
            char_array_3[1] = static_cast<unsigned char>(((char_array_4[1] & 0x0f) << 4) + ((char_array_4[2] & 0x3c) >> 2));
            char_array_3[2] = static_cast<unsigned char>(((char_array_4[2] & 0x03) << 6) + char_array_4[3]);

            for (i = 0; i < 3; ++i) {
                decoded.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i > 0) {
        for (j = i; j < 4; ++j) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; ++j) {
            const char * pos = std::strchr(k_base64_chars, char_array_4[j]);
            char_array_4[j] = static_cast<unsigned char>(pos ? pos - k_base64_chars : 0);
        }

        char_array_3[0] = static_cast<unsigned char>((char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4));
        char_array_3[1] = static_cast<unsigned char>(((char_array_4[1] & 0x0f) << 4) + ((char_array_4[2] & 0x3c) >> 2));
        char_array_3[2] = static_cast<unsigned char>(((char_array_4[2] & 0x03) << 6) + char_array_4[3]);

        for (j = 0; j < i - 1; ++j) {
            decoded.push_back(char_array_3[j]);
        }
    }

    return decoded;
}

void write_binary_file(const std::filesystem::path & path, const std::vector<std::uint8_t> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

std::string make_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm local_now{};
#ifdef _WIN32
    localtime_s(&local_now, &now);
#else
    local_now = *std::localtime(&now);
#endif
    char buffer[32];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%04d%02d%02d-%02d%02d%02d",
        local_now.tm_year + 1900,
        local_now.tm_mon + 1,
        local_now.tm_mday,
        local_now.tm_hour,
        local_now.tm_min,
        local_now.tm_sec);
    return buffer;
}

std::string join_prompt_args(int argc, char ** argv) {
    if (argc <= 1) {
        return "A cinematic rainy cyberpunk street in Shanghai at night, reflective wet pavement, glowing neon Chinese signs, detailed storefronts, atmospheric fog, ultra detailed, high contrast lighting";
    }

    std::string prompt;
    for (int i = 1; i < argc; ++i) {
        if (i > 1) {
            prompt += ' ';
        }
        prompt += argv[i];
    }
    return prompt;
}

struct command_line_options {
    std::string prompt;
    std::string ready_file;
    std::string wait_file;
    std::string generate_ready_file;
    std::string generate_wait_file;
    std::string stop_after_phase;
    int repeat = 1;
    int steps = 8;
    int width = 1024;
    int height = 1024;
    int seed = 123456;
    bool flash_attention = false;
    bool vae_cpu = false;
    bool clip_cpu = true;
    bool diffusion_conv_direct = false;
    bool vae_conv_direct = false;
};

bool env_flag_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    std::string normalized;
    for (const char * cur = value; *cur; ++cur) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*cur))));
    }
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    return default_value;
}

command_line_options parse_options(int argc, char ** argv) {
    command_line_options options;
    options.flash_attention = env_flag_enabled("ZIMAGE_FLASH_ATTN", options.flash_attention);
    options.vae_cpu = env_flag_enabled("ZIMAGE_VAE_CPU", options.vae_cpu);
    options.clip_cpu = env_flag_enabled("ZIMAGE_CLIP_CPU", options.clip_cpu);
    options.diffusion_conv_direct = env_flag_enabled("ZIMAGE_DIFFUSION_CONV_DIRECT", options.diffusion_conv_direct);
    options.vae_conv_direct = env_flag_enabled("ZIMAGE_VAE_CONV_DIRECT", options.vae_conv_direct);
    std::vector<std::string> prompt_parts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt_parts.push_back(argv[++i]);
            continue;
        }
        if (arg == "--ready-file" && i + 1 < argc) {
            options.ready_file = argv[++i];
            continue;
        }
        if (arg == "--wait-file" && i + 1 < argc) {
            options.wait_file = argv[++i];
            continue;
        }
        if (arg == "--generate-ready-file" && i + 1 < argc) {
            options.generate_ready_file = argv[++i];
            continue;
        }
        if (arg == "--generate-wait-file" && i + 1 < argc) {
            options.generate_wait_file = argv[++i];
            continue;
        }
        if (arg == "--stop-after-phase" && i + 1 < argc) {
            options.stop_after_phase = argv[++i];
            continue;
        }
        if (arg == "--repeat" && i + 1 < argc) {
            options.repeat = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--steps" && i + 1 < argc) {
            options.steps = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--width" && i + 1 < argc) {
            options.width = std::max(64, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--height" && i + 1 < argc) {
            options.height = std::max(64, std::atoi(argv[++i]));
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            options.seed = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--flash-attn") {
            options.flash_attention = true;
            continue;
        }
        if (arg == "--no-flash-attn") {
            options.flash_attention = false;
            continue;
        }
        if (arg == "--vae-cpu") {
            options.vae_cpu = true;
            continue;
        }
        if (arg == "--no-vae-cpu") {
            options.vae_cpu = false;
            continue;
        }
        if (arg == "--clip-cpu") {
            options.clip_cpu = true;
            continue;
        }
        if (arg == "--no-clip-cpu") {
            options.clip_cpu = false;
            continue;
        }
        if (arg == "--diffusion-conv-direct") {
            options.diffusion_conv_direct = true;
            continue;
        }
        if (arg == "--no-diffusion-conv-direct") {
            options.diffusion_conv_direct = false;
            continue;
        }
        if (arg == "--vae-conv-direct") {
            options.vae_conv_direct = true;
            continue;
        }
        if (arg == "--no-vae-conv-direct") {
            options.vae_conv_direct = false;
            continue;
        }
        prompt_parts.push_back(arg);
    }

    if (prompt_parts.empty()) {
        options.prompt = join_prompt_args(0, nullptr);
    } else {
        for (std::size_t i = 0; i < prompt_parts.size(); ++i) {
            if (i > 0) {
                options.prompt += ' ';
            }
            options.prompt += prompt_parts[i];
        }
    }

    return options;
}

void write_text_file(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open text output file: " + path.string());
    }
    out << text;
}

void wait_for_file(const std::filesystem::path & path) {
    while (!std::filesystem::exists(path)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const command_line_options options = parse_options(argc, argv);
        const std::filesystem::path executable_dir =
            std::filesystem::absolute(argc > 0 ? argv[0] : ".").parent_path();
        const std::filesystem::path example_dir = executable_dir.parent_path();
        const std::filesystem::path output_dir = example_dir / "output";
        const std::filesystem::path weights_dir = example_dir / "models" / "zimage";

        std::filesystem::create_directories(output_dir);
        std::filesystem::current_path(executable_dir);

        const std::string executable_root = executable_dir.generic_string() + "/";
        executable_path = executable_root;

        const std::string model_path = (weights_dir / "z-image-turbo-Q4_K_S.gguf").string();
        const std::string clip1_path = (weights_dir / "Qwen3-4B-Q4_K_S.gguf").string();
        const std::string vae_path = (weights_dir / "zimage-vae.safetensors").string();
        const std::string prompt = options.prompt;

        const sd_load_model_inputs load_inputs{
            model_path.c_str(),
            executable_root.c_str(),
            0,
            "0",
            8,
            0,
            options.flash_attention,
            false,
            options.vae_cpu,
            options.clip_cpu,
            options.diffusion_conv_direct,
            options.vae_conv_direct,
            false,
            768,
            "",
            clip1_path.c_str(),
            "",
            vae_path.c_str(),
            0,
            nullptr,
            nullptr,
            0,
            "",
            "",
            0,
            0,
            "",
            false,
            1,
        };

        if (!sdtype_load_model(load_inputs)) {
            std::cerr << "sdtype_load_model failed\n";
            return 1;
        }

        if (!options.stop_after_phase.empty()) {
            setenv("ZIMAGE_STOP_AFTER_PHASE", options.stop_after_phase.c_str(), 1);
        } else {
            unsetenv("ZIMAGE_STOP_AFTER_PHASE");
        }

        if (!options.ready_file.empty()) {
            write_text_file(options.ready_file, "ready\n");
        }
        if (!options.wait_file.empty()) {
            wait_for_file(options.wait_file);
        }

        std::filesystem::path output_path;
        std::size_t png_size = 0;
        bool phase_only_completed = false;

        for (int generation_index = 0; generation_index < options.repeat; ++generation_index) {
            if (!options.generate_ready_file.empty()) {
                write_text_file(options.generate_ready_file, "generate_ready\n");
            }
            if (!options.generate_wait_file.empty()) {
                wait_for_file(options.generate_wait_file);
            }

            const sd_generation_inputs gen_inputs{
                prompt.c_str(),
                "blurry, low quality, deformed, distorted, text, watermark",
                "",
                "",
                0,
                nullptr,
                false,
                0.6f,
                1.0f,
                0.0f,
                0,
                0.0f,
                options.steps,
                options.width,
                options.height,
                options.seed + generation_index,
                "default",
                "default",
                -1,
                1,
                0,
                false,
                false,
                false,
                "",
                "",
                false,
                0,
                nullptr,
                nullptr,
            };

            const sd_generation_outputs result = sdtype_generate(gen_inputs);
            if (result.status != 1) {
                std::cerr << "sdtype_generate returned no image data\n";
                return 1;
            }
            if (result.data == nullptr || result.data[0] == '\0') {
                if (!options.stop_after_phase.empty()) {
                    phase_only_completed = true;
                    continue;
                }
                std::cerr << "sdtype_generate returned no image data\n";
                return 1;
            }

            const std::vector<std::uint8_t> png_bytes = base64_decode(result.data);
            if (png_bytes.empty()) {
                std::cerr << "failed to decode generated PNG\n";
                return 1;
            }

            output_path = output_dir / ("zimage-direct-" + make_timestamp() + ".png");
            write_binary_file(output_path, png_bytes);
            png_size = png_bytes.size();
        }

        if (phase_only_completed && output_path.empty()) {
            std::cout << "txt2img phase-only succeeded\n";
            std::cout << "phase_stop_after: " << options.stop_after_phase << '\n';
            std::cout << "image: <none>\n";
            std::cout << "bytes: 0\n";
            return 0;
        }

        std::cout << "txt2img succeeded\n";
        std::cout << "image: " << output_path.string() << '\n';
        std::cout << "bytes: " << png_size << '\n';
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}

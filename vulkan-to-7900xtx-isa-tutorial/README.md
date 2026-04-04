# Vulkan To 7900 XTX ISA Tutorial

这个子目录面向 `RX 7900 XTX + RADV` 的 Linux Vulkan compute 调优路径。

目标不是泛泛地讲 Vulkan API，而是把下面这条链真正做通：

1. 写一个可运行的 Vulkan compute shader
2. 把它编译成 SPIR-V
3. 让 AMD 驱动把它编译成实际执行的 AMDGPU ISA
4. 抓 `.rgp`
5. 用 `rgp-analyzer-cli` 看 code object、资源占用、runtime 指标和热点 PC

## Chapters

- `ch01`
  基本开发环境、最小 compute 实验、以及 `Vulkan -> SPIR-V -> AMDGPU ISA -> .rgp` 的对应关系。

## Chapter Layout

每一章都使用同样的结构：

```text
vulkan-to-7900xtx-isa-tutorial/
  chXX/
    README.md
    src/
      ... self-contained experiment code and scripts ...
```

每章都要求：

- `src/` 代码自包含，可直接构建和运行
- 能在 Linux + RADV 上抓 `.rgp`
- 能接上 `rgp-analyzer-cli`

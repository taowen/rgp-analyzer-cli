# Medium Vulkan Graph RGP Example

This example sits between the single-purpose microbenchmark and the full `z-image-turbo` workload.

It models a small compute graph with:

- three compute pipelines
- two storage buffers
- explicit phase barriers
- optional batched submit / per-phase submit / multi-command-buffer submission
- Vulkan debug labels and timestamp queries

Use it to answer one question:

When a Vulkan compute workload becomes graph-like, at what point does `.rgp` stop carrying dense runtime evidence?

Current role in the repo:

- calibration workload for field semantics
- bridge between `minimal-vulkan-compute-rgp` and `z-image-turbo-rgp`
- safe place to perturb register pressure, memory pressure, and barrier intensity

## Layout

```text
examples/medium-vulkan-graph-rgp/
  README.md
  src/
    main.cpp
    build.sh
    compile-shaders.sh
    run.sh
    capture-rgp.sh
    analyze-latest.sh
    shaders/
      preprocess.comp
      mix.comp
      reduce.comp
```

## Build

```bash
cd ~/projects/rgp-analyzer-cli/examples/medium-vulkan-graph-rgp/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run.sh
bash ./run.sh --submit-mode phase
bash ./run.sh --submit-mode multi-cmdbuf --graph-iterations 24
```

## Capture

```bash
bash ./capture-rgp.sh
bash ./capture-rgp.sh --submit-mode phase
bash ./capture-rgp.sh --submit-mode multi-cmdbuf --graph-iterations 24
```

## Analyze

```bash
cd ~/projects/rgp-analyzer-cli/examples/medium-vulkan-graph-rgp/src
bash ./analyze-latest.sh
```

## Field calibration

```bash
cd ~/projects/rgp-analyzer-cli/examples/medium-vulkan-graph-rgp/src
bash ./probe-field-semantics.sh
```

This produces a small calibration matrix:

- `field-baseline`
- `field-reg-pressure`
- `field-memory-heavy`
- `field-barrier-heavy`

This example is intentionally more structured than `minimal-vulkan-compute-rgp`, but still fully controlled and easy to rebuild.

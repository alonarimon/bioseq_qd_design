# Offline Model-Based Optimisation for Controllable Biological Sequence Design

This repository implements a framework for de novo biological sequence design using offline model-based optimisation (MBO) combined with quality-diversity (QD) algorithms and foundation models. It extends the MAP-Elites algorithm to work in a one-shot, offline setting—where access to the true objective function (oracle) is restricted—and integrates pretrained sequence models to improve both generation and evaluation.

This work was developed as part of the MSc Individual Project at Imperial College London.

> This repository is a fork of OpenELM (https://github.com/CarperAI/OpenELM) , an open-source library by CarperAI released under the MIT License. We adapted their Quality-Diversity framework to biological sequence design, and significantly extended the mutation, scoring, and descriptor logic for the offline setting.


## Report

A detailed description of the motivation, methodology, experiments, and results can be found in the [project report](./Project_Report.pdf).

## Project Overview

### Motivation

Biological sequence design is a critical task in synthetic biology, drug discovery, and mRNA therapeutics. Due to the cost and time required for wet-lab experiments, in-silico optimisation methods are used to pre-select candidate sequences. However, designing biologically plausible, diverse, and high-performing sequences under offline constraints is challenging.

This project explores the use of:
- **Offline MBO**: surrogate-based optimisation using only a static dataset of labeled sequences.
- **Quality-Diversity algorithms (MAP-Elites)**: to generate diverse and useful sequences across a defined behaviour space.
- **Foundation models (e.g., Helix-mRNA)**: for both generative mutation and conservative surrogate scoring.

### Key Components
- **Task**: Optimisation of 5'UTR mRNA sequences for high ribosome load.
- **Surrogates**: Conservative Objective Models (COMs) and fine-tuned Helix-mRNA.
- **Mutators**: Random and Helix-mRNA-based substring replacement.
- **Descriptors**: Nucleotide frequency-based behaviour space.
- **Final Batch**: Downsampled using Centroidal Voronoi Tessellation (CVT) to 128 diverse high-fitness candidates.

### Highlights
- Demonstrated how Helix-guided mutation avoids surrogate exploitation.
- Compared mutation-surrogate pairings in terms of fitness, diversity, and novelty.
- Proposed and analysed biologically meaningful distance metrics (e.g., structural edit distance).

## Setup

This project runs inside an [Apptainer](https://apptainer.org/) container to ensure a consistent environment across machines.

### 1. Build the Container

To build the container locally, run:

```bash
./build_container.sh
```

Alternatively, if a pre-built container.sif file already exists in the ./apptainer/ directory, you can skip this step.

### 2. Use the Container
You can interact with the container in two ways:

* Option A: Run the container directly
    ```bash
    apptainer run --nv ./apptainer/container.sif
    ```
* Option B: Open a shell inside the container 
    ```bash
    ./shell_container.sh
    ```
    From the shell, you can run Python scripts or interactively explore the environment.

## Dependencies and Acknowledgements

This project builds on the following foundational works and public repositories:

- **OpenELM** [[1]](https://github.com/CarperAI/OpenELM): This repository is a fork of *OpenELM* by Bradley et al., an open-source library by CarperAI released under the MIT License. We adapted their Quality-Diversity framework to the biological sequence design domain, significantly extending the mutation strategies, scoring models, and behaviour descriptor logic for offline optimisation.
- **Design-Bench** [[1]](https://github.com/rail-berkeley/design-bench): for offline datasets and oracle models.
- **Helical** [[3]](https://github.com/helicalAI/helical): a package providing pretrained foundation models for biological sequence design. We use the `helix` model for both generative mutation and surrogate evaluation of mRNA sequences.

We thank the authors of these works for making their code and models available.

### Citations

[1] Bradley H., Fan H., Carvalho F., Fisher M., Castricato L., reciprocated, et al. *OpenELM*, 2023. Available at: [https://github.com/CarperAI/OpenELM](https://github.com/CarperAI/OpenELM)  

[2] Trabucco, B., Geng, X., Kumar, A., & Levine, S. (2022). *Design-Bench: Benchmarks for data-driven offline model-based optimization*. ICML 2022.

[3] Donà, J., Flajolet, A., Marginean, A., Cully, A., & Pierrot, T. (2023). *Quality-Diversity for One-Shot Biological Sequence Design*.

[4] Wood, M., Klop, M., & Allard, M. (2025). *Helix-mRNA: A Hybrid Foundation Model for Full Sequence mRNA Therapeutics*. [arXiv:2502.13785](https://arxiv.org/abs/2502.13785)

## License

This repository is a fork of [OpenELM](https://github.com/CarperAI/OpenELM), which is licensed under the MIT License. The original code from OpenELM remains under the MIT License — see the [`LICENSE`](./LICENSE) file for full details.

### Submodules

This repository includes the following Git submodules, each with their own respective licenses:

- [Design-Bench](https://github.com/rail-berkeley/design-bench) – MIT License
- [Helical](https://github.com/helicalAI/helical) – GNU Affero General Public License v3.0

These submodules are governed by their own license terms (see the `LICENSE` file inside each submodule directory).

---

⚠️ Original work created in this repository by Alona Rimon is **not yet licensed** for redistribution or reuse. A separate license for these additions may be added in the future.

If you have questions or would like to reuse original components of this work, please [contact me](mailto:rimonalona@gmail.com).



## Authors

**Alona Rimon** – MSc Advanced Computing, Imperial College London  
Supervisor: Dr. Antoine Cully  
Second Supervisors: Maxime Allard, Hannah Janmohamed




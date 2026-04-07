# UniCore Hardware Simulator

We reproduce the simulator-side artifact-evaluation figures with the scripts under `ae_scripts/`.
All commands below should be run from this repository root.

## Environment

### Validated local platform

The simulator AE flow is lightweight and only needs a CPU Python environment plus a host compiler for `DRAMsim3`.
The current repository is validated on the following local machine setup:

+ Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz
+ Ubuntu 22.04.5 LTS
+ GCC 11.4.0
+ G++ 11.4.0
+ GNU Make 4.3
+ Python 3.10.18

### Prerequisite

A minimal Linux setup only needs:

+ Ubuntu 22.04 or a similar Linux environment
+ `bash`
+ `make`
+ `gcc` / `g++` for building `DRAMsim3`
+ Conda or another Python virtual-environment tool
+ Python 3.10
+ `torch`, `numpy`, and `matplotlib`

This simulator flow does not require CUDA, `transformers`, or `pandas`.
A CPU-only PyTorch build is sufficient.

The AE scripts use the current shell Python environment directly.
They do not call `conda activate` internally, so please activate the intended environment before running any script.

You can quickly check the local toolchain with:

```bash
python --version
gcc --version
g++ --version
make --version
```

### Conda

Create and activate a clean simulator environment:

```bash
conda create -n unicore_sim python=3.10.18 -y
conda activate unicore_sim
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then verify the minimal Python packages:

```bash
python -c "import torch, numpy, matplotlib; print('simulator env ready')"
```

The current pinned simulator requirements are:

- `torch==2.6.0+cpu`
- `numpy==1.26.4`
- `matplotlib==3.8.4`


## Evaluation

### AE automation scripts

One-click artifact-evaluation scripts are provided under:

```bash
ae_scripts/
├── common.sh
├── figure17-18/
│   └── run_prefill_ae.sh
└── figure19-20/
    ├── calc_avg_speedup.py
    └── run_decode_ae.sh
```

These scripts keep the flow simple and only replace manual shell commands.
Each script does the following:

1. use the current shell Python environment
2. build `DRAMsim3`
3. regenerate DRAM summary numbers from `DRAMsim3`
4. run `run_from_configuration.py`
5. generate the target figures

### Run figures

Prefill:

```bash
conda activate unicore_sim
bash ae_scripts/figure17-18/run_prefill_ae.sh --memory ddr4 --cxt-len 8192 --fused-attn  # figure 17-18 # About 10 minutes
```

Decode:

```bash
conda activate unicore_sim
bash ae_scripts/figure19-20/run_decode_ae.sh --memory ddr4 --cxt-len 8192 --batch-size 128  # figure 19
bash ae_scripts/figure19-20/run_decode_ae.sh --memory hbm2 --cxt-len 8192 --batch-size 128  # figure 20
bash ae_scripts/figure19-20/run_decode_ae.sh --memory both --cxt-len 8192 --batch-size 128  # figure 19-20 # About 15 minutes
```

### Clean space

```bash
bash ae_scripts/clean_ae.sh  # clean all generated outputs
```


- `--memory ddr4|hbm2|both`
- `--cxt-len <int>`
- `--batch-size <int>` for decode
- `--fused-attn` for prefill fused-attention modeling
- `--skip-dramsim` to reuse the existing `DRAMsim3/runs_by_type/summary.csv`
- `--dry-run` to print commands without executing them

### Output

Generated outputs are written under:

```bash
results_ddr4_8Gb_x8_3200_ctx_sweep*/
results_ddr4_8Gb_x8_3200_ctx_sweep_decode_bs*/
results_hbm2_8Gb_x128_ctx_sweep*/
results_hbm2_8Gb_x128_ctx_sweep_decode_bs*/
figure/
```

Figure mapping for the current AE scripts:

- `plot/figure17_prefill_speedup.py`: prefill speedup plotting script for Figure 17
- `plot/figure18_prefill_energy.py`: prefill energy plotting script for Figure 18
- `plot/figure19-20.py`: decode plotting script for Figure 19 and Figure 20
- `ae_scripts/figure17-18/run_prefill_ae.sh --memory ddr4 --cxt-len 8192 --fused-attn`: writes `figure/figure17.{pdf,png}` and `figure/figure18.{pdf,png}`
- `ae_scripts/figure19-20/run_decode_ae.sh --memory both --cxt-len 8192 --batch-size 128`: writes `figure/figure19.{pdf,png}` and `figure/figure20.{pdf,png}`

### Cleanup

To clean AE-generated artifacts and return to the pre-run repository shape:

```bash
bash ae_scripts/clean_ae.sh
```

Preview the cleanup commands without deleting anything:

```bash
bash ae_scripts/clean_ae.sh --dry-run
```

## Acknowledgement

The simulator is based on HPCA'25 BitMoD. We add fused-attention support to reduce modeled IO overhead.

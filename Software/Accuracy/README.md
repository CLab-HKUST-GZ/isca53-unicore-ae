# UniCore Quantization & Approximation

We reproduce the artifact-evaluation results for Table 1-4 with the scripts under `ae_scripts/`.
All commands below should be run from this `Software/Accuracy/` directory.

## Environment

### Paper's hardware configuration

+ INTEL(R) XEON(R) GOLD 6544Y
+ 4 * NVIDIA RTX 6000 Ada GPUs (48GB)

### Prerequisite

The validated setup uses:

+ Ubuntu 22.04.5 LTS
+ CUDA 13.1
+ GCC 11.4.0
+ Python 3.10.18
+ `torch==2.6.0+cu124`

This artifact requires a local CUDA toolkit and a working host compiler because `unicore_kernel` is JIT-compiled on first use.
You can quickly check the local toolchain with:

```bash
nvcc --version
gcc --version
```

### Conda

Create and activate the AE conda environment:

```bash
conda create -n unicore python=3.10.18 -y
conda activate unicore
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Note

- reviewer runs may automatically download missing models or datasets
- the scripts use Hugging Face model IDs by default; if you prefer local checkpoints, replace the entries in the corresponding `models=(...)` array before running it
- if your CUDA toolkit is not on the default path, set `CUDA_HOME` to your local CUDA root before running the scripts

## Evaluation

### Run A Whole Table

Each table provides a one-click `run.sh` entrypoint:

```bash
bash ae_scripts/table1/run.sh
bash ae_scripts/table2/run.sh
bash ae_scripts/table3/run.sh
bash ae_scripts/table4/run.sh
```

Useful variants:

```bash
# preview without launching
DRY_RUN=1 bash ae_scripts/table1/run.sh

# run on a specific GPU
device=0 bash ae_scripts/table1/run.sh

# run only scripts whose path contains the given substring
bash ae_scripts/table1/run.sh w4a4kv16
bash ae_scripts/table3/run.sh 4/fpma.sh
bash ae_scripts/table4/run.sh gs32

# include 70B scripts when available
INCLUDE_70B=1 bash ae_scripts/table1/run.sh
```

By default, each `run.sh` executes scripts in sorted order and skips `*_70b.sh` unless `INCLUDE_70B=1` is set.
The optional filter argument uses substring matching on the script path, so a broader keyword may match multiple scripts.

### Run A Single Script

You can also run any script directly. Examples:

```bash
device=0 bash ae_scripts/table1/w16a16kv16/wikitext_ppl_fp.sh
device=0 bash ae_scripts/table1/w4a4kv16/unicore.sh
device=0 bash ae_scripts/table2/w4a4kv16/unicore.sh
device=1 bash ae_scripts/table3/8/fpma_cg.sh
device=0 bash ae_scripts/table4/gs32/unicore.sh
```

### Output

Generated outputs are written under `results/`:

```bash
results/table1/...
results/table2/...
results/table3/...
results/table4/...
```

## Expected Results


The following values are copied from the paper tables and can be used as the expected results reference.
Table I, Table III, and Table IV report WikiText-2 perplexity (PPL).
Table II reports LM-Eval task accuracy on ARC-Easy, HellaSwag, PIQA, and WinoGrande, plus the macro average (`Avg.`). Results of perplexity and task accuracy evaluations can be reproduced with slight random error.

### Table I: WikiText-2 Perplexity

| Method | Bits (W/A/KV) | OPT-6.7B | LLaMA2-7B | LLaMA2-70B | LLaMA3-8B | Qwen3-8B | Qwen3-14B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 16/16/16 | 10.86 | 5.47 | 3.32 | 6.14 | 9.72 | 8.64 |
| UNICORE | 16/16/16 | 10.88 | 5.48 | 3.32 | 6.17 | 9.69 | 8.64 |
| INT | 8/8/16 | 22.85 | 5.58 | 3.48 | 6.27 | 9.63 | 8.70 |
| UNICORE | 8/8/16 | 10.98 | 5.50 | 3.34 | 6.20 | 9.77 | 8.72 |
| INT | 4/4/16 | 11.18 | 5.95 | 3.62 | 7.39 | 10.51 | 9.22 |
| MXFP4 | 4/4/16 | 11.26 | 5.99 | 3.61 | 7.50 | 10.74 | 9.50 |
| BitMoD | 4/4/16 | 11.08 | 5.82 | 3.56 | 7.13 | 10.33 | 9.20 |
| M-ANT | 4/4/16 | 11.14 | 5.85 | 3.56 | 7.11 | 10.33 | 9.29 |
| UNICORE | 4/4/16 | 11.15 | 5.81 | 3.55 | 7.05 | 10.37 | 9.16 |
| UNICORE-Q | 4/4/16 | 10.93 | 5.76 | 3.51 | 6.85 | 10.09 | 9.07 |
| BitMoD | 4/8/16 | 45.95 | 5.67 | 3.55 | 6.60 | 9.92 | 8.84 |
| M-ANT | 4/8/16 | 44.12 | 5.70 | 3.55 | 6.59 | 9.86 | 8.90 |
| UNICORE | 4/8/16 | 11.11 | 5.67 | 3.44 | 6.66 | 10.26 | 8.97 |
| UNICORE-Q | 4/8/16 | 10.88 | 5.62 | 3.40 | 6.46 | 9.93 | 8.90 |
| BitMoD | 3/8/16 | 190.12 | 6.13 | 3.84 | 7.82 | 11.20 | 9.61 |
| UNICORE | 3/8/16 | 14.60 | 6.40 | 3.94 | 9.27 | 12.32 | 9.95 |
| UNICORE-Q | 3/8/16 | 11.77 | 5.95 | 3.63 | 7.42 | 10.45 | 9.44 |
| INT | 4/4/4 | 11.33 | 6.73 | 4.57 | 8.38 | 11.27 | 10.00 |
| UNICORE | 4/4/4 | 11.54 | 6.23 | 3.82 | 7.86 | 10.78 | 9.39 |
| UNICORE-Q | 4/4/4 | 11.10 | 6.19 | 3.78 | 7.59 | 10.45 | 9.28 |

### Table II: Zero-Shot Performance

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Method</th>
      <th rowspan="2">Bits (W/A/KV)</th>
      <th colspan="5">Accuracy</th>
    </tr>
    <tr>
      <th>Arc-e</th>
      <th>Hella.</th>
      <th>Piqa</th>
      <th>Wino.</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">LLaMA3-8B</td>
      <td>FP16</td>
      <td>16/16/16</td>
      <td>77.82</td><td>79.25</td><td>80.79</td><td>73.09</td><td>77.74</td>
    </tr>
    <tr><td>INT</td><td>8/8/16</td><td>77.36</td><td>78.99</td><td>79.98</td><td>73.64</td><td>77.49</td></tr>
    <tr><td>UNICORE</td><td>8/8/16</td><td>77.61</td><td>79.09</td><td>80.41</td><td>73.24</td><td>77.59</td></tr>
    <tr><td>INT</td><td>4/4/16</td><td>75.13</td><td>76.78</td><td>79.98</td><td>70.09</td><td>75.49</td></tr>
    <tr><td>M-ANT</td><td>4/4/16</td><td>73.48</td><td>76.99</td><td>78.94</td><td>70.09</td><td>74.88</td></tr>
    <tr><td>BitMoD</td><td>4/4/16</td><td>73.65</td><td>76.75</td><td>78.40</td><td>69.77</td><td>74.64</td></tr>
    <tr><td>UNICORE</td><td>4/4/16</td><td>73.61</td><td>76.66</td><td>78.24</td><td>70.88</td><td>74.85</td></tr>
    <tr><td>UNICORE-Q</td><td>4/4/16</td><td>75.21</td><td>77.72</td><td>79.16</td><td>70.64</td><td>75.68</td></tr>
    <tr>
      <td rowspan="8">Qwen3-8B</td>
      <td>FP16</td>
      <td>16/16/16</td>
      <td>80.64</td><td>74.94</td><td>77.37</td><td>67.72</td><td>75.17</td>
    </tr>
    <tr><td>INT</td><td>8/8/16</td><td>79.34</td><td>75.06</td><td>77.48</td><td>67.80</td><td>74.92</td></tr>
    <tr><td>UNICORE</td><td>8/8/16</td><td>80.72</td><td>74.53</td><td>77.53</td><td>67.88</td><td>75.16</td></tr>
    <tr><td>INT</td><td>4/4/16</td><td>73.82</td><td>71.53</td><td>75.24</td><td>65.98</td><td>71.64</td></tr>
    <tr><td>M-ANT</td><td>4/4/16</td><td>76.05</td><td>71.90</td><td>74.97</td><td>65.82</td><td>72.19</td></tr>
    <tr><td>BitMoD</td><td>4/4/16</td><td>78.03</td><td>71.41</td><td>74.97</td><td>65.19</td><td>72.40</td></tr>
    <tr><td>UNICORE</td><td>4/4/16</td><td>78.24</td><td>72.74</td><td>75.90</td><td>66.85</td><td>73.43</td></tr>
    <tr><td>UNICORE-Q</td><td>4/4/16</td><td>78.28</td><td>73.01</td><td>76.01</td><td>66.30</td><td>73.40</td></tr>
  </tbody>
</table>

### Table III: FPMA Compensation Ablation

| Method | Bits | OPT-6.7B | LLaMA2-7B | LLaMA3-8B | Qwen3-8B | Qwen3-14B |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 | 16 | 10.86 | 5.47 | 6.14 | 9.72 | 8.64 |
| UNICORE | 16 | 10.88 | 5.48 | 6.17 | 9.69 | 8.64 |
| FP8 | 8 | 10.98 | 5.50 | 6.20 | 9.76 | 8.71 |
| FPMA+Cg | 8 | 11.02 | 5.52 | 6.23 | 9.84 | 8.77 |
| UNICORE | 8 | 10.98 | 5.50 | 6.20 | 9.77 | 8.72 |
| FP4 | 4 | 11.15 | 5.81 | 7.05 | 10.37 | 9.14 |
| FPMA | 4 | 1.1E+4 | 3.4E+4 | 3.6E+5 | 4.9E+6 | 1.5E+6 |
| UNICORE | 4 | 11.15 | 5.81 | 7.05 | 10.37 | 9.14 |

### Table IV: Group-Size Ablation

WikiText-2 PPL on LLaMA2-7B under the `W4A4` setting. FP16 baseline: `5.47`.

| GS | UNICORE | M-ANT | BitMoD | ANT | INT |
| --- | --- | --- | --- | --- | --- |
| GS-128 | 5.98 | 6.36 | 6.39 | 6.50 | 6.54 |
| GS-64 | 5.84 | 6.02 | 5.99 | 6.10 | 6.14 |
| GS-32 | 5.76 | 5.85 | 5.82 | 5.91 | 5.95 |

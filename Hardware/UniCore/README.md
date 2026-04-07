# UniCore: Hardware Design & Functional Verification


## Project Overview
This document outlines the hardware design and functional verification process for the UniCore project. The core is implemented using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/master/index.html), a modern, high-level hardware description language that facilitates efficient and flexible hardware design.


## Directory Structure
Hardware files are organized as follows:

* **SpinalHDL Source Code:** `hw/spinal/UniCore`
* **Verification & Testbenches:** `hw/spinal/UniCore/Testing`
* **Generated Verilog RTL:** `hw/gen/UniCore/` (This directory is created after running the generation)


## Environment Setup
### Recommended: Pre-configured Environment (only for the Hardware part)

We have prepared a ready-to-use development environment for you, which can be accessed via docker pull.

```bash
docker pull victorchan433/unicore-dev

docker run --rm -it victorchan433/unicore-dev /bin/bash

# After entered
cd /workspace
git clone https://github.com/CLab-HKUST-GZ/isca53-unicore-ae.git

# Navigate to the project's root directory
cd isca53-unicore-ae/Hardware/UniCore/
```


## Generating Verilog RTL

Follow these steps to generate the Verilog RTL from the SpinalHDL source code. 

The process uses SBT (Simple Build Tool) to compile the Scala-based SpinalHDL code and execute the generator. 

All SpinalHDL components can generate Verilog RTL.
We provide some examples below:

```bash
# Launch the SBT (Simple Build Tool) interactive shell
cs launch sbt

# Within the sbt shell, compile the project's source code
compile

# Generate Verilog files for composable PE (4b-8b)
runMain UniCore.ComposablePE.Scalable_4b8b.ScalablePE_4b8b_Gen

# Generate Verilog files for composable PE (4b-8b-16b)
runMain UniCore.ComposablePE.Scalable_4b8b16b.ScalablePE_4b8b16b_Gen

# Wait for the "[success]" message, which indicates completion
```
The generated Verilog files can be found in the following output directory:
`hw/gen/UniCore/`.


## Functional Verification with Iverilog

To run the functional verification suite, you'll use the SBT environment to launch simulations. 

The testbench will report the DUT (Device Under Test) results and log a comparison against a golden reference model directly to your terminal.

### Running the Complete Test Suite
This command executes all verification tests in sequence.

```bash
# Make sure you're still in the sbt shell

# (Optional) If you've modified the source code, re-compile the project first
compile

# Run the complete functional test suite
runMain UniCore.Testing.OverallFunctionalTest
```

### Running the Individual Test Suite
The `OverallFunctionalTest` suite is composed of several independent test modules below. 

You can also execute these tests individually. 

Make sure you are inside the SBT shell before running these commands.

```bash
# Make sure you're still in the sbt shell

# 1. Test for Scalable Floating-Point Multiplication Approximation (S-FPMA) with 4b-8b scalability
# Verifies the correctness and scalability of the S-FPMA unit with dual-path compensation.
# Including 2 tese cases: W4A4 / W8A8 modes.
runMain UniCore.Testing.TestCase.Test_SFPMA_4b8b

# 2. Test for Scalable Floating-Point Multiplication Approximation (S-FPMA) with 4b-8b-16b scalability
# Verifies the correctness and scalability of the S-FPMA unit with dual-path compensation.
# Including 3 tese cases: W4A4 / W8A8 / W16A16 modes.
runMain UniCore.Testing.TestCase.Test_SFPMA_4b8b16b

# 3. Test for Composable PE (C-PE) with 4b-8b scalability
# Verifies the Multiply-Accumulate (MAC) functionality and scalability of the Composable PE.
# Including 2 tese cases: W4A4 / W8A8 modes.
runMain UniCore.Testing.TestCase.Test_CPE_4b8b

# 4. Test for Composable PE (C-PE) with 4b-8b-16b scalability
# Verifies the Multiply-Accumulate (MAC) functionality and scalability of the Composable PE.
# Including 3 tese cases: W4A4 / W8A8 / W16A16 modes.
runMain UniCore.Testing.TestCase.Test_CPE_4b8b16b

# Tests are finished. To exit the sbt shell, type
exit
```
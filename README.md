# Basic Block ML Toolkit
Collection of tools related to gathering & analyzing basic blocks, and training & running ML models on them.

## Logger (`/logger`)
A [DynamoRIO](https://dynamorio.org/) program to collect basic blocks of a passed in executable.

**Usage**

First, compile the logger into a shared / dynamically linked library:
```bash
# add --release to build in release mode
$ cargo build --package=logger
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
```
Run the logger:
```bash
# -file=<save path> to specify the output trace path
# -filter to only save basic blocks from the application specified, ignoring any from other libraries
# if you built in release mode, replace 'debug' with 'release' in the path to the shared object
$ ./vendor/drio-11.90/bin64/drrun -c ./target/debug/liblogger.so [-file=<save path>] [-filter] -- <process name>
```
For example, to trace `ls`:
```bash
$ ./vendor/drio-11.90/bin64/drrun -c ./target/debug/liblogger.so -- ls
Saved trace to "traces/ls_trace-68792aea.trace".
Summary:
TraceSummary {
    application: Application {
        name: "ls",
        address: 124220501499904,
    },
    counts: BlockCountStats {
        num_blocks: 4067,
        num_unique_blocks: 3895,
        num_unique_symbolized: 2221,
    },
    targeted: Some(
        BlockCountStats {
            num_blocks: 466,
            num_unique_blocks: 458,
            num_unique_symbolized: 295,
        },
    ),
    unique_apps: Some(
        5,
    ),
}
```
> The `.trace` file is a binary file, and to read it, it needs to be deserialized.

## Machine Learning (`/ml`)
Set of tools relating to tokenizing, training models, and running inference.

**Setup**

Make sure the virutal environment is active & dependencies are installed:
```
source .venv/bin/activate
uv pip install -e .
```

Build ffi components:
```
cd ./ffi
cargo build
maturin develop
cd ../
```

**Usage**

The `/ml` folder contains various utilities relating to machine learning.
- The tokenizer is built in Rust, and exposed via ffi at `bb_toolkit.BasicBlockTokenizer`
- The model is a GPT-2 transformer neural network model, located at `ml.model.BasicBlockPredictor`
- Tools related to handling training & model data are available in `ml.dataset`
- A training loop is available in `ml.train`
- A function for making inferences is available at `ml.inference`

Additionally, `ml/main.py` has a demo script that tokenizes the basic blocks, trains a model, and runs inference. To use it:
1. Collect traces of a wide selection of programs using the logger (`/logger`).
   To do this, you may want to use the (WIP) `bulk_collect` binary available on the `incremental` branch: [analyze/src/bin/bulk_collect.rs](../incremental/analyze/src/bin/bulk_collect.rs).
2. Update the paths in `ml/main.py` to point to your traces, and set model & tokenizer save paths.
   You may need to create a run directory.
   ```
   # if your tokenizer & model save paths still point to ./run
   mkdir run
   ```
3. Run the script with `uv`.
   ```
   uv run --active ml/main.py
   ```

## Setup
First, clone this repository:
```
git clone https://github.com/abscosmos/BasicBlockMLToolkit.git
cd BasicBlockMLToolkit
```

Install the necessary packages:
```bash
# cc & bindgen build DynamoRIO from source before generating bindings, so tools for building DynamoRIO are necessary
sudo apt install -y cmake gcc g++ g++-multilib clang zlib1g-dev libunwind-dev libsnappy-dev liblz4-dev libxxhash-dev perl binutils
# install rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# install uv (python package manager)
wget -qO- https://astral.sh/uv/install.sh | sh
# maturin builds the rust->python ffi components
cargo install --locked maturin
```

Create and activate a python virtual environment and install dependencies:
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

Next, install the DynamoRIO 11.90 executable itself:
```bash
wget https://github.com/DynamoRIO/dynamorio/releases/download/cronbuild-11.90.20281/DynamoRIO-Linux-11.90.20281.tar.gz
tar xvf DynamoRIO-Linux-11.90.20281.tar.gz
rm DynamoRIO-Linux-11.90.20281.tar.gz
mkdir vendor
mv DynamoRIO-Linux-11.90.20281.tar.gz vendor/drio-11.90/
```

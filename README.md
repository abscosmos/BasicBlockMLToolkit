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
> The `.trace` file is a binary file, and to read it needs to be deserialized.

## Setup
First, clone this repository:
```
git clone https://github.com/abscosmos/BasicBlockMLToolkit.git
cd BasicBlockMLToolkit
```

The logger (`/logger`) uses [`dynamorio-sys`](https://crates.io/crates/dynamorio-sys) for FFI bindings to DynamoRIO.
To generate the bindings, `dynamorio-sys` compiles DynamoRIO 11.90 from [source](https://github.com/DynamoRIO/dynamorio) using [`cc`](https://crates.io/crates/cc), and then uses [`bindgen`](https://crates.io/crates/bindgen).
Because of this, you need the necessary build tools to build DynamoRIO, which are listed [here](https://dynamorio.org/page_building.html).

Install the necessary packages:
```bash
# cc & bindgen build DynamoRIO from source before generating bindings, so tools for building DynamoRIO are necessary
sudo apt install -y cmake gcc g++ g++-multilib clang zlib1g-dev libunwind-dev libsnappy-dev liblz4-dev libxxhash-dev perl binutils
# install rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

Next, install the DynamoRIO 11.90 executable itself:
```bash
wget https://github.com/DynamoRIO/dynamorio/releases/download/cronbuild-11.90.20281/DynamoRIO-Linux-11.90.20281.tar.gz
tar xvf DynamoRIO-Linux-11.90.20281.tar.gz
rm DynamoRIO-Linux-11.90.20281.tar.gz
mkdir vendor
mv DynamoRIO-Linux-11.90.20281.tar.gz vendor/drio-11.90/
```

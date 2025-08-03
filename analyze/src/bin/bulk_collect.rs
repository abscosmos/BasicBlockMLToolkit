use std::path::Path;
use std::process::{Command, Stdio};
use anyhow::Context;
use itertools::Itertools;

fn main() -> anyhow::Result<()> {
    let logger_path = Path::new("./target/debug/liblogger.so");
    let drrun_path = Path::new("./vendor/drio-11.90/bin64/drrun");
    let base_directory = Path::new("./bulk_collect/");

    let trace_directory = Path::new("traces");
    let incremental_file = Path::new("incremental.bin");

    let programs = include_str!("./programs.txt");

    std::fs::create_dir_all(base_directory).context("failed to create trace directory")?;

    let incremental_path = base_directory.join(incremental_file);

    for line in programs.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut line_parts = line.split_whitespace();

        let program = line_parts.next().expect("line isn't empty");
        let args = line_parts.collect_vec();

        let trace_path = base_directory
            .join(trace_directory)
            .join(program)
            .with_extension("trace");

        let res = Command::new(drrun_path)
            .arg("-c").arg(logger_path)
            .arg(format!("-file={}", trace_path.display()))
            .arg(format!("-incremental={}", incremental_path.display()))
            .arg("--")
            .arg(program)
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match res {
            Ok(_) => println!("Traced \"{program}\". -> {}", trace_path.display()),
            Err(err) => println!("Failed to run \"{program}\": {err}"),
        }
    }

    Ok(())
}
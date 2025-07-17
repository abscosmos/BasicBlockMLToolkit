use std::fs;
use hashbrown::HashSet;
use std::ffi::OsStr;
use std::path::Path;
use anyhow::Context;
use itertools::Itertools;
use logger_core::{BlockCountStats, TraceData};

fn main() {
    let mut traces = load_traces(None::<&[&str]>).unwrap();
    traces.dedup_by_key(|t| t.targeted.name.clone());

    println!("Loaded traces: {:?}", traces.iter().map(|t| &t.targeted.name).collect::<Vec<_>>());

    println!("Stats:\n{:#?}", stats(&traces));
}

fn load_traces(applications: Option<&[impl AsRef<str>]>) -> anyhow::Result<Vec<TraceData>> {
    let search_dir = Path::new("./traces");

    let mut traces = Vec::new();

    for entry in fs::read_dir(search_dir).context("failed to read \"./traces\"")? {
        let path = entry.context("failed to get dir entry")?.path();

        if path.is_file()
            && path.extension() == Some(OsStr::new("trace"))
            && path.file_name().map(OsStr::to_str)
                .flatten()
                .is_some_and(|file_name| applications.is_none_or(|apps|
                    apps.iter().any(|app| file_name.starts_with(app.as_ref())
                )))
        {
            let bytes = fs::read(path).context("failed to read bytes")?;

            let trace_data = postcard::from_bytes(&bytes)
                .context("failed to deserialize trace data")?;

            traces.push(trace_data);
        }
    }

    Ok(traces)
}

fn stats<'a>(traces: impl IntoIterator<Item=&'a TraceData>) -> BlockCountStats {
    let traces = traces.into_iter();

    let mut all_block_loc = HashSet::new();
    let mut collective_unique = HashSet::new();

    for trace in traces {
        all_block_loc.extend(trace.blocks.keys());
        collective_unique.extend(trace.blocks.values().unique());
    }

    BlockCountStats {
        num_blocks: all_block_loc.len(),
        num_unique_blocks: collective_unique.len(),
        num_unique_symbolized: collective_unique.iter().unique_by(|b| b.symbolize()).count(),
    }
}
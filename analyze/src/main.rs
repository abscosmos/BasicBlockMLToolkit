use std::fs;
use hashbrown::HashMap;
use std::ffi::OsStr;
use std::num::NonZeroUsize;
use std::path::Path;
use anyhow::Context;
use logger_core::{Application, BlockCountStats, TraceData};

fn main() {
    let mut traces = load_traces(None::<&[&str]>).unwrap();
    traces.dedup_by_key(|t| t.targeted.name.clone());

    println!("Loaded traces: ");
    for trace in &traces {
        println!("{:#?}", trace.summary());
    }

    let naive_merge = naive_stats(&traces);

    println!("Naive merge:\n{:#?}", naive_merge);
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

fn naive_stats<'a>(traces: impl IntoIterator<Item=&'a TraceData>) -> BlockCountStats {
    let mut blocks = HashMap::new();

    for trace in traces {
        blocks.extend(trace.blocks.clone());
    }

    BlockCountStats::from_iter(blocks.values())
}
use std::{fs, io};
use hashbrown::HashSet;
use std::ffi::OsStr;
use std::path::Path;
use anyhow::Context;
use logger_core::TraceData;

fn main() {
    let traces = load_traces(&["ls", "cd"]).unwrap();

    let trace_apps = traces.iter()
        .map(|t| t.targeted.name.as_ref())
        .collect::<HashSet<_>>();

    println!("Loaded traces for {trace_apps:?}");
}

fn load_traces(applications: &[impl AsRef<str>]) -> anyhow::Result<Vec<TraceData>> {
    let search_dir = Path::new("./traces");

    let mut traces = Vec::new();

    for entry in fs::read_dir(search_dir).context("failed to read \"./traces\"")? {
        let path = entry.context("failed to get dir entry")?.path();

        if path.is_file()
            && path.extension() == Some(OsStr::new("trace"))
            && path.file_name().map(OsStr::to_str)
                .flatten()
                .is_some_and(|file_name|
                    applications.iter().any(|app|
                        file_name.starts_with(app.as_ref())
                    )
                )
        {
            let bytes = fs::read(path).context("failed to read bytes")?;

            let trace_data = postcard::from_bytes(&bytes)
                .context("failed to deserialize trace data")?;

            traces.push(trace_data);
        }
    }

    Ok(traces)
}
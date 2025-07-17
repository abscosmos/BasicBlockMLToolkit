use std::{fs, io};
use std::ffi::OsStr;
use std::path::Path;
use logger_core::TraceData;

fn main() {
    println!("Hello, world!");
}

fn load_traces(applications: &[impl AsRef<str>]) -> io::Result<Vec<TraceData>> {
    let search_dir = Path::new("./traces");

    let mut traces = Vec::new();

    for entry in fs::read_dir(search_dir)? {
        let path = entry?.path();

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
            let bytes = fs::read(path)?;

            let trace_data = postcard::from_bytes(&bytes)
                .expect("TODO: error handling; binary file wasn't valid");

            traces.push(trace_data);
        }
    }

    Ok(traces)
}
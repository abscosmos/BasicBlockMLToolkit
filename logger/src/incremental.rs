use std::{fs, io};
use std::path::Path;
use logger_core::{BasicBlock, Incremental};

// TODO: error handling for this function
pub fn write_incremental<'a>(blocks: impl IntoIterator<Item=BasicBlock>, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();

    let set = if path.exists() {
        let bytes = fs::read(path)?;
        
        let mut set = postcard::from_bytes::<Incremental>(&bytes).expect("should be valid binary data");
        
        set.0.extend(blocks);
        
        set
    } else {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        Incremental(blocks.into_iter().collect())
    };
    
    let bytes = postcard::to_allocvec(&set).expect("should serialize properly");
    
    fs::write(path, &bytes)?;
    
    Ok(())
}
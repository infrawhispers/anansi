use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let original_out_dir = PathBuf::from(env::var("OUT_DIR")?);
    tonic_build::configure()
        .out_dir(original_out_dir.clone())
        .file_descriptor_set_path(original_out_dir.join("api.bin"))
        .compile(&["proto/api.proto"], &["proto"])?;
    Ok(())
}

use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

use anyhow::bail;
use bytes::Bytes;
use futures::StreamExt;
use md5::{Digest, Md5};
use reqwest::header::HeaderMap;
use tokio::fs::File as tkFile;
use tokio::fs::OpenOptions as tkOpenOptions;
use tokio::io as tkio;
use tokio::io::AsyncReadExt;
use tokio::io::BufReader as tkBufReader;
use tracing::info;

fn get_md5_sync(file_path: &Path) -> anyhow::Result<String> {
    let f = OpenOptions::new().read(true).open(file_path)?;
    let mut reader = BufReader::new(f);
    let mut hasher = Md5::new();
    let mut buf: [u8; 8192] = [0; 8192]; //chunk size (8K, 65536, etc)
    while let Ok(size) = reader.read(&mut buf[..]) {
        if size == 0 {
            break;
        }
        hasher.update(&buf[0..size]);
    }
    let hash = hasher.finalize();
    Ok(base16ct::lower::encode_string(&hash))
}

async fn get_md5(file_path: &Path) -> anyhow::Result<String> {
    // let mut file = std::fs::File::open(file_path)?;
    // let mut hasher = Md5::new();
    // let bytes_written = std::io::copy(&mut file, &mut hasher)?;
    // let hash_bytes = hasher.finalize();
    // let base64_hash = base16ct::lower::encode_string(&hash_bytes);
    // Ok(base64_hash)
    let f = tkOpenOptions::new().read(true).open(file_path).await?;
    let mut reader = tkBufReader::new(f);
    let mut hasher = Md5::new();
    let mut buf: [u8; 8192] = [0; 8192]; //chunk size (8K, 65536, etc)
    while let Ok(size) = reader.read(&mut buf[..]).await {
        if size == 0 {
            break;
        }
        hasher.update(&buf[0..size]);
    }
    let hash = hasher.finalize();
    Ok(base16ct::lower::encode_string(&hash))
}

pub fn download_image_sync(uri: &str) -> anyhow::Result<Bytes> {
    let client = reqwest::blocking::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", "Mozilla/5.0".parse()?);
    let res = client.get(uri).headers(headers).send()?;
    Ok(res.bytes()?)
}
pub async fn download_image(uri: &str) -> anyhow::Result<Bytes> {
    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", "Mozilla/5.0".parse()?);
    let res = client.get(uri).headers(headers).send().await?;
    Ok(res.bytes().await?)
}

pub fn download_model_sync(
    model_name: &str,
    uri: &str,
    with_resume: bool,
    dest_path: &Path,
    md5sum: &str,
) -> anyhow::Result<()> {
    let tmp_file_path = dest_path.with_extension("onnx.part");
    let mut f: File;
    if tmp_file_path.exists() {
        info!(
            model = model_name,
            "opening the existing file: {:?}", tmp_file_path
        );
        f = OpenOptions::new()
            .write(true)
            .read(true)
            .append(true)
            .open(tmp_file_path.clone())?;
    } else {
        info!(model = model_name, "creating the file: {:?}", tmp_file_path);
        f = File::create(tmp_file_path.clone())?;
    }
    let resume_byte_pos = fs::metadata(tmp_file_path)?.len();
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", "Mozilla/5.0".parse()?);
    if resume_byte_pos != 0 && with_resume {
        info!(
            model = model_name,
            "resuming from position: {}", resume_byte_pos
        );
        headers.insert("Range", format!("bytes={}-", resume_byte_pos).parse()?);
    }
    let client = reqwest::blocking::Client::new();
    let mut res = client.get(uri).headers(headers).send()?;
    if res.status() == 416 {
        let res = get_md5_sync(&dest_path.with_extension("onnx.part"))?;
        if res == md5sum {
            std::fs::rename(dest_path.with_extension("onnx.part"), dest_path)?;
        } else {
            bail!("md5sum: {} does not match, got: {} ", md5sum, res)
        }
        return Ok(());
    }

    if res.status() != 206 && res.status() != 200 {
        bail!("recived status code: {}, bailing", res.status());
    }
    let bytes_total: u64;
    match res.content_length() {
        Some(bytes) => bytes_total = bytes,
        None => bytes_total = u64::MAX,
    };
    info!(
        model = model_name,
        "total size: {} MB",
        bytes_total / 1_000_000
    );

    let mut buffer = [0; 8192];
    let mut do_read: bool = true;
    let mut checkpoints: [bool; 10] = [false; 10];
    let mut bytes_downloaded: u64 = 0;
    while do_read {
        let bytes_written = res.read(&mut buffer)?;
        // println!("{}", bytes_written);
        if bytes_written == 0 {
            do_read = false
        }
        bytes_downloaded += bytes_written as u64;
        let pct = (bytes_downloaded as f64) / (bytes_total as f64) * 100.0;
        let idx = (((pct / 10.0) as u64) % 10) as usize;
        if idx < checkpoints.len() && !checkpoints[idx] {
            info!(
                model = model_name,
                "downloaded: {:.2}% ({}MB of {}MB)",
                pct,
                bytes_downloaded / 1_000_000,
                bytes_total / 1_000_000,
            );
            checkpoints[idx] = true
        }
        std::io::copy(&mut &buffer[0..bytes_written], &mut f)?;
    }
    let calc_md5sum = get_md5_sync(&dest_path.with_extension("onnx.part"))?;
    info!(model = model_name, "md5 calculated: {}", calc_md5sum);
    if calc_md5sum == md5sum {
        std::fs::rename(dest_path.with_extension("onnx.part"), dest_path)?;
    } else {
        info!(
            model = model_name,
            "md5 mismatch: expected {}, got: {}", calc_md5sum, md5sum
        );
        bail!("md5sum: {} does not match, got: {} ", md5sum, calc_md5sum)
    }
    info!(model = model_name, "md5 matches, download completed.");
    Ok(())
}

pub async fn download_model(
    uri: &str,
    with_resume: bool,
    dest_path: &Path,
    md5sum: &str,
) -> anyhow::Result<()> {
    let tmp_file_path = dest_path.with_extension("onnx.part");
    let mut f: tkFile;
    if tmp_file_path.exists() {
        println!("opening the existing file: {:?}", tmp_file_path);
        f = tkOpenOptions::new()
            .write(true)
            .read(true)
            .append(true)
            .open(tmp_file_path)
            .await?;
    } else {
        println!("creating the file: {:?}", tmp_file_path);
        f = tkFile::create(tmp_file_path).await?;
    }
    let resume_byte_pos = f.metadata().await?.len();
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", "Mozilla/5.0".parse()?);
    if resume_byte_pos != 0 && with_resume {
        headers.insert("Range", format!("bytes={}-", resume_byte_pos).parse()?);
    }
    let client = reqwest::Client::new();
    let res = client.get(uri).headers(headers).send().await?;
    if res.status() == 416 {
        let res = get_md5(&dest_path.with_extension("onnx.part")).await?;
        if res == md5sum {
            tokio::fs::rename(dest_path.with_extension("onnx.part"), dest_path).await?;
        } else {
            bail!("md5sum: {} does not match, got: {} ", md5sum, res)
        }
        return Ok(());
    }
    if res.status() != 206 && res.status() != 200 {
        bail!("recived status code: {}, bailing", res.status());
    }
    let total_bytes: u64;
    match res.content_length() {
        Some(bytes) => total_bytes = bytes,
        None => total_bytes = u64::MAX,
    };
    println!("total_bytes: {:?}", total_bytes);
    let mut stream = res.bytes_stream();
    while let Some(pb) = stream.next().await {
        let b = pb?;
        tkio::copy(&mut &*b, &mut f).await?;
    }
    // now compute the MD5 of the file to ensure that we are good to go.
    let res = get_md5(&dest_path.with_extension("onnx.part")).await?;
    if res == md5sum {
        tokio::fs::rename(dest_path.with_extension("onnx.part"), dest_path).await?;
    } else {
        bail!("md5sum: {} does not match, got: {} ", md5sum, res)
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tracing::Level;

    #[test]
    fn test_donwload_sync() {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("unable to create the tracing subscriber");

        let uri = "https://d1wz516niig2xr.cloudfront.net/scratch/download.bin";
        let with_resume = true;
        let dest_path = PathBuf::from("scratch/test");
        if !dest_path.exists() {
            fs::create_dir_all(dest_path.clone()).expect("unable to create the directory");
        }

        let md5sum = "e8982e2fd2982e8744a74fa9607bd118";
        download_model_sync(
            &"dowload.bin".to_string(),
            uri,
            with_resume,
            &dest_path.join("download.bin"),
            md5sum,
        )
        .expect("error while downloading the test file");
        assert_eq!(dest_path.join("download.bin").exists(), true);
    }
}

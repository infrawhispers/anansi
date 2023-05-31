use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use std::path::Path;
use tokio::task::JoinSet;

use anyhow::{bail, Context};
use tokio_uring::fs::File;
use tracing::info;

use futures::future::{self, join_all, try_join_all};

pub struct AlignedRead<'a> {
    pub offset: u64,
    pub len: usize,
    pub buf: &'a mut [u8],
}

pub struct Reader {
    runtime: Arc<tokio_uring::Runtime>,
    file: Option<tokio_uring::fs::File>,
}

impl Reader {
    pub fn get_runtime() -> anyhow::Result<Arc<tokio_uring::Runtime>> {
        Ok(Arc::new(
            tokio_uring::Runtime::new(
                tokio_uring::builder()
                    .entries(64)
                    .uring_builder(tokio_uring::uring_builder().setup_cqsize(1024)),
            )
            .with_context(|| "unable to create the tokio_uring runtime")?,
        ))
    }

    pub fn new(
        filepath: &std::path::Path,
        runtime: Arc<tokio_uring::Runtime>,
    ) -> anyhow::Result<Self> {
        info!("initiating reader for: {filepath:?}");
        let mut obj = Reader {
            runtime: runtime,
            file: None,
        };
        obj.populate_file(filepath)?;
        Ok(obj)
    }
    fn populate_file(&mut self, filepath: &Path) -> anyhow::Result<()> {
        let file = self
            .runtime
            .block_on(async move { File::open(filepath).await })?;
        self.file = Some(file);
        Ok(())
    }
    /// submits n requests to the kernel in order to run the read operations, this is expected
    /// to
    pub fn read(&self, read_reqs: &mut [AlignedRead]) -> anyhow::Result<()> {
        self.runtime.block_on(async move {
            let file = self
                .file
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no attached file, cannot issue read requests"))?;
            let ops: Vec<_> = read_reqs
                .iter_mut()
                .map(|read_req| async {
                    let buf: Vec<u8> = vec![0u8; read_req.buf.len()];
                    let (res, buf) = file.read_at(buf, read_req.offset).await;
                    // println!("this is done: offset is: {}", );
                    // println!(
                    //     "offset: {}, read {} bytes",
                    //     read_req.offset,
                    //     read_req.buf.len()
                    // );
                    let n = res?;
                    if n != read_req.buf.len() {
                        bail!(
                            "mismatch in bytes read - got: {} bytes expected: {} bytes",
                            n,
                            read_req.buf.len()
                        )
                    }
                    read_req.buf[..].copy_from_slice(&buf);
                    Ok(())
                })
                .collect();
            let res = join_all(ops).await;
            for res in res {
                match res {
                    Ok(()) => {}
                    Err(err) => return Err(err),
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;
    #[test]
    fn test_reader() {
        let runtime = Reader::get_runtime().expect("unable to create the runtime");
        let reader = Reader::new(&Path::new(".disk/disk.bin"), runtime.clone())
            .expect("should be able to create the reader");
        let mut buf1 = crate::av_store::AlignedDataStore::<u8>::new(1, 4096);
        let mut buf2 = crate::av_store::AlignedDataStore::<u8>::new(1, 4096);
        let mut buf3 = crate::av_store::AlignedDataStore::<u8>::new(1, 4096);
        let mut read_reqs: Vec<AlignedRead> = vec![
            AlignedRead {
                len: 4096,
                buf: &mut buf1.data[..],
                offset: 4096,
            },
            AlignedRead {
                len: 4096,
                buf: &mut buf2.data[..],
                offset: 2 * 4096,
            },
            AlignedRead {
                len: 4096,
                buf: &mut buf3.data[..],
                offset: 3 * 4096,
            },
        ];
        reader.read(&mut read_reqs).expect("reading should be fine");
        assert_eq!(buf1.data[0..4], [0u8, 0u8, 4u8, 66u8]);
    }
}

//! Crash-safe file writes.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temporary sibling of `path`: same directory (so the final rename stays on
/// one file system and is atomic), hidden, and unique per process and call.
fn temp_sibling(path: &Path) -> io::Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let unique = format!(
        ".{}.tmp-{}-{}",
        name.to_string_lossy(),
        std::process::id(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    Ok(path.with_file_name(unique))
}

/// Atomically replace `path` with the bytes produced by `fill`.
///
/// The data goes to a temporary file in the same directory, is flushed to
/// disk (`fsync`), and is then renamed over `path`. Readers see either the
/// old file or the complete new one, never a partial write. If `fill` or any
/// I/O step fails, the temporary file is removed and `path` is untouched.
pub fn atomic_write_with(
    path: &Path,
    fill: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let tmp = temp_sibling(path)?;
    let result = (|| {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        fill(&mut f)?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        // Persist the directory entry too (best effort; not supported everywhere).
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty())
            && let Ok(d) = File::open(dir)
        {
            let _ = d.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Atomically replace `path` with `bytes`.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    atomic_write_with(path, |f| f.write_all(bytes))
}

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A fresh empty directory under the system temp dir, removed on drop.
    pub struct TempDir(pub PathBuf);
    impl TempDir {
        pub fn new(tag: &str) -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let p = std::env::temp_dir().join(format!(
                "mklight-engine-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        pub fn path(&self) -> &std::path::Path {
            &self.0
        }
        pub fn entries(&self) -> Vec<String> {
            let mut v: Vec<String> = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::TempDir;
    use super::*;

    #[test]
    fn replaces_file_and_leaves_no_temp() {
        let dir = TempDir::new("atomic");
        let p = dir.path().join("a.mklight");
        atomic_write(&p, b"first").unwrap();
        atomic_write(&p, b"second").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"second");
        assert_eq!(dir.entries(), vec!["a.mklight".to_string()]);
    }

    #[test]
    fn failed_write_keeps_original() {
        let dir = TempDir::new("atomic-fail");
        let p = dir.path().join("a.mklight");
        atomic_write(&p, b"original").unwrap();
        let err = atomic_write_with(&p, |f| {
            f.write_all(b"partial garbage")?;
            Err(io::Error::other("simulated crash mid-write"))
        })
        .unwrap_err();
        assert_eq!(err.to_string(), "simulated crash mid-write");
        assert_eq!(fs::read(&p).unwrap(), b"original");
        assert_eq!(dir.entries(), vec!["a.mklight".to_string()]);
    }

    #[test]
    fn missing_directory_is_an_error() {
        let dir = TempDir::new("atomic-missing");
        let p = dir.path().join("no/such/dir/a.mklight");
        assert!(atomic_write(&p, b"x").is_err());
        assert!(dir.entries().is_empty());
    }
}

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
/// disk (`File::sync_all`, which is `F_FULLFSYNC` on Apple platforms and
/// `fsync` elsewhere), and is then renamed over `path`. Readers see either
/// the old file or the complete new one, never a partial write. If `fill` or
/// any I/O step fails, the temporary file is removed and `path` is untouched.
///
/// The replacement is transparent for an existing file: a symbolic link is
/// followed, so the link stays and its target is replaced; the file's
/// permissions are carried over; and a read-only file is refused with
/// `PermissionDenied`, exactly as an in-place write would be.
pub fn atomic_write_with(
    path: &Path,
    fill: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    // Resolve symlinks (and `..`) for an existing file; a new file keeps
    // the path as given.
    let target = match fs::canonicalize(path) {
        Ok(p) => p,
        Err(e) if e.kind() == io::ErrorKind::NotFound => path.to_path_buf(),
        Err(e) => return Err(e),
    };
    let existing = match fs::metadata(&target) {
        Ok(m) => Some(m.permissions()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if existing.as_ref().is_some_and(|p| p.readonly()) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{} is read-only", target.display()),
        ));
    }
    let tmp = temp_sibling(&target)?;
    let result = (|| {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        fill(&mut f)?;
        f.flush()?;
        if let Some(perm) = existing {
            f.set_permissions(perm)?;
        }
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, &target)?;
        // Persist the directory entry too (best effort; not supported everywhere).
        if let Some(dir) = target.parent().filter(|d| !d.as_os_str().is_empty())
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

    #[cfg(unix)]
    #[test]
    fn symlinks_and_permissions_are_preserved() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let dir = TempDir::new("atomic-link");
        let real = dir.path().join("real.mklight");
        atomic_write(&real, b"v1").unwrap();
        fs::set_permissions(&real, fs::Permissions::from_mode(0o640)).unwrap();
        let link = dir.path().join("link.mklight");
        symlink(&real, &link).unwrap();
        atomic_write(&link, b"v2").unwrap();
        // The link is still a link, and its target got the new bytes.
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&real).unwrap(), b"v2");
        assert_eq!(
            fs::metadata(&real).unwrap().permissions().mode() & 0o777,
            0o640
        );
        assert_eq!(
            dir.entries(),
            vec!["link.mklight".to_string(), "real.mklight".to_string()]
        );
        // A read-only file is refused and left alone.
        fs::set_permissions(&real, fs::Permissions::from_mode(0o444)).unwrap();
        let err = atomic_write(&link, b"v3").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(fs::read(&real).unwrap(), b"v2");
        assert_eq!(dir.entries().len(), 2);
    }

    #[test]
    fn missing_directory_is_an_error() {
        let dir = TempDir::new("atomic-missing");
        let p = dir.path().join("no/such/dir/a.mklight");
        assert!(atomic_write(&p, b"x").is_err());
        assert!(dir.entries().is_empty());
    }
}

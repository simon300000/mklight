//! Compilation of one unit with NVRTC (loaded at run time). Shared by the
//! crate (units without an embedded image) and its build script (every unit
//! ahead of time), so it depends only on `cudarc` and [`crate::sources`].

use std::ffi::CString;

use cudarc::nvrtc::sys as nv;

/// NVRTC's version, or `None` when the library cannot be loaded.
pub fn version() -> Option<(i32, i32)> {
    if !unsafe { nv::is_culib_present() } {
        return None;
    }
    let (mut a, mut b) = (0, 0);
    let r = unsafe { nv::nvrtcVersion(&mut a, &mut b) };
    (r == nv::nvrtcResult::NVRTC_SUCCESS).then_some((a, b))
}

/// Compile `source` (named `name`) for `target`: `sm_XY` gives a CUBIN,
/// `compute_XY` NUL-terminated PTX (what `cuModuleLoadData` expects). The
/// headers are [`super::sources::HEADERS`], the options
/// [`super::sources::OPTIONS`]. Returns the image and NVRTC's log.
pub fn compile(name: &str, source: &str, target: &str) -> Result<(Vec<u8>, String), String> {
    let headers = super::sources::HEADERS;
    let cstr = |s: &str| CString::new(s).map_err(|e| e.to_string());
    let src = cstr(source)?;
    let file = cstr(&format!("{name}.cu"))?;
    let hdr: Vec<CString> = headers
        .iter()
        .map(|(_, s)| cstr(s))
        .collect::<Result<_, _>>()?;
    let hdr_names: Vec<CString> = headers
        .iter()
        .map(|(n, _)| cstr(n))
        .collect::<Result<_, _>>()?;
    let hp: Vec<*const i8> = hdr.iter().map(|c| c.as_ptr()).collect();
    let hn: Vec<*const i8> = hdr_names.iter().map(|c| c.as_ptr()).collect();
    let mut opts: Vec<CString> = super::sources::OPTIONS
        .iter()
        .map(|o| cstr(o))
        .collect::<Result<_, _>>()?;
    opts.push(cstr(&format!("--gpu-architecture={target}"))?);
    let op: Vec<*const i8> = opts.iter().map(|c| c.as_ptr()).collect();
    let ok = |r: nv::nvrtcResult, what: &str| {
        if r == nv::nvrtcResult::NVRTC_SUCCESS {
            Ok(())
        } else {
            Err(format!("NVRTC {what} ({name}, {target}): {r:?}"))
        }
    };
    // SAFETY: every pointer outlives the calls; the program is destroyed on
    // every path after creation.
    unsafe {
        let mut prog = std::mem::MaybeUninit::uninit();
        ok(
            nv::nvrtcCreateProgram(
                prog.as_mut_ptr(),
                src.as_ptr(),
                file.as_ptr(),
                hp.len() as i32,
                hp.as_ptr(),
                hn.as_ptr(),
            ),
            "create",
        )?;
        let mut prog = prog.assume_init();
        let result = (|| {
            let r = nv::nvrtcCompileProgram(prog, op.len() as i32, op.as_ptr());
            let mut log_size = 0usize;
            nv::nvrtcGetProgramLogSize(prog, &mut log_size);
            let mut log = vec![0u8; log_size.max(1)];
            nv::nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut i8);
            let log = String::from_utf8_lossy(&log[..log_size.saturating_sub(1)]).into_owned();
            if r != nv::nvrtcResult::NVRTC_SUCCESS {
                return Err(format!("compiling {name} for {target}: {r:?}\n{log}"));
            }
            let mut size = 0usize;
            let image = if target.starts_with("compute_") {
                ok(nv::nvrtcGetPTXSize(prog, &mut size), "PTX size")?;
                let mut ptx = vec![0u8; size];
                ok(nv::nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut i8), "PTX")?;
                // nvrtcGetPTXSize counts the terminating NUL.
                ptx
            } else {
                ok(nv::nvrtcGetCUBINSize(prog, &mut size), "CUBIN size")?;
                let mut cubin = vec![0u8; size];
                ok(
                    nv::nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut i8),
                    "CUBIN",
                )?;
                cubin
            };
            Ok((image, log))
        })();
        nv::nvrtcDestroyProgram(&mut prog);
        result
    }
}

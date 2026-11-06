//! Command-line grammar (clap derive). `docs/CLI.md` documents every command.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "mklight",
    version,
    about = "MKLight headless optical design engine",
    long_about = "MKLight headless optical design engine.\n\n\
        Exit codes: 0 ok, 1 usage, 2 invalid model, 3 analysis failed, 4 I/O error, \
        130 cancelled."
)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalOpts,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Args, Debug, Clone, Default)]
pub struct GlobalOpts {
    /// Print machine-readable JSON on stdout instead of tables.
    #[arg(long, global = true)]
    pub json: bool,
    /// Omit wall-clock fields (started_unix_ms, duration_ms, elapsed_ms,
    /// metadata created/modified) so repeated runs print identical bytes.
    #[arg(long, global = true)]
    pub no_timing: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Engine and ABI versions, compute backends and their precision.
    Info,
    /// Create a project file (empty, or from a built-in sample).
    New(NewArgs),
    /// Validate a project (structure, materials, coatings, apertures).
    Validate(ProjectArg),
    /// Lens data table and system settings.
    Show(ShowArgs),
    /// Upgrade a project file to the current schema version (keeps a backup).
    Migrate(MigrateArgs),
    /// Import a lens prescription file (.zmx or .seq) into a new project.
    Import(ImportArgs),
    /// User glass catalogs (AGF or JSON). Vendor catalogs are never shipped.
    #[command(subcommand)]
    Catalogs(CatalogsCmd),
    /// List registered analyses with their settings schema and defaults.
    Analyses(AnalysesArgs),
    /// Run a registered analysis through the engine job scheduler.
    Run(RunArgs),
    /// Trace one real ray with the CPU f64 reference tracer.
    Trace(TraceArgs),
    /// First-order (paraxial) properties at every system wavelength.
    Paraxial(ParaxialArgs),
    /// Set one parameter through its ParamPath, validate and save atomically.
    Set(SetArgs),
}

#[derive(Args, Debug)]
pub struct ProjectArg {
    /// Project file (.mklight).
    pub project: PathBuf,
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Project file to create.
    pub file: PathBuf,
    /// Start from a built-in sample (`singlet`, `achromat`, `cooke_triplet`,
    /// `double_gauss`, `newtonian`, `petzval_asphere`, `relay`).
    #[arg(long)]
    pub sample: Option<String>,
    /// Overwrite an existing file.
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct ImportArgs {
    /// Lens prescription file (.zmx or .seq).
    pub file: PathBuf,
    /// Project file to create.
    pub project: PathBuf,
    /// Glass catalog (AGF or .json) to resolve glasses against; repeatable.
    /// Recorded in the project.
    #[arg(long)]
    pub catalog: Vec<PathBuf>,
    /// Overwrite an existing project file.
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    pub project: PathBuf,
    /// Do not trace rays for automatic semi-diameters (faster; shows "auto").
    #[arg(long)]
    pub no_semi_diameters: bool,
}

#[derive(Args, Debug)]
pub struct MigrateArgs {
    pub project: PathBuf,
    /// Report what would change without writing anything.
    #[arg(long)]
    pub dry_run: bool,
    /// Do not keep a backup of the original file.
    #[arg(long)]
    pub no_backup: bool,
}

#[derive(Subcommand, Debug)]
pub enum CatalogsCmd {
    /// Catalogs available to a project (built-in first).
    List(CatalogScope),
    /// Glasses of one catalog.
    Show(CatalogShowArgs),
    /// Load an AGF (or .json) catalog and record it in a project.
    Import(CatalogImportArgs),
}

#[derive(Args, Debug)]
pub struct CatalogScope {
    /// Include the user catalogs this project references.
    #[arg(long)]
    pub project: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct CatalogShowArgs {
    /// Catalog name (case-insensitive), e.g. BUILTIN.
    pub name: String,
    #[arg(long)]
    pub project: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct CatalogImportArgs {
    /// AGF catalog file (or a GlassCatalog .json file).
    pub file: PathBuf,
    /// Project that records the catalog reference (saved atomically).
    #[arg(long)]
    pub project: PathBuf,
}

#[derive(Args, Debug)]
pub struct AnalysesArgs {
    /// Show one analysis in detail.
    pub name: Option<String>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierChoice {
    /// Design-grade budgets and sampling.
    Design,
    /// Cheaper sampling and aiming within the preview budgets; labelled in `run_info.tier`.
    Preview,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AimStopChoice {
    /// Stop real-aim solves at the project's requested tolerance.
    Requested,
    /// Stop at the residual derived from the propagated error (CONVENTIONS §7).
    Conditioned,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// CPU f64 reference.
    Cpu,
    /// Metal GPU backend (fails if unavailable or not implemented for the analysis).
    Metal,
    /// Chosen per request by the engine's cost model (docs/GPU.md §9.2), else the CPU reference.
    Auto,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    pub project: PathBuf,
    /// Analysis name (see `mklight analyses`).
    pub analysis: String,
    /// Settings as JSON text, or `@file` to read them from a file. Missing
    /// keys take the schema defaults.
    #[arg(long)]
    pub settings: Option<String>,
    /// Random seed (sets `settings.seed`; only for analyses that take one).
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,
    /// Accuracy tier (recorded in `run_info.tier`).
    #[arg(long, value_enum, default_value_t = TierChoice::Design)]
    pub tier: TierChoice,
    /// Aim-stop policy; the engine default when omitted (recorded in `run_info.aim_stop`).
    #[arg(long, value_enum)]
    pub aim_stop: Option<AimStopChoice>,
    /// Write the result envelope (JSON) to this file instead of stdout.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Report progress on stderr while the job runs.
    #[arg(long)]
    pub progress: bool,
    /// Cancel the job after this many seconds (exit 130).
    #[arg(long)]
    pub timeout: Option<f64>,
    /// Include buffer contents (`data` arrays) in the JSON envelope.
    #[arg(long)]
    pub buffers: bool,
}

#[derive(Args, Debug)]
pub struct TraceArgs {
    pub project: PathBuf,
    /// System field index (0-based).
    #[arg(long, default_value_t = 0)]
    pub field: usize,
    /// System wavelength index (0-based); default: the primary wavelength.
    #[arg(long)]
    pub wavelength: Option<usize>,
    /// Normalised pupil coordinates `px,py` on the unit disk (`|p| ≤ 1`).
    #[arg(long, default_value = "0,0", allow_hyphen_values = true)]
    pub pupil: String,
    /// Print per-surface ray data.
    #[arg(long)]
    pub surfaces: bool,
    /// Ignore fixed semi-diameters and apertures.
    #[arg(long)]
    pub no_clip: bool,
}

#[derive(Args, Debug)]
pub struct ParaxialArgs {
    pub project: PathBuf,
    /// Also print the paraxial marginal and chief ray tables (primary wavelength).
    #[arg(long)]
    pub rays: bool,
}

#[derive(Args, Debug)]
pub struct SetArgs {
    pub project: PathBuf,
    /// Parameter path as JSON, e.g. '{"param":"thickness","surface":3}'.
    pub path: String,
    /// New value (a number, or inf / -inf).
    #[arg(allow_hyphen_values = true)]
    pub value: String,
    /// Validate and report without saving.
    #[arg(long)]
    pub dry_run: bool,
    /// Save to this file instead of overwriting the project.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

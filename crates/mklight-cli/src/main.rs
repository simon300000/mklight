//! `mklight`: the headless command line of MKLight.
//!
//! A thin front end over `mklight-engine` (the facade shared with the macOS
//! app and the Python bindings): projects, sessions, the analysis registry
//! and the job scheduler. Human-readable tables by default, `--json` for
//! machine output, deterministic bytes with `--no-timing`. Commands, output
//! formats and exit codes are documented in `docs/CLI.md`.

mod cli;
mod cmd;
mod error;
mod output;
mod signal;

use clap::Parser;
use serde_json::json;
use std::io::Write;

use cli::{CatalogsCmd, Cli, Command};
use cmd::{Ctx, Report};
use error::{CliResult, Exit};

fn main() {
    std::process::exit(real_main());
}

fn real_main() -> i32 {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            // Help and version requests succeed; every other parse problem is
            // a usage error (exit 1, not clap's default 2, which is reserved
            // for invalid models).
            let _ = e.print();
            return if e.exit_code() == 0 {
                Exit::Ok.code()
            } else {
                Exit::Usage.code()
            };
        }
    };
    let ctx = Ctx::new(cli.global.clone());
    let outcome = dispatch(&ctx, cli.command);
    let g = &ctx.global;
    match outcome {
        Ok(mut r) => {
            if g.json {
                if g.no_timing {
                    output::strip_timing(&mut r.json);
                }
                emit_stdout(&output::to_pretty(&r.json));
            } else if r.text_to_stderr {
                eprint!("{}", r.text);
            } else {
                emit_stdout(&r.text);
            }
            r.exit.code()
        }
        Err(e) => {
            eprintln!("mklight: error: {}", e.message);
            if g.json {
                let doc = json!({"error": {
                    "exit_code": e.exit.code(),
                    "kind": e.exit.kind(),
                    "message": e.message,
                }});
                emit_stdout(&output::to_pretty(&doc));
            }
            e.exit.code()
        }
    }
}

/// Write to stdout, ignoring a closed pipe (`mklight … | head`).
fn emit_stdout(s: &str) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(s.as_bytes());
    let _ = out.flush();
}

fn dispatch(ctx: &Ctx, command: Command) -> CliResult<Report> {
    match command {
        Command::Info => cmd::info::info(ctx),
        Command::New(a) => cmd::project::new(ctx, &a),
        Command::Validate(a) => cmd::project::validate(ctx, &a.project),
        Command::Show(a) => cmd::project::show(ctx, &a),
        Command::Migrate(a) => cmd::project::migrate(ctx, &a),
        Command::Set(a) => cmd::project::set(ctx, &a),
        Command::Catalogs(c) => match c {
            CatalogsCmd::List(a) => cmd::catalogs::list(ctx, &a),
            CatalogsCmd::Show(a) => cmd::catalogs::show(ctx, &a),
            CatalogsCmd::Import(a) => cmd::catalogs::import(ctx, &a),
        },
        Command::Analyses(a) => cmd::analyses::list(ctx, &a),
        Command::Run(a) => cmd::analyses::run(ctx, &a),
        Command::Trace(a) => cmd::trace::trace(ctx, &a),
        Command::Paraxial(a) => cmd::trace::paraxial(ctx, &a),
    }
}

//! `marsh` binary entry point.

use clap::Parser;
use marsh::cli::{run, Cli};

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

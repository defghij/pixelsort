use clap::{Command, Arg, ArgMatches}; 
use std::path::PathBuf;


fn main() {

    // Argument Parsing
    let args = Command::new("pixel-sort")
        .author("Chuck Norris, cnorri17@jhu.edu")
        .version("0.0.1")
        .about("This program sorts pixel in a given image file.")
        .args([
            Arg::new("image-in")
                .short('i')
                .value_parser(clap::value_parser!(std::path::PathBuf))
                .required(true)
                .help("Input file name."),
            Arg::new("image-out")
                .short('o')
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .required(true)
                .help("Output file name."),
            Arg::new("sort-type")
                .short('s')
                .value_parser(["column-separate", "row-separate", "column-collective", "row-collective"])
                .default_value("column-separate")
                .help("Output file name"),
        ])
        .after_help("Longer explanation to appear after the options when \
              displaying the help information from --help or -h")
        .get_matches();

    let image_file = args.get_one::<std::path::PathBuf>("image-in");

}

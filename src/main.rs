use clap::{Command, Arg};
use std::{path::Path, ops::Deref};

use pixelsort::functions::*;



fn main()  {

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
    let test_file = "./src/test_files/multi_pixel.png";
    let pixel_array = PixelArray::from_path(&Path::new(test_file).to_path_buf());
    let mut p_array = pixel_array.pixels();
    p_array.sort_unstable();
    
    
    //let elems = pa.width * pa.height;
    //let mut npixels: Vec<NormalizedPixel> = Vec::from(pa.pixels());
    //for i in 0..elems {

    //}


    ()
}

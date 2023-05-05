use clap::{Command, Arg};
use image::{ImageEncoder};
use std::fs::File;
use std::{path::{PathBuf}, ops::Deref/*, ops::Deref*/};

use pixelsort::pixels::*;
use pixelsort::bitonic;
use pixelsort::timing;


fn main()  {

    // Argument Parsing
    let args = Command::new("pixel-sort")
        .author("Chuck Norris, cnorri17@jhu.edu")
        .version("0.0.1")
        .about("This program sorts pixels in a given image file.")
        .args([
            Arg::new("image-in")
                .short('i')
                .long("image-in")
                .value_parser(clap::value_parser!(std::path::PathBuf))
                .required(false)
                .help("Input file name."),
            Arg::new("image-out")
                .short('o')
                .long("image-out")
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .required(false)
                .help("Output file name."),
            Arg::new("timing-info")
                .short('t')
                .long("timing-graphs")
                .value_parser(["comparative", "isolated", "image", "profile"])
                .required(false)
                .help(
                      "Run timing functions and output appropiate graph.
                      If this argument is provide then all other provided arguments
                      are ignored."
                ),
            // Arg::new("sort-type")
            //     .short('s')
            //     .value_parser(["column-separate", "row-separate", "column-collective", "row-collective"])
            //     .default_value("column-separate")
            //     .help("Output file name"),
        ])
        .after_help("Longer explanation to appear after the options when \
              displaying the help information from --help or -h")
        .get_matches();

    // Check if we want timing info
    // Should we request timing info then we dont sort an image and
    // we ignore other arguments.
    let timing_info: Option<&String> = args.get_one::<String>("timing-info");
    if timing_info.is_some() {
        match timing_info.unwrap().as_str() {
            "comparative" => timing::small_single_thread_random_data_comparative(),
            "isolated"    => timing::large_single_thread_random_data_isolated(),
            "profile"    => timing::general_profile_random_data(),
            "image"       => {
                
                // Get pixel array and dimensions from provided arguments
                let pixel_array = PixelArray::from_path(args.get_one::<PathBuf>("image-in").unwrap());
                timing::image_comparative(pixel_array)
            },
            _             => panic!("Unreachable! Clap should have caught any non-valid strings")
        }
        return ();
    }

    let pixel_array = PixelArray::from_path(args.get_one::<PathBuf>("image-in").unwrap());
    let width: u32       = pixel_array.width.clone() as u32;
    let height: u32      = pixel_array.height.clone() as u32;
    let pixel_count: i64 = width as i64 * height as i64;

    let mut network = bitonic::network::Network::new(pixel_count as usize);
    let network = network.set_comparitors().clone();
    
    let mut array: Box<Vec<OrdinalPixel>> = pixel_array.pixels();
    network.sort(&mut array).unwrap();

    let sorted: Vec<u8> = array.deref()
                        .iter()
                        .map(|p| {p.pixel.clone()})
                        .flatten()
                        .collect();

    let buff = File::create(args.get_one::<String>("image-out").unwrap()).expect("Couldnt create file");
    image::codecs::png::PngEncoder::new(buff).write_image(sorted[..].as_ref(), width, height, image::ColorType::Rgba8).expect("Fuuu");
    ()
}

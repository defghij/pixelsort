use clap::{Command, Arg};
use image::{ImageEncoder};
use std::cmp::Ordering;
use std::fs::File;
use std::{path::{PathBuf}, ops::Deref/*, ops::Deref*/};

use pixelsort::pixels::*;
use pixelsort::bitonic;



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
                .required(true)
                .help("Input file name."),
            Arg::new("image-out")
                .short('o')
                .long("image-out")
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .required(true)
                .help("Output file name.")/*,
            Arg::new("sort-type")
                .short('s')
                .value_parser(["column-separate", "row-separate", "column-collective", "row-collective"])
                .default_value("column-separate")
                .help("Output file name"), */
        ])
        .after_help("Longer explanation to appear after the options when \
              displaying the help information from --help or -h")
        .get_matches();


    let pixel_array = PixelArray::from_path(args.get_one::<PathBuf>("image-in").unwrap());
    let width: u32       = pixel_array.width.clone() as u32;
    let height: u32      = pixel_array.height.clone() as u32;
    let pixel_count: i64 = width as i64 * height as i64;

    let network: Vec<bitonic::network::Node> = bitonic::recursive::network(pixel_count);
    let mut array: Box<Vec<OrdinalPixel>> = pixel_array.pixels();

    for node in network.iter() {
        let (i, j, direction): (usize, usize, Ordering) = node.details();
        if array[i].cmp(&array[j]) != direction {
            array.swap(i, j);
        }
    }

    let sorted: Vec<u8> = array.deref()
                        .iter()
                        .map(|p| {p.pixel.clone()})
                        .flatten()
                        .collect();

    let buff = File::create(args.get_one::<String>("image-out").unwrap()).expect("Couldnt create file");
    image::codecs::png::PngEncoder::new(buff).write_image(sorted[..].as_ref(), width, height, image::ColorType::Rgba8).expect("Fuuu");

    ()
}

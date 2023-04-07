pub mod functions {
    use std::{path::PathBuf};

    pub struct NormalizedPixel {
        pub pixel: Vec<u8>,
        pub norm: u64
    } impl NormalizedPixel {
        pub fn from_pixel(pixel: Vec<u8>) -> NormalizedPixel {
            if pixel.len() != 3 {
                panic!("Invalid vector for 3d norm!");
            }
            let x = pixel[0];
            let y = pixel[1];
            let z = pixel[2];
            let norm = (( (x as u64).pow(2) + (y as u64).pow(2) + (z as u64).pow(2) ) as f64).sqrt() as u64;
            NormalizedPixel { pixel, norm  }
        }
    }

    pub struct PixelArray {
        height: usize,
        width: usize,
        pixels: Box<Vec<u8>>,
        pixel_stride: usize
    } impl PixelArray {
        pub fn from_path(path: &PathBuf) -> PixelArray {
            let img_buf = image::io::Reader::open(path).expect("Failed to open image")
                                        .decode().expect("Failed to decode image")
                                        .into_rgb8();
            let dim = img_buf.dimensions();
            let width = dim.0 as usize;
            let height = dim.1 as usize;
            let pixels = Box::new(img_buf.into_raw());
            let pixel_stride = 3;
            PixelArray { 
                height,
                width,
                pixels,
                pixel_stride
            }                            

        }
        pub fn pixels(self) -> Box<Vec<u8>>{
            self.pixels
        }

        pub fn pixel(&self, w: usize, h: usize) -> Vec<u8> {
            let index = (self.width * h + w) * self.pixel_stride;
            println!("{}", index);
            self.pixels[index..(index + self.pixel_stride)].to_vec().clone()
        }
    }


    pub fn bitonic_sort(list: &mut Vec<u8>) -> &Vec<u8> {
        unimplemented!("Not Implemented Yet!");
    }
}

#[cfg(test)]
mod pixel_tests {
    use super::functions::*;
    use std::{path::Path};

    #[test]
    fn read_multi_pixel() {
        let test_file = "./src/test_files/multi_pixel.png";
        let pixel_array = PixelArray::from_path(&Path::new(test_file).to_path_buf());
        let img_buf = image::io::Reader::open(test_file).expect("Failed to open image")
                                    .decode().expect("Failed to decode image")
                                    .into_rgb8();

        // Transparent (0,0)
        for w in 0..800 {
            for h in 0..800 {
                let p1 = pixel_array.pixel(w,h);
                let p2 = img_buf.get_pixel(w as u32, h as u32);
                assert_eq!(p1[0], p2[0]);
                assert_eq!(p1[1], p2[1]);
                assert_eq!(p1[2], p2[2]);
            }
        }
    }
    /*
    #[test]
    fn bitonic_sort_small() {
        let mut list: Vec<u8>= vec![7,6,5,4,0,1,2,3];

        assert_eq!(*bitonic_sort(&mut list), [0,1,2,3,4,5,6,7]);

    }
    */
    #[test]
    fn sort_three_pixels() {
        let test_file = "./src/test_files/multi_pixel.png";
        let pixel_array = PixelArray::from_path(&Path::new(test_file).to_path_buf());
        let p0 = NormalizedPixel::from_pixel(pixel_array.pixel(338, 65));  //   0,   0,   0
        let p1 = NormalizedPixel::from_pixel(pixel_array.pixel(345, 72));  // 255, 255, 255
        let p2 = NormalizedPixel::from_pixel(pixel_array.pixel(346, 205)); //   0,   0, 255
        let mut v = vec![p0.norm, p1.norm, p2.norm];
        v.sort();
        assert_eq!(v, [p0.norm, p2.norm, p1.norm]);

    }
}
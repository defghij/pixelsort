pub mod functions {
    use std::{path::PathBuf};
    use std::cmp::Ordering;


    #[derive(Debug, Clone)]
    pub struct OrdinalPixel {
        pub pixel: Vec<u8>,
        pub norm: f32,
        strict: bool
    } impl OrdinalPixel {

        pub fn set_strict(&mut self, strict: bool) -> &mut Self {
            self.strict = strict;
            self
        }

    } impl From<Vec<u8>> for OrdinalPixel {
        fn from(pixel: Vec<u8>) -> Self {
            if pixel.len() < 3  || 4 < pixel.len() {
                panic!("Invalid vector for pixel! Expected a vector of length 3 or 4, found {}", pixel.len());
            }
            let mut norm: u32 = 0;
            for comp in pixel.iter() {
                norm = norm + (*comp as u32).pow(2);
            }
            let norm = (norm as f32).sqrt();
            OrdinalPixel { pixel, norm, strict: false }
        }

        
    } impl PartialOrd for OrdinalPixel {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            if self.strict { // Use Cosine Similarity as well.
                self.norm.partial_cmp(&other.norm)
            } else {
                self.norm.partial_cmp(&other.norm)
            }
        }
    } impl Ord for OrdinalPixel {
        fn cmp(&self, other: &Self) -> Ordering {
            if self.strict {
                self.norm.total_cmp(&other.norm)
            } else {
                self.norm.total_cmp(&other.norm)
            }

            
        }
    } impl PartialEq for OrdinalPixel {
        fn eq(&self, other: &Self) -> bool {
            if self.strict {
                self.cmp(&other).is_eq()
            } else {
                self.cmp(&other).is_eq()
            }
        }
    } impl Eq for OrdinalPixel {}


   #[allow(dead_code)]
    pub struct PixelArray {
        pub height: usize,
        pub width: usize,
        pixels: Box<Vec<OrdinalPixel>>,
        pixel_stride: usize
    } impl PixelArray {
        pub fn from_path(path: &PathBuf) -> PixelArray {
            let img_buf = image::io::Reader::open(path).expect("Failed to open image")
                                        .decode().expect("Failed to decode image")
                                        .into_rgba8();

            let pixel_dimensions = 4;
            let dim = img_buf.dimensions();
            let width = dim.0 as usize;
            let height = dim.1 as usize;
            let vector = img_buf.into_raw();
            let mut pixels: Vec<OrdinalPixel> = Vec::with_capacity(vector.len() / pixel_dimensions);
            for i in 0..pixels.capacity() {
                let pixel: Vec<u8> = vector[(i * 4).. (4 * i + 4)].to_vec();
                pixels.push(OrdinalPixel::from(pixel));
            }
            let pixels: Box<Vec<OrdinalPixel>> = Box::new(pixels);

            let pixel_stride: usize= 4;
            PixelArray { 
                height,
                width,
                pixels,
                pixel_stride
            }                            

        }

        pub fn pixels(self) -> Box<Vec<OrdinalPixel>>{
            self.pixels.clone()
        }

        pub fn pixel(&self, w: usize, h: usize) -> OrdinalPixel {
            let index = self.width * h + w;
            self.pixels[index].clone()
        }
    }  

    pub fn bitonic_sort(list: &mut Vec<u8>) -> &Vec<u8> {
        println!("{:?}", list);
        unimplemented!("Not Implemented Yet!");
    }
}

#[cfg(test)]
mod io_tests {
    use super::functions::*;
    use std::{path::Path};

    #[test]
    fn read_multi_pixel() {
        let test_file = "./src/test_files/multi_pixel.png";
        let pixel_array = PixelArray::from_path(&Path::new(test_file).to_path_buf());
        let img_buf = image::io::Reader::open(test_file).expect("Failed to open image")
                                    .decode().expect("Failed to decode image")
                                    .into_rgba8();

        // Transparent (0,0)
        for w in 0..800 {
            for h in 0..800 {
                let p1 = pixel_array.pixel(w,h);
                let p2 = img_buf.get_pixel(w as u32, h as u32);
                assert_eq!(p1.pixel[0], p2[0]);
                assert_eq!(p1.pixel[1], p2[1]);
                assert_eq!(p1.pixel[2], p2[2]);
            }
        }
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::functions::*;

    #[test]
    fn euclidean_norm_test() {
        let test_vectors: Vec<Vec<u8>>= vec![vec![255,255,255], vec![197,17,23] ];
        let expected_values: Vec<f32> = vec![441.67294, 199.06532];
        for test_set in test_vectors.iter().zip(expected_values.iter()){
            let (test_vector, expected_value) = test_set;
            let npixel  = OrdinalPixel::from(test_vector.clone());
            assert_eq!(npixel.norm, *expected_value);

        }
    }

    #[test]
    fn normalized_pixel_ord() {
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {

                    let p1 = OrdinalPixel::from( vec![i, j, k ]); 
                    let p2 = OrdinalPixel::from( vec![i, j, k + 1] );
                    assert!(p1.norm < p2.norm);
                    assert!( p1 < p2);

                    let p1 = OrdinalPixel::from( vec![i, j, k ]); 
                    let p2 = OrdinalPixel::from( vec![i, j + 1, k] );
                    assert!(p1.norm < p2.norm);
                    assert!( p1 < p2);

                    let p1 = OrdinalPixel::from( vec![i, j, k ]); 
                    let p2 = OrdinalPixel::from( vec![i + 1, j, k] );
                    assert!(p1.norm < p2.norm);
                    assert!( p1 < p2);
                }
            }
        }
    }
    
    #[test]
    fn normalized_pixel_eq() {
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {

                    let p1 = OrdinalPixel::from( vec![ i, j, k ] ); 
                    let p2 = OrdinalPixel::from( vec![ i, j, k ] );
                    assert_eq!(p1.norm, p2.norm);
                    assert_eq!(p1, p2);

                    let p1 = OrdinalPixel::from( vec![ i    , j, k ]); 
                    let p2 = OrdinalPixel::from( vec![ i + 1, j, k ]);
                    assert_ne!(p1.norm, p2.norm);
                    assert_ne!(p1, p2);

                    let p1 = OrdinalPixel::from( vec![ i, j    , k ]); 
                    let p2 = OrdinalPixel::from( vec![ i, j + 1, k ] );
                    assert_ne!(p1.norm, p2.norm);
                    assert_ne!(p1, p2);

                    let p1 = OrdinalPixel::from( vec![ i, j, k    ] ); 
                    let p2 = OrdinalPixel::from( vec![ i, j, k + 1] );
                    assert_ne!(p1.norm, p2.norm);
                    assert_ne!(p1, p2);


                }
            }
        }
    }

    #[test]
    fn sort_three_pixels() {
        let p0 = OrdinalPixel::from( vec![0  ,0  ,255] );
        let p1 = OrdinalPixel::from( vec![0  ,255,255] ); 
        let p2 = OrdinalPixel::from( vec![255,255,255] );
        let ev = vec![&p0, &p1, &p2];
        
        let mut test_vectors = vec![
            vec![&p0, &p1, &p2],
            vec![&p0, &p2, &p1],
            vec![&p1, &p2, &p0],
            vec![&p1, &p0, &p2],
            vec![&p2, &p1, &p0],
            vec![&p2, &p0, &p1],
        ];
        for tv in test_vectors.iter_mut() {
            tv.sort();
            assert_eq!(*tv, ev);
        }

    }
}
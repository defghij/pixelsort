pub mod pixels {
    use std::{path::PathBuf};
    use std::cmp::Ordering;


    #[derive(Debug, Clone)]
    pub struct OrdinalPixel {
        pub pixel: Vec<u8>,
        pub norm: f32,
    } impl OrdinalPixel {
        pub fn new(pixel: Vec<u8>) -> OrdinalPixel {
            let mut norm: u32 = 0;
            for comp in pixel.iter() {
                norm = norm + (*comp as u32).pow(2);
            }
            let norm = (norm as f32).sqrt();
            OrdinalPixel { pixel, norm }

        }
    } impl From<Vec<u8>> for OrdinalPixel {
        fn from(pixel: Vec<u8>) -> Self{
            OrdinalPixel::new(pixel)
        }
    } impl PartialOrd for OrdinalPixel {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            match self.norm.partial_cmp(&other.norm).unwrap() {
                Ordering::Less    => { return Some(Ordering::Less) },
                Ordering::Greater => { return Some(Ordering::Greater) },
                Ordering::Equal => {
                    for (a,b) in self.pixel.iter().zip(other.pixel.iter()).rev() {
                        if a == b { continue;                      } else 
                        if a < b  { return Some(Ordering::Less)    } else 
                        if a > b  { return Some(Ordering::Greater) }
                    }
                    Some(Ordering::Equal)
                }
            }
        }
    } impl Ord for OrdinalPixel {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.norm.partial_cmp(&other.norm).unwrap() {
                Ordering::Less    => { return Ordering::Less },
                Ordering::Greater => { return Ordering::Greater },
                Ordering::Equal => {
                    for (a,b) in self.pixel.iter().zip(other.pixel.iter()).rev() {
                        if a == b { continue;                } else 
                        if a < b  { return Ordering::Less    } else 
                        if a > b  { return Ordering::Greater }
                    }
                    Ordering::Equal
                }
            }

            
        }
    } impl PartialEq for OrdinalPixel {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(&other).is_eq()
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

}

pub mod bitonic_indices {
    use std::cmp::Ordering;

    // TODO: Generic over T: PartialOrd.
    pub fn bitonic_sort(list: &mut Vec<(i64, i64, Ordering)>, lo: i64, n: i64, dir: Ordering) {
        if n > 1 {
            let m: i64 = n / 2;
            let direction: Ordering = dir;
            let opposite: Ordering = match dir {
                Ordering::Less => Ordering::Greater,
                Ordering::Greater => Ordering::Less,
                Ordering::Equal => panic!("Invalid ordering for bitonic sort")
            };
            bitonic_sort( list, lo,       m, opposite);
            bitonic_sort( list, lo + m, n-m, direction);
            bitonic_merge(list, lo,       n, direction);
        }
    }

    #[inline]
    fn bitonic_merge(list: &mut Vec<(i64, i64, Ordering)>, lo: i64, n: i64, dir: Ordering) {
        if n > 1 { // Exit condition: n == 1
            let  m: i64 = greatest_power_of_two_less_than(n);
            for i in 0..lo+n-m {
                if list[i as usize].cmp(&list[(i+m) as usize]) != dir {
                    list.swap(i as usize, (i+m) as usize);
                }
            }
            bitonic_merge(list, lo, m, dir);
            bitonic_merge(list, lo+m, n-m, dir);
        }
    }


    /// Use the implementation found here: https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
    /// as a basis for an arbitrary n bitonic sort. Note that we want to find the indicies first so that 
    /// we can avoid using recursion and use a sorting network (fixed comaparisons).
    /// 
    /// Specifically, standard bitonic sort uses a comparator network B_{p} where p is a power of 2 as the basic 
    /// building block. We derive network Bn for arbitrary n from network B_{p} where p is the next-greatest 
    /// power of 2 by using only the first n – p/2 comparators of B_{p}. 
    pub fn calculate_indicies(len: i64) -> Vec<(i64, i64, Ordering)> {
        let mut compares: Vec<(i64, i64, Ordering)> = Vec::new();
        bitonic_sort_index_calculation(&mut compares, 0, len, Ordering::Greater);
        compares
        
    }

    /// Bitonic Sort Network generated for six input array.
    ///             COMPARES
    ///         0  1  2  3  4  5  6  7  
    ///     0------⊛--⊛--⊛-----⊛-----⊛--
    ///            ∆  ∆  |     |     ∇  
    ///     1---⊛--|--⊛--|--⊛--|--⊛--⊛--
    /// I       ∆  |     |  |  ∇  |     
    /// N   2---⊛--⊛-----|--|--⊛--|--⊛--
    /// P                |  |     ∇  ∇  
    /// U   3------⊛--⊛--|--|-----⊛--⊛--
    /// T          |  ∇  ∇  |           
    /// S   4---⊛--|--⊛--⊛--|--------⊛--
    ///         ∇  ∇        ∇        ∇  
    ///     5---⊛--⊛--------⊛--------⊛--
    /// Note that compares are only between two elements of which
    /// are denotes with '⊛' in the direction of either '∇' or '∆'
    pub fn bitonic_sort_index_calculation(compares: &mut Vec<(i64, i64, Ordering)>, lo: i64, n: i64, dir: Ordering) {
        if n > 1 {
            let m: i64 = n / 2;
            let direction: Ordering = dir;
            let opposite: Ordering = match dir {
                Ordering::Less => Ordering::Greater,
                Ordering::Greater => Ordering::Less,
                Ordering::Equal => panic!("Invalid ordering for bitonic sort")
            };
            bitonic_sort_index_calculation( compares, lo,       m, opposite);
            bitonic_sort_index_calculation( compares, lo + m, n-m, direction);
            bitonic_merge_index_calculation(compares, lo,       n, direction);
        }
    }

    #[inline]
    fn bitonic_merge_index_calculation(compares: &mut Vec<(i64, i64, Ordering)>, lo: i64, n: i64, dir: Ordering) {
        if n > 1 { // Exit condition: n == 1
            let  m: i64 = greatest_power_of_two_less_than(n);
            for i in 0..lo+n-m {
                compares.push( (i, i+m, dir) );
            }
            bitonic_merge_index_calculation(compares, lo, m, dir);
            bitonic_merge_index_calculation(compares, lo+m, n-m, dir);
        }
        /*
        let mut x = n;
        let mut l = lo;
        while x > 1 {
            let  y: i64 = greatest_power_of_two_less_than(x);
            for i in 0..l+x-y {
                compares.push( (i, i+y, dir) );
            }
            x = y;    
        }
        x = n;
        while x > 1 {
            let  y: i64 = greatest_power_of_two_less_than(x);
            for i in 0..l+x-y {
                compares.push( (i, i+y, dir) );
            }
            l = l + y;
            x = x - y;    
        }
        */
    }


    // n>=2  and  n<=Integer.MAX_VALUE
    fn greatest_power_of_two_less_than(n: i64) -> i64 {
        let mut k: i64 = 1;
        while k>0 && k< n  { k <<= 1; }
        k.rotate_right(1)
    }
}

#[cfg(test)]
mod sorting {
    use std::cmp::Ordering;

    use super::pixels::*;
    use super::bitonic_indices::*;
    
    /// Rely only on norm ordering of the vector space.
    #[test]
    fn weak_ordering() {
        let p0: OrdinalPixel = OrdinalPixel::from( vec![255,   0,   0,   0] );
        let p1: OrdinalPixel = OrdinalPixel::from( vec![255, 255,   0,   0] ); 
        let p2: OrdinalPixel = OrdinalPixel::from( vec![255, 255, 255,   0] );
        let p3: OrdinalPixel = OrdinalPixel::from( vec![255, 255, 255, 255] );
        let ev = vec![&p0, &p1, &p2, &p3];
        
        let mut test_vectors = vec![
            vec![&p0, &p1, &p2, &p3],
            vec![&p1, &p2, &p3, &p0],
            vec![&p2, &p3, &p0, &p1],
            vec![&p3, &p0, &p1, &p2],
            vec![&p3, &p2, &p1, &p0],
            vec![&p2, &p1, &p0, &p3],
            vec![&p1, &p0, &p3, &p2],
            vec![&p0, &p3, &p2, &p1],
            vec![&p0, &p2, &p1, &p3],
            vec![&p2, &p1, &p3, &p0],
            vec![&p1, &p3, &p0, &p2],
            vec![&p3, &p0, &p2, &p1],
            vec![&p1, &p0, &p2, &p3],
            vec![&p0, &p2, &p3, &p1],
            vec![&p2, &p3, &p1, &p0],
            vec![&p3, &p1, &p0, &p2],
        ];
        for tv in test_vectors.iter_mut() {
            tv.sort();
            assert_eq!(*tv, ev);
        }
    }

    /// Rely only on lexigraphical ordering of the vector space.
    #[test]
    fn strong_ordering() {
        let p0: OrdinalPixel = OrdinalPixel::from( vec![255,   0,   0,   0] );
        let p1: OrdinalPixel = OrdinalPixel::from( vec![0  , 255,   0,   0] ); 
        let p2: OrdinalPixel = OrdinalPixel::from( vec![0  ,   0, 255,   0] );
        let p3: OrdinalPixel = OrdinalPixel::from( vec![0  ,   0,   0, 255] );
        let ev = vec![&p0, &p1, &p2, &p3];
        
        let mut test_vectors = vec![
            vec![&p0, &p1, &p2, &p3],
            vec![&p1, &p2, &p3, &p0],
            vec![&p2, &p3, &p0, &p1],
            vec![&p3, &p0, &p1, &p2],
            vec![&p3, &p2, &p1, &p0],
            vec![&p2, &p1, &p0, &p3],
            vec![&p1, &p0, &p3, &p2],
            vec![&p0, &p3, &p2, &p1],
            vec![&p0, &p2, &p1, &p3],
            vec![&p2, &p1, &p3, &p0],
            vec![&p1, &p3, &p0, &p2],
            vec![&p3, &p0, &p2, &p1],
            vec![&p1, &p0, &p2, &p3],
            vec![&p0, &p2, &p3, &p1],
            vec![&p2, &p3, &p1, &p0],
            vec![&p3, &p1, &p0, &p2],
        ];
        for tv in test_vectors.iter_mut() {
            tv.sort();
            assert_eq!(*tv, ev);
        }
    }

    #[test]
    fn sort_using_indices() {
        let p0: OrdinalPixel = OrdinalPixel::from( vec![255,   0,   0,   0] );
        let p1: OrdinalPixel = OrdinalPixel::from( vec![0  , 255,   0,   0] ); 
        let p2: OrdinalPixel = OrdinalPixel::from( vec![0  ,   0, 255,   0] );
        let p3: OrdinalPixel = OrdinalPixel::from( vec![0  ,   0,   0, 255] );
        let expected: Vec<&OrdinalPixel> = vec![&p3, &p2, &p1, &p0];
        let mut test_vectors = vec![
            vec![&p0, &p1, &p2, &p3],
            vec![&p1, &p2, &p3, &p0],
            vec![&p2, &p3, &p0, &p1],
            vec![&p3, &p0, &p1, &p2],
            vec![&p3, &p2, &p1, &p0],
            vec![&p2, &p1, &p0, &p3],
            vec![&p1, &p0, &p3, &p2],
            vec![&p0, &p3, &p2, &p1],
            vec![&p0, &p2, &p1, &p3],
            vec![&p2, &p1, &p3, &p0],
            vec![&p1, &p3, &p0, &p2],
            vec![&p3, &p0, &p2, &p1],
            vec![&p1, &p0, &p2, &p3],
            vec![&p0, &p2, &p3, &p1],
            vec![&p2, &p3, &p1, &p0],
            vec![&p3, &p1, &p0, &p2],
        ];
        let compares: Vec<(i64,i64,Ordering)> = calculate_indicies(4);
        for tv in test_vectors.iter_mut() {
            for compare in compares.iter() {
                let i = compare.0 as usize;
                let j = compare.1 as usize;
                let direction = compare.2;
                if tv[i].cmp(&tv[j]) != direction {
                    tv.swap(i, j);
                }
            }
            assert_eq!(tv, &expected);
        }
    }

    #[test]
    fn random_sort_integers() {
        use rand::{distributions::Standard, Rng};
        for list_size in 10..=100 {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            let mut expected: Vec<u64> = test_vector.clone();

            let compares: Vec<(i64,i64,Ordering)> = calculate_indicies(list_size as i64);
            for compare in compares.iter() {
                let i = compare.0 as usize;
                let j = compare.1 as usize;
                let direction = compare.2;
                if test_vector[i].cmp(&test_vector[j]) != direction {
                    test_vector.swap(i, j);
                }
            }
            expected.sort();
            expected.reverse();
            for (i,(x,y)) in expected.iter().zip(test_vector.iter()).enumerate(){
                if x!= y {
                    println!("[{} of {}] {} != {}",i, list_size, x, y);
                }
            }
            assert_eq!(test_vector, expected);
        }
    }

    /*
    #[test]
    fn random_sort_bitonic() {
        use rand::{distributions::Standard, Rng};
        for list_size in 10..=100 {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            let mut expected: Vec<u64> = test_vector.clone();
            bitonic_sort(&mut test_vector, 0, test_vector.len(), Ordering::Greater);
            expected.sort();
            expected.reverse();
            for (i,(x,y)) in expected.iter().zip(test_vector.iter()).enumerate(){
                if x!= y {
                    println!("[{} of {}] {} != {}",i, list_size, x, y);
                }
            }
            assert_eq!(test_vector, expected);
        }
    }
    */
}

#[cfg(test)]
mod io {
    use super::pixels::*;
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
mod ordinality {
    use super::pixels::*;
    const MAX_SCALAR: u8 = 5;

    fn lt(p1: OrdinalPixel, p2: OrdinalPixel) {
        assert!(p1 < p2, "{:?} < {:?} is {}; returned {:?}", p1, p2, p1 < p2, p1.cmp(&p2));
    }
    fn gt(p1: OrdinalPixel, p2: OrdinalPixel) {
        assert!(p1 > p2, "{:?} > {:?} is {}; returned {:?}", p1, p2, p1 > p2, p1.cmp(&p2));
    }
    fn eq(p1: OrdinalPixel, p2: OrdinalPixel) {
        assert!(p1 == p2, "{:?} == {:?} is {}; returned {:?}", p1, p2, p1 == p2, p1.cmp(&p2));
    }

    #[test]
    fn euclidean_norm() {
        let test_vectors: Vec<Vec<u8>>= vec![vec![255,255,255], vec![197,17,23] ];
        let expected_values: Vec<f32> = vec![441.67294, 199.06532];
        for test_set in test_vectors.iter().zip(expected_values.iter()){
            let (test_vector, expected_value) = test_set;
            let npixel  = OrdinalPixel::from(test_vector.clone());
            assert_eq!(npixel.norm, *expected_value);

        }
    }

    #[test]
    fn strong() {
        for i in 0..MAX_SCALAR {
            for j in 0..MAX_SCALAR {
                for k in 0..MAX_SCALAR {
                    let p1: OrdinalPixel = OrdinalPixel::from( vec![i, j, k ]); 
                    let p2: OrdinalPixel = OrdinalPixel::from( vec![j, k, i ] );
                    // Test for contradictions as well.
                    if p1 == p2 {
                        assert!(p1.norm == p2.norm, "{} == {} is false", p1.norm, p2.norm);
                        for (a,b) in p1.pixel.iter().zip(p2.pixel.iter()) {
                            assert!(a == b, "{} == {} is false", a, b);
                        }
                    } else 
                    if p1 < p2 {
                        if p1.norm == p2.norm {
                            let v1: Vec<u8> = p1.pixel.clone();
                            let v2: Vec<u8> = p2.pixel.clone();
                            if v1[2] <  v2[2] {         lt(p1, p2); } else
                            if v1[2] >  v2[2] {         gt(p1, p2); } else // Shouldn't happen
                            if v1[2] == v2[2] {
                                if v1[1] <  v2[1] {     lt(p1, p2); } else
                                if v1[1] >  v2[1] {     gt(p1, p2); } else // Shouldn't happen
                                if v1[1] == v2[1] {
                                    if v1[0] <  v2[0] { lt(p1, p2); } else
                                    if v1[0] >  v2[0] { gt(p1, p2); } else // Shouldn't happen 
                                    if v1[0] == v2[0] { eq(p1, p2); }
                                }
                            }
                        } else 
                        if p1.norm <  p2.norm { lt(p1, p2);  } 
                        else
                        { assert!(false, "{} & {} do not obey the triangle equality!", p1.norm, p2.norm)} 
                    } else
                    if p1 > p2 {
                        if p1.norm == p2.norm {
                            let v1: Vec<u8> = p1.pixel.clone();
                            let v2: Vec<u8> = p2.pixel.clone();
                            if v1[2] <  v2[2] {         lt(p1, p2); } else // Shouldn't happen
                            if v1[2] >  v2[2] {         gt(p1, p2); } else 
                            if v1[2] == v2[2] {
                                if v1[1] <  v2[1] {     lt(p1, p2); } else // Shouldn't happen
                                if v1[1] >  v2[1] {     gt(p1, p2); } else 
                                if v1[1] == v2[1] {
                                    if v1[0] <  v2[0] { lt(p1, p2); } else // Shouldn't happen
                                    if v1[0] >  v2[0] { gt(p1, p2); } else 
                                    if v1[0] == v2[0] { eq(p1, p2); }
                                }
                            }
                        } else 
                        if p1.norm <  p2.norm { lt(p1, p2);  } 
                        else
                        { assert!(false, "{} & {} do not obey the triangle equality!", p1.norm, p2.norm)} 
                    } else {
                        assert!(false, "{:?} & {:?} do not obey the triangle equality!", p1, p2);
                    }
                }
            }
        }
    }

    #[test]
    fn weak() {
        for i in 0..MAX_SCALAR {
            for j in 0..MAX_SCALAR {
                for k in 0..MAX_SCALAR {

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
    
}



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

pub mod bitonic {
    pub mod network {
        use std::cmp::Ordering;

        #[derive(Debug, Clone)]
        pub struct Node {
            i: usize,
            j: usize,
            direction: Ordering
        } impl Node {
            #[inline(always)]
            pub fn new(i: usize, j: usize, direction: Ordering) -> Node {
                Node {i, j, direction}
            }

            #[inline(always)]
            pub fn uninit() -> Node {
                Node::new(0, 0, Ordering::Equal)
            }

            #[inline(always)]
            pub fn details(&self) -> (usize, usize, Ordering) {
                (self.i, self.j, self.direction)
            }
        }

        #[derive(Debug, Clone)]
        pub struct Group {
            /// Indices of comparitors composing this Group
            pub nodes: (usize, usize),
        } impl Group {
            #[inline(always)]
            pub fn new(start: usize, stop: usize) -> Group {
                Group {
                    nodes: (start, stop)
                }
            }
            #[inline(always)]
            pub fn uninit() -> Group {
                Group {
                    nodes: (0,0)
                }
            }
        }

        #[derive(Debug, Clone)]
        pub struct Phase {
            /// Collection of Groups composing this Phase
            pub groups: Vec<(usize, usize)>
            //direction: Ordering,
        } impl Phase {
            pub fn new() -> Phase {
                Phase {
                    groups: Vec::new()
                }
            }
        }

        #[derive(Debug, Clone)]
        pub struct Network {
            /// Vector of tuples, (i,j), describing where in Box<[Node]> the Phases are bounded
            pub phases: Vec<Vec<(usize, usize)>>,

            /// Vector of tuples, (i,j), describing where in Box<[Node]> the Groups are bounded
            pub groups: Vec<(usize, usize)>,

            /// Comparitor operations which compose the bitonic sorting network
            pub nodes: Box<[Node]>,

            /// Silly value to help me mentally track current working comparitor
            pub comparitor: usize,

            pub comparitor_count: usize,

            pub wires_virtual: usize,

            pub wires_real: usize,

            pub comparitors_set: bool

        } impl Network {
            pub fn new(n: usize) -> Network {
                let wires_virtual: usize = n.next_power_of_two();// least_power_of_two_greater_than(n as i64).unwrap() as usize;
                let wires_real = n - 1;
                let number_of_comparitors: usize = wires_virtual * ((wires_virtual as i64).ilog2().pow(2) as usize);
                // Allocate memory for nodes upfront.
                let nodes: Box<[Node]> = vec![Node::uninit(); number_of_comparitors].into_boxed_slice();
                let groups: Vec<(usize, usize)> = Vec::new();
                let phases: Vec<Vec<(usize, usize)>> = Vec::new();
                Network {
                    phases,
                    groups,
                    nodes,
                    comparitor: 0,
                    comparitor_count: number_of_comparitors,
                    wires_virtual,
                    wires_real,
                    comparitors_set: false
                }
            }
        }
    }

    pub mod recursive {
        use std::cmp::Ordering;
        use super::helper::greatest_power_of_two_less_than;
        use super::network;

        pub fn sort<T: PartialOrd>(list: &mut Vec<T>, lo: i64, n: i64, dir: Ordering) {
            if n > 1 {
                let m: i64 = n / 2;
                let direction: Ordering = dir;
                let opposite: Ordering = match dir {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                    Ordering::Equal => panic!("Invalid ordering for bitonic sort")
                };
                sort( list, lo,       m, opposite);
                sort( list, lo + m, n-m, direction);
                merge(list, lo,       n, direction);
            }
        }
        
        #[inline(always)]
        fn merge<T: PartialOrd>(list: &mut Vec<T>, lo: i64, n: i64, dir: Ordering) {
            if n > 1 { // Exit condition: n == 1
                let  m: i64 = greatest_power_of_two_less_than(n).unwrap();
                for i in 0..lo+n-m {
                    if list[i as usize].partial_cmp(&list[(i+m) as usize]).unwrap() != dir {
                        list.swap(i as usize, (i+m) as usize);
                    }
                }
                merge(list, lo, m, dir);
                merge(list, lo+m, n-m, dir);
            }
        }

        /// Use the implementation found here: https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
        /// as a basis for an arbitrary n bitonic sort. Note that we want to find the indicies first so that 
        /// we can avoid using recursion and use a sorting network (fixed comaparisons).
        /// 
        /// Specifically, standard bitonic sort uses a comparator network B_{p} where p is a power of 2 as the basic 
        /// building block. We derive network Bn for arbitrary n from network B_{p} where p is the next-greatest 
        /// power of 2 by using only the first n – p/2 comparators of B_{p}. 
        pub fn network(len: i64) -> Vec<network::Node> {
            let mut compares: Vec<network::Node> = Vec::new();
            nodes(&mut compares, 0, len, Ordering::Greater);
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
        fn nodes(compares: &mut Vec<network::Node>, lo: i64, n: i64, dir: Ordering) {
            if n > 1 {
                let m: i64 = n / 2;
                let direction: Ordering = dir;
                let opposite: Ordering = match dir {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                    Ordering::Equal => panic!("Invalid ordering for bitonic sort")
                };
                nodes( compares, lo,       m, opposite);
                nodes( compares, lo + m, n-m, direction);
                nodes_merge(compares, lo,       n, direction);
            }
        }

        #[inline(always)]
        fn nodes_merge(compares: &mut Vec<network::Node>, lo: i64, n: i64, dir: Ordering) {
            if n > 1 { // Exit condition: n == 1
                let  m: i64 = greatest_power_of_two_less_than(n).unwrap();
                for i in 0..lo+n-m {
                    compares.push( network::Node::new(i as usize, (i+m) as usize, dir));
                }
                nodes_merge(compares, lo, m, dir);
                nodes_merge(compares, lo+m, n-m, dir);
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
    }

    pub mod iterative {
        use std::cmp::Ordering;

        use super::{
            network::{
                Node,
                Network
            }
        };

        pub fn sort<T: PartialOrd>(list: &mut Vec<T>) -> Result<&mut Vec<T>, &str> {
            let mut sorting_network: Network = Network::new(list.len());
            sorting_network.set_comparitors().clone().sort(list)
        }

        impl Network {

            pub fn sort<T: PartialOrd>(mut self, list: &mut Vec<T>) -> Result<&mut Vec<T>, &str>{
                if !self.comparitors_set { return Err("Network comparitors not set!"); }

                self.wires_real = list.len() - 1;


                for phase in self.phases.iter() {
                    for group in phase.iter() {
                        let (start, end) = *group;
                        for node in &self.nodes[start..end] {
                            let (i, j, direction): (usize, usize, Ordering) = node.details();
                            if list[i].partial_cmp(&list[j]) != Some(direction) {
                                list.swap(i, j);
                            }
                        }
                    }
                }
                Ok(list)
            }
            
            #[inline(always)]
            fn add_phase(&mut self) -> &mut Self{
                let mut groups: Vec<(usize, usize)> = Vec::with_capacity(self.groups.len());
                groups.extend(self.groups.drain(..));
                self.phases.push(groups);
                self
            }

            #[inline(always)]
            fn add_group(&mut self, i: usize, j: usize) -> &mut Self {
                self.groups.push((i, j));
                self
            }

            #[inline(always)]
            fn clear_groups(&mut self) -> &mut Self {
                self.groups = Vec::new();
                self
            }
            
            #[inline(always)]
            fn add_node(&mut self, node:Node) -> &mut Self {
                if self.comparitor == self.comparitor_count {
                    let nodes: Box<[Node]> = vec![Node::uninit(); self.comparitor_count].into_boxed_slice();
                    self.nodes = vec![self.nodes.clone(), nodes].concat().into_boxed_slice();
                    self.comparitor_count *= 2;
                }
                self.nodes[self.comparitor] = node;
                self.comparitor += 1;
                self
            }

            #[inline(always)]
            pub fn set_input_length(&mut self, len: usize) -> &mut Self {
                self.wires_real = len-1;
                self
            }

            #[inline(always)]
            pub fn set_comparitors(&mut self) -> &mut Self {
                // Credit to: 
                //      John Mellor-Crummy & Thomas Anastasio
                //      https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_overview.html
                self.comparitors_set = true;
                let n: i64 = self.wires_virtual as i64;
                let mut s: i64 = 2;
                while s <= n {
                    for i in 0..n {

                        self.merge( i, s,  Ordering::Greater)
                            .merge( i + s, s, Ordering::Less)
                            .add_phase()
                            .clear_groups();


                    }
                    s *= 2;
                }
                self
            }
            #[inline(always)]
            fn merge(&mut self, lo: i64, n: i64, direction: Ordering) -> &mut Self {
                /*
                void merge_up(int *arr, int n) {
                    int step=n/2,s,i,j,temp;
                    while (step > 0) {
                        for (s=0; s < n; s+=step*2) {
                            for (i=s,j=0;j < step;i++,j++) {
                                if (arr[i] > arr[i+step]) { // list.push(i, j+step, '>');
                                    // swap
                                    temp = arr[i];
                                    arr[i]=arr[i+step];
                                    arr[i+step]=temp;
                                }
                            }
                        }
                        step /= 2;
                    }
                }
                */
                let start: usize = self.comparitor;

                let base: usize = lo as usize;
                let mut step: i64 = (n as i64) /2;
                let mut i: usize;
                while step > 0 {
                    for s in (0..n).step_by((step as usize)*2) {
                        i = s as usize;
                        for j in 0..(step as usize) {
                            let a = base + i;
                            let b = base + j + step as usize;

                            if b > self.wires_real || a > self.wires_real { break; }

                            self.add_node(Node::new(a, b, direction));
                            i += 1;
                        }
                    }
                    step /= 2;
                }
                let end: usize = self.comparitor;
                self.add_group(start, end)
            }
        }
    }

    pub mod helper{
        pub fn greatest_power_of_two_less_than(n: i64) -> Result<i64, &'static str> {
            let mut k: i64 = 1;
            while k>0 && k< n  { k <<= 1; }
            Ok(k.rotate_right(1))
        }

        pub fn least_power_of_two_greater_than(n: i64) -> Result<i64, &'static str> {
            if n < 1  || i64::MAX < n { return Err("input value out of range"); }

            let e = n.ilog2(); // Log base 2 of n, rounded down.
            let k = 2i64.pow(e);
            let k = if n <= k { k }  else { 2i64.pow(e + 1)};

            if k < 1  || i64::MAX < k.into() { return Err("resulting value out of range"); }
            else {                      return Ok(k.into()); }

            
        }


    }
}

#[cfg(test)]
mod multithreading {
    use rayon::*;
    use itertools::Itertools;
    use std::cmp::Ordering;
    use std::time::Duration;
    use std::time::Instant;
    use super::bitonic;

    #[test]
    fn simple_mt_pool() {
        let cores = num_cpus::get_physical();
        let pool = ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
        let mut count: usize = 0;
        let mut tids: Vec<usize> = Vec::new();
        for _ in 0..cores {
            tids.push(pool.install(|| { self::current_thread_index().unwrap() } ));
            count += count;
        }

        // Test that we dont run only on thread 0
        let reduced = tids.iter().fold(0, |acc, e| acc + e);
        assert!( reduced != 0, "Cores = {}, Count = {}", cores, count);
        
        // Test that we get more than one thread id.
        let uniq: Vec<&usize> = tids.iter().unique().collect();
        assert!(uniq.len() != 1, "Only {} unique TID: {}", uniq.len(), uniq[0]);
    }

/*
    #[test]
    fn simple_node_sort_random() {
        use rand::{distributions::Standard, Rng};

        let cores = num_cpus::get();
        let pool = ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
    
        for list_size in (100..=500).step_by(100) {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            let mut expected: Vec<u64> = test_vector.clone();

            let start: Instant = Instant::now();
            let mut sorting_network = bitonic::network::Network::new(list_size);
            let sorting_network = sorting_network.set_input_length(list_size).set_comparitors();
            let duration: Duration = start.elapsed();
            println!("Time spent generating sorting network: {:?}", duration);

            let nodes = sorting_network.nodes.clone();

            let start: Instant = Instant::now();
            for group in sorting_network.phases.iter() {
                for (start, stop) in group.iter() {
                    for node in *start..*stop {
                        pool.install(|| {
                                    let (i, j, direction): (usize, usize, Ordering) = nodes[node].details();
                                    if test_vector[i].partial_cmp(&test_vector[j]) != Some(direction) {
                                        test_vector.swap(i, j);
                                    }
                        });
                    }
                }

            }
            let bitonic: Duration = start.elapsed();

            let start: Instant = Instant::now();
            expected.sort();
            let rust_sort: Duration = start.elapsed();

            println!("Sort time:\n\tBitonic: {:?}, .sort: {:?}", bitonic, rust_sort);
            expected.reverse();

            for (i,(x,y)) in test_vector.iter().zip(expected.iter()).enumerate(){
                if x!= y {
                    println!("[{} of {}] {} != {}",i, list_size, x, y);
                }
            }
            assert_eq!(test_vector, expected);
        }
    }
*/

    #[test]
    fn simple_group_sort_random() {
        use rand::{distributions::Standard, Rng};

        let cores = num_cpus::get();
        let pool = ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
    
        for list_size in (100..=500).step_by(100) {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            let mut expected: Vec<u64> = test_vector.clone();

            let start: Instant = Instant::now();
            let mut sorting_network = bitonic::network::Network::new(list_size);
            let sorting_network = sorting_network.set_input_length(list_size).set_comparitors();
            let duration: Duration = start.elapsed();
            println!("Time spent generating sorting network: {:?}", duration);

            let nodes = sorting_network.nodes.clone();

            let start: Instant = Instant::now();
            for group in sorting_network.phases.iter() {
                for (start, stop) in group.iter() {

                    pool.install(|| {
                            for node in *start..*stop {
                                let (i, j, direction): (usize, usize, Ordering) = nodes[node].details();
                                if test_vector[i].partial_cmp(&test_vector[j]) != Some(direction) {
                                    test_vector.swap(i, j);
                                }
                            }
                    });
                }

            }
            let bitonic: Duration = start.elapsed();

            let start: Instant = Instant::now();
            expected.sort();
            let rust_sort: Duration = start.elapsed();

            println!("Sort time:\n\tBitonic: {:?}, .sort: {:?}", bitonic, rust_sort);
            expected.reverse();

            for (i,(x,y)) in test_vector.iter().zip(expected.iter()).enumerate(){
                if x!= y {
                    println!("[{} of {}] {} != {}",i, list_size, x, y);
                }
            }
            assert_eq!(test_vector, expected);
        }
    }

    #[test]
    fn simple_phase_sort_random() {
        use rand::{distributions::Standard, Rng};

        let cores = num_cpus::get();
        let pool = ThreadPoolBuilder::new().num_threads(cores).build().unwrap();
    
        for list_size in (100..=500).step_by(100) {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            let mut expected: Vec<u64> = test_vector.clone();

            let start: Instant = Instant::now();
            let mut sorting_network = bitonic::network::Network::new(list_size);
            let sorting_network = sorting_network.set_input_length(list_size).set_comparitors();
            let duration: Duration = start.elapsed();
            println!("Time spent generating sorting network: {:?}", duration);

            let nodes = sorting_network.nodes.clone();

            let start: Instant = Instant::now();
            for group in sorting_network.phases.iter() {

                pool.install(|| {
                    for (start, stop) in group.iter() {
                        for node in *start..*stop {
                            let (i, j, direction): (usize, usize, Ordering) = nodes[node].details();
                            if test_vector[i].partial_cmp(&test_vector[j]) != Some(direction) {
                                test_vector.swap(i, j);
                            }
                        }
                    }
                });

            }
            let bitonic: Duration = start.elapsed();

            let start: Instant = Instant::now();
            expected.sort();
            let rust_sort: Duration = start.elapsed();

            println!("Sort time:\n\tBitonic: {:?}, .sort: {:?}", bitonic, rust_sort);
            expected.reverse();

            for (i,(x,y)) in test_vector.iter().zip(expected.iter()).enumerate(){
                if x!= y {
                    println!("[{} of {}] {} != {}",i, list_size, x, y);
                }
            }
            assert_eq!(test_vector, expected);
        }
    }
}


#[cfg(test)]
mod helper {
    use crate::bitonic::helper::least_power_of_two_greater_than;


    #[test]
    fn power_of_two_greater() {
        for (a,b) in [1, 2, 3, 5, 6,  8,  9, 12, 14, 15, 16, 17, 24, 31, 32, 33].iter()
                           .zip([1, 2, 4, 8, 8,  8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 64 ]) {
            assert_eq!(least_power_of_two_greater_than(*a).unwrap(), b);

        }
    }
    /*FIXME: I shouldn't be using signed integers.
    #[test]
    fn power_of_two_lesser() {
        for (a,b) in [1, 2, 3, 5, 6,  8,  9, 12, 14, 15, 16, 17, 24, 31, 32, 33].iter()
                           .zip([1, 2, 2, 4, 4,  8,  8,  8,  8,  8, 16, 16, 16, 16, 32, 32]) {
            let a = greatest_power_of_two_less_than(*a).unwrap();
            if a != b { println!("{} {}", a, b); }
            assert_eq!(a as u64, b as u64);

        }
    }
    */
}

#[cfg(test)]
mod sorting {
    use std::cmp::Ordering;

    use super::pixels::*;
    use super::bitonic;
    
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
    fn bitonic_recursive_indices_fixed() {
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
        use bitonic::network::Node;
        let network: Vec<Node> = bitonic::recursive::network(4);
        for vector in test_vectors.iter_mut() {
            for node in network.iter() {
                let (i, j, direction): (usize, usize, Ordering) = node.details();
                if vector[i].cmp(&vector[j]) != direction {
                    vector.swap(i, j);
                }
            }
            assert_eq!(vector, &expected);
        }
    }

    #[test]
    fn bitonic_recursive_random() {
        use rand::{distributions::Standard, Rng};
        for list_size in 50..=100 {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            let mut expected: Vec<u64> = test_vector.clone();
            bitonic::recursive::sort(&mut test_vector, 0 as i64, list_size as i64, Ordering::Greater);
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

    #[test]
    fn bitonic_iterative_indices_fixed() {
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
        for vector in test_vectors.iter_mut() {
            let sorted = bitonic::iterative::sort(vector);
            assert_eq!(&expected, sorted.unwrap());
        }
    }
    
    #[test]
    fn bitonic_iterative_random() {
        use std::time::{Duration, Instant};

        use rand::{distributions::Standard, Rng};
        

        for list_size in (100..=500).step_by(100) {
            let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
            //let tv_clone = test_vector.clone();
            let mut expected: Vec<u64> = test_vector.clone();

            let start: Instant = Instant::now();
            let mut sorting_network = bitonic::network::Network::new(500);
            let sorting_network = sorting_network.set_input_length(list_size).set_comparitors();
            let duration: Duration = start.elapsed();
            println!("Time spent generating sorting network: {:?}", duration);

            let network = sorting_network.clone();

            let start: Instant = Instant::now();
            network.sort(&mut test_vector).unwrap();
            let bitonic: Duration = start.elapsed();

            let start: Instant = Instant::now();
            expected.sort();
            let rust_sort: Duration = start.elapsed();

            println!("Sort time:\n\tBitonic: {:?}, .sort: {:?}", bitonic, rust_sort);

            expected.reverse();
            for (i,(x,y)) in test_vector.iter().zip(expected.iter()).enumerate(){
                if x!= y {
                    println!("[{} of {}] {} != {}",i, list_size, x, y);
                }
            }
            assert_eq!(test_vector, expected);
        }
    }

    #[test]
    fn bitonic_iterative_single() {
        use std::time::{Duration, Instant};

        use rand::{distributions::Standard, Rng};
        
        let start: Instant = Instant::now();
        let mut binding = bitonic::network::Network::new(500);
        let sorting_network = binding.set_comparitors();
        let duration: Duration = start.elapsed();
        println!("Time spent generating sorting network: {:?}", duration);

        let list_size = 500;
        let mut test_vector: Vec<u64> = rand::thread_rng().sample_iter(Standard).take(list_size).collect();
        //let tv_clone = test_vector.clone();
        let mut expected: Vec<u64> = test_vector.clone();
        let network = sorting_network.clone();

        let start: Instant = Instant::now();
        network.sort(&mut test_vector).unwrap();
        let bitonic: Duration = start.elapsed();

        let start: Instant = Instant::now();
        expected.sort();
        let rust_sort: Duration = start.elapsed();

        println!("Sort time:\n\tBitonic: {:?}, .sort: {:?}", bitonic, rust_sort);

        expected.reverse();
        for (i,(x,y)) in test_vector.iter().zip(expected.iter()).enumerate(){
            if x!= y {
                println!("[{} of {}] {} != {}",i, list_size, x, y);
            }
        assert_eq!(test_vector, expected);
        }
    }
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
use std::io::BufReader;

use bytemuck::{Pod, Zeroable};
use tobj::{MTLLoadResult, Model};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
    _normal: [f32; 3],
}

pub(crate) trait Mesh {
    fn new() -> Self;
    fn data() -> (Vec<[f32; 8]>, Vec<u16>);
    fn vtx(&self) -> (Vec<Vertex>, Vec<u16>) {
        let data = Self::data();
        (
            data.0
                .iter()
                .map(|v| Vertex {
                    _pos: [v[0], v[1], v[2], 1.0],
                    _tex_coord: [v[3], v[4]],
                    _normal: [0.0; 3],
                })
                .collect::<Vec<Vertex>>(),
            data.1.iter().map(|f| *f as u16).collect(),
        )
    }
}

fn from_tobj(o: &Model) -> (Vec<[f32; 8]>, Vec<u16>) {
    (
        o.mesh
            .positions
            .chunks_exact(3)
            .enumerate()
            .zip(o.mesh.texcoords.chunks_exact(2))
            .zip(o.mesh.normals.chunks_exact(3))
            .map(|f| {
                [
                    // pos
                    f.0 .0 .1[0],
                    f.0 .0 .1[1],
                    f.0 .0 .1[2], // tex
                    f.0 .1[1],
                    f.0 .1[1], //  norm
                    f.1[0],
                    f.1[1],
                    f.1[2],
                ]
            })
            .collect(),
        o.mesh
            .indices
            .iter()
            .map(|f| (*f) as u16)
            .collect::<Vec<u16>>()
            .clone(),
    )
}

macro_rules! use_obj {
    // `()` indicates that the macro takes no argument.
    ($file:expr) => {
        fn new() -> Self {
            Self
        }
        fn data() -> (Vec<[f32; 8]>, Vec<u16>) {
            let o = tobj::load_obj_buf(
                &mut BufReader::new(&include_bytes!($file)[..]),
                &tobj::LoadOptions {
                    single_index: true,
                    ..tobj::GPU_LOAD_OPTIONS
                },
                |_| MTLLoadResult::Ok((vec![], ahash::AHashMap::new())),
            )
            .unwrap()
            .0;
            from_tobj(&o[0])
        }
    };
}

pub(crate) struct Sphere;
impl Mesh for Sphere {
    use_obj!("assets/usphere.obj");
}

pub(crate) struct Icosphere;
impl Mesh for Icosphere {
    use_obj!("assets/ico.obj");
}

pub(crate) struct Suzane;
impl Mesh for Suzane {
    use_obj!("assets/suz.obj");
}

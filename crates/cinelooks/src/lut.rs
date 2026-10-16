//! `.cube` LUTs (the Resolve / Adobe text format): an optional 1D table (`LUT_1D_SIZE`,
//! a shaper when both are present) followed by a 3D lattice (`LUT_3D_SIZE`, red fastest), with
//! `DOMAIN_MIN` / `DOMAIN_MAX` (or `LUT_*_INPUT_RANGE`) input domains. Inputs outside the
//! domain clamp to it (the table has no data there). Malformed files are errors that say
//! what is wrong and where; they never load half a table.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Largest 3D lattice side accepted (129³ ≈ 2.1 M entries).
pub const MAX_3D: usize = 129;
/// Largest 1D table accepted.
pub const MAX_1D: usize = 65_536;
/// Largest `.cube` file read.
pub const MAX_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    pub title: String,
    /// 1D table: `n1` RGB rows (0 = none).
    pub n1: usize,
    pub d1: Vec<[f32; 3]>,
    pub min1: [f32; 3],
    pub max1: [f32; 3],
    /// 3D lattice side (0 = none) and `n3³` RGB entries, red fastest.
    pub n3: usize,
    pub d3: Vec<[f32; 3]>,
    pub min3: [f32; 3],
    pub max3: [f32; 3],
}

fn parse3(it: &mut std::str::SplitWhitespace, line: usize) -> Result<[f32; 3], String> {
    let mut v = [0.0f32; 3];
    for x in &mut v {
        let s = it.next().ok_or_else(|| format!("line {line}: expected three numbers"))?;
        *x = s.parse().map_err(|_| format!("line {line}: `{s}` is not a number"))?;
        if !x.is_finite() {
            return Err(format!("line {line}: `{s}` is not a finite number"));
        }
    }
    Ok(v)
}

/// Parse `.cube` text.
pub fn parse_cube(text: &str) -> Result<Lut, String> {
    let mut lut = Lut { title: String::new(), n1: 0, d1: vec![], min1: [0.0; 3], max1: [1.0; 3], n3: 0, d3: vec![], min3: [0.0; 3], max3: [1.0; 3] };
    let mut rows: Vec<[f32; 3]> = Vec::new();
    // Generic DOMAIN_* applies to both tables; the INPUT_RANGE forms to their own.
    let mut domain: Option<([f32; 3], [f32; 3])> = None;
    let (mut range1, mut range3): (Option<[f32; 2]>, Option<[f32; 2]>) = (None, None);
    let (mut dmin, mut dmax) = ([0.0f32; 3], [1.0f32; 3]);
    let mut have_min = false;
    let mut have_max = false;
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let l = raw.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let mut it = l.split_whitespace();
        let Some(key) = it.next() else { continue };
        match key {
            "TITLE" => lut.title = l.trim_start_matches("TITLE").trim().trim_matches('"').chars().take(200).collect(),
            "LUT_1D_SIZE" | "LUT_3D_SIZE" => {
                let s = it.next().ok_or_else(|| format!("line {line}: {key} needs a size"))?;
                let n: usize = s.parse().map_err(|_| format!("line {line}: {key} `{s}` is not a size"))?;
                if key == "LUT_1D_SIZE" {
                    if !(2..=MAX_1D).contains(&n) {
                        return Err(format!("line {line}: LUT_1D_SIZE {n} is outside 2..{MAX_1D}"));
                    }
                    lut.n1 = n;
                } else {
                    if !(2..=MAX_3D).contains(&n) {
                        return Err(format!("line {line}: LUT_3D_SIZE {n} is outside 2..{MAX_3D}"));
                    }
                    lut.n3 = n;
                }
            }
            "DOMAIN_MIN" => {
                dmin = parse3(&mut it, line)?;
                have_min = true;
            }
            "DOMAIN_MAX" => {
                dmax = parse3(&mut it, line)?;
                have_max = true;
            }
            "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => {
                let a: f32 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("line {line}: {key} needs two numbers"))?;
                let b: f32 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("line {line}: {key} needs two numbers"))?;
                if key == "LUT_1D_INPUT_RANGE" {
                    range1 = Some([a, b]);
                } else {
                    range3 = Some([a, b]);
                }
            }
            k if k.starts_with(|c: char| c.is_ascii_alphabetic()) => {
                // Unknown keywords (LUT_IN_VIDEO_RANGE, vendor extensions) are ignored.
            }
            _ => {
                let mut all = l.split_whitespace();
                rows.push(parse3(&mut all, line)?);
                if all.next().is_some() {
                    return Err(format!("line {line}: a data row has more than three numbers"));
                }
                if rows.len() > MAX_1D + MAX_3D * MAX_3D * MAX_3D {
                    return Err("too many data rows".into());
                }
            }
        }
    }
    if have_min || have_max {
        domain = Some((dmin, dmax));
    }
    if lut.n1 == 0 && lut.n3 == 0 {
        return Err("no LUT_1D_SIZE or LUT_3D_SIZE: not a .cube LUT".into());
    }
    let need = lut.n1 + lut.n3 * lut.n3 * lut.n3;
    if rows.len() != need {
        return Err(format!("the header promises {need} data rows but the file has {}", rows.len()));
    }
    let (d1, d3) = rows.split_at(lut.n1);
    lut.d1 = d1.to_vec();
    lut.d3 = d3.to_vec();
    let check = |lo: [f32; 3], hi: [f32; 3]| -> Result<(), String> {
        if (0..3).any(|i| hi[i].partial_cmp(&lo[i]) != Some(std::cmp::Ordering::Greater)) { Err("DOMAIN_MAX must be greater than DOMAIN_MIN in every channel".into()) } else { Ok(()) }
    };
    if let Some((lo, hi)) = domain {
        check(lo, hi)?;
        (lut.min1, lut.max1, lut.min3, lut.max3) = (lo, hi, lo, hi);
    }
    if let Some([a, b]) = range1 {
        check([a; 3], [b; 3])?;
        (lut.min1, lut.max1) = ([a; 3], [b; 3]);
    }
    if let Some([a, b]) = range3 {
        check([a; 3], [b; 3])?;
        (lut.min3, lut.max3) = ([a; 3], [b; 3]);
    }
    Ok(lut)
}

#[inline]
fn norm(c: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| {
        let t = (c[i] - lo[i]) / (hi[i] - lo[i]).max(1e-9);
        if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) }
    })
}

impl Lut {
    /// Apply: 1D table (per channel, linear interpolation), then the lattice (tetrahedral, or
    /// trilinear when `trilinear`).
    pub fn apply(&self, c: [f32; 3], trilinear: bool) -> [f32; 3] {
        let mut c = c;
        if self.n1 >= 2 {
            let t = norm(c, self.min1, self.max1);
            let s = (self.n1 - 1) as f32;
            c = [0, 1, 2].map(|i| {
                let x = t[i] * s;
                let i0 = (x as usize).min(self.n1 - 2);
                let f = x - i0 as f32;
                let a = self.d1.get(i0).map_or(0.0, |r| r[i]);
                let b = self.d1.get(i0 + 1).map_or(a, |r| r[i]);
                a + (b - a) * f
            });
        }
        if self.n3 >= 2 {
            let t = norm(c, self.min3, self.max3);
            c = if trilinear { self.trilinear(t) } else { self.tetrahedral(t) };
        }
        c
    }

    #[inline]
    fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        let n = self.n3;
        self.d3.get(r + g * n + b * n * n).copied().unwrap_or([0.0; 3])
    }

    #[inline]
    fn cell(&self, x: f32) -> (usize, f32) {
        let v = x * (self.n3 - 1) as f32;
        let i = (v as usize).min(self.n3 - 2);
        (i, v - i as f32)
    }

    pub fn trilinear(&self, t: [f32; 3]) -> [f32; 3] {
        let (r, fr) = self.cell(t[0]);
        let (g, fg) = self.cell(t[1]);
        let (b, fb) = self.cell(t[2]);
        let l = |a: [f32; 3], b: [f32; 3], f: f32| [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
        let c00 = l(self.at(r, g, b), self.at(r + 1, g, b), fr);
        let c10 = l(self.at(r, g + 1, b), self.at(r + 1, g + 1, b), fr);
        let c01 = l(self.at(r, g, b + 1), self.at(r + 1, g, b + 1), fr);
        let c11 = l(self.at(r, g + 1, b + 1), self.at(r + 1, g + 1, b + 1), fr);
        l(l(c00, c10, fg), l(c01, c11, fg), fb)
    }

    pub fn tetrahedral(&self, t: [f32; 3]) -> [f32; 3] {
        let (r, fr) = self.cell(t[0]);
        let (g, fg) = self.cell(t[1]);
        let (b, fb) = self.cell(t[2]);
        let c000 = self.at(r, g, b);
        let c111 = self.at(r + 1, g + 1, b + 1);
        let (w1, w2, w3, p1, p2) = if fr > fg {
            if fg > fb {
                (fr, fg, fb, self.at(r + 1, g, b), self.at(r + 1, g + 1, b))
            } else if fr > fb {
                (fr, fb, fg, self.at(r + 1, g, b), self.at(r + 1, g, b + 1))
            } else {
                (fb, fr, fg, self.at(r, g, b + 1), self.at(r + 1, g, b + 1))
            }
        } else if fb > fg {
            (fb, fg, fr, self.at(r, g, b + 1), self.at(r, g + 1, b + 1))
        } else if fb > fr {
            (fg, fb, fr, self.at(r, g + 1, b), self.at(r, g + 1, b + 1))
        } else {
            (fg, fr, fb, self.at(r, g + 1, b), self.at(r + 1, g + 1, b))
        };
        [0, 1, 2].map(|i| c000[i] + w1 * (p1[i] - c000[i]) + w2 * (p2[i] - p1[i]) + w3 * (c111[i] - p2[i]))
    }

    /// `.cube` text of this LUT.
    pub fn to_cube(&self) -> String {
        let mut s = String::new();
        if !self.title.is_empty() {
            s.push_str(&format!("TITLE \"{}\"\n", self.title.replace('"', "'")));
        }
        if self.n1 >= 2 {
            s.push_str(&format!("LUT_1D_SIZE {}\n", self.n1));
        }
        if self.n3 >= 2 {
            s.push_str(&format!("LUT_3D_SIZE {}\n", self.n3));
        }
        let f = |v: [f32; 3]| format!("{:.6} {:.6} {:.6}", v[0], v[1], v[2]);
        if self.min3 != [0.0; 3] || self.max3 != [1.0; 3] {
            s.push_str(&format!("DOMAIN_MIN {}\nDOMAIN_MAX {}\n", f(self.min3), f(self.max3)));
        }
        for r in self.d1.iter().chain(&self.d3) {
            s.push_str(&f(*r));
            s.push('\n');
        }
        s
    }
}

type Cache = Mutex<HashMap<String, (Option<std::time::SystemTime>, Result<Arc<Lut>, String>)>>;

/// Load a LUT: from the file at `path` when it can be read, else from the embedded `.cube`
/// text `data` (projects keep a copy so they survive a moved LUT). Files are cached by path
/// and modification time. The error explains a missing or malformed LUT.
pub fn load(path: &str, data: &str) -> Result<Arc<Lut>, String> {
    let path = path.trim();
    if !path.is_empty() {
        match read_file(path) {
            Ok(l) => return Ok(l),
            Err(e) if data.trim().is_empty() => return Err(e),
            Err(_) => {}
        }
    }
    if data.trim().is_empty() {
        return Err("no LUT chosen".into());
    }
    static EMBED: OnceLock<Mutex<HashMap<u64, Result<Arc<Lut>, String>>>> = OnceLock::new();
    let key = fnv(data.as_bytes());
    let cache = EMBED.get_or_init(Default::default);
    if let Some(v) = cache.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
        return v.clone();
    }
    let v = parse_cube(data).map(Arc::new);
    let mut m = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if m.len() > 64 {
        m.clear();
    }
    m.insert(key, v.clone());
    v
}

fn read_file(path: &str) -> Result<Arc<Lut>, String> {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let name = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path.to_string());
    let meta = std::fs::metadata(path).map_err(|_| format!("LUT file not found: {name}"))?;
    if meta.len() > MAX_BYTES {
        return Err(format!("{name} is larger than {} MB", MAX_BYTES >> 20));
    }
    let mtime = meta.modified().ok();
    let cache = CACHE.get_or_init(Default::default);
    if let Some((t, v)) = cache.lock().unwrap_or_else(PoisonError::into_inner).get(path)
        && *t == mtime
    {
        return v.clone();
    }
    let v = std::fs::read_to_string(path)
        .map_err(|e| format!("can't read {name}: {e}"))
        .and_then(|t| parse_cube(&t))
        .map(Arc::new)
        .map_err(|e| format!("{name}: {e}"));
    let mut m = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if m.len() > 64 {
        m.clear();
    }
    m.insert(path.to_string(), (mtime, v.clone()));
    v
}

/// FNV-1a 64-bit (cache keys).
pub fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in b {
        h ^= *x as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// An identity 3D LUT of side `n` (tests, cube export).
pub fn identity(n: usize) -> Lut {
    let n = n.clamp(2, MAX_3D);
    let s = (n - 1) as f32;
    let mut d3 = Vec::with_capacity(n * n * n);
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                d3.push([r as f32 / s, g as f32 / s, b as f32 / s]);
            }
        }
    }
    Lut { title: String::new(), n1: 0, d1: vec![], min1: [0.0; 3], max1: [1.0; 3], n3: n, d3, min3: [0.0; 3], max3: [1.0; 3] }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(n: usize, f: impl Fn([f32; 3]) -> [f32; 3]) -> String {
        let mut s = format!("TITLE \"t\"\nLUT_3D_SIZE {n}\n");
        let k = (n - 1) as f32;
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let o = f([r as f32 / k, g as f32 / k, b as f32 / k]);
                    s.push_str(&format!("{} {} {}\n", o[0], o[1], o[2]));
                }
            }
        }
        s
    }

    #[test]
    fn identity_and_affine_luts_are_exact() {
        let l = parse_cube(&cube(17, |c| c)).unwrap();
        for c in [[0.1, 0.5, 0.9], [0.33, 0.33, 0.8], [1.0, 0.0, 0.25]] {
            for tri in [false, true] {
                let o = l.apply(c, tri);
                for i in 0..3 {
                    assert!((o[i] - c[i]).abs() < 1e-5, "{c:?} {o:?}");
                }
            }
        }
        // Tetrahedral interpolation reproduces any affine map exactly.
        let l = parse_cube(&cube(5, |c| [0.2 + 0.5 * c[1], c[0] * 0.8, 1.0 - c[2]])).unwrap();
        let c = [0.37, 0.61, 0.13];
        let o = l.apply(c, false);
        assert!((o[0] - (0.2 + 0.5 * c[1])).abs() < 1e-5 && (o[1] - c[0] * 0.8).abs() < 1e-5 && (o[2] - (1.0 - c[2])).abs() < 1e-5);
    }

    #[test]
    fn shaper_domain_and_round_trip() {
        let text = "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n0 0 0\n1 1 1\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let l = parse_cube(text).unwrap();
        // 1.0 sits in the middle of the 0..2 domain → 0.5 after the shaper → 0.5/… of the lattice.
        let o = l.apply([1.0, 1.0, 1.0], false);
        assert!((o[0] - 0.25).abs() < 1e-5, "{o:?}");
        // Out-of-domain input clamps (to 1.0 out of the shaper, the lattice's middle).
        assert_eq!(l.apply([9.0, 9.0, 9.0], false), [0.5, 0.5, 0.5]);
        let back = parse_cube(&l.to_cube()).unwrap();
        assert_eq!(back.n3, 2);
    }

    #[test]
    fn malformed_files_say_why() {
        assert!(parse_cube("").unwrap_err().contains("not a .cube"));
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 0\n").unwrap_err().contains("promises 8"));
        assert!(parse_cube("LUT_3D_SIZE 99999\n").unwrap_err().contains("outside"));
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 x\n").unwrap_err().contains("line 2"));
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 nan\n").is_err());
        assert!(parse_cube("LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n").is_err());
        assert!(load("/definitely/not/here.cube", "").unwrap_err().contains("not found"));
        // A missing file falls back to the embedded copy.
        assert!(load("/definitely/not/here.cube", &cube(2, |c| c)).is_ok());
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 0 0\n").is_err());
    }

    #[test]
    fn nan_input_is_safe() {
        let l = parse_cube(&cube(3, |c| c)).unwrap();
        let o = l.apply([f32::NAN, 0.5, f32::INFINITY], false);
        assert!(o.iter().all(|v| v.is_finite()));
    }
}

//! Tone curves: control points in 0..1 joined by a monotone cubic (Fritsch–Carlson), so the
//! curve never overshoots between points. Curves are baked into [`Table`]s that the CPU and
//! the GPU kernel sample the same way: linear interpolation inside 0..1 and the end slopes
//! outside it (so over-range values keep their shape instead of clipping).

/// Samples per baked table.
pub const TABLE_SIZE: usize = 1024;
/// Most control points a curve keeps.
pub const MAX_POINTS: usize = 64;

/// Control points, sorted by x, x strictly increasing, all in 0..1.
#[derive(Clone, Debug, PartialEq)]
pub struct Points(pub Vec<[f32; 2]>);

impl Points {
    pub fn identity() -> Points {
        Points(vec![[0.0, 0.0], [1.0, 1.0]])
    }

    /// Parse `"x,y x,y …"` leniently: bad pairs are skipped, values clamped into 0..1, points
    /// sorted, duplicates in x merged; fewer than two points is the identity.
    pub fn parse(s: &str) -> Points {
        let mut v: Vec<[f32; 2]> = s
            .split_whitespace()
            .filter_map(|pair| {
                let (a, b) = pair.split_once(',')?;
                let x: f32 = a.trim().parse().ok()?;
                let y: f32 = b.trim().parse().ok()?;
                (x.is_finite() && y.is_finite()).then(|| [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
            })
            .take(MAX_POINTS)
            .collect();
        v.sort_by(|a, b| a[0].total_cmp(&b[0]));
        v.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-4);
        if v.len() < 2 { Points::identity() } else { Points(v) }
    }

    pub fn format(&self) -> String {
        self.0.iter().map(|p| format!("{},{}", trim(p[0]), trim(p[1]))).collect::<Vec<_>>().join(" ")
    }

    pub fn is_identity(&self) -> bool {
        self.0.iter().all(|p| (p[0] - p[1]).abs() < 1e-5)
    }

    /// The monotone cubic through the points at `x` (flat outside the first and last point).
    pub fn eval(&self, x: f32) -> f32 {
        let p = &self.0;
        let (Some(first), Some(last)) = (p.first(), p.last()) else { return x };
        if x <= first[0] {
            return first[1];
        }
        if x >= last[0] {
            return last[1];
        }
        let m = tangents(p);
        // Segment containing x.
        let i = p.partition_point(|q| q[0] <= x).saturating_sub(1).min(p.len().saturating_sub(2));
        let (Some(a), Some(b), Some(ma), Some(mb)) = (p.get(i), p.get(i + 1), m.get(i), m.get(i + 1)) else { return x };
        let h = b[0] - a[0];
        if h <= 0.0 {
            return a[1];
        }
        let t = (x - a[0]) / h;
        let (t2, t3) = (t * t, t * t * t);
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * a[1] + h10 * h * ma + h01 * b[1] + h11 * h * mb
    }

    /// Insert a point at `x` (on the curve), returning its index.
    pub fn insert(&mut self, x: f32, y: f32) -> usize {
        let x = x.clamp(0.0, 1.0);
        if let Some(i) = self.0.iter().position(|p| (p[0] - x).abs() < 1e-3) {
            if let Some(p) = self.0.get_mut(i) {
                p[1] = y.clamp(0.0, 1.0);
            }
            return i;
        }
        let i = self.0.partition_point(|p| p[0] < x);
        self.0.insert(i, [x, y.clamp(0.0, 1.0)]);
        i
    }

    /// Move point `i`, keeping it between its neighbours; end points keep their x.
    pub fn move_point(&mut self, i: usize, x: f32, y: f32) {
        let n = self.0.len();
        let lo = if i == 0 { 0.0 } else { self.0.get(i - 1).map_or(0.0, |p| p[0] + 1e-3) };
        let hi = if i + 1 >= n { 1.0 } else { self.0.get(i + 1).map_or(1.0, |p| p[0] - 1e-3) };
        if let Some(p) = self.0.get_mut(i) {
            p[0] = if lo <= hi { x.clamp(lo, hi) } else { p[0] };
            p[1] = y.clamp(0.0, 1.0);
        }
    }

    /// Remove point `i` unless only two remain.
    pub fn remove(&mut self, i: usize) -> bool {
        if self.0.len() <= 2 || i >= self.0.len() {
            return false;
        }
        self.0.remove(i);
        true
    }

    pub fn bake(&self) -> Table {
        let n = TABLE_SIZE;
        let v: Vec<f32> = (0..n).map(|i| self.eval(i as f32 / (n - 1) as f32)).collect();
        Table::new(v)
    }
}

fn trim(v: f32) -> String {
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" { "0".into() } else { s.into() }
}

/// Fritsch–Carlson tangents.
fn tangents(p: &[[f32; 2]]) -> Vec<f32> {
    let n = p.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let d: Vec<f32> = p.windows(2).map(|w| (w[1][1] - w[0][1]) / (w[1][0] - w[0][0]).max(1e-6)).collect();
    let mut m = vec![0.0f32; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 {
        m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) * 0.5 };
    }
    for i in 0..n - 1 {
        if d[i].abs() < 1e-9 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let a = m[i] / d[i];
        let b = m[i + 1] / d[i];
        let s = a * a + b * b;
        if s > 9.0 {
            let t = 3.0 / s.sqrt();
            m[i] = t * a * d[i];
            m[i + 1] = t * b * d[i];
        }
    }
    m
}

/// A function of one variable sampled over 0..1, extended linearly outside with its end slopes.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub v: Vec<f32>,
    pub slope_lo: f32,
    pub slope_hi: f32,
}

impl Table {
    pub fn new(v: Vec<f32>) -> Table {
        let n = v.len();
        let s = (n.max(2) - 1) as f32;
        let slope_lo = match (v.first(), v.get(1)) {
            (Some(a), Some(b)) => (b - a) * s,
            _ => 1.0,
        };
        let slope_hi = match (n.checked_sub(2).and_then(|i| v.get(i)), v.last()) {
            (Some(a), Some(b)) => (b - a) * s,
            _ => 1.0,
        };
        Table { v, slope_lo, slope_hi }
    }

    pub fn identity() -> Table {
        Points::identity().bake()
    }

    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let n = self.v.len();
        let (Some(&first), Some(&last)) = (self.v.first(), self.v.last()) else { return x };
        if x.is_nan() {
            // NaN reads as 0.
            return first;
        }
        if x <= 0.0 {
            return first + x * self.slope_lo;
        }
        if x >= 1.0 {
            return last + (x - 1.0) * self.slope_hi;
        }
        let f = x * (n - 1) as f32;
        let i = (f as usize).min(n.saturating_sub(2));
        let t = f - i as f32;
        let a = self.v.get(i).copied().unwrap_or(first);
        let b = self.v.get(i + 1).copied().unwrap_or(last);
        a + (b - a) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_format_and_identity() {
        let p = Points::parse("1,1 0,0 0.5,0.6 junk 0.5,0.7 nan,1");
        assert_eq!(p.0.len(), 3);
        assert_eq!(p.format(), "0,0 0.5,0.6 1,1");
        assert!(Points::parse("").is_identity());
        assert!(Points::parse("0,0 1,1").bake().v.iter().enumerate().all(|(i, v)| (v - i as f32 / 1023.0).abs() < 1e-6));
    }

    #[test]
    fn monotone_and_through_points() {
        let p = Points::parse("0,0 0.25,0.1 0.5,0.8 0.75,0.82 1,1");
        for q in &p.0 {
            assert!((p.eval(q[0]) - q[1]).abs() < 1e-5);
        }
        let t = p.bake();
        for w in t.v.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "not monotone");
        }
    }

    #[test]
    fn table_extrapolates_with_end_slopes() {
        let t = Table::identity();
        assert!((t.eval(2.0) - 2.0).abs() < 1e-4);
        assert!((t.eval(-0.5) + 0.5).abs() < 1e-4);
        assert_eq!(t.eval(f32::NAN), 0.0);
        let flat = Points::parse("0,0.2 0.3,0.2 1,0.9").bake();
        assert!((flat.eval(0.1) - 0.2).abs() < 1e-4);
    }

    #[test]
    fn editing() {
        let mut p = Points::identity();
        let i = p.insert(0.5, 0.7);
        assert_eq!(i, 1);
        p.move_point(1, 2.0, 0.3);
        assert!(p.0[1][0] < 1.0);
        assert!(p.remove(1));
        assert!(!p.remove(0));
    }
}

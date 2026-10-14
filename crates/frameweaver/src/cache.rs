//! A bounded, process-wide cache of estimated flow pairs, keyed by content.
//!
//! The key is a 64-bit hash of everything that determines the result: both analysis planes
//! (size and every sample), every estimator parameter, and any extra inputs the caller mixes in
//! (imported-vector settings, guide positions). Content addressing means a cached entry can
//! never be stale: change a frame, a parameter or a guide and the key changes. Entries are
//! evicted least-recently-used beyond [`MAX_ENTRIES`] or [`MAX_BYTES`]. The lock is recovered
//! if a thread panicked while holding it (the cache is cleared; it only ever holds
//! recomputable data).
//!
//! Layers of caching in FrameWeaver: decoded source frames are the renderer's (frame cache);
//! the analysis planes are cheap and recomputed; **flows with their occlusion and confidence
//! maps are cached here** (the expensive part, reused across every output frame between the
//! same two source frames and while scrubbing); the synthesised frames are cached by the
//! renderer's layer cache, keyed on the layer time.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use crate::consistency::FlowPair;
use crate::flow::FlowParams;
use crate::plane::Plane;

/// At most this many flow pairs are kept.
pub const MAX_ENTRIES: usize = 6;
/// At most this many bytes of flow data are kept.
pub const MAX_BYTES: usize = 256 << 20;

#[derive(Default)]
struct Lru {
    /// Least recently used first.
    entries: Vec<(u64, Arc<FlowPair>)>,
}

impl Lru {
    fn bytes(&self) -> usize {
        self.entries.iter().map(|(_, p)| p.bytes()).sum()
    }

    fn get(&mut self, key: u64) -> Option<Arc<FlowPair>> {
        let i = self.entries.iter().position(|(k, _)| *k == key)?;
        let e = self.entries.remove(i);
        let v = e.1.clone();
        self.entries.push(e);
        Some(v)
    }

    fn insert(&mut self, key: u64, v: Arc<FlowPair>) {
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, v));
        while self.entries.len() > 1 && (self.entries.len() > MAX_ENTRIES || self.bytes() > MAX_BYTES) {
            self.entries.remove(0);
        }
    }
}

static CACHE: OnceLock<Mutex<Lru>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

fn lock() -> MutexGuard<'static, Lru> {
    let m = CACHE.get_or_init(|| Mutex::new(Lru::default()));
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            // A thread panicked while holding the lock: drop what it may have left half-done
            // and un-poison, so the cache keeps working for the rest of the session.
            let mut g = poisoned.into_inner();
            g.entries.clear();
            drop(g);
            m.clear_poison();
            m.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }
}

/// Cache statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub bytes: usize,
}

pub fn stats() -> Stats {
    let g = lock();
    Stats { hits: HITS.load(Ordering::Relaxed), misses: MISSES.load(Ordering::Relaxed), entries: g.entries.len(), bytes: g.bytes() }
}

/// Drop every entry and reset the counters.
pub fn clear() {
    lock().entries.clear();
    HITS.store(0, Ordering::Relaxed);
    MISSES.store(0, Ordering::Relaxed);
}

/// FNV-1a over 32-bit words.
struct Hasher(u64);

impl Hasher {
    fn new() -> Hasher {
        Hasher(0xcbf2_9ce4_8422_2325)
    }
    fn word(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn f(&mut self, v: f32) {
        self.word(v.to_bits());
    }
}

/// The cache key of a pair of planes, estimator parameters and extra inputs.
pub fn key(a: &Plane, b: &Plane, p: &FlowParams, extra: &[f32]) -> u64 {
    let mut h = Hasher::new();
    for pl in [a, b] {
        h.word(pl.w as u32);
        h.word(pl.h as u32);
        for &v in &pl.d {
            h.f(v);
        }
    }
    h.word(p.max_long);
    h.word(p.patch as u32);
    h.f(p.overlap);
    h.word(p.iters);
    h.word(p.refine_outer);
    h.word(p.refine_inner);
    h.f(p.alpha);
    h.word(p.median);
    h.word(p.symmetrize as u32);
    h.word(p.global_fallback as u32);
    h.word(extra.len() as u32);
    for &v in extra {
        h.f(v);
    }
    h.0
}

/// The cached pair for `key`, or the result of `compute` (inserted when `Some`).
pub fn get_or_compute(key: u64, compute: impl FnOnce() -> Option<FlowPair>) -> Option<Arc<FlowPair>> {
    if let Some(hit) = lock().get(key) {
        HITS.fetch_add(1, Ordering::Relaxed);
        return Some(hit);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // Computed outside the lock: other threads keep using the cache meanwhile.
    let v = Arc::new(compute()?);
    lock().insert(key, v.clone());
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Quality;
    use crate::testkit::render;

    /// The cache is process-wide; tests that read its counters take this lock.
    pub(crate) static SERIAL: Mutex<()> = Mutex::new(());

    fn pair_of(shift: f32, w: usize) -> (Plane, Plane) {
        (render(w, 40, |x, y| (x, y)), render(w, 40, |x, y| (x - shift, y)))
    }

    #[test]
    fn hit_on_identical_content_miss_on_a_one_pixel_change() {
        let _g = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        clear();
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        let (a, b) = pair_of(2.0, 48);
        let k = key(&a, &b, &p, &[]);
        let calls = std::cell::Cell::new(0);
        let make = || {
            calls.set(calls.get() + 1);
            FlowPair::estimate(&a, &b, &p).ok()
        };
        assert!(get_or_compute(k, make).is_some());
        assert!(get_or_compute(k, make).is_some());
        assert_eq!(calls.get(), 1, "second lookup must hit");
        let mut b2 = b.clone();
        b2.d[7] += 1e-3;
        assert_ne!(key(&a, &b2, &p, &[]), k);
        assert_ne!(key(&a, &b, &p, &[1.0]), k);
        let mut p2 = p;
        p2.alpha *= 2.0;
        assert_ne!(key(&a, &b, &p2, &[]), k);
        let s = stats();
        assert_eq!((s.hits, s.misses, s.entries), (1, 1, 1));
    }

    #[test]
    fn entry_count_is_bounded_and_lru_order_kept() {
        let _g = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        clear();
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        for i in 0..(MAX_ENTRIES + 3) {
            let (a, b) = pair_of(1.0 + i as f32 * 0.1, 40);
            let k = key(&a, &b, &p, &[]);
            get_or_compute(k, || FlowPair::estimate(&a, &b, &p).ok());
        }
        assert_eq!(stats().entries, MAX_ENTRIES);
        // The oldest is gone, the newest is present.
        let (a, b) = pair_of(1.0, 40);
        let before = stats().misses;
        get_or_compute(key(&a, &b, &p, &[]), || FlowPair::estimate(&a, &b, &p).ok());
        assert_eq!(stats().misses, before + 1);
    }

    #[test]
    fn byte_bound_is_respected() {
        let mut l = Lru::default();
        let big = Arc::new(FlowPair { w: 4096, h: 4096, ..FlowPair::default() });
        // FlowPair::bytes() is computed from the dimensions, so no real allocation is needed.
        assert!(big.bytes() > MAX_BYTES / 2);
        for k in 0..4 {
            l.insert(k, big.clone());
        }
        assert!(l.bytes() <= MAX_BYTES || l.entries.len() == 1);
        assert!(l.entries.len() < 4);
    }

    #[test]
    fn poisoned_lock_is_recovered() {
        let _g = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        clear();
        let r = std::panic::catch_unwind(|| {
            let _held = CACHE.get_or_init(|| Mutex::new(Lru::default())).lock();
            std::panic::resume_unwind(Box::new("poison the cache lock"));
        });
        assert!(r.is_err());
        let p = FlowParams::for_quality(Quality::Draft, 0.7);
        let (a, b) = pair_of(2.0, 40);
        assert!(get_or_compute(1, || FlowPair::estimate(&a, &b, &p).ok()).is_some());
        assert!(get_or_compute(2, || FlowPair::estimate(&a, &b, &p).ok()).is_some());
        // Un-poisoned: both entries survive (the cache is not cleared on every access).
        assert_eq!(stats().entries, 2);
    }
}

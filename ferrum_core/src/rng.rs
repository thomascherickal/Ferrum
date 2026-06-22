//! Seeded xorshift64* PRNG — deterministic, dependency-free.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let actual_seed = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        crate::vprintln!("[rng::Rng::new] seed={} (effective={})", seed, actual_seed);
        Self {
            state: actual_seed,
        }
    }

    /// The raw internal state — pair with [`Rng::from_state`] to checkpoint and
    /// resume a generator exactly where it left off (T6).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Rebuild a generator from a raw [`Rng::state`] value (no zero-remapping —
    /// the state already came from a live generator).
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// A uniformly random permutation of `0..n` via Fisher–Yates, so a training
    /// epoch can visit every index exactly once (without replacement) instead of
    /// sampling with replacement. Returns an empty vector for `n == 0`.
    pub fn shuffled_indices(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let seq = |seed| {
            (0..50)
                .map(|_| {
                    let mut r = Rng::new(seed);
                    r.next_u64()
                })
                .last()
        };
        assert_eq!(
            {
                let mut r = Rng::new(42);
                (0..50).map(|_| r.next_u64()).collect::<Vec<_>>()
            },
            {
                let mut r = Rng::new(42);
                (0..50).map(|_| r.next_u64()).collect::<Vec<_>>()
            }
        );
        let _ = seq; // silence unused warning
    }

    #[test]
    fn f32_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x), "out of [0,1): {x}");
        }
    }

    #[test]
    fn zero_seed_remapped() {
        // A zero seed should not lock the generator at 0 forever.
        let mut r = Rng::new(0);
        let vals: Vec<u64> = (0..10).map(|_| r.next_u64()).collect();
        assert!(vals.iter().any(|&v| v != 0));
    }

    #[test]
    fn normal_mean_near_zero() {
        let mut r = Rng::new(99);
        let mean: f32 = (0..50_000).map(|_| r.next_normal()).sum::<f32>() / 50_000.0;
        assert!(mean.abs() < 0.05, "mean = {mean}");
    }

    #[test]
    fn shuffled_indices_is_a_permutation() {
        let mut r = Rng::new(7);
        let n = 100;
        let perm = r.shuffled_indices(n);
        assert_eq!(perm.len(), n);
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>(), "not a permutation of 0..n");
        // Empty / singleton edge cases.
        assert!(Rng::new(1).shuffled_indices(0).is_empty());
        assert_eq!(Rng::new(1).shuffled_indices(1), vec![0]);
    }

    #[test]
    fn shuffled_indices_is_deterministic_and_actually_shuffles() {
        assert_eq!(Rng::new(5).shuffled_indices(64), Rng::new(5).shuffled_indices(64));
        // Overwhelmingly likely to differ from the identity for n=64.
        assert_ne!(Rng::new(5).shuffled_indices(64), (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn normal_std_near_one() {
        let mut r = Rng::new(88);
        let xs: Vec<f32> = (0..50_000).map(|_| r.next_normal()).collect();
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;
        assert!((var.sqrt() - 1.0).abs() < 0.05, "std = {}", var.sqrt());
    }
}

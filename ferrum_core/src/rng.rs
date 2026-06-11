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
    fn normal_std_near_one() {
        let mut r = Rng::new(88);
        let xs: Vec<f32> = (0..50_000).map(|_| r.next_normal()).collect();
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;
        assert!((var.sqrt() - 1.0).abs() < 0.05, "std = {}", var.sqrt());
    }
}

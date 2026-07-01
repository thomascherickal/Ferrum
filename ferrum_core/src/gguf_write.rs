//! Pure-`std` GGUF *writer* — serialize a Ferrum `LlamaModel` back to a GGUF v3
//! file that runs in the wider ecosystem (llama.cpp / ollama / LM Studio).
//!
//! The reader in [`crate::gguf`] is the specification: every block encoder here
//! is the exact inverse of the matching `dequant_*` decoder, and is verified by
//! round-tripping through [`crate::gguf::Gguf`].

use crate::gguf::{
    GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_Q8_1,
};

/// The on-disk GGUF tensor type the writer should emit for weight matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufQuant {
    F32,
    F16,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
}

impl GgufQuant {
    /// Parse a CLI/API name (case-insensitive). Accepts the reader's spellings.
    /// API is deliberately `GgufQuant::from_str`, not implementing `FromStr` trait.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "none" => Some(Self::F32),
            "f16" | "fp16" => Some(Self::F16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            "q8_1" => Some(Self::Q8_1),
            "q4_0" | "q4" => Some(Self::Q4_0),
            "q4_1" => Some(Self::Q4_1),
            "q4_k" | "q4k" => Some(Self::Q4K),
            "q5_k" | "q5k" => Some(Self::Q5K),
            "q6_k" | "q6k" => Some(Self::Q6K),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_F32,
            Self::F16 => GGML_F16,
            Self::Q8_0 => GGML_Q8_0,
            Self::Q8_1 => GGML_Q8_1,
            Self::Q4_0 => GGML_Q4_0,
            Self::Q4_1 => GGML_Q4_1,
            Self::Q4K => GGML_Q4_K,
            Self::Q5K => GGML_Q5_K,
            Self::Q6K => GGML_Q6_K,
        }
    }

    /// The GGUF `general.file_type` enum id closest to this target (metadata hint).
    #[allow(dead_code)]
    pub(crate) fn file_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q8_0 => 7,
            Self::Q8_1 => 9,
            Self::Q4K => 15, // MOSTLY_Q4_K_M
            Self::Q5K => 17, // MOSTLY_Q5_K_M
            Self::Q6K => 18, // MOSTLY_Q6_K
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_kquant(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K)
    }
}

/// Encode an `f32` as an IEEE-754 half (`u16`), round-to-nearest-ties-to-even.
/// Inf/NaN, subnormals, and overflow-to-Inf are all handled. Exact inverse of
/// [`crate::gguf::f16_to_f32`] on values representable as f16.
pub fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp32 = ((x >> 23) & 0xFF) as i32;
    let mant32 = x & 0x007F_FFFF;

    if exp32 == 0xFF {
        // Inf (mant 0) or NaN (mant != 0 → a canonical quiet NaN).
        return if mant32 != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7C00
        };
    }

    let mut exp16 = exp32 - 127 + 15;
    if exp16 >= 0x1F {
        return sign | 0x7C00; // Overflow → Inf.
    }
    if exp16 <= 0 {
        // Subnormal f16, or underflow to signed zero.
        if exp16 < -10 {
            return sign;
        }
        let mant = mant32 | 0x0080_0000; // restore implicit leading 1
        let shift = (14 - exp16) as u32; // 14..=24
        let mut half = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half as u16;
    }

    // Normalized f16.
    let mut mant16 = (mant32 >> 13) as u16;
    let rem = mant32 & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (mant16 & 1) == 1) {
        mant16 += 1;
        if mant16 == 0x0400 {
            mant16 = 0;
            exp16 += 1;
            if exp16 >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp16 as u16) << 10) | mant16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::f16_to_f32;

    #[test]
    fn f32_to_f16_is_exact_inverse_of_f16_to_f32() {
        // For every representable f16 bit pattern, encoding its f32 value must
        // return the same bits. NaNs are canonicalized; ±0 compared by bits.
        for h in 0u32..=0xFFFF {
            let h = h as u16;
            let exp = (h >> 10) & 0x1F;
            let mant = h & 0x3FF;
            if exp == 0x1F && mant != 0 {
                continue; // skip NaN encodings (many bit patterns, one canonical out)
            }
            let f = f16_to_f32(h);
            assert_eq!(f32_to_f16(f), h, "roundtrip failed for f16 bits {h:#06x}");
        }
    }

    #[test]
    fn f32_to_f16_spot_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF); // max normal f16
        assert_eq!(f32_to_f16(1e5), 0x7C00); // overflow → +Inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn gguf_quant_parses_and_maps_types() {
        use crate::gguf::{GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0};
        assert_eq!(GgufQuant::from_str("f32").unwrap().ggml_type(), GGML_F32);
        assert_eq!(GgufQuant::from_str("f16").unwrap().ggml_type(), GGML_F16);
        assert_eq!(GgufQuant::from_str("q8_0").unwrap().ggml_type(), GGML_Q8_0);
        assert_eq!(GgufQuant::from_str("q4_0").unwrap().ggml_type(), GGML_Q4_0);
        assert_eq!(GgufQuant::from_str("q4_k").unwrap().ggml_type(), GGML_Q4_K);
        assert_eq!(GgufQuant::from_str("q6_k").unwrap().ggml_type(), GGML_Q6_K);
        assert!(GgufQuant::from_str("q4_k").unwrap().is_kquant());
        assert!(!GgufQuant::from_str("q8_0").unwrap().is_kquant());
        assert!(GgufQuant::from_str("bogus").is_none());
    }
}

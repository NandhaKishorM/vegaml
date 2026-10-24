//! Data-only parsing: the checkpoint is plain little-endian f32 arrays. Nothing is executed, and
//! any dtype or layout we do not expect is refused rather than guessed.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub type Tensors = HashMap<String, Tensor>;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// Parse a safetensors file holding only F32 tensors.
pub fn read_safetensors(path: &Path) -> Result<Tensors> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_safetensors(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn parse_safetensors(bytes: &[u8]) -> Result<Tensors> {
    // Every offset comes from the file, so all slicing goes through checked `get`.
    let n = usize::try_from(u64::from_le_bytes(bytes.get(..8).context("file too short")?.try_into()?))?;
    let header_end = n.checked_add(8).filter(|&e| e <= bytes.len()).context("header length out of range")?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..header_end])?;
    let data = &bytes[header_end..];
    let mut out = Tensors::new();
    for (name, v) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().ok_or_else(|| anyhow!("{name}: no dtype"))?;
        if dtype != "F32" {
            bail!("{name}: dtype {dtype} (only F32 is supported)");
        }
        let shape: Vec<usize> = serde_json::from_value(v["shape"].clone())?;
        let offs: [usize; 2] = serde_json::from_value(v["data_offsets"].clone())?;
        let raw = data.get(offs[0]..offs[1]).with_context(|| format!("{name}: data offsets out of range"))?;
        out.insert(name.clone(), tensor(shape, raw).with_context(|| name)?);
    }
    Ok(out)
}

/// A tensor from raw little-endian f32 bytes, refusing shapes whose element count overflows or does not match.
fn tensor(shape: Vec<usize>, raw: &[u8]) -> Result<Tensor> {
    let numel = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).context("shape overflows")?;
    ensure!(numel.checked_mul(4) == Some(raw.len()), "{} bytes for shape {shape:?}", raw.len());
    Ok(Tensor { shape, data: f32s(raw) })
}

/// Parse a NumPy `.npz` written by `np.savez` (stored, not deflated) holding `<f4` C-order arrays.
pub fn read_npz(path: &Path) -> Result<Tensors> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_npz(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn parse_npz(bytes: &[u8]) -> Result<Tensors> {
    let u16_at = |p: usize| -> Result<usize> { Ok(u16::from_le_bytes(bytes.get(p..p + 2).context("truncated")?.try_into()?) as usize) };
    let u32_at = |p: usize| -> Result<usize> { Ok(u32::from_le_bytes(bytes.get(p..p + 4).context("truncated")?.try_into()?) as usize) };
    let mut out = Tensors::new();
    let mut i = 0usize;
    // Local file headers come first; the central directory (PK\x01\x02) ends the walk.
    while bytes.get(i..i + 4) == Some(b"PK\x03\x04".as_slice()) {
        let flags = u16_at(i + 6)?;
        let method = u16_at(i + 8)?;
        let mut size = u32_at(i + 18)?;
        let name_len = u16_at(i + 26)?;
        let extra_len = u16_at(i + 28)?;
        let name_start = i.checked_add(30).context("offset overflow")?;
        let extra_start = name_start + name_len;
        let start = extra_start + extra_len;
        let name = std::str::from_utf8(bytes.get(name_start..extra_start).context("truncated name")?)?.to_string();
        let extra = bytes.get(extra_start..start).context("truncated extra field")?;
        // ZIP64: the 32-bit size is 0xFFFFFFFF and the real one sits in the extra field (id 0x0001).
        if size == 0xFFFF_FFFF {
            let mut e = 0usize;
            while let Some(h) = extra.get(e..e + 4) {
                let id = u16::from_le_bytes([h[0], h[1]]);
                let len = u16::from_le_bytes([h[2], h[3]]) as usize;
                if id == 1 && len >= 8 {
                    let field = extra.get(e + 4..e + 12).context("truncated ZIP64 field")?;
                    size = usize::try_from(u64::from_le_bytes(field.try_into()?))?;
                }
                e += 4 + len;
            }
        }
        ensure!(method == 0, "{name}: compressed entries are not supported");
        ensure!(flags & 0x08 == 0, "{name}: streamed entries are not supported");
        let end = start.checked_add(size).filter(|&e| e <= bytes.len()).with_context(|| format!("{name}: truncated"))?;
        let key = name.trim_end_matches(".npy").to_string();
        out.insert(key, parse_npy(&bytes[start..end]).with_context(|| name.clone())?);
        i = end;
    }
    ensure!(!out.is_empty(), "no arrays found");
    Ok(out)
}

fn parse_npy(b: &[u8]) -> Result<Tensor> {
    ensure!(b.get(..6) == Some(b"\x93NUMPY".as_slice()), "not an .npy array");
    let (hlen, hstart): (usize, usize) = match b.get(6) {
        Some(1) => (u16::from_le_bytes(b.get(8..10).context("truncated")?.try_into()?) as usize, 10),
        Some(2 | 3) => (u32::from_le_bytes(b.get(8..12).context("truncated")?.try_into()?) as usize, 12),
        v => bail!("npy version {v:?}"),
    };
    let data_start = hstart.checked_add(hlen).context("header length overflow")?;
    let header = std::str::from_utf8(b.get(hstart..data_start).context("truncated header")?)?;
    ensure!(header.contains("'descr': '<f4'"), "only little-endian float32 arrays are supported: {header}");
    ensure!(header.contains("'fortran_order': False"), "only C-order arrays are supported");
    let shape_txt = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or_else(|| anyhow!("no shape in {header}"))?;
    let shape: Vec<usize> = shape_txt.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse()).collect::<Result<_, _>>()?;
    tensor(shape, &b[data_start..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_safetensors_file() {
        let header = br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"format":"pt"}}"#;
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header);
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice(&(-2.0f32).to_le_bytes());
        let t = parse_safetensors(&b).unwrap();
        assert_eq!(t["w"].data, vec![1.5, -2.0]);
    }

    #[test]
    fn truncated_or_hostile_files_are_errors_not_panics() {
        let mut st = (u64::MAX).to_le_bytes().to_vec();
        st.extend_from_slice(b"{}");
        assert!(parse_safetensors(&st).is_err());
        let header = br#"{"w":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,8]}}"#;
        let mut big = (header.len() as u64).to_le_bytes().to_vec();
        big.extend_from_slice(header);
        big.extend_from_slice(&[0; 8]);
        assert!(parse_safetensors(&big).is_err());
        let mut zip = b"PK\x03\x04".to_vec();
        zip.extend_from_slice(&[0; 14]);
        zip.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        zip.extend_from_slice(&[0; 4]);
        zip.extend_from_slice(&200u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        for cut in 0..zip.len() {
            assert!(parse_npz(&zip[..cut]).is_err());
        }
        assert!(parse_npz(&zip).is_err());
        assert!(parse_npy(b"\x93NUMPY\x02\x00\xff\xff\xff\xff").is_err());
    }

    #[test]
    fn refuses_other_dtypes() {
        let header = br#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header);
        b.extend_from_slice(&[0, 0]);
        assert!(parse_safetensors(&b).is_err());
    }
}

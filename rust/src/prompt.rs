//! Mirrors `vegaml._common.option_texts` and `build_prompt`. Each piece is tokenized on its own
//! (as the Python code does), so span boundaries are exact token positions.

use crate::QType;

use crate::engine::qtype_name;

/// One typed question for the engine.
#[derive(Debug, Clone)]
pub struct VegaQuestion {
    pub kind: QType,
    pub instructions: String,
    /// Option texts in label order; boolean is always `[no…, yes…]`.
    pub options: Vec<String>,
}

impl VegaQuestion {
    /// `criteria` maps each choice label to an optional description (`"label - description"`).
    pub fn choice(instructions: &str, criteria: &[(&str, Option<&str>)]) -> Self {
        let options = criteria.iter().map(|(k, v)| match v {
            Some(v) if !v.is_empty() => format!("{k} - {v}"),
            _ => k.to_string(),
        });
        Self { kind: QType::Choice, instructions: instructions.into(), options: options.collect() }
    }

    /// Levels 0..n, each with its description.
    pub fn score(instructions: &str, levels: &[&str]) -> Self {
        let options = levels.iter().enumerate().map(|(i, c)| format!("level {i} - {c}"));
        Self { kind: QType::Score, instructions: instructions.into(), options: options.collect() }
    }

    /// Optional descriptions of what "no" and "yes" mean; defaults are VegaML's.
    pub fn boolean(instructions: &str, no: Option<&str>, yes: Option<&str>) -> Self {
        let no = format!("no - {}", no.unwrap_or("the statement does not hold"));
        let yes = format!("yes - {}", yes.unwrap_or("the statement holds"));
        Self { kind: QType::Boolean, instructions: instructions.into(), options: vec![no, yes] }
    }
}

/// Token spans `[start, end)` inside the prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spans {
    pub state: (usize, usize),
    pub question: (usize, usize),
    pub options: Vec<(usize, usize)>,
}

const HEAD: &str = "Observation:\n";
const END: &str = "\nOutcome:";
const INS_MAX_TOKENS: usize = 256;

fn question_text(q: &VegaQuestion) -> String {
    format!("\n\nMeasurement ({}): {}\nPossible outcomes:", qtype_name(q.kind), q.instructions)
}

fn query_first_text(q: &VegaQuestion) -> String {
    format!("Question to keep in mind while reading ({}): {}\n\n", qtype_name(q.kind), q.instructions)
}

/// Template-0 prompt for `state` and `q`, options in label order, state truncated on the right to fit `max_len`.
/// `enc` tokenizes plain text without special tokens; `bos` is the reader's BOS token if it has one.
pub fn build_prompt(enc: &dyn Fn(&str) -> Vec<i32>, bos: Option<i32>, state: &str, q: &VegaQuestion, max_len: usize, opt_max_tokens: usize, query_first_max_state: usize) -> (Vec<i32>, Spans) {
    let bos: Vec<i32> = bos.into_iter().collect();
    let head = enc(HEAD);
    let mut qids = enc(&question_text(q));
    qids.truncate(INS_MAX_TOKENS);
    let mut oids: Vec<Vec<i32>> = q
        .options
        .iter()
        .map(|o| {
            let mut t = enc(&format!("\n* {o}"));
            t.truncate(opt_max_tokens);
            t
        })
        .collect();
    let end = enc(END);
    let opt_sum = |o: &Vec<Vec<i32>>| o.iter().map(Vec::len).sum::<usize>();
    let mut fixed = bos.len() + head.len() + qids.len() + opt_sum(&oids) + end.len();
    if fixed + 16 > max_len {
        // Too many or too long options: shrink every option evenly, then the instructions.
        let room = max_len.saturating_sub(16 + bos.len() + head.len() + end.len() + qids.len().min(64));
        let per = (room / oids.len().max(1)).max(3);
        oids.iter_mut().for_each(|o| o.truncate(per));
        let keep = max_len.saturating_sub(16 + bos.len() + head.len() + end.len() + opt_sum(&oids)).max(8);
        qids.truncate(keep);
        fixed = bos.len() + head.len() + qids.len() + opt_sum(&oids) + end.len();
    }
    let mut st = enc(state);
    let mut pre = Vec::new();
    if query_first_max_state > 0 && st.len() < query_first_max_state {
        pre = enc(&query_first_text(q));
        pre.truncate(128.min(INS_MAX_TOKENS));
        fixed += pre.len();
    }
    st.truncate(max_len.saturating_sub(fixed));

    let mut ids = bos.clone();
    ids.extend(pre);
    ids.extend(head);
    let s0 = ids.len();
    let state_span = if st.is_empty() { (bos.len(), s0) } else { (s0, s0 + st.len()) };
    ids.extend(st);
    let q0 = ids.len();
    ids.extend(qids);
    let question = (q0, ids.len());
    let mut options = Vec::with_capacity(oids.len());
    for o in oids {
        let a = ids.len();
        ids.extend(o);
        options.push((a, ids.len()));
    }
    ids.extend(end);
    (ids, Spans { state: state_span, question, options })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character keeps span arithmetic easy to check by hand.
    fn chars(s: &str) -> Vec<i32> {
        s.chars().map(|c| c as i32).collect()
    }

    #[test]
    fn spans_cover_state_question_and_each_option() {
        let q = VegaQuestion::boolean("Is it an invoice?", None, None);
        let (ids, sp) = build_prompt(&chars, None, "Pay me", &q, 10_000, 32, 0);
        let text: String = ids.iter().map(|&c| char::from_u32(c as u32).unwrap()).collect();
        let slice = |(a, b): (usize, usize)| text.chars().skip(a).take(b - a).collect::<String>();
        assert_eq!(slice(sp.state), "Pay me");
        assert_eq!(slice(sp.question), "\n\nMeasurement (boolean): Is it an invoice?\nPossible outcomes:");
        assert_eq!(slice(sp.options[1]), "\n* yes - the statement holds".chars().take(32).collect::<String>());
        assert!(text.starts_with("Observation:\n") && text.ends_with("\nOutcome:"));
    }

    #[test]
    fn short_states_are_read_with_the_question_first() {
        let q = VegaQuestion::boolean("Q?", None, None);
        let (ids, sp) = build_prompt(&chars, Some(1), "abc", &q, 10_000, 32, 4096);
        assert_eq!(ids[0], 1);
        let pre = query_first_text(&q).chars().count();
        assert_eq!(sp.state.0, 1 + pre + HEAD.len());
    }

    #[test]
    fn long_states_are_truncated_to_fit() {
        let q = VegaQuestion::boolean("Q?", None, None);
        let state = "x".repeat(5000);
        let (ids, sp) = build_prompt(&chars, None, &state, &q, 1000, 32, 0);
        assert_eq!(ids.len(), 1000);
        assert!(sp.state.1 - sp.state.0 < 1000);
    }
}

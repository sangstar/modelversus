use crate::tokf1::token_f1_score;
use crate::utils::Sequence;
use std::collections::HashMap;
const MAX_NGRAMS: usize = 6;

fn get_ngrams_from_word_vec(word_vec: &Vec<String>, ngrams: usize) -> Vec<String> {
    if ngrams == 1 {
        return word_vec.to_owned();
    }
    let len = word_vec.len();
    let mut res: Vec<String> = vec![];
    for i in 0..=(len - ngrams) {
        let word_grouping = word_vec[i..i + ngrams].join(" ");
        res.push(word_grouping);
    }
    res
}

fn get_ngram_counts(ngrams_vec: &Vec<String>) -> HashMap<String, usize> {
    let mut hashmap: HashMap<String, usize> = HashMap::new();
    for ngram in ngrams_vec {
        *hashmap.entry(ngram.clone()).or_insert(0) += 1;
    }
    hashmap
}

fn get_matches_clipped(
    pred_map: &HashMap<String, usize>,
    gold_map: &HashMap<String, usize>,
) -> usize {
    let mut matches: usize = 0;
    for (ng, pred_count) in pred_map {
        if let Some(gold_count) = gold_map.get(ng) {
            matches += pred_count.min(gold_count);
        }
    }
    matches
}

fn calc_bleu_for_ngram(
    pred_word_vector: &Vec<String>,
    gold_word_vector: &Vec<String>,
    ngrams: usize,
) -> f64 {
    let pred_ngrams_vec = get_ngrams_from_word_vec(pred_word_vector, ngrams);
    let gold_ngrams_vec = get_ngrams_from_word_vec(gold_word_vector, ngrams);

    let total_pred_ngrams_count = pred_ngrams_vec.len();
    if total_pred_ngrams_count == 0 {
        return 0.0;
    }

    let pred_ngrams_counts = get_ngram_counts(&pred_ngrams_vec);
    let gold_ngrams_counts = get_ngram_counts(&gold_ngrams_vec);

    let matches = get_matches_clipped(&pred_ngrams_counts, &gold_ngrams_counts) as f64;
    matches / total_pred_ngrams_count as f64
}

fn geometric_mean(values: &Vec<f64>) -> f64 {
    if values.is_empty() || values.iter().any(|&v| v == 0.0) {
        return 0.0;
    }

    let log_sum: f64 = values.iter().map(|v| v.ln()).sum();
    (log_sum / values.len() as f64).exp()
}

pub fn bleu_score(pred: &Sequence, gold: &Sequence) -> f64 {
    let pred_word_vector_len = pred.word_vector.len();
    let gold_word_vector_len = gold.word_vector.len();

    // Make sure there we don't ask for more ngrams than the
    // two strings can provide
    let max_ngrams = MAX_NGRAMS
        .min(pred.word_vector.len())
        .min(gold.word_vector.len());

    let mut bleu_scores: Vec<f64> = vec![];
    for i in 1..=max_ngrams {
        bleu_scores.push(calc_bleu_for_ngram(&pred.word_vector, &gold.word_vector, i))
    }
    let mean = geometric_mean(&bleu_scores);

    let brevity_penalty: f64;

    if pred_word_vector_len > gold_word_vector_len {
        brevity_penalty = 1.0
    } else {
        let brevity_penalty_arg: f64 =
            1.0 - (gold.word_vector.len() as f64 / pred.word_vector.len() as f64);
        brevity_penalty = brevity_penalty_arg.exp();
    }

    brevity_penalty * mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bleu_perfect_match() {
        let pred = "the quick brown fox";
        let gold = "the quick brown fox";
        let score = bleu_score(&Sequence::new(pred), &Sequence::new(gold));
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Expected BLEU 1.0, got {}",
            score
        );
    }

    // This won't work for 3 ngrams, as pred has no
    // "the quick fox" sequence, and BLEU punishes this
    // severely with the geometric mean
    #[test]
    fn test_bleu_partial_match() {
        let pred = "the quick brown fox";
        let gold = "the quick fox";
        let score = bleu_score(&Sequence::new(pred), &Sequence::new(gold));
        assert_eq!(score, 0.0, "Expected BLEU 0.0, got {}", score);
    }

    #[test]
    fn test_bleu_no_match() {
        let pred = "cats sleep all day";
        let gold = "the quick brown fox";
        let score = bleu_score(&Sequence::new(pred), &Sequence::new(gold));
        assert_eq!(score, 0.0, "Expected BLEU 0.0, got {}", score);
    }

    #[test]
    fn test_bleu_repeated_pred() {
        let pred = "the the the the";
        let gold = "the the";
        let score = bleu_score(&Sequence::new(pred), &Sequence::new(gold));
        // BLEU-1 should be 0.5 due to clipped count: min(4, 2) / 4
        assert!(
            (score - 0.5).abs() < 0.1,
            "Expected BLEU ≈ 0.5, got {}",
            score
        );
    }

    #[test]
    fn test_bleu_empty_pred() {
        let pred = Sequence::new("");
        let gold = Sequence::new("the quick brown fox");
        let score = bleu_score(&pred, &gold);
        assert_eq!(
            score, 0.0,
            "Expected BLEU 0.0 for empty prediction, got {}",
            score
        );
    }

    #[test]
    fn test_bleu_empty_gold() {
        let pred = Sequence::new("the quick brown fox");
        let gold = Sequence::new("");
        let score = bleu_score(&pred, &gold);
        assert_eq!(
            score, 0.0,
            "Expected BLEU 0.0 for empty gold, got {}",
            score
        );
    }
}

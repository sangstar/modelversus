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

fn calc_bleu(pred_word_vec: &Vec<String>, gold_word_vec: &Vec<String>, ngrams: usize) -> f32 {
    let pred_ngrams_vec = get_ngrams_from_word_vec(pred_word_vec, ngrams);
    let gold_ngrams_vec = get_ngrams_from_word_vec(gold_word_vec, ngrams);

    let total_pred_ngrams_count = pred_ngrams_vec.len();
    if total_pred_ngrams_count == 0 {
        return 0.0;
    }

    let pred_ngrams_counts = get_ngram_counts(&pred_ngrams_vec);
    let gold_ngrams_counts = get_ngram_counts(&gold_ngrams_vec);

    let matches = get_matches_clipped(&pred_ngrams_counts, &gold_ngrams_counts) as f32;
    matches / total_pred_ngrams_count as f32
}

fn geometric_mean(values: &Vec<f32>) -> f32 {
    if values.is_empty() || values.iter().any(|&v| v == 0.0) {
        return 0.0;
    }

    let log_sum: f32 = values.iter().map(|v| v.ln()).sum();
    (log_sum / values.len() as f32).exp()
}

fn calculate_bleu(pred: &str, gold: &str) -> f32 {
    let pred_word_vec: Vec<String> = pred.split_whitespace().map(|s| s.to_string()).collect();
    let gold_word_vec: Vec<String> = gold.split_whitespace().map(|s| s.to_string()).collect();

    let pred_word_vec_len = pred_word_vec.len();
    let gold_word_vec_len = gold_word_vec.len();

    // Make sure there we don't ask for more ngrams than the
    // two strings can provide
    let max_ngrams = MAX_NGRAMS.min(pred_word_vec.len()).min(gold_word_vec.len());

    let mut bleu_scores: Vec<f32> = vec![];
    for i in 1..=max_ngrams {
        bleu_scores.push(calc_bleu(&pred_word_vec, &gold_word_vec, i))
    }
    let mean = geometric_mean(&bleu_scores);

    let brevity_penalty: f32;

    if pred_word_vec_len > gold_word_vec_len {
        brevity_penalty = 1.0
    } else {
        let brevity_penalty_arg: f32 =
            1.0 - (gold_word_vec.len() as f32 / pred_word_vec.len() as f32);
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
        let score = calculate_bleu(pred, gold);
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
        let score = calculate_bleu(pred, gold);
        assert_eq!(score, 0.0, "Expected BLEU 0.0, got {}", score);
    }

    #[test]
    fn test_bleu_no_match() {
        let pred = "cats sleep all day";
        let gold = "the quick brown fox";
        let score = calculate_bleu(pred, gold);
        assert_eq!(score, 0.0, "Expected BLEU 0.0, got {}", score);
    }

    #[test]
    fn test_bleu_repeated_pred() {
        let pred = "the the the the";
        let gold = "the the";
        let score = calculate_bleu(pred, gold);
        // BLEU-1 should be 0.5 due to clipped count: min(4, 2) / 4
        assert!(
            (score - 0.5).abs() < 0.1,
            "Expected BLEU ≈ 0.5, got {}",
            score
        );
    }

    #[test]
    fn test_bleu_empty_pred() {
        let pred = "";
        let gold = "the quick brown fox";
        let score = calculate_bleu(pred, gold);
        assert_eq!(
            score, 0.0,
            "Expected BLEU 0.0 for empty prediction, got {}",
            score
        );
    }

    #[test]
    fn test_bleu_empty_gold() {
        let pred = "the quick brown fox";
        let gold = "";
        let score = calculate_bleu(pred, gold);
        assert_eq!(
            score, 0.0,
            "Expected BLEU 0.0 for empty gold, got {}",
            score
        );
    }
}

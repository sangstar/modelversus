use std::collections::HashSet;

// This deserves a low weight for the final score prediction, as:
// Reference: "the quick brown fox"
// Prediction: "fox quick the brown"
// Are a perfect score just because they have the same overlap
fn get_token_f1(pred: &str, gold: &str) -> f32 {
    let pred_set: HashSet<&str> = pred.split_whitespace().collect();
    let gold_set: HashSet<&str> = gold.split_whitespace().collect();

    // If one of the outputs are empty and the other isn't,
    // this warrants a score of 0.
    if pred_set.is_empty() || gold_set.is_empty() {
        return 0.0;
    }

    // If both are empty, this is a perfect match
    if pred_set.is_empty() && gold_set.is_empty() {
        return 1.0;
    }

    let num_common = gold_set.intersection(&pred_set).count() as f32;

    let precision: f32 = num_common / pred_set.len() as f32;
    let recall: f32 = num_common / gold_set.len() as f32;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_match() {
        let pred = "the quick brown fox";
        let gold = "the quick brown fox";

        let score = get_token_f1(pred, gold);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_full_overlap_different_order() {
        let pred = "fox quick the brown";
        let gold = "the quick brown fox";

        let score = get_token_f1(pred, gold);
        assert_eq!(score, 1.0); // same tokens, different order
    }

    #[test]
    fn test_partial_overlap() {
        let pred = "the fox jumps";
        let gold = "the quick brown fox";

        // common: "the", "fox"
        // precision = 2/3
        // recall = 2/4
        // f1 = 2 * (2/3 * 2/4) / (2/3 + 2/4) = 4/5 = 0.5714...
        let score = get_token_f1(pred, gold);
        assert!((score - 0.5714).abs() < 0.01);
    }

    #[test]
    fn test_no_overlap() {
        let pred = "apples and oranges";
        let gold = "the quick brown fox";

        let score = get_token_f1(pred, gold);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_empty_pred() {
        let pred = "";
        let gold = "something";

        let score = get_token_f1(pred, gold);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_empty_gold() {
        let pred = "something";
        let gold = "";

        let score = get_token_f1(pred, gold);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_both_empty() {
        let pred = "";
        let gold = "";

        let score = get_token_f1(pred, gold);
        assert_eq!(score, 0.0); // no prediction and no target means undefined precision/recall
    }
}

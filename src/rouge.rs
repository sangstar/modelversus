use crate::tokf1::token_f1_score;
use crate::utils::{DPMatrix, Sequence};

fn get_lcs(pred: &Sequence, gold: &Sequence) -> usize {
    // Add +1 to both rows and cols to accommodate the empty prefix
    // of the dynamic programming matrix
    let mut dp: DPMatrix = DPMatrix::new(pred.n_words + 1, gold.n_words + 1);
    for i in 1..=pred.n_words {
        // In this inner loop, we're keeping the row
        // index constant and looping over columns.
        // This means we're taking the word at
        // pred.word_vector[i-1], and seeing what the
        // longest subsequence we can make with it from
        // the words in gold.word_vector
        for j in 1..=gold.n_words {
            // If there's a match here for the pred and
            // gold word vector words here, increment the
            // LCS. Once again, there's an intentional off-by-one
            // here between the DP and the word vector indexing
            // due to the empty prefix for the dp

            if pred.word_vector[i - 1] == gold.word_vector[j - 1] {
                dp.set(i, j, dp.at(i - 1, j - 1) + 1)
            } else {
                // If not a match, we try to see if we can skip one of
                // the words in either vector, since LCS doesn't require
                // contiguity of sequences, just maintaining order, we can
                // do this. A skip in either word in the two word vectors
                // either involves taking the LCS from the previous pred
                // or gold vector word. Whichever is larger we'll take.
                dp.set(i, j, dp.at(i - 1, j).max(dp.at(i, j - 1)));
            }
        }
    }
    dp.at(pred.n_words, gold.n_words) as usize
}

pub fn rouge_l_score(pred: &Sequence, gold: &Sequence) -> f64 {
    let lcs_len = get_lcs(pred, gold) as f64;
    let pred_len = pred.n_words as f64;
    let gold_len = gold.n_words as f64;

    if pred_len == 0.0 || gold_len == 0.0 {
        return 0.0;
    }

    let precision = lcs_len / pred_len;
    let recall = lcs_len / gold_len;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Sequence;

    fn sample(text: &str) -> Sequence {
        Sequence::new(text)
    }

    #[test]
    fn test_lcs_identical_strings() {
        let pred = sample("the cat sat on the mat");
        let gold = sample("the cat sat on the mat");
        assert_eq!(get_lcs(&pred, &gold), pred.n_words);
    }

    #[test]
    fn test_lcs_no_overlap() {
        let pred = sample("cats sleep all day");
        let gold = sample("the quick brown fox");
        assert_eq!(get_lcs(&pred, &gold), 0);
    }

    #[test]
    fn test_lcs_partial_match() {
        let pred = sample("quick brown fox jumps");
        let gold = sample("the quick brown fox");
        assert_eq!(get_lcs(&pred, &gold), 3); // "quick", "brown", "fox"
    }

    #[test]
    fn test_lcs_order_matters() {
        let pred = sample("fox brown quick the");
        let gold = sample("the quick brown fox");
        assert_eq!(get_lcs(&pred, &gold), 1); // only one can be used in-order
    }

    #[test]
    fn test_rouge_l_perfect_match() {
        let pred = sample("the cat sat");
        let gold = sample("the cat sat");
        assert_eq!(rouge_l_score(&pred, &gold), 1.0);
    }

    #[test]
    fn test_rouge_l_empty_pred() {
        let pred = sample("");
        let gold = sample("the cat sat");
        assert_eq!(rouge_l_score(&pred, &gold), 0.0);
    }

    #[test]
    fn test_rouge_l_empty_gold() {
        let pred = sample("the cat sat");
        let gold = sample("");
        assert_eq!(rouge_l_score(&pred, &gold), 0.0);
    }

    #[test]
    fn test_rouge_l_partial_match() {
        let pred = sample("quick brown fox");
        let gold = sample("the quick brown fox");
        let score = rouge_l_score(&pred, &gold);
        assert!((score - 0.8571).abs() < 0.01); // LCS = 3, prec = 1.0, recall = 0.75 → F1 ≈ 0.857
    }

    #[test]
    fn test_rouge_l_no_match() {
        let pred = sample("cats eat cheese");
        let gold = sample("dogs chase balls");
        assert_eq!(rouge_l_score(&pred, &gold), 0.0);
    }
}

use crate::tokf1::token_f1_score;
use crate::utils::{DPMatrix, Sequence};

fn edit_distance(pred_words_vec: &Vec<String>, gold_words_vec: &Vec<String>) -> i16 {
    let n_words_pred = pred_words_vec.len();
    let n_words_gold = gold_words_vec.len();

    let mut dp = DPMatrix::new(n_words_pred + 1, n_words_gold + 1);

    // This part essentially does this:
    //     ""  s
    // "" [ 0  1 ]
    //  k [ 1  ? ]
    // Where ? are the values to still be filled
    // To fill them, we consider deletion operations,
    // insertion operations, or subsitution operations,
    // consider the cost of each, and then choose the lowest cost

    // For words, same concept.
    for i in 0..=n_words_pred {
        dp.set(i, 0, i as i16);
    }
    for j in 0..=n_words_gold {
        dp.set(0, j, j as i16);
    }

    let mut cost: i16 = 0;
    for i in 1..=n_words_pred {
        for j in 1..=n_words_gold {
            // Looking at a cell with ?
            // First case: do the words already match?
            // i-1 and j-1 here because the -1 starts at
            // the first word
            if pred_words_vec[i - 1] == gold_words_vec[j - 1] {
                cost = 0;
            } else {
                cost = 1;
            }

            let delete_cost = dp.at(i - 1, j) + 1;
            let insert_cost = dp.at(i, j - 1) + 1;
            let substitution_cost = dp.at(i - 1, j - 1) + cost;

            let min = delete_cost.min(insert_cost).min(substitution_cost);
            dp.set(i, j, min);
        }
    }
    dp.at(n_words_pred, n_words_gold)
}

pub fn fuzzy_match_score(pred: &Sequence, gold: &Sequence) -> f64 {
    let mut normalizing_constant = pred.word_vector.len().max(gold.word_vector.len());
    if normalizing_constant == 0 {
        normalizing_constant = 1;
    }

    let edit_distance = edit_distance(&pred.word_vector, &gold.word_vector) as f64;
    1.0 - (edit_distance / normalizing_constant as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::str_to_word_vec;

    #[test]
    fn test_levenshtein_distance_words() {
        let pred = "kitten sat on the mat";
        let gold = "sitting sat on mat";

        // Expected word-level edit distance:
        // kitten → sitting      => substitution
        // "the" → (deleted)     => deletion
        // Total = 2
        let distance = edit_distance(&str_to_word_vec(pred), &str_to_word_vec(gold));
        assert_eq!(distance, 2);
    }

    #[test]
    fn test_identical_sentences() {
        let pred = "the quick brown fox";
        let gold = "the quick brown fox";

        let distance = edit_distance(&str_to_word_vec(pred), &str_to_word_vec(gold));
        assert_eq!(distance, 0);
    }

    #[test]
    fn test_empty_input() {
        let pred = "";
        let gold = "hello world";

        let distance = edit_distance(&str_to_word_vec(pred), &str_to_word_vec(gold));
        assert_eq!(distance, 2);
    }

    #[test]
    fn identical_strings() {
        let score = fuzzy_match_score(
            &Sequence::new("the quick brown fox"),
            &Sequence::new("the quick brown fox"),
        );
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn completely_different_strings() {
        let score = fuzzy_match_score(
            &Sequence::new("cat dog mouse"),
            &Sequence::new("apple banana orange"),
        );
        assert_eq!(score, 0.0);
    }

    #[test]
    fn single_substitution() {
        let score = fuzzy_match_score(
            &Sequence::new("the quick brown fox"),
            &Sequence::new("the quick blue fox"),
        );
        // Only 1 word difference out of 4
        let expected = 1.0 - 1.0 / 4.0;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn insertion_case() {
        let score = fuzzy_match_score(
            &Sequence::new("quick brown fox"),
            &Sequence::new("the quick brown fox"),
        );
        // 1 word inserted: "the"
        let expected = 1.0 - 1.0 / 4.0;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn deletion_case() {
        let score = fuzzy_match_score(
            &Sequence::new("the quick brown fox"),
            &Sequence::new("quick brown fox"),
        );
        // 1 word deleted: "the"
        let expected = 1.0 - 1.0 / 4.0;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn reordering_is_penalized() {
        let score = fuzzy_match_score(
            &Sequence::new("the quick brown fox"),
            &Sequence::new("fox brown quick the"),
        );
        assert!(score == 0.0);
    }

    #[test]
    fn empty_vs_empty() {
        let score = fuzzy_match_score(&Sequence::new(""), &Sequence::new(""));
        assert_eq!(score, 1.0);
    }

    #[test]
    fn empty_vs_nonempty() {
        let score = fuzzy_match_score(
            &Sequence::new(""),
            &Sequence::new("some non empty sentence"),
        );
        assert_eq!(score, 0.0);

        let score = fuzzy_match_score(&Sequence::new("hello"), &Sequence::new(""));
        assert_eq!(score, 0.0);
    }
}

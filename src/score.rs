use crate::bleu::bleu_score;
use crate::fuzzy::fuzzy_match_score;
use crate::rouge::rouge_l_score;
use crate::tokf1::token_f1_score;
use crate::utils::Sequence;

pub async fn get_word_vec_ops_score(pred_batch: Vec<String>, gold_batch: Vec<String>) -> Vec<f64> {
    let mut results: Vec<PerformanceContext> = vec![];
    let zipped_contents = pred_batch.into_iter().zip(gold_batch.into_iter());
    for (pred, gold) in zipped_contents {
        results.push(PerformanceContext::from_str(pred, gold));
    }
    let tasks = results
        .into_iter()
        .map(|input| async move { input.get_unified_score() });
    futures::future::join_all(tasks).await
}

type ScoreFn = fn(&Sequence, &Sequence) -> f64;
const WEIGHTED_SCORE_FNS: &[(f64, ScoreFn)] = &[
    (1.0, bleu_score),
    (1.0, fuzzy_match_score),
    (1.0, rouge_l_score),
    (1.0, token_f1_score),
];

pub struct PerformanceContext {
    pred: Sequence,
    gold: Sequence,
    scores: &'static [(f64, ScoreFn)],
}

impl PerformanceContext {
    pub fn new(pred: Sequence, gold: Sequence) -> Self {
        return PerformanceContext {
            pred,
            gold,
            scores: WEIGHTED_SCORE_FNS,
        };
    }

    pub fn from_str(pred: String, gold: String) -> Self {
        let pred_seq = Sequence::new(pred.as_str());
        let gold_seq = Sequence::new(gold.as_str());
        return PerformanceContext {
            pred: pred_seq,
            gold: gold_seq,
            scores: WEIGHTED_SCORE_FNS,
        };
    }
    pub fn get_unified_score(&self) -> f64 {
        let mut raw_score = 0.0;

        for &(scalar, score_fn) in self.scores {
            raw_score += scalar * score_fn(&self.pred, &self.gold);
        }

        raw_score / self.scores.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Sequence;
    use tokio::runtime::Runtime;

    fn pc(pred: &str, gold: &str) -> PerformanceContext {
        PerformanceContext::new(Sequence::new(pred), Sequence::new(gold))
    }

    fn close_enough(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_unified_perfect_match() {
        let ctx = pc("the cat sat", "the cat sat");
        let score = ctx.get_unified_score();
        assert!(
            close_enough(score, 1.0, 0.01),
            "Expected ~4.0, got {}",
            score
        );
    }

    #[test]
    fn test_unified_no_match() {
        let ctx = pc("dogs bark loud", "the cat sat");
        let score = ctx.get_unified_score();
        assert!(
            close_enough(score, 0.0, 0.01),
            "Expected 0.0, got {}",
            score
        );
    }

    #[test]
    fn test_unified_partial_match() {
        let ctx = pc("cat sleeps", "the cat sat");
        let score = ctx.get_unified_score();
        assert!(
            score > 0.0 && score < 1.0,
            "Expected partial score, got {}",
            score
        );
    }

    #[test]
    fn test_unified_empty_prediction() {
        let ctx = pc("", "the cat sat");
        let score = ctx.get_unified_score();
        assert!(
            close_enough(score, 0.0, 0.01),
            "Expected 0.0 for empty pred, got {}",
            score
        );
    }

    #[test]
    fn test_unified_empty_gold() {
        let ctx = pc("the cat sat", "");
        let score = ctx.get_unified_score();
        assert!(
            close_enough(score, 0.0, 0.01),
            "Expected 0.0 for empty gold, got {}",
            score
        );
    }

    #[test]
    fn test_unified_token_scrambled() {
        let ctx = pc("sat cat the", "the cat sat");
        let score = ctx.get_unified_score();
        // Token F1 will be 0.25, others will penalize order
        assert!(
            score > 0.25 && score < 1.0,
            "Expected intermediate score, got {}",
            score
        );
    }

    #[test]
    fn test_unified_ranking() {
        let gold = "the cat sat";
        let s1 = pc("the cat sat", gold).get_unified_score(); // perfect
        let s2 = pc("cat sat", gold).get_unified_score(); // high
        let s3 = pc("cat", gold).get_unified_score(); // lower
        let s4 = pc("dog ran", gold).get_unified_score(); // very low

        assert!(s1 > s2 && s2 > s3 && s3 > s4, "Expected strict ranking");
    }

    #[test]
    fn test_batch_scorer() {
        let preds = vec![String::from("the cat sat"), String::from("a dog barked")];
        let golds = vec![
            String::from("the cat sat"),
            String::from("the dog barked loudly"),
        ];

        let rt = Runtime::new().expect("Failed to create async runtime");
        let scores = rt.block_on(get_word_vec_ops_score(preds, golds));

        assert_eq!(scores.len(), 2);
        for s in scores {
            println!("score: {}", s);
            assert!(s >= 0.0 && s <= 1.0);
        }
    }
}

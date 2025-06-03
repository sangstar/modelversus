use crate::bleu::bleu_score;
use crate::fuzzy::fuzzy_match_score;
use crate::rogue::rogue_l_score;
use crate::tokf1::token_f1_score;
use crate::utils::Sequence;

type ScoreFn = fn(&Sequence, &Sequence) -> f32;
const SCORERS: &[ScoreFn] = &[bleu_score, fuzzy_match_score, rogue_l_score, token_f1_score];

pub struct PerformanceContext {
    pred: Sequence,
    gold: Sequence,
    scores: &'static [ScoreFn],
}

impl PerformanceContext {
    pub fn new(pred: Sequence, gold: Sequence) -> Self {
        return PerformanceContext {
            pred,
            gold,
            scores: SCORERS,
        };
    }
    pub fn get_unified_score(&self) -> f32 {
        let mut raw_score = 0.0;
        for score in self.scores {
            raw_score += score(&self.pred, &self.gold);
        }
        raw_score / self.scores.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::Sequence;

    use super::*;

    fn pc(pred: &str, gold: &str) -> PerformanceContext {
        PerformanceContext::new(Sequence::new(pred), Sequence::new(gold))
    }

    fn close_enough(a: f32, b: f32, eps: f32) -> bool {
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
}

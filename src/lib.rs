mod bert_score;
mod bleu;
mod fuzzy;
mod rouge;
mod score;
mod tokf1;
mod utils;
// TODO: Can delegate N unique workers per M rows, each worker doing all the metrics calcs for each
//       row that they're assigned to, aggregating all in to averages. Then workers avg their avgs
//       Each worker can also delegate F subworkers for each metric calc. Experiment with this

use crate::bert_score::MODEL;
use crate::score::{get_word_vec_ops_score, PerformanceContext};
use crate::utils::Sequence;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use tokio::join;
use tokio::runtime::Runtime;

#[pyfunction]
fn score_batch(py: Python, preds: Vec<String>, golds: Vec<String>) -> PyResult<Vec<f64>> {
    let scores = py.allow_threads(|| {
        let rt = Runtime::new().expect("Failed to create async runtime");
        rt.block_on(get_word_vec_ops_score(preds, golds))
    });

    Ok(scores)
}

/// Takes a batch of pre-processed texts `preds`, a corresponding batch of pre-processed texts
/// `golds`, input ids and attention masks for each sequence from the two texts (before they were
/// pre-processed), and calculates, per sequence index i in the batch, a score of semantic
/// similarity between the text in preds[i] and golds[i].
///
/// Asynchronously (GIL-free) computes an aggregated score based on semantic similarity metrics
/// on word vectors with `get_word_vec_ops_scores`, while also using the input ids and attention
/// masks to compute batched cosine similarity scores between the original `preds` and `golds`
/// sequences by encoding the vectors with the final hidden state of a BERT model. These two
/// scores are then aggregated to produce the per-sequence similarity score.
///
/// # Arguments
///
/// * `py`: The Python marker token
/// * `preds`: A `list[str]` from a Python caller of pre-processed texts (stemmed, lowercased etc)
/// * `golds`: A `list[str]` from a Python caller of pre-processed texts, to compare with `preds`
/// * `pred_input_ids`: The original input IDs of each sequence in `preds` before pre-processing
/// * `pred_attention_mask`: The original attention_mask of each sequence in `preds`
/// * `gold_input_ids`: The original input IDs of each sequence in `golds`
/// * `gold_attention_mask`:The original attention_mask of each sequence in `golds`
///
/// returns: `Result<Vec<f64, Global>, PyErr>`
#[pyfunction]
fn get_unified_score(
    py: Python,
    preds: Vec<String>,
    golds: Vec<String>,
    pred_input_ids: PyReadonlyArray2<i64>,
    pred_attention_mask: PyReadonlyArray2<i64>,
    gold_input_ids: PyReadonlyArray2<i64>,
    gold_attention_mask: PyReadonlyArray2<i64>,
) -> PyResult<Vec<f64>> {

    // Python objects cannot be mutated without locking
    // the GIL, so copy and own Rust views of the data
    // to release it
    let pred_ids = pred_input_ids.as_array().to_owned();
    let pred_attn_mask = pred_attention_mask.as_array().to_owned();
    let gold_ids = gold_input_ids.as_array().to_owned();
    let gold_attn_mask = gold_attention_mask.as_array().to_owned();

    let scores = py.allow_threads(|| {
        // TODO: Don't make a new runtime each call; expensive
        let rt = Runtime::new().expect("Failed to create async runtime");
        let (result_1, result_2) = rt.block_on(async {
            join!(
                get_word_vec_ops_score(preds, golds),
                MODEL.bert_score(
                    pred_ids.view(),
                    pred_attn_mask.view(),
                    gold_ids.view(),
                    gold_attn_mask.view()
                ),
            )
        });

        // This result does a weighted elementwise addition
        // of the two scores, weighing the word_vec_ops_score
        // by half as much
        result_1
            .iter()
            .zip(result_2.iter())
            .map(|(x, y)| 0.5 * x + y)
            .collect()
    });

    Ok(scores)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_unified_score, m)?)?;
    Ok(())
}

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

// TODO: All pred and gold text here is presumed to be PRE-PROCESSED.
//       ie lowercased, stemmed, with stopwords removed

use crate::bert_score::MODEL;
use crate::score::{get_results_from_batch, PerformanceContext};
use crate::utils::Sequence;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use tokio::join;
use tokio::runtime::Runtime;

#[pyfunction]
fn score_batch(py: Python, preds: Vec<String>, golds: Vec<String>) -> PyResult<Vec<f64>> {
    let scores = py.allow_threads(|| {
        let rt = Runtime::new().expect("Failed to create async runtime");
        rt.block_on(get_results_from_batch(preds, golds))
    });

    Ok(scores)
}

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
    // Copy the py objects to Rust data so that
    // we can release the GIL
    let pred_ids = pred_input_ids.as_array().to_owned();
    let pred_attn_mask = pred_attention_mask.as_array().to_owned();
    let gold_ids = gold_input_ids.as_array().to_owned();
    let gold_attn_mask = gold_attention_mask.as_array().to_owned();

    let scores = py.allow_threads(|| {
        let rt = Runtime::new().expect("Failed to create async runtime");
        let (result_1, result_2) = rt.block_on(async {
            join!(
                get_results_from_batch(preds, golds),
                MODEL.bert_score(
                    pred_ids.view(),
                    pred_attn_mask.view(),
                    gold_ids.view(),
                    gold_attn_mask.view()
                ),
            )
        });
        dbg!(&result_1);
        dbg!(&result_2);
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

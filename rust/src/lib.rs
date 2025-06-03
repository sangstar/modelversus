mod bleu;
mod fuzzy;
mod rogue;
mod score;
mod tokf1;
mod utils;
// TODO: Can delegate N unique workers per M rows, each worker doing all the metrics calcs for each
//       row that they're assigned to, aggregating all in to averages. Then workers avg their avgs
//       Each worker can also delegate F subworkers for each metric calc. Experiment with this

// TODO: All pred and gold text here is presumed to be PRE-PROCESSED.
//       ie lowercased, stemmed, with stopwords removed

use pyo3::prelude::*;
use crate::score::{PerformanceContext, get_results_from_batch};
use crate::utils::Sequence;
use tokio::runtime::Runtime;

#[pyfunction]
fn score_batch(preds: Vec<String>, golds: Vec<String>) -> PyResult<Vec<f32>> {
    let rt = Runtime::new().expect("Failed to create async runtime");
    let scores = rt.block_on(get_results_from_batch(preds, golds));

    Ok(scores)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_batch, m)?)?;
    Ok(())
}

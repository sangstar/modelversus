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

/// Dummy function; adds a and b and sets result as string
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Dummy function for module scaffolding
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

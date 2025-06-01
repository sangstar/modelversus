mod bleu;
mod fuzzy;
mod rogue;
mod tokf1;

// TODO: Can delegate N unique workers per M rows, each worker doing all the metrics calcs for each
//       row that they're assigned to, aggregating all in to averages. Then workers avg their avgs
//       Each worker can also delegate F subworkers for each metric calc. Experiment with this

use pyo3::prelude::*;

/// Dummy function; adds a and b and sets result as string
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Dummy function for module scaffolding
#[pymodule]
fn _rustbindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

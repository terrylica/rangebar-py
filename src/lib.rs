use pyo3::prelude::*;

/// A minimal Python module to verify build works
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

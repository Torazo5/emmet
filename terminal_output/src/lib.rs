use pyo3::prelude::*;
use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
};
use std::io::{stdout, Write};

#[pyfunction]
fn display_styled_output(text: &str, color: &str) -> PyResult<()> {
    let mut stdout = stdout();
    let color = match color {
        "red" => Color::Red,
        "green" => Color::Green,
        "blue" => Color::Blue,
        _ => Color::White,
    };

    execute!(
        stdout,
        SetForegroundColor(color),
        Print(text),
        ResetColor
    )?;

    Ok(())
}

#[pymodule]
fn terminal_output(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(display_styled_output, m)?)?;
    Ok(())
}



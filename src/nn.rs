//! Общее для нейросетей на видеокарте: одно окружение ONNX Runtime и сессии с DirectML.

use anyhow::{Result, anyhow};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::{Mutex, OnceLock};

/// Выходы сети: форма и данные каждого.
pub type Outputs = Vec<(Vec<usize>, Vec<f32>)>;

/// `PHOTOSTAB_CPU_NN=1` — считать сети детекции на процессоре (для сравнения и отладки).
pub fn cpu_only() -> bool {
    std::env::var_os("PHOTOSTAB_CPU_NN").is_some()
}

pub fn env() -> Result<&'static Environment> {
    static E: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    E.get_or_init(|| ort::init().with_name("photostab").build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow!("ONNX Runtime: {e}"))
}

/// Сессия на видеокарте (DirectML). Если DirectML не поднялся — ошибка, а не тихий откат на CPU:
/// у вызывающего есть свой запасной путь.
pub fn gpu_session(model: &[u8]) -> Result<Session> {
    Session::builder(env()?)
        .map_err(|e| anyhow!("{e}"))?
        .with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()])
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_memory(model)
        .map_err(|e| anyhow!("{e}"))
}

/// Один прогон сети с одним входом f32.
pub fn run(session: &Mutex<Session>, shape: [usize; 4], input: &[f32]) -> Result<Outputs> {
    let mut s = session.lock().map_err(|_| anyhow!("сеть занята"))?;
    let t = TensorRef::from_array_view((shape, input)).map_err(|e| anyhow!("{e}"))?;
    let out = s.run(ort::inputs![t]).map_err(|e| anyhow!("{e}"))?;
    out.iter()
        .map(|(_, v)| {
            let (sh, d) = v.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
            Ok((sh.iter().map(|&x| x.max(0) as usize).collect(), d.to_vec()))
        })
        .collect()
}

/// Фиксирует форму входа модели и стирает записанные формы промежуточных тензоров и выходов —
/// они посчитаны под другой размер входа. DirectML быстрее всего с заранее известными формами.
pub fn with_input_shape(model: &[u8], shape: &[i64]) -> Result<Vec<u8>> {
    use prost::Message;
    use tract_onnx::pb::tensor_shape_proto::dimension::Value as Dim;
    use tract_onnx::pb::type_proto::Value as Ty;
    let mut m = tract_onnx::pb::ModelProto::decode(model).map_err(|e| anyhow!("ONNX: {e}"))?;
    let g = m.graph.as_mut().ok_or_else(|| anyhow!("ONNX: нет графа"))?;
    g.value_info.clear();
    let consts: std::collections::HashSet<String> = g.initializer.iter().map(|t| t.name.clone()).collect();
    let input = g.input.iter_mut().find(|i| !consts.contains(&i.name)).ok_or_else(|| anyhow!("ONNX: нет входа"))?;
    let Some(Ty::TensorType(t)) = input.r#type.as_mut().and_then(|t| t.value.as_mut()) else {
        return Err(anyhow!("ONNX: вход не тензор"));
    };
    let dims = &mut t.shape.get_or_insert_with(Default::default).dim;
    if dims.len() != shape.len() {
        return Err(anyhow!("ONNX: у входа {} измерений, а не {}", dims.len(), shape.len()));
    }
    for (d, &v) in dims.iter_mut().zip(shape) {
        d.value = Some(Dim::DimValue(v));
    }
    for out in g.output.iter_mut() {
        if let Some(Ty::TensorType(t)) = out.r#type.as_mut().and_then(|t| t.value.as_mut()) {
            if let Some(s) = t.shape.as_mut() {
                for d in s.dim.iter_mut() {
                    d.value = None;
                }
            }
        }
    }
    Ok(m.encode_to_vec())
}

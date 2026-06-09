use mlxforge::{abi_version, version, Engine, Sampling};

#[test]
fn reports_version_and_abi() {
    assert!(!version().is_empty());
    assert_eq!(abi_version(), 1);
}

#[test]
fn generate_and_embed_if_model_present() {
    let dir = match std::env::var("MLXFORGE_MODEL_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("MLXFORGE_MODEL_DIR not set; skipping model test");
            return;
        }
    };
    let engine = Engine::load(&dir).expect("load");

    let mut s = Sampling::greedy();
    s.max_tokens = 16;
    let out = engine
        .chat(&[("user", "What is the capital of France?")], &s)
        .expect("chat");
    assert!(!out.is_empty(), "expected non-empty generation");

    // Embeddings: unit-normalized + semantic ordering.
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let a = engine.embed("The cat sat on the warm mat.", 0).expect("embed a");
    let b = engine.embed("A kitten rests on a soft rug.", 0).expect("embed b");
    let c = engine
        .embed("The stock market fell sharply amid economic fears.", 0)
        .expect("embed c");
    assert!((dot(&a, &a) - 1.0).abs() < 0.02, "embedding must be unit-normalized");
    assert!(dot(&a, &b) > dot(&a, &c), "semantically similar text should score higher");
}

#[test]
fn constrained_json_if_model_present() {
    let dir = match std::env::var("MLXFORGE_MODEL_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let engine = Engine::load(&dir).expect("load");
    let mut s = Sampling::greedy();
    s.max_tokens = 96;
    s.json_schema = Some("json".to_string());
    let out = engine
        .chat(&[("user", "Describe a city as a JSON object.")], &s)
        .expect("chat");
    // A clean stop yields parseable JSON; otherwise it is a valid prefix.
    assert!(out.trim_start().starts_with(['{', '[', '"']) || out.trim_start().starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == 't' || c == 'f' || c == 'n'));
}

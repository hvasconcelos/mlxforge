// Minimal CLI: stream-free chat completion, plus a quick embedding.
//   cargo run --example chat -- <model-spec> [prompt]
use mlxforge::{Engine, Sampling};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = match args.get(1) {
        Some(s) => s.clone(),
        None => {
            eprintln!("usage: chat <model-spec> [prompt]");
            std::process::exit(2);
        }
    };
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "What is the capital of France?".into());

    let engine = Engine::load(&spec).expect("failed to load model");
    eprintln!("loaded: {}", engine.model_name());

    let mut s = Sampling::greedy();
    s.max_tokens = 64;
    let reply = engine.chat(&[("user", prompt.as_str())], &s).expect("chat failed");
    println!("{reply}");

    let v = engine.embed("The cat sat on the mat.", 0).expect("embed failed");
    eprintln!("embedding dim = {}", v.len());
}

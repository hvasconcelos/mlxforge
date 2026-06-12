#include "inspect/schematic_html.h"

#include <cstring>
#include <stdexcept>

namespace mlxforge::inspect {

namespace {

// Splice point for the schema JSON inside the data <script> tag. Replaced with
// find+replace (not fmt) so the template can contain braces freely.
constexpr const char* kMarker = "__MLXFORGE_SCHEMA_JSON__";

// JSON-in-script-tag hardening: every '<' becomes the \\u003c escape so a
// hostile tensor name ("</script>...") can never terminate the data element. JSON.parse restores
// the original characters; the JS side renders all data via textContent.
std::string escape_json_for_html(const std::string& json) {
  std::string out;
  out.reserve(json.size());
  for (char c : json) {
    if (c == '<') {
      out += "\\u003c";
    } else {
      out += c;
    }
  }
  return out;
}

// The page template: a pastel "drafting paper" single-file infographic.
// Inline CSS/JS only — no external fonts, scripts or styles — so the file
// renders offline. Adjacent raw literals concatenate.
const char* kTemplate =
    R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>model schematic · mlxforge</title>
<style>
:root{
  --bg:#f7f5ef; --panel:#fdfcf8; --panel2:#f2efe7; --line:#e5dfd2; --line2:#d6cebd;
  --ink:#4d5a66; --dim:#8a93a0; --faint:#b4bbc5; --bright:#2c3742;
  --cyan:#5fb6c9; --amber:#e8a04c; --green:#7fbf8a; --violet:#a08fd8;
  --pink:#e08cb0; --teal:#5fbfa8; --blue:#7fa3dd; --gray:#9aa5b1;
}
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{
  font-family:"SF Mono","Menlo",ui-monospace,Monaco,"Cascadia Mono",monospace;
  color:var(--ink); min-height:100vh; padding:0 0 64px;
  background:
    radial-gradient(1100px 520px at 75% -8%, rgba(95,182,201,.12), transparent 62%),
    radial-gradient(900px 500px at -10% 30%, rgba(160,143,216,.09), transparent 60%),
    linear-gradient(rgba(95,182,201,.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(95,182,201,.06) 1px, transparent 1px),
    var(--bg);
  background-size:auto,auto,30px 30px,30px 30px,auto;
}
body::after{ /* faint paper grain for the print feel */
  content:""; position:fixed; inset:0; pointer-events:none; z-index:9;
  background:repeating-linear-gradient(0deg, rgba(44,55,66,.012) 0 1px, transparent 1px 3px);
}
main,header.page{max-width:1180px;margin:0 auto;padding:0 28px}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
header.page,section{animation:rise .55s cubic-bezier(.2,.7,.2,1) both}
header.page{animation-delay:.03s}
section:nth-of-type(1){animation-delay:.10s}
section:nth-of-type(2){animation-delay:.17s}
section:nth-of-type(3){animation-delay:.24s}
section:nth-of-type(4){animation-delay:.31s}

header.page{padding-top:44px}
.eyebrow{color:var(--cyan);font-size:.72rem;letter-spacing:.32em;text-transform:uppercase}
.eyebrow::before{content:"▞ ";color:var(--amber)}
h1{color:var(--bright);font-size:2.05rem;font-weight:700;letter-spacing:-.01em;
   margin:10px 0 14px;word-break:break-word}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:22px}
.chip{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;padding:4px 10px;
      border:1px solid var(--line2);border-radius:3px;color:var(--dim);background:rgba(255,255,255,.65)}
.chip.accent{color:var(--cyan);border-color:rgba(95,182,201,.55);background:rgba(95,182,201,.10)}
.chip.warm{color:var(--amber);border-color:rgba(232,160,76,.55);background:rgba(232,160,76,.10)}
.bigstats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
          background:var(--line);border:1px solid var(--line);margin-bottom:8px}
.bigstat{background:var(--panel);padding:14px 16px}
.bigstat b{display:block;color:var(--bright);font-size:1.28rem;font-weight:600;margin-top:4px}
.bigstat span{font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}
.bigstat small{color:var(--faint);font-size:.72rem}

section{margin-top:40px}
h2{color:var(--bright);font-size:.85rem;font-weight:600;letter-spacing:.26em;
   text-transform:uppercase;margin-bottom:16px;display:flex;align-items:center;gap:12px}
h2::before{content:"";width:18px;height:1px;background:var(--amber)}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
h3{color:var(--dim);font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;margin:22px 0 10px}

.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}
.panel{border:1px solid var(--line);background:var(--panel);padding:18px 20px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:7px 22px;font-size:.84rem}
.kv dt{color:var(--dim)}
.kv dd{color:var(--bright);text-align:right}
.kv dd small{color:var(--dim)}

/* ---- forward-pass diagram ---- */
#flow{display:flex;flex-direction:column;align-items:center;padding:26px 12px 8px;
      border:1px solid var(--line);background:
        linear-gradient(rgba(95,182,201,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(95,182,201,.05) 1px, transparent 1px),
        var(--panel);
      background-size:15px 15px,15px 15px,auto}
.node{border:1px solid var(--line2);background:var(--panel2);padding:9px 18px;text-align:center;
      border-left:3px solid var(--gray);min-width:230px;position:relative}
.node b{color:var(--bright);font-weight:600;font-size:.88rem}
.node i{display:block;font-style:normal;color:var(--dim);font-size:.74rem;margin-top:3px}
.conn{display:flex;flex-direction:column;align-items:center;position:relative;height:30px}
.conn::before{content:"";width:1px;flex:1;background:var(--faint)}
.conn::after{content:"";border:4px solid transparent;border-top-color:var(--faint);margin-top:-1px}
.conn span{position:absolute;left:calc(50% + 12px);top:50%;transform:translateY(-58%);
           white-space:nowrap;color:var(--faint);font-size:.7rem}
.block{border:1px dashed var(--line2);padding:20px 22px 16px;position:relative;margin:0 6px;
       display:flex;flex-direction:column;align-items:center;background:rgba(253,252,248,.7)}
.block>.badge{position:absolute;top:-11px;right:14px;background:var(--bg);border:1px solid var(--amber);
        color:var(--amber);font-size:.72rem;padding:2px 10px;letter-spacing:.08em}
.block>.bname{position:absolute;top:-10px;left:14px;background:var(--bg);color:var(--dim);
        font-size:.7rem;padding:1px 8px;letter-spacing:.18em;text-transform:uppercase}
.row{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.proj{border:1px solid var(--line2);background:var(--panel);padding:7px 12px;text-align:center;min-width:108px}
.proj b{display:block;color:var(--bright);font-size:.78rem;font-weight:600}
.proj i{display:block;font-style:normal;color:var(--cyan);font-size:.72rem;margin-top:2px}
.proj em{display:block;font-style:normal;color:var(--faint);font-size:.64rem;margin-top:2px}
.proj .qt{display:inline-block;margin-top:3px;font-size:.6rem;color:var(--amber);
          border:1px solid rgba(232,160,76,.5);padding:0 5px;border-radius:2px}
.rownote{color:var(--dim);font-size:.7rem;margin-top:8px;text-align:center}
.hsplit{display:flex;gap:26px;flex-wrap:wrap;justify-content:center;align-items:stretch}
.expgrid{display:grid;gap:2px;margin:10px auto 2px;width:max-content}
.expgrid i{width:7px;height:7px;background:var(--line2);display:block}
.expgrid i.hot{background:var(--amber);box-shadow:0 0 6px rgba(232,160,76,.45)}
.vlane{border:1px solid rgba(95,191,168,.45);background:rgba(95,191,168,.10);padding:14px 18px;
       margin-bottom:2px;text-align:center}
.vlane b{color:var(--teal);font-size:.8rem;letter-spacing:.14em;text-transform:uppercase}
.vlane .row{margin-top:10px}

/* ---- distribution ---- */
.stack{display:flex;height:34px;border:1px solid var(--line);background:var(--panel);overflow:hidden}
.stack i{display:block;height:100%;width:0;transition:width .9s cubic-bezier(.2,.7,.2,1)}
.legend{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:14px;font-size:.8rem}
.legend .it{display:flex;align-items:center;gap:8px}
.legend .sw{width:10px;height:10px;border-radius:2px}
.legend b{color:var(--bright);font-weight:600}
.legend span{color:var(--dim)}
#layerbars{display:flex;align-items:flex-end;gap:2px;height:96px;padding:10px 12px 6px;
           border:1px solid var(--line);background:var(--panel)}
#layerbars i{flex:1;min-width:2px;background:var(--cyan);opacity:.78;height:0;
             transition:height .7s cubic-bezier(.2,.7,.2,1)}
#layerbars i:hover{opacity:1}
.axis{display:flex;justify-content:space-between;color:var(--faint);font-size:.68rem;margin-top:6px}

/* ---- explorer ---- */
.toolbar{display:flex;gap:14px;align-items:center;margin-bottom:14px}
#search{flex:1;max-width:430px;background:var(--panel);border:1px solid var(--line2);color:var(--bright);
        font:inherit;font-size:.84rem;padding:8px 12px;outline:none}
#search:focus{border-color:var(--cyan)}
#search::placeholder{color:var(--faint)}
#tcount{color:var(--dim);font-size:.76rem}
details.grp{border:1px solid var(--line);border-bottom:none;background:var(--panel)}
details.grp:last-child{border-bottom:1px solid var(--line)}
details.grp>summary{display:flex;align-items:center;gap:14px;padding:9px 16px;cursor:pointer;
                    list-style:none;font-size:.82rem;user-select:none}
details.grp>summary::-webkit-details-marker{display:none}
details.grp>summary::before{content:"▸";color:var(--faint);font-size:.72rem;width:12px}
details[open].grp>summary::before{content:"▾";color:var(--cyan)}
details.grp>summary:hover{background:var(--panel2)}
.gtitle{color:var(--bright);font-weight:600;min-width:88px}
.gkind{font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;padding:1px 7px;border-radius:2px;
       border:1px solid var(--line2);color:var(--dim)}
.gmeta{color:var(--dim);font-size:.74rem;flex:1}
.gshare{width:140px;height:5px;background:var(--line);border-radius:2px;overflow:hidden}
.gshare i{display:block;height:100%;background:var(--cyan);width:0;
          transition:width .8s cubic-bezier(.2,.7,.2,1)}
table.tt{width:100%;border-collapse:collapse;font-size:.78rem}
table.tt td{padding:5px 16px;border-top:1px solid var(--line);white-space:nowrap}
table.tt tr:hover td{background:rgba(95,182,201,.08)}
td.tname{color:var(--ink);max-width:380px;overflow:hidden;text-overflow:ellipsis}
td.tname span{color:var(--faint)}
td.tshape{color:var(--cyan)}
td.tdtype{color:var(--dim)}
td.tquant span{font-size:.64rem;color:var(--amber);border:1px solid rgba(232,160,76,.5);
               padding:0 5px;border-radius:2px}
td.tnum{text-align:right;color:var(--bright)}
td.tbar{width:120px}
td.tbar i{display:block;height:4px;background:linear-gradient(90deg,var(--cyan),var(--violet));
          border-radius:2px;min-width:1px}
footer{max-width:1180px;margin:54px auto 0;padding:14px 28px 0;border-top:1px solid var(--line);
       color:var(--faint);font-size:.72rem;letter-spacing:.06em}
@media (max-width:700px){html{font-size:13.5px}.gshare{display:none}}
</style>
</head>
<body>
<header class="page">
  <div class="eyebrow">mlxforge · model schematic</div>
  <h1 id="mname"></h1>
  <div class="chips" id="chips"></div>
  <div class="bigstats" id="bigstats"></div>
</header>
<main>
  <section>
    <h2>architecture</h2>
    <div class="grid2" id="archpanels"></div>
  </section>
  <section>
    <h2>forward pass</h2>
    <div id="flow"></div>
  </section>
  <section>
    <h2>parameter distribution</h2>
    <div class="stack" id="stackbar"></div>
    <div class="legend" id="legend"></div>
    <h3 id="layerbars-title">per-layer parameters</h3>
    <div id="layerbars"></div>
    <div class="axis" id="layeraxis"></div>
  </section>
  <section>
    <h2>tensor explorer</h2>
    <div class="toolbar">
      <input id="search" type="search" placeholder="filter tensors… (name / dtype / quant)" autocomplete="off">
      <span id="tcount"></span>
    </div>
    <div id="groups"></div>
  </section>
</main>
<footer>generated by <b>mlxforge-cli schematic</b> — metadata-only read of the checkpoint
(safetensors headers / GGUF tensor directory); no weights were loaded.</footer>
<script id="schema-data" type="application/json">__MLXFORGE_SCHEMA_JSON__</script>
)HTML"
    R"HTML(<script>
"use strict";
const D = JSON.parse(document.getElementById("schema-data").textContent);
const COLORS = {embed:"#a08fd8", attn:"#5fb6c9", linear_attn:"#7fa3dd", mlp:"#7fbf8a",
                moe:"#e8a04c", norm:"#9aa5b1", lm_head:"#e08cb0", vision:"#5fbfa8",
                other:"#8a93a0"};
const q = s => document.querySelector(s);
function el(tag, cls, text){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const fmtP = n => n >= 1e9 ? (n/1e9).toFixed(2)+" B" : n >= 1e6 ? (n/1e6).toFixed(1)+" M"
              : n >= 1e3 ? (n/1e3).toFixed(1)+" K" : String(n);
const fmtB = n => n >= 2**30 ? (n/2**30).toFixed(2)+" GiB" : n >= 2**20 ? (n/2**20).toFixed(1)+" MiB"
              : n >= 1024 ? (n/1024).toFixed(1)+" KiB" : n+" B";
const fmtI = n => Number(n).toLocaleString("en-US");
const shapeStr = s => "[" + s.join(" × ") + "]";
const H = D.header, A = D.arch, DV = D.derived;

/* ---------- header ---------- */
document.title = H.name + " · mlxforge schematic";
q("#mname").textContent = H.name;
{
  const chips = q("#chips");
  const add = (t, cls) => chips.appendChild(el("span", "chip " + (cls||""), t));
  add(H.family, "accent");
  add(H.format);
  add(H.quant, "warm");
  if (H.tied_embeddings) add("tied embeddings");
  if (A.qk_norm) add("qk-norm");
  if (A.moe) add("mixture of experts");
  if (A.hybrid) add("hybrid attention");
  if (A.vision) add("vision-language");
}
{
  const bs = q("#bigstats");
  const add = (label, value, sub) => {
    const d = el("div", "bigstat");
    d.appendChild(el("span", "", label));
    d.appendChild(el("b", "", value));
    if (sub) d.appendChild(el("small", "", sub));
    bs.appendChild(d);
  };
  add("parameters", fmtP(H.params), fmtI(H.params));
  add("on disk", fmtB(H.bytes), H.quant);
  add("context", fmtI(H.context_length), "tokens");
  add("vocab", fmtI(H.vocab), "entries");
  add("kv cache", fmtB(DV.kv_bytes_per_token) + " /tok",
      fmtB(DV.kv_bytes_per_token * H.context_length) + " @ full ctx");
}

/* ---------- architecture panels ---------- */
function panel(title, pairs){
  const p = el("div", "panel");
  p.appendChild(el("h3", "", title));
  const dl = el("dl", "kv");
  for (const [k, v, sub] of pairs){
    if (v === undefined || v === null || v === "") continue;
    dl.appendChild(el("dt", "", k));
    const dd = el("dd", "", String(v));
    if (sub) { dd.appendChild(document.createTextNode(" ")); dd.appendChild(el("small", "", sub)); }
    dl.appendChild(dd);
  }
  p.appendChild(dl);
  return p;
}
{
  const root = q("#archpanels");
  const core = [
    ["hidden size", fmtI(A.hidden)],
    ["layers", fmtI(A.n_layers),
      A.hybrid ? `${DV.n_full_attn_layers} full + ${A.hybrid.n_linear_layers} linear`
      : A.moe && A.moe.n_moe_layers < A.n_layers ? `${A.moe.n_moe_layers} MoE` : ""],
    ["attention heads", `${A.n_heads} q / ${A.n_kv_heads} kv`, `GQA ${A.gqa_ratio}:1`],
    ["head dim", fmtI(A.head_dim)],
    ["mlp width", fmtI(A.intermediate_size)],
    ["rope", "θ " + fmtI(A.rope.theta),
      A.rope.type !== "none" ? A.rope.type + (A.rope.factor ? " ×" + A.rope.factor : "") : ""],
  ];
  root.appendChild(panel("core", core));
  if (A.moe) {
    root.appendChild(panel("mixture of experts", [
      ["experts / layer", fmtI(A.moe.experts)],
      ["active per token", A.moe.top_k, `${(100 * A.moe.top_k / A.moe.experts).toFixed(1)}% of experts`],
      ["expert width", fmtI(A.moe.moe_intermediate)],
      ["moe layers", `${A.moe.n_moe_layers} of ${A.n_layers}`],
    ]));
  }
  if (A.hybrid) {
    root.appendChild(panel("hybrid linear attention", [
      ["full attn every", `${A.hybrid.full_attention_interval} layers`],
      ["linear layers", A.hybrid.n_linear_layers, "gated DeltaNet"],
      ["linear k/v heads", `${A.hybrid.linear_num_key_heads} / ${A.hybrid.linear_num_value_heads}`],
      ["linear head dims", `${A.hybrid.linear_key_head_dim} / ${A.hybrid.linear_value_head_dim}`],
      ["conv kernel", A.hybrid.conv_kernel],
    ]));
  }
  if (A.vision) {
    root.appendChild(panel("vision tower (ViT)", [
      ["blocks", A.vision.depth],
      ["hidden / heads", `${fmtI(A.vision.hidden)} / ${A.vision.num_heads}`],
      ["patch", `${A.vision.patch_size}px`, `merge ${A.vision.spatial_merge_size}×${A.vision.spatial_merge_size} → 1 token`],
      ["output dim", fmtI(A.vision.out_hidden_size), "= LLM hidden"],
      ["deepstack taps", (A.vision.deepstack_indexes || []).join(", ")],
    ]));
  }
  root.appendChild(panel("kv cache / token", [
    ["fp16", fmtB(DV.kv_bytes_per_token), `${DV.n_full_attn_layers} attn layers`],
    ["kv-bits 8", fmtB(DV.kv_bytes_per_token_kv8), "approx"],
    ["kv-bits 4", fmtB(DV.kv_bytes_per_token_kv4), "approx"],
    ["@ 4k tokens", fmtB(DV.kv_bytes_per_token * 4096), "fp16"],
    A.hybrid ? ["linear layers", "fixed-size recurrent state", "no per-token KV"] : ["", ""],
  ]));
}
</script>
)HTML"
    R"HTML(<script>
"use strict";
/* ---------- forward-pass diagram ---------- */
(() => {
  const D2 = D, A2 = D2.arch, H2 = D2.header;
  const flow = q("#flow");
  const mms = D2.derived.decode_matmuls || [];
  const byPrefix = p => mms.filter(m => m.name.startsWith(p));
  const first = (arr, name) => arr.find(m => m.name.endsWith(name));

  function node(title, sub, color){
    const n = el("div", "node");
    n.style.borderLeftColor = color || "var(--gray)";
    n.appendChild(el("b", "", title));
    if (sub) n.appendChild(el("i", "", sub));
    return n;
  }
  function conn(label){
    const c = el("div", "conn");
    if (label) c.appendChild(el("span", "", label));
    return c;
  }
  function projBox(m, color){
    const p = el("div", "proj");
    p.style.borderTopColor = color || "var(--line2)";
    p.appendChild(el("b", "", m.name.split(".").pop()));
    p.appendChild(el("i", "", fmtI(m.in) + " → " + fmtI(m.out)));
    if (m.note) p.appendChild(el("em", "", m.note));
    if (m.quant) p.appendChild(el("span", "qt", m.quant));
    return p;
  }
  function row(items){
    const r = el("div", "row");
    items.forEach(i => r.appendChild(i));
    return r;
  }
  function expertsGrid(total, hot){
    const cap = Math.min(total, 256);
    const cols = Math.min(32, Math.ceil(Math.sqrt(cap) * 2));
    const g = el("div", "expgrid");
    g.style.gridTemplateColumns = `repeat(${cols}, 7px)`;
    for (let i = 0; i < cap; ++i)
      g.appendChild(el("i", i < hot ? "hot" : ""));
    return g;
  }

  // Attention sub-assembly of one decoder block (full softmax attention):
  // q/k/v project IN PARALLEL from the same normed input, the attention
  // computation runs on their outputs, and only then does o_proj map the
  // concatenated heads back to the residual stream.
  function attnRows(container){
    const attn = byPrefix("self_attn.");
    if (!attn.length) return;
    const qkv = attn.filter(m => !m.name.endsWith("o_proj"));
    const o = attn.find(m => m.name.endsWith("o_proj"));
    container.appendChild(row(qkv.map(m => projBox(m, "var(--cyan)"))));
    container.appendChild(el("div", "rownote", "in parallel, from the same input"));
    container.appendChild(conn("k,v → kv cache"));
    container.appendChild(node("scaled dot-product attention",
      `softmax(q·kᵀ/√${A2.head_dim})·v · ${A2.n_heads} heads × ${A2.head_dim}` +
      (A2.gqa_ratio > 1 ? ` · GQA ${A2.gqa_ratio}:1` : "") +
      (A2.qk_norm ? " · qk-norm" : ""), "var(--cyan)"));
    if (o){
      container.appendChild(conn("concat heads"));
      container.appendChild(row([projBox(o, "var(--cyan)")]));
    }
  }
  // Dense MLP sub-assembly: gate and up project IN PARALLEL, their outputs
  // combine elementwise, then down projects back to the residual stream.
  function denseMlpRows(container, mods, color, widthNote){
    const gateUp = mods.filter(m => !m.name.endsWith("down_proj"));
    const down = mods.find(m => m.name.endsWith("down_proj"));
    container.appendChild(row(gateUp.map(m => projBox(m, color))));
    container.appendChild(el("div", "rownote", "in parallel, from the same input"));
    container.appendChild(conn());
    container.appendChild(node("silu(gate) ⊙ up", "elementwise gate · " + widthNote, color));
    if (down){
      container.appendChild(conn());
      container.appendChild(row([projBox(down, color)]));
    }
  }
  // MLP / MoE sub-assembly.
  function mlpRows(container){
    const sw = byPrefix("mlp.switch_mlp."), router = first(mms, "mlp.gate");
    if (sw.length){
      if (router)
        container.appendChild(row([projBox({...router, name:"router", note:"scores all experts"}, "var(--amber)")]));
      container.appendChild(conn("top-" + A2.moe.top_k + " select"));
      container.appendChild(expertsGrid(A2.moe.experts, A2.moe.top_k));
      container.appendChild(el("div", "rownote",
        `${A2.moe.top_k} of ${A2.moe.experts} experts run per token`));
      container.appendChild(conn());
      denseMlpRows(container, sw, "var(--amber)", "width " + fmtI(A2.moe.moe_intermediate));
    } else {
      const mlp = byPrefix("mlp.").filter(m => !m.name.includes("switch_mlp"));
      if (mlp.length)
        denseMlpRows(container, mlp, "var(--green)", "SwiGLU width " + fmtI(A2.intermediate_size));
    }
  }
  function fullBlock(badge, name){
    const b = el("div", "block");
    b.appendChild(el("span", "badge", badge));
    b.appendChild(el("span", "bname", name));
    b.appendChild(node("rmsnorm", "input_layernorm", "var(--gray)"));
    b.appendChild(conn());
    attnRows(b);
    b.appendChild(conn("+ residual"));
    b.appendChild(node("rmsnorm", "post_attention_layernorm", "var(--gray)"));
    b.appendChild(conn());
    mlpRows(b);
    return b;
  }
  function linearBlock(badge){
    const b = el("div", "block");
    b.appendChild(el("span", "badge", badge));
    b.appendChild(el("span", "bname", "linear block"));
    b.appendChild(node("rmsnorm", "input_layernorm", "var(--gray)"));
    b.appendChild(conn());
    const lin = byPrefix("linear_attn.");
    if (lin.length){
      b.appendChild(row(lin.map(m => projBox(m, "var(--blue)"))));
      b.appendChild(el("div", "rownote",
        `gated DeltaNet · ${A2.hybrid.linear_num_key_heads}k/${A2.hybrid.linear_num_value_heads}v heads · conv ${A2.hybrid.conv_kernel}`));
    } else {
      b.appendChild(node("gated DeltaNet", "linear attention", "var(--blue)"));
    }
    b.appendChild(conn("+ residual"));
    b.appendChild(node("rmsnorm", "post_attention_layernorm", "var(--gray)"));
    b.appendChild(conn());
    mlpRows(b);
    return b;
  }

  // Vision lane (Qwen3-VL): image -> ViT -> merger -> tokens into the decoder.
  if (A2.vision){
    const v = el("div", "vlane");
    v.appendChild(el("b", "", "vision tower"));
    v.appendChild(row([
      projBox({name:"patch_embed", in:A2.vision.patch_size*A2.vision.patch_size*3,
               out:A2.vision.hidden, note:A2.vision.patch_size+"px patches"}, "var(--teal)"),
      projBox({name:"ViT × "+A2.vision.depth, in:A2.vision.hidden, out:A2.vision.hidden,
               note:A2.vision.num_heads+" heads"}, "var(--teal)"),
      projBox({name:"merger", in:A2.vision.hidden*A2.vision.spatial_merge_size**2,
               out:A2.vision.out_hidden_size,
               note:A2.vision.spatial_merge_size+"×"+A2.vision.spatial_merge_size+" → 1"}, "var(--teal)"),
    ]));
    flow.appendChild(v);
    flow.appendChild(conn("image tokens · interleaved M-RoPE"));
  }

  flow.appendChild(node("embed_tokens", shapeStr([H2.vocab, A2.hidden]), COLORS.embed));
  flow.appendChild(conn("hidden = " + fmtI(A2.hidden)));

  if (A2.hybrid){
    const split = el("div", "hsplit");
    split.appendChild(linearBlock("× " + A2.hybrid.n_linear_layers));
    split.appendChild(fullBlock("× " + D2.derived.n_full_attn_layers, "full-attention block"));
    flow.appendChild(split);
    flow.appendChild(el("div", "rownote",
      `every ${A2.hybrid.full_attention_interval}ᵗʰ layer is full attention; the rest are linear`));
  } else {
    flow.appendChild(fullBlock("× " + A2.n_layers, "decoder block"));
  }

  flow.appendChild(conn());
  flow.appendChild(node("rmsnorm", "model.norm", "var(--gray)"));
  flow.appendChild(conn());
  flow.appendChild(node("lm_head",
    H2.tied_embeddings ? "tied ← embed_tokens " + shapeStr([A2.hidden, H2.vocab])
                       : shapeStr([H2.vocab, A2.hidden]),
    COLORS.lm_head));
  flow.appendChild(conn("logits = " + fmtI(H2.vocab)));
  flow.appendChild(el("div", "rownote", "argmax / sample → next token"));
  flow.lastChild.style.marginBottom = "14px";
})();

/* ---------- parameter distribution ---------- */
(() => {
  const comps = Object.entries(D.components)
      .filter(([, v]) => v.params > 0)
      .sort((a, b) => b[1].params - a[1].params);
  const total = comps.reduce((s, [, v]) => s + Number(v.params), 0) || 1;
  const bar = q("#stackbar"), legend = q("#legend");
  const widths = [];
  for (const [name, v] of comps){
    const seg = el("i");
    seg.style.background = COLORS[name] || COLORS.other;
    seg.title = `${name} · ${fmtP(v.params)} (${(100*v.params/total).toFixed(1)}%) · ${fmtB(v.bytes)}`;
    bar.appendChild(seg);
    widths.push([seg, 100 * v.params / total]);
    const it = el("div", "it");
    const sw = el("span", "sw");
    sw.style.background = COLORS[name] || COLORS.other;
    it.appendChild(sw);
    it.appendChild(el("b", "", name));
    it.appendChild(el("span", "", `${fmtP(v.params)} · ${(100*v.params/total).toFixed(1)}% · ${fmtB(v.bytes)}`));
    legend.appendChild(it);
  }
  const layers = D.layers || [];
  const lb = q("#layerbars"), heights = [];
  if (layers.length){
    const max = Math.max(...layers.map(l => Number(l.params))) || 1;
    for (const l of layers){
      const b = el("i");
      b.style.background = l.kind === "moe" ? COLORS.moe : l.kind === "linear" ? COLORS.linear_attn : COLORS.attn;
      b.title = `layer ${l.idx} · ${l.kind} · ${fmtP(l.params)} · ${fmtB(l.bytes)}`;
      lb.appendChild(b);
      heights.push([b, Math.max(3, 100 * l.params / max)]);
    }
    q("#layeraxis").appendChild(el("span", "", "layer 0"));
    q("#layeraxis").appendChild(el("span", "", "layer " + (layers.length - 1)));
  } else {
    lb.style.display = "none";
    q("#layerbars-title").style.display = "none";
  }
  requestAnimationFrame(() => requestAnimationFrame(() => {
    widths.forEach(([s, w]) => s.style.width = w + "%");
    heights.forEach(([b, h]) => b.style.height = h + "%");
  }));
})();

/* ---------- tensor explorer ---------- */
(() => {
  const tensors = D.tensors || [];
  const groupsRoot = q("#groups");
  const maxBytes = Math.max(...tensors.map(t => Number(t.bytes)), 1);
  const layerKind = {};
  (D.layers || []).forEach(l => layerKind[l.idx] = l.kind);

  const groups = [];  // {title, kind, rows, prefix}
  const globals = tensors.filter(t => t.layer < 0 && t.component !== "vision");
  if (globals.length) groups.push({title: "globals", kind: "embed · head", rows: globals, prefix: ""});
  const vision = tensors.filter(t => t.component === "vision");
  if (vision.length) groups.push({title: "vision", kind: "vit", rows: vision, prefix: ""});
  const byLayer = new Map();
  for (const t of tensors){
    if (t.layer < 0) continue;
    if (!byLayer.has(t.layer)) byLayer.set(t.layer, []);
    byLayer.get(t.layer).push(t);
  }
  [...byLayer.keys()].sort((a, b) => a - b).forEach(li => {
    groups.push({title: "layer " + li, kind: layerKind[li] || "attn",
                 rows: byLayer.get(li), prefix: `model.layers.${li}.`});
  });

  const totalBytes = tensors.reduce((s, t) => s + Number(t.bytes), 0) || 1;
  q("#tcount").textContent = fmtI(tensors.length) + " tensors";

  function buildTable(g){
    const table = el("table", "tt");
    const tb = el("tbody");
    for (const t of g.rows){
      const tr = el("tr");
      tr.dataset.search = (t.name + " " + t.dtype + " " + t.quant + " " + t.component).toLowerCase();
      const name = el("td", "tname");
      if (g.prefix && t.name.startsWith(g.prefix)){
        name.appendChild(el("span", "", g.prefix));
        name.appendChild(document.createTextNode(t.name.slice(g.prefix.length)));
      } else {
        name.textContent = t.name;
      }
      name.title = t.name + (t.stored_shape ? "  (stored " + shapeStr(t.stored_shape) + ")" : "");
      tr.appendChild(name);
      tr.appendChild(el("td", "tshape", shapeStr(t.shape)));
      tr.appendChild(el("td", "tdtype", t.dtype));
      const qt = el("td", "tquant");
      if (t.quant) qt.appendChild(el("span", "", t.quant));
      tr.appendChild(qt);
      tr.appendChild(el("td", "tnum", fmtP(t.params)));
      tr.appendChild(el("td", "tnum", fmtB(t.bytes)));
      const bar = el("td", "tbar");
      const i = el("i");
      i.style.width = Math.max(1, 100 * t.bytes / maxBytes) + "%";
      bar.appendChild(i);
      tr.appendChild(bar);
      tb.appendChild(tr);
    }
    table.appendChild(tb);
    return table;
  }

  const dets = [];
  for (const g of groups){
    const det = el("details", "grp");
    const sum = el("summary");
    sum.appendChild(el("span", "gtitle", g.title));
    sum.appendChild(el("span", "gkind", g.kind));
    const params = g.rows.reduce((s, t) => s + Number(t.params), 0);
    const bytes = g.rows.reduce((s, t) => s + Number(t.bytes), 0);
    sum.appendChild(el("span", "gmeta",
      `${g.rows.length} tensors · ${fmtP(params)} · ${fmtB(bytes)}`));
    const share = el("span", "gshare");
    const fill = el("i");
    share.appendChild(fill);
    sum.appendChild(share);
    det.appendChild(sum);
    det.addEventListener("toggle", () => {  // lazy row build on first open
      if (det.open && !det.dataset.built){
        det.appendChild(buildTable(g));
        det.dataset.built = "1";
      }
    });
    groupsRoot.appendChild(det);
    dets.push({det, g, fill, share: 100 * bytes / totalBytes});
  }
  requestAnimationFrame(() => requestAnimationFrame(() => {
    dets.forEach(d => d.fill.style.width = Math.max(1, d.share) + "%");
  }));

  q("#search").addEventListener("input", e => {
    const needle = e.target.value.trim().toLowerCase();
    let shown = 0;
    for (const {det, g} of dets){
      if (!needle){
        det.style.display = "";
        det.open = false;
        det.querySelectorAll("tr").forEach(tr => tr.style.display = "");
        continue;
      }
      if (!det.dataset.built){ det.appendChild(buildTable(g)); det.dataset.built = "1"; }
      let hits = 0;
      det.querySelectorAll("tr").forEach(tr => {
        const hit = tr.dataset.search.includes(needle);
        tr.style.display = hit ? "" : "none";
        if (hit) ++hits;
      });
      det.style.display = hits ? "" : "none";
      det.open = hits > 0;
      shown += hits;
    }
    q("#tcount").textContent = needle ? fmtI(shown) + " / " + fmtI(tensors.length) + " tensors"
                                      : fmtI(tensors.length) + " tensors";
  });
})();
</script>
</body>
</html>
)HTML";

}  // namespace

std::string render_schematic_html(const nlohmann::json& schema) {
  std::string page = kTemplate;
  const size_t pos = page.find(kMarker);
  if (pos == std::string::npos) {
    throw std::logic_error("schematic_html: template is missing the schema marker");
  }
  page.replace(pos, std::strlen(kMarker), escape_json_for_html(schema.dump()));
  return page;
}

}  // namespace mlxforge::inspect

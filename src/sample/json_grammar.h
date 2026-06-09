// Byte-level incremental JSON grammar for constrained decoding.
//
// A JsonGrammar validates a JSON value one byte at a time. The sampler uses it
// to mask, at each decode step, any token whose bytes would make the output
// stop being a valid JSON prefix — so the model can only emit well-formed JSON.
//
// Two modes:
//   * any_value()        — accept any JSON value (the "json mode" of structured
//                          output: response_format = json).
//   * from_schema(json)  — a supported subset: the top level must be an object
//                          whose keys are exactly the schema's `properties`, in
//                          order, all required, each a scalar type
//                          (string/number/integer/boolean) or a free value.
//                          (Nested-object/array schemas fall back to free
//                          values; documented in doc/embedding.md.)
//
// The matcher is pure host logic (no MLX), so it is unit-tested directly and
// runs on the worker thread to build per-row token masks. It is cheap to copy,
// which is how `accepts(bytes)` probes a candidate token without committing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mlxforge {

class JsonGrammar {
 public:
  // Free-form: accept any single JSON value.
  static JsonGrammar any_value();

  // Compile a (subset of) JSON Schema into a grammar. Unsupported constructs
  // degrade to a free value rather than throwing, so a schema never blocks
  // generation; the supported subset is enforced exactly.
  static JsonGrammar from_schema(const nlohmann::json& schema);

  // Parse a schema given as a JSON string; an empty/invalid string => any_value.
  static JsonGrammar from_schema_string(const std::string& schema_json);

  // Would feeding `bytes` from the current state keep the output a valid JSON
  // prefix? Does not mutate this grammar (probes a copy).
  bool accepts(const std::string& bytes) const;

  // Commit `bytes` (the chosen token's output) to the current state. Behavior is
  // undefined unless accepts(bytes) was true.
  void advance(const std::string& bytes);

  // Is the output so far a complete JSON value (so generation may stop / EOS is
  // allowed)? True only at the top level with no value in progress.
  bool complete() const;

  // Compact mode: forbid whitespace *between* tokens (whitespace inside strings
  // is always allowed). Used during generation so greedy decoding cannot stall
  // emitting endless separator whitespace; left on for plain validation.
  void set_compact(bool compact) { allow_ws_ = !compact; }

  // True while a string value/key is being consumed (whitespace is content).
  bool in_string() const { return scalar_ == Scalar::String; }

 private:
  // What the parser expects next at the current structural position. The
  // *OrEnd variants are the positions right after an opening bracket, where the
  // container may also close immediately (empty); the plain Value/ObjectKey are
  // after a comma, where a trailing close would be an invalid trailing comma.
  enum class Expect : uint8_t {
    Value,            // a value is required (root, after ':' , after ',' in array)
    ArrayValueOrEnd,  // right after '[': a value or ']'
    ObjectKeyOrEnd,   // right after '{': a '"' key or '}'
    ObjectKey,        // a '"' key is required (after ',' in object)
    ObjectColon,      // ':' between key and value
    ObjectNext,       // ',' or '}' after a member value
    ArrayNext,        // ',' or ']' after an element
    End,              // a complete top-level value parsed; only whitespace remains
  };

  // Sub-state while a scalar literal is being consumed byte by byte.
  enum class Scalar : uint8_t {
    None,
    String,        // inside a string, normal chars
    StringEsc,     // just saw a backslash
    StringU,       // inside a \uXXXX escape (u_left_ hex digits remain)
    Number,        // inside a number (number_ sub-state via num_)
    Lit,           // inside true/false/null (matching lit_ from lit_pos_)
  };

  // Number DFA sub-state (JSON number grammar). Terminable states (a valid
  // complete number ends here) are IntZero, IntDigits, FracDigits, ExpDigits.
  enum class Num : uint8_t {
    Sign,         // optional '-' consumed; expect first int digit
    IntZero,      // leading '0'; no more int digits (only '.', 'e'/'E')
    IntDigits,    // one+ int digits
    FracDot,      // '.' consumed; expect first frac digit
    FracDigits,   // frac digits
    ExpStart,     // 'e'/'E' consumed; optional sign or first exp digit
    ExpAfterSign, // exp sign consumed; expect first exp digit
    ExpDigits,    // exp digits
  };

  // Schema-driven typing of the next value (first-byte gating). Free = any value.
  enum class Type : uint8_t { Free, String, Number, Integer, Boolean };

  // Feed one byte; returns false if it makes the output invalid JSON.
  bool feed(uint8_t b);
  // Begin a value with `t` (schema type gating for the first byte).
  bool begin_value(uint8_t b, Type t);
  // Advance the in-progress number DFA by `b`; false if `b` does not extend it.
  bool number_continue(uint8_t b);
  // Finish the in-progress scalar (if terminable) so byte `b` can be handled as
  // structure; returns false if the scalar cannot terminate here.
  bool finish_scalar_then(uint8_t b);
  // Handle a structural byte `b` given the current Expect/stack (no scalar open).
  bool structural(uint8_t b);
  // Schema mode: are all the schema's required keys already consumed (so no
  // further key may start at the top-level object)?
  bool schema_keys_exhausted() const;
  void start_key();              // begin reading an object key string
  void close_container(char c);  // pop a '{' or '[' and mark its value done
  void push_value_done();        // a value just completed: advance Expect/stack.

  // For a schema object: the ordered keys, their types, and how far we are.
  struct ObjectSchema {
    std::vector<std::string> keys;
    std::vector<Type> types;
  };

  // Parser state.
  Expect expect_ = Expect::Value;
  Scalar scalar_ = Scalar::None;
  Num num_ = Num::Sign;
  bool num_int_only_ = false;  // schema "integer": forbid '.'/'e'/'E' in the number
  std::string lit_;       // literal being matched ("true"/"false"/"null")
  size_t lit_pos_ = 0;
  int u_left_ = 0;        // remaining hex digits in a \uXXXX escape

  std::vector<char> stack_;  // '{' or '[' for each open container
  bool allow_ws_ = true;     // permit inter-token whitespace (off in compact mode)

  // Schema mode (empty schema_keys_ => free JSON).
  bool schema_ = false;
  ObjectSchema root_schema_;
  // Index of the next key to expect at the top-level object (schema mode).
  size_t schema_key_idx_ = 0;
  // Pending value type for the value about to start (schema mode).
  Type pending_type_ = Type::Free;
  // The key string currently being read/checked (schema mode).
  std::string cur_key_;
  bool reading_key_ = false;
};

}  // namespace mlxforge

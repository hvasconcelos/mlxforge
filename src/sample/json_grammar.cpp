#include "sample/json_grammar.h"

namespace mlxforge {

namespace {

bool is_ws(uint8_t b) { return b == ' ' || b == '\t' || b == '\n' || b == '\r'; }
bool is_digit(uint8_t b) { return b >= '0' && b <= '9'; }
bool is_hex(uint8_t b) {
  return is_digit(b) || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F');
}

// Map a JSON-Schema "type" string to a Type code matching the Type enum order
// (Free=0, String=1, Number=2, Integer=3, Boolean=4) — kept as ints so this free
// helper needs no access to the private Type enum.
int type_code(const std::string& s) {
  if (s == "string") return 1;
  if (s == "number") return 2;
  if (s == "integer") return 3;
  if (s == "boolean") return 4;
  return 0;  // object/array/null -> free value
}

// Extract ordered (key, type_code) pairs from a schema's `properties`. Templated
// on the JSON type so `ordered_json` can preserve the declared key order (the
// default nlohmann::json sorts object keys, which would reorder the schema).
template <class J>
bool extract_properties(const J& schema, std::vector<std::string>& keys,
                        std::vector<int>& codes) {
  if (!schema.is_object() || !schema.contains("properties") ||
      !schema["properties"].is_object())
    return false;
  for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
    keys.push_back(it.key());
    int code = 0;
    const auto& prop = it.value();
    if (prop.is_object() && prop.contains("type") && prop["type"].is_string())
      code = type_code(prop["type"].template get<std::string>());
    codes.push_back(code);
  }
  return true;
}

}  // namespace

// ---- Construction ---------------------------------------------------------

JsonGrammar JsonGrammar::any_value() { return JsonGrammar{}; }

JsonGrammar JsonGrammar::from_schema(const nlohmann::json& schema) {
  JsonGrammar g;
  std::vector<std::string> keys;
  std::vector<int> codes;
  if (!extract_properties(schema, keys, codes)) return g;  // unsupported -> free
  g.schema_ = true;
  g.root_schema_.keys = std::move(keys);
  for (int c : codes) g.root_schema_.types.push_back(static_cast<Type>(c));
  return g;
}

JsonGrammar JsonGrammar::from_schema_string(const std::string& schema_json) {
  if (schema_json.empty() || schema_json == "json") return any_value();  // free JSON
  // Parse with ordered_json so the schema's property order is preserved.
  auto parsed = nlohmann::ordered_json::parse(schema_json, nullptr, /*allow_exceptions=*/false);
  if (parsed.is_discarded()) return any_value();
  JsonGrammar g;
  std::vector<std::string> keys;
  std::vector<int> codes;
  if (!extract_properties(parsed, keys, codes)) return g;
  g.schema_ = true;
  g.root_schema_.keys = std::move(keys);
  for (int c : codes) g.root_schema_.types.push_back(static_cast<Type>(c));
  return g;
}

// ---- Probing / committing --------------------------------------------------

bool JsonGrammar::accepts(const std::string& bytes) const {
  JsonGrammar probe = *this;  // cheap copy; feed without mutating self
  for (unsigned char c : bytes)
    if (!probe.feed(c)) return false;
  return true;
}

void JsonGrammar::advance(const std::string& bytes) {
  for (unsigned char c : bytes) feed(c);
}

bool JsonGrammar::complete() const {
  if (expect_ == Expect::End) return true;
  // A bare top-level number has no terminator; it is complete whenever it is in
  // a terminable number state at the root.
  if (stack_.empty() && scalar_ == Scalar::Number) {
    switch (num_) {
      case Num::IntZero:
      case Num::IntDigits:
      case Num::FracDigits:
      case Num::ExpDigits:
        return true;
      default:
        return false;
    }
  }
  return false;
}

// ---- The byte automaton ----------------------------------------------------

bool JsonGrammar::feed(uint8_t b) {
  // 1) A scalar in progress consumes bytes until it ends.
  switch (scalar_) {
    case Scalar::String: {
      if (b == '"') {
        scalar_ = Scalar::None;
        if (reading_key_) {
          // Key closed: in schema mode it must equal the expected key exactly.
          if (schema_ && stack_.size() == 1) {
            if (cur_key_.size() != root_schema_.keys[schema_key_idx_].size()) return false;
            pending_type_ = root_schema_.types[schema_key_idx_];
          }
          reading_key_ = false;
          expect_ = Expect::ObjectColon;
        } else {
          push_value_done();
        }
        return true;
      }
      if (b == '\\') {
        if (reading_key_ && schema_ && stack_.size() == 1) return false;  // simple keys only
        scalar_ = Scalar::StringEsc;
        return true;
      }
      if (b < 0x20) return false;  // unescaped control char
      if (reading_key_ && schema_ && stack_.size() == 1) {
        const std::string& want = root_schema_.keys[schema_key_idx_];
        if (cur_key_.size() >= want.size() ||
            static_cast<uint8_t>(want[cur_key_.size()]) != b)
          return false;
        cur_key_.push_back(static_cast<char>(b));
      }
      return true;  // any other byte (incl. UTF-8 continuation) is string content
    }
    case Scalar::StringEsc: {
      switch (b) {
        case '"': case '\\': case '/': case 'b':
        case 'f': case 'n': case 'r': case 't':
          scalar_ = Scalar::String;
          return true;
        case 'u':
          scalar_ = Scalar::StringU;
          u_left_ = 4;
          return true;
        default:
          return false;
      }
    }
    case Scalar::StringU: {
      if (!is_hex(b)) return false;
      if (--u_left_ == 0) scalar_ = Scalar::String;
      return true;
    }
    case Scalar::Lit: {
      if (lit_pos_ >= lit_.size()) return false;  // defensive
      if (b != static_cast<uint8_t>(lit_[lit_pos_])) return false;
      if (++lit_pos_ == lit_.size()) {
        scalar_ = Scalar::None;
        push_value_done();
      }
      return true;
    }
    case Scalar::Number: {
      if (number_continue(b)) return true;
      // The number ends here; it must be in a terminable state, then byte b is
      // re-handled as structure / whitespace.
      if (!finish_scalar_then(b)) return false;
      return true;
    }
    case Scalar::None:
      break;
  }

  // 2) No scalar open: whitespace between tokens (suppressed in compact mode).
  if (is_ws(b)) return allow_ws_;
  return structural(b);
}

// True if `b` extends the in-progress number; advances num_ when so.
bool JsonGrammar::number_continue(uint8_t b) {
  switch (num_) {
    case Num::Sign:
      if (b == '0') { num_ = Num::IntZero; return true; }
      if (b >= '1' && b <= '9') { num_ = Num::IntDigits; return true; }
      return false;
    case Num::IntZero:
      if (num_int_only_) return false;  // integer: no fraction/exponent
      if (b == '.') { num_ = Num::FracDot; return true; }
      if (b == 'e' || b == 'E') { num_ = Num::ExpStart; return true; }
      return false;  // a digit after a leading 0 is invalid
    case Num::IntDigits:
      if (is_digit(b)) return true;
      if (num_int_only_) return false;  // integer: no fraction/exponent
      if (b == '.') { num_ = Num::FracDot; return true; }
      if (b == 'e' || b == 'E') { num_ = Num::ExpStart; return true; }
      return false;
    case Num::FracDot:
      if (is_digit(b)) { num_ = Num::FracDigits; return true; }
      return false;
    case Num::FracDigits:
      if (is_digit(b)) return true;
      if (b == 'e' || b == 'E') { num_ = Num::ExpStart; return true; }
      return false;
    case Num::ExpStart:
      if (b == '+' || b == '-') { num_ = Num::ExpAfterSign; return true; }
      if (is_digit(b)) { num_ = Num::ExpDigits; return true; }
      return false;
    case Num::ExpAfterSign:
      if (is_digit(b)) { num_ = Num::ExpDigits; return true; }
      return false;
    case Num::ExpDigits:
      if (is_digit(b)) return true;
      return false;
  }
  return false;
}

bool JsonGrammar::finish_scalar_then(uint8_t b) {
  // Only a terminable number may end implicitly.
  const bool terminable = num_ == Num::IntZero || num_ == Num::IntDigits ||
                          num_ == Num::FracDigits || num_ == Num::ExpDigits;
  if (!terminable) return false;
  scalar_ = Scalar::None;
  push_value_done();
  if (is_ws(b)) return allow_ws_;
  return structural(b);
}

bool JsonGrammar::begin_value(uint8_t b, Type t) {
  // Scalar type gating (schema mode): restrict the first byte to the type.
  switch (t) {
    case Type::String:
      if (b != '"') return false;
      break;
    case Type::Integer:
    case Type::Number:
      if (b != '-' && !is_digit(b)) return false;
      break;
    case Type::Boolean:
      if (b != 't' && b != 'f') return false;
      break;
    case Type::Free:
      break;
  }

  switch (b) {
    case '"':
      scalar_ = Scalar::String;
      return true;
    case '{':
      stack_.push_back('{');
      expect_ = Expect::ObjectKeyOrEnd;
      return true;
    case '[':
      stack_.push_back('[');
      expect_ = Expect::ArrayValueOrEnd;
      return true;
    case 't':
      scalar_ = Scalar::Lit; lit_ = "true"; lit_pos_ = 1;
      return true;
    case 'f':
      scalar_ = Scalar::Lit; lit_ = "false"; lit_pos_ = 1;
      return true;
    case 'n':
      scalar_ = Scalar::Lit; lit_ = "null"; lit_pos_ = 1;
      return true;
    case '-':
      scalar_ = Scalar::Number; num_ = Num::Sign; num_int_only_ = (t == Type::Integer);
      return true;
    case '0':
      scalar_ = Scalar::Number; num_ = Num::IntZero; num_int_only_ = (t == Type::Integer);
      return true;
    default:
      if (b >= '1' && b <= '9') {
        scalar_ = Scalar::Number; num_ = Num::IntDigits; num_int_only_ = (t == Type::Integer);
        return true;
      }
      return false;
  }
}

bool JsonGrammar::structural(uint8_t b) {
  switch (expect_) {
    case Expect::Value:
      // A schema constrains the top level to an object.
      if (schema_ && stack_.empty() && b != '{') return false;
      return begin_value(b, pending_type_);

    case Expect::ArrayValueOrEnd:
      if (b == ']') { close_container(']'); return true; }
      return begin_value(b, Type::Free);  // array elements are unconstrained

    case Expect::ObjectKeyOrEnd:
      if (b == '}') {
        if (schema_ && stack_.size() == 1 &&
            schema_key_idx_ != root_schema_.keys.size())
          return false;  // missing required keys
        close_container('}');
        return true;
      }
      if (b == '"') {
        if (schema_keys_exhausted()) return false;  // no more keys allowed
        start_key();
        return true;
      }
      return false;

    case Expect::ObjectKey:
      if (b == '"') {
        if (schema_keys_exhausted()) return false;
        start_key();
        return true;
      }
      return false;

    case Expect::ObjectColon:
      if (b == ':') { expect_ = Expect::Value; return true; }
      return false;

    case Expect::ObjectNext:
      if (b == ',') {
        if (schema_keys_exhausted()) return false;  // no more keys: trailing comma
        expect_ = Expect::ObjectKey;
        return true;
      }
      if (b == '}') {
        if (schema_ && stack_.size() == 1 &&
            schema_key_idx_ != root_schema_.keys.size())
          return false;
        close_container('}');
        return true;
      }
      return false;

    case Expect::ArrayNext:
      if (b == ',') { expect_ = Expect::Value; pending_type_ = Type::Free; return true; }
      if (b == ']') { close_container(']'); return true; }
      return false;

    case Expect::End:
      return false;  // only whitespace after the top-level value
  }
  return false;
}

bool JsonGrammar::schema_keys_exhausted() const {
  return schema_ && stack_.size() == 1 && schema_key_idx_ >= root_schema_.keys.size();
}

void JsonGrammar::start_key() {
  scalar_ = Scalar::String;
  reading_key_ = true;
  cur_key_.clear();
}

void JsonGrammar::close_container(char close) {
  if (stack_.empty()) return;  // defensive
  stack_.pop_back();
  (void)close;
  push_value_done();
}

void JsonGrammar::push_value_done() {
  // A value just completed. If schema mode and it was a top-level member value,
  // advance to the next required key.
  if (!stack_.empty() && stack_.back() == '{' && schema_ && stack_.size() == 1)
    ++schema_key_idx_;

  if (stack_.empty()) {
    expect_ = Expect::End;
    return;
  }
  pending_type_ = Type::Free;
  expect_ = (stack_.back() == '{') ? Expect::ObjectNext : Expect::ArrayNext;
}

}  // namespace mlxforge

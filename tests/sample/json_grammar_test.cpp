// Unit tests for the byte-level JSON grammar used by constrained decoding.
// Pure logic, no model — always run.
#include <doctest/doctest.h>

#include <string>

#include "sample/json_grammar.h"

using namespace mlxforge;

namespace {

// A string is a "valid complete value" if every byte keeps it a valid prefix
// and the grammar is complete at the end.
bool valid_complete(JsonGrammar g, const std::string& s) {
  if (!g.accepts(s)) return false;
  g.advance(s);
  return g.complete();
}

// A "valid prefix" (accepted incrementally) but not necessarily complete.
bool valid_prefix(const JsonGrammar& g, const std::string& s) { return g.accepts(s); }

}  // namespace

TEST_CASE("json grammar accepts well-formed JSON values") {
  for (const char* s : {
           "42", "-3.14e10", "0", "0.5", "1E+9", "-0",
           "\"hello\"", "\"with \\\"escape\\\" and \\u00e9\"", "\"\"",
           "true", "false", "null",
           "[]", "[1,2,3]", "[ 1 , 2 ]", "[[1],[2,3]]",
           "{}", "{\"a\":1}", "{ \"a\" : 1 , \"b\" : [true,null] }",
           "{\"nested\":{\"x\":\"y\"}}", "  {\"a\":1}  ",
       }) {
    CAPTURE(s);
    CHECK(valid_complete(JsonGrammar::any_value(), s));
  }
}

TEST_CASE("json grammar rejects malformed JSON") {
  for (const char* s : {
           "01", "1.", "1.e5", "+1", "--1", ".5", "1..2",
           "[1,]", "[,]", "[1 2]", "{,}", "{\"a\"}", "{\"a\":}", "{\"a\":1,}",
           "tru", "True", "nul", "\"unterminated", "{\"a\":1}x", "[1]]",
           "}", "]", ":",
       }) {
    CAPTURE(s);
    CHECK_FALSE(valid_complete(JsonGrammar::any_value(), s));
  }
}

TEST_CASE("json grammar tracks completeness incrementally") {
  JsonGrammar g = JsonGrammar::any_value();
  CHECK(valid_prefix(g, "{\"a\":"));     // valid prefix
  CHECK_FALSE(valid_complete(JsonGrammar::any_value(), "{\"a\":"));  // not complete
  CHECK(valid_complete(JsonGrammar::any_value(), "{\"a\":1}"));      // now complete

  // A bare top-level number is complete once it reaches a terminable state.
  JsonGrammar n = JsonGrammar::any_value();
  n.advance("12");
  CHECK(n.complete());
  CHECK(n.accepts("3"));        // can still extend the number
  CHECK_FALSE(n.accepts("x"));  // but not with a non-number byte
}

TEST_CASE("accepts() does not mutate the grammar") {
  JsonGrammar g = JsonGrammar::any_value();
  g.advance("[1");
  CHECK(g.accepts(",2]"));
  CHECK(g.accepts(",99]"));  // still works: the earlier accepts() did not commit
  g.advance(",2]");
  CHECK(g.complete());
}

TEST_CASE("schema grammar enforces an object with ordered typed keys") {
  const std::string schema =
      R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}})";
  auto make = [&] { return JsonGrammar::from_schema_string(schema); };

  // Exactly the schema shape, in order.
  CHECK(valid_complete(make(), R"({"name":"Ada","age":36})"));
  CHECK(valid_complete(make(), R"({ "name" : "Ada" , "age" : 36 })"));

  // Wrong first key, wrong order, wrong type, missing key, extra key.
  CHECK_FALSE(valid_complete(make(), R"({"age":36,"name":"Ada"})"));   // order
  CHECK_FALSE(valid_complete(make(), R"({"name":36,"age":36})"));       // type: name must be string
  CHECK_FALSE(valid_complete(make(), R"({"name":"Ada","age":3.5})"));   // type: age must be integer
  CHECK_FALSE(valid_complete(make(), R"({"name":"Ada"})"));             // missing required key
  CHECK_FALSE(valid_complete(make(), R"({"name":"Ada","age":3,"x":1})"));  // extra key
  CHECK_FALSE(valid_complete(make(), R"(42)"));                        // root must be object

  // Incremental: the wrong key byte is rejected as soon as it diverges.
  JsonGrammar g = make();
  CHECK(g.accepts("{\"n"));          // matches "name" so far
  CHECK_FALSE(g.accepts("{\"x"));    // 'x' is not the next key byte
}

TEST_CASE("schema grammar accepts an empty-properties object") {
  auto g = JsonGrammar::from_schema_string(R"({"type":"object","properties":{}})");
  CHECK(valid_complete(g, "{}"));
  CHECK_FALSE(valid_complete(JsonGrammar::from_schema_string(R"({"type":"object","properties":{}})"),
                             R"({"a":1})"));
}

TEST_CASE("invalid or empty schema falls back to free JSON") {
  CHECK(valid_complete(JsonGrammar::from_schema_string(""), "[1,2,3]"));
  CHECK(valid_complete(JsonGrammar::from_schema_string("json"), "[1,2,3]"));
  CHECK(valid_complete(JsonGrammar::from_schema_string("not json"), "[1,2,3]"));
}

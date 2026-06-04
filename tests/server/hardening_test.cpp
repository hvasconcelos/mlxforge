// server config parsing + bounded waiting queue (no GPU).
#include <doctest/doctest.h>

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "server/config.h"

using namespace mlxforge;

namespace {

// Writes `body` to a unique temp file and returns its path. The file is left on
// disk for the duration of the test process (tiny; the OS temp dir is cleaned up
// out of band) — keeps each test hermetic without a committed fixture.
std::string write_temp_config(const std::string& body) {
  char tmpl[] = "/tmp/mlxforge_config_XXXXXX.json";
  int fd = mkstemps(tmpl, 5 /* len of ".json" suffix */);
  REQUIRE(fd != -1);
  close(fd);  // reopen via ofstream below; we only needed a unique name
  std::string path(tmpl);
  std::ofstream(path) << body;
  return path;
}

}  // namespace

TEST_CASE("ServerConfig parses flags with defaults") {
  ServerConfig d = ServerConfig::parse({"-m", "/models/llama"});
  CHECK(d.model_dir == "/models/llama");
  CHECK(d.host == "0.0.0.0");
  CHECK(d.port == 8080);
  CHECK(d.max_ctx == 8192);

  ServerConfig c = ServerConfig::parse(
      {"--model=/models/llama", "--host", "127.0.0.1", "--port=9000", "--max-ctx", "2048",
       "--max-waiting", "16"});
  CHECK(c.model_dir == "/models/llama");
  CHECK(c.host == "127.0.0.1");
  CHECK(c.port == 9000);
  CHECK(c.max_ctx == 2048);
  CHECK(c.max_waiting == 16);
}

TEST_CASE("ServerConfig rejects unknown and positional args") {
  CHECK_THROWS_AS(ServerConfig::parse({"-m", "/m", "--bogus", "x"}), std::runtime_error);
  CHECK_THROWS_AS(ServerConfig::parse({"-m"}), std::runtime_error);    // missing value
  CHECK_THROWS_AS(ServerConfig::parse({"/models/llama"}), std::runtime_error);  // bare positional
}

TEST_CASE("ServerConfig loads values from a JSON config file") {
  std::string path = write_temp_config(
      R"({"model":"/models/from-file","host":"127.0.0.1","port":9000,)"
      R"("max_ctx":4096,"max_waiting":64,"kv_budget":1024})");
  ServerConfig c = ServerConfig::parse({"-c", path});
  CHECK(c.model_dir == "/models/from-file");
  CHECK(c.host == "127.0.0.1");
  CHECK(c.port == 9000);
  CHECK(c.max_ctx == 4096);
  CHECK(c.max_waiting == 64);
  CHECK(c.kv_budget_bytes == 1024);

  // --config=path form is equivalent.
  ServerConfig c2 = ServerConfig::parse({"--config=" + path});
  CHECK(c2.port == 9000);
}

TEST_CASE("CLI flags override config file values") {
  std::string path = write_temp_config(R"({"host":"127.0.0.1","port":9000,"max_ctx":4096})");
  ServerConfig c = ServerConfig::parse({"-m", "/models/cli", "-c", path, "--port=9999"});
  CHECK(c.model_dir == "/models/cli");  // -m supplies the model (absent from file)
  CHECK(c.port == 9999);                // CLI wins
  CHECK(c.host == "127.0.0.1");         // untouched file value retained
  CHECK(c.max_ctx == 4096);
}

TEST_CASE("config precedence: CLI > env > file") {
  std::string path = write_temp_config(R"({"port":9000})");

  setenv("MLXFORGE_PORT", "7000", 1);
  ServerConfig env_wins = ServerConfig::parse({"-c", path});
  CHECK(env_wins.port == 7000);  // env overrides file

  ServerConfig cli_wins = ServerConfig::parse({"-c", path, "--port", "5000"});
  CHECK(cli_wins.port == 5000);  // CLI overrides env (and file)
  unsetenv("MLXFORGE_PORT");
}

TEST_CASE("config file can supply the model; -m overrides it") {
  std::string path = write_temp_config(R"({"model":"/models/from-file"})");
  ServerConfig from_file = ServerConfig::parse({"-c", path});
  CHECK(from_file.model_dir == "/models/from-file");

  ServerConfig overridden = ServerConfig::parse({"-m", "/models/cli", "-c", path});
  CHECK(overridden.model_dir == "/models/cli");
}

TEST_CASE("config file validation rejects bad input") {
  // Unknown key (typo).
  CHECK_THROWS_AS(ServerConfig::parse({"-c", write_temp_config(R"({"prot":1})")}),
                  std::runtime_error);
  // Wrong type.
  CHECK_THROWS_AS(ServerConfig::parse({"-c", write_temp_config(R"({"port":"abc"})")}),
                  std::runtime_error);
  // Out of range.
  CHECK_THROWS_AS(ServerConfig::parse({"-c", write_temp_config(R"({"port":70000})")}),
                  std::runtime_error);
  // Malformed JSON.
  CHECK_THROWS_AS(ServerConfig::parse({"-c", write_temp_config("{not json")}),
                  std::runtime_error);
  // Root is not an object.
  CHECK_THROWS_AS(ServerConfig::parse({"-c", write_temp_config("[1,2,3]")}), std::runtime_error);
  // Missing file.
  CHECK_THROWS_AS(ServerConfig::parse({"-c", "/no/such/config/file.json"}), std::runtime_error);
  // Missing value for -c.
  CHECK_THROWS_AS(ServerConfig::parse({"-c"}), std::runtime_error);
}

TEST_CASE("bounded waiting queue rejects on overflow (429)") {
  Scheduler sched;
  sched.set_max_waiting(2);
  CHECK(sched.submit(std::make_shared<Request>()));
  CHECK(sched.submit(std::make_shared<Request>()));
  CHECK_FALSE(sched.submit(std::make_shared<Request>()));  // full -> reject (429)

  // Draining a slot lets a new request in.
  sched.take_waiting(1);
  CHECK(sched.submit(std::make_shared<Request>()));
}

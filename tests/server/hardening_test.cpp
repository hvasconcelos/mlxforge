// server config parsing + bounded waiting queue (no GPU).
#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "server/config.h"

using namespace mlxforge;

TEST_CASE("ServerConfig parses flags with defaults") {
  ServerConfig d = ServerConfig::parse({"/models/llama"});
  CHECK(d.model_dir == "/models/llama");
  CHECK(d.host == "0.0.0.0");
  CHECK(d.port == 8080);
  CHECK(d.max_ctx == 8192);

  ServerConfig c = ServerConfig::parse(
      {"/models/llama", "--host", "127.0.0.1", "--port=9000", "--max-ctx", "2048",
       "--max-waiting", "16"});
  CHECK(c.model_dir == "/models/llama");
  CHECK(c.host == "127.0.0.1");
  CHECK(c.port == 9000);
  CHECK(c.max_ctx == 2048);
  CHECK(c.max_waiting == 16);
}

TEST_CASE("ServerConfig rejects unknown flags") {
  CHECK_THROWS_AS(ServerConfig::parse({"/m", "--bogus", "x"}), std::runtime_error);
  CHECK_THROWS_AS(ServerConfig::parse({"/m", "--port"}), std::runtime_error);  // missing value
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

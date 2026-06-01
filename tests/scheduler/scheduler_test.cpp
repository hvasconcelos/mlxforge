// XLLM-016: three-queue handoff + bounded token queue (no model/GPU).
#include <doctest/doctest.h>

#include <memory>
#include <thread>

#include "scheduler/request.h"
#include "scheduler/scheduler.h"

using namespace xllm;

TEST_CASE("XLLM-016: requests enqueue from one thread and are taken by another") {
  Scheduler sched;
  auto r1 = std::make_shared<Request>();
  auto r2 = std::make_shared<Request>();

  // Enqueue from a separate thread.
  std::thread producer([&] {
    sched.submit(r1);
    sched.submit(r2);
  });
  producer.join();

  CHECK(sched.waiting_size() == 2);
  auto got = sched.take_waiting(8);
  CHECK(got.size() == 2);
  CHECK(got[0] == r1);
  CHECK(got[1] == r2);
  CHECK(sched.waiting_size() == 0);
}

TEST_CASE("XLLM-016: next_waiting blocks then returns; stop drains") {
  Scheduler sched;
  auto r1 = std::make_shared<Request>();

  std::thread consumer([&] {
    auto req = sched.next_waiting();  // blocks until submit
    CHECK(req == r1);
    auto none = sched.next_waiting();  // unblocked by stop, drained
    CHECK(none == nullptr);
  });

  sched.submit(r1);
  sched.stop();
  consumer.join();
  CHECK(sched.stopping());
}

TEST_CASE("XLLM-016: bounded token queue pushes, pops, and closes") {
  TokenQueue q(/*capacity=*/4);
  q.push(10);
  q.push(20);
  q.close();  // producer done

  int v = 0;
  CHECK(q.pop(v));
  CHECK(v == 10);
  CHECK(q.pop(v));
  CHECK(v == 20);
  CHECK_FALSE(q.pop(v));  // closed and drained
}

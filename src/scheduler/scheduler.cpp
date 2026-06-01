#include "scheduler/scheduler.h"

namespace xllm {

void Scheduler::submit(const std::shared_ptr<Request>& req) {
  {
    std::lock_guard<std::mutex> lk(m_);
    waiting_.push_back(req);
  }
  cv_.notify_one();
}

std::shared_ptr<Request> Scheduler::next_waiting() {
  std::unique_lock<std::mutex> lk(m_);
  cv_.wait(lk, [&] { return !waiting_.empty() || stop_; });
  if (waiting_.empty()) return nullptr;  // stopping and drained
  auto req = waiting_.front();
  waiting_.pop_front();
  return req;
}

std::vector<std::shared_ptr<Request>> Scheduler::take_waiting(int max) {
  std::lock_guard<std::mutex> lk(m_);
  std::vector<std::shared_ptr<Request>> out;
  while (!waiting_.empty() && static_cast<int>(out.size()) < max) {
    out.push_back(waiting_.front());
    waiting_.pop_front();
  }
  return out;
}

size_t Scheduler::waiting_size() const {
  std::lock_guard<std::mutex> lk(m_);
  return waiting_.size();
}

void Scheduler::stop() {
  {
    std::lock_guard<std::mutex> lk(m_);
    stop_ = true;
  }
  cv_.notify_all();
}

bool Scheduler::stopping() const {
  std::lock_guard<std::mutex> lk(m_);
  return stop_;
}

}  // namespace xllm

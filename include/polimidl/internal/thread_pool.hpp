#ifndef POLIMIDL_THREADING_THREAD_POOL_HPP
#define POLIMIDL_THREADING_THREAD_POOL_HPP

#include <algorithm>
#include <condition_variable>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace polimidl {
namespace internal {
class thread_pool {
 public:
  explicit thread_pool(unsigned int number_of_workers) :
    number_of_active_tasks_(number_of_workers),
    is_running_(true) {
    if (number_of_workers == 1) {
      number_of_active_tasks_ = 0;
      return;
    }
    std::size_t count = 0;
    std::generate_n(std::back_inserter(workers_),
                    number_of_workers,
                    [&count, pool = this](){
      return std::thread([worker = count++, pool = pool]() {
        std::function<void(unsigned int)> task;
        while (pool->is_running_) {
          {
            std::unique_lock<std::mutex> tasks_lock(pool->tasks_mutex_);
            pool->notify_task_completed();
            pool->tasks_condition_variable_.wait(tasks_lock, [=]() {
              return !pool->tasks_.empty();
            });
            std::swap(task, pool->tasks_.front());
            pool->tasks_.pop();
            pool->notify_task_started();
          }
          task(worker);
        }
        {
          std::unique_lock<std::mutex> tasks_lock(pool->tasks_mutex_);
          pool->notify_task_completed();
        }
      });
    });
    wait();
  }
  thread_pool() = delete;
  thread_pool(const thread_pool& other) = delete;

  thread_pool& operator=(const thread_pool& other) = delete;

  ~thread_pool() {
    is_running_ = false;
    for (std::size_t i = 0; i < workers_.size(); ++i) {
      schedule([](unsigned int worker){});
    }
    wait();
    for (auto&& thread : workers_) {
      thread.join();
    }
  }

  unsigned int number_of_workers() const {
    return std::max(static_cast<unsigned int>(1),
                    static_cast<unsigned int>(workers_.size()));
  }

  template <typename task_t>
  void schedule(task_t&& task) {
    if (workers_.empty()) {
      task(0);
      return;
    }
    std::unique_lock<std::mutex> lock(tasks_mutex_);
    tasks_.push(std::function<void(unsigned int)>(std::forward<task_t>(task)));
    tasks_condition_variable_.notify_one();
  }

  void wait() {
    if (workers_.empty()) {
      return;
    }
    std::future<void> future;
    {
    	std::unique_lock<std::mutex> lock_tasks(tasks_mutex_);
    	if (number_of_active_tasks_ == 0 && tasks_.empty()) {
      	return;
      }
    	std::unique_lock<std::mutex> lock_awaiters(awaiters_mutex_);
    	awaiters_.push(std::promise<void>());
    	future = awaiters_.back().get_future();
    }
    future.wait();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void(unsigned int)>> tasks_;
  std::size_t number_of_active_tasks_;
  std::mutex tasks_mutex_;
  std::condition_variable tasks_condition_variable_;
  std::queue<std::promise<void>> awaiters_;
  std::mutex awaiters_mutex_;
  bool is_running_;

  void notify_task_started() {
    ++number_of_active_tasks_;
  }

  void notify_task_completed() {
    --number_of_active_tasks_;
    if (number_of_active_tasks_ == 0 && tasks_.empty()) {
      std::unique_lock<std::mutex> awaiters_lock(awaiters_mutex_);
      while (!awaiters_.empty()) {
        awaiters_.front().set_value();
        awaiters_.pop();
      }
    }
  }
};

class scheduler {
 private:
  thread_pool* pool_;

 public:
  explicit scheduler(thread_pool* pool) : pool_(pool) {}
  scheduler() = delete;

  unsigned int number_of_workers() const {
    return pool_->number_of_workers();
  }

  template <typename T>
  void schedule(T task) const { pool_->schedule(task); }

  void wait() const { pool_->wait(); }
};
}
}

#endif // POLIMIDL_THREADING_THREAD_POOL_HPP

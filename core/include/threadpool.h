// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace geniex {

// Worker pool for overlapping host-side CPU work with HTP execution during decode.
// `wait()` is the barrier callers use before reading a queued job's result.
//
// The pool also owns an optional set of "clock-keeper" spinner threads. On a
// load-driven CPU governor, the decode dispatch thread idles while it waits for
// the accelerator, which can let the governor down-clock the cluster and slow
// every token. The spinners busy-wait on a pinned core set to keep the governor's
// measured load high. They do no real work and run only across the decode window.
class ThreadPool {
   public:
    ThreadPool() = default;
    ~ThreadPool();

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // `cpu_mask` pins workers to CPUs whose bit is set (0 = no pin). `poll` busy-spins
    // idle workers instead of blocking (lower latency, more heat). No-op if started.
    void start(unsigned n_threads, uint64_t cpu_mask = 0, bool poll = false);

    // Join all workers; idempotent. Does not drain — call wait() first if needed.
    // Also stops the clock keeper.
    void stop();

    // Queue one job; returns immediately.
    void enqueue(std::function<void()> job);

    // Block until all queued jobs finish; rethrows the first job exception, if any.
    void wait();

    size_t size() const { return threads_.size(); }
    bool   started() const { return !threads_.empty(); }

    // Clock keeper: spin up / tear down `n_spinners` busy-wait threads pinned to
    // `cpu_mask` (0 = no pin) that keep the CPU cluster from down-clocking.
    // Independent of the job workers — usable even when the pool runs zero workers.
    // startClockKeeper is a no-op if n_spinners == 0 or already running.
    void startClockKeeper(unsigned n_spinners, uint64_t cpu_mask = 0);
    void stopClockKeeper();
    bool clockKeeperRunning() const { return !spinners_.empty(); }

   private:
    void loop();

    std::vector<std::thread>          threads_;
    std::queue<std::function<void()>> jobs_;
    mutable std::mutex                mutex_;
    std::condition_variable           job_cv_;
    std::condition_variable           done_cv_;
    size_t                            outstanding_ = 0;  // queued + in-flight jobs
    bool                              terminate_   = false;
    bool                              poll_        = false;
    uint64_t                          cpu_mask_    = 0;
    std::exception_ptr                first_error_;  // first exception thrown by a job

    // Clock-keeper spinner threads (separate from the job workers above).
    std::vector<std::thread> spinners_;
    std::atomic<bool>        spinners_stop_{false};
};

}  // namespace geniex

// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "threadpool.h"

#if defined(__linux__)
#include <sched.h>
#endif

namespace geniex {

ThreadPool::~ThreadPool() { stop(); }

void ThreadPool::start(unsigned n_threads, uint64_t cpu_mask, bool poll) {
    if (!threads_.empty() || n_threads == 0) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        terminate_ = false;
        poll_      = poll;
        cpu_mask_  = cpu_mask;
    }
    threads_.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; ++i) threads_.emplace_back([this] { loop(); });
}

void ThreadPool::stop() {
    stopClockKeeper();
    if (threads_.empty()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        terminate_ = true;
    }
    job_cv_.notify_all();
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
    // Drop any unrun jobs and reset state so the pool can be reused.
    std::lock_guard<std::mutex>       lock(mutex_);
    std::queue<std::function<void()>> empty;
    jobs_.swap(empty);
    outstanding_ = 0;
    terminate_   = false;
}

void ThreadPool::enqueue(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push(std::move(job));
        ++outstanding_;
    }
    job_cv_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return outstanding_ == 0; });
    // Surface the first job exception (if any) to the caller, then clear it.
    if (first_error_) {
        std::exception_ptr e = first_error_;
        first_error_         = nullptr;
        std::rethrow_exception(e);
    }
}

void ThreadPool::loop() {
#if defined(__linux__)
    if (cpu_mask_ != 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int cpu = 0; cpu < 64; ++cpu)
            if (cpu_mask_ & (uint64_t{1} << cpu)) CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (poll_) {
                // Busy-spin: re-check under lock without blocking on the cv.
                while (!terminate_ && jobs_.empty()) {
                    lock.unlock();
                    std::this_thread::yield();
                    lock.lock();
                }
            } else {
                job_cv_.wait(lock, [this] { return terminate_ || !jobs_.empty(); });
            }
            if (terminate_ && jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
        }

        try {
            job();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!first_error_) first_error_ = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--outstanding_ == 0) done_cv_.notify_all();
        }
    }
}

void ThreadPool::startClockKeeper(unsigned n_spinners, uint64_t cpu_mask) {
    if (!spinners_.empty() || n_spinners == 0) return;
    spinners_stop_.store(false, std::memory_order_relaxed);
    spinners_.reserve(n_spinners);
    for (unsigned i = 0; i < n_spinners; ++i) {
        spinners_.emplace_back([this, cpu_mask] {
#if defined(__linux__)
            if (cpu_mask != 0) {
                cpu_set_t set;
                CPU_ZERO(&set);
                for (int cpu = 0; cpu < 64; ++cpu)
                    if (cpu_mask & (uint64_t{1} << cpu)) CPU_SET(cpu, &set);
                sched_setaffinity(0, sizeof(set), &set);
            }
#endif
            // Tight busy-wait: keeping the core out of idle is the point — it
            // keeps the governor's measured load high so the cluster stays clocked up.
            while (!spinners_stop_.load(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
                __asm__ __volatile__("yield");
#endif
            }
        });
    }
}

void ThreadPool::stopClockKeeper() {
    if (spinners_.empty()) return;
    spinners_stop_.store(true, std::memory_order_relaxed);
    for (auto& t : spinners_)
        if (t.joinable()) t.join();
    spinners_.clear();
}

}  // namespace geniex

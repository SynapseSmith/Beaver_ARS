#!/usr/bin/env python3
"""
Load Testing Script for Beaver ARS
Tests API performance under load
"""

import time
import json
import statistics
import concurrent.futures
from typing import List, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LoadTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = self._create_session()
        self.results: List[float] = []
        self.errors: List[str] = []
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def send_request(self, payload: Dict) -> float:
        """Send single request and measure response time"""
        try:
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/order",
                json=payload,
                timeout=30
            )
            end = time.time()
            
            if response.status_code == 200:
                return end - start
            else:
                self.errors.append(f"Status {response.status_code}")
                return -1
        except Exception as e:
            self.errors.append(str(e))
            return -1
    
    def run_concurrent_test(self, num_requests: int, num_workers: int = 10):
        """Run concurrent load test"""
        print("=" * 60)
        print(f"Load Testing: {num_requests} requests with {num_workers} workers")
        print("=" * 60)
        print()
        
        test_payloads = [
            {"user_message": "김치찌개 2개 주문할게요"},
            {"user_message": "영업시간이 언제인가요?"},
            {"user_message": "배달 가능한가요?"},
            {"user_message": "불고기 3인분 주문하고 싶어요"},
            {"user_message": "메뉴 추천해주세요"},
        ]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_requests):
                payload = test_payloads[i % len(test_payloads)]
                futures.append(executor.submit(self.send_request, payload))
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                response_time = future.result()
                if response_time > 0:
                    self.results.append(response_time)
                completed += 1
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{num_requests} requests completed")
        
        end_time = time.time()
        self.print_results(end_time - start_time)
    
    def print_results(self, total_time: float):
        """Print test results"""
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        
        if not self.results:
            print("❌ All requests failed")
            return
        
        # Calculate statistics
        successful = len(self.results)
        failed = len(self.errors)
        total = successful + failed
        
        avg_time = statistics.mean(self.results)
        median_time = statistics.median(self.results)
        min_time = min(self.results)
        max_time = max(self.results)
        
        if len(self.results) > 1:
            stdev = statistics.stdev(self.results)
            p95 = sorted(self.results)[int(len(self.results) * 0.95)]
            p99 = sorted(self.results)[int(len(self.results) * 0.99)]
        else:
            stdev = 0
            p95 = avg_time
            p99 = avg_time
        
        requests_per_sec = total / total_time
        
        # Print statistics
        print(f"Total Requests:        {total}")
        print(f"Successful:            {successful} ({successful/total*100:.1f}%)")
        print(f"Failed:                {failed} ({failed/total*100:.1f}%)")
        print(f"Total Time:            {total_time:.2f}s")
        print(f"Requests/sec:          {requests_per_sec:.2f}")
        print()
        print("Response Times (seconds):")
        print(f"  Average:             {avg_time:.3f}")
        print(f"  Median:              {median_time:.3f}")
        print(f"  Min:                 {min_time:.3f}")
        print(f"  Max:                 {max_time:.3f}")
        print(f"  Std Dev:             {stdev:.3f}")
        print(f"  95th Percentile:     {p95:.3f}")
        print(f"  99th Percentile:     {p99:.3f}")
        
        if self.errors:
            print()
            print("Errors:")
            error_counts = {}
            for error in self.errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
        
        print("=" * 60)
        
        # Pass/Fail criteria
        success_rate = successful / total
        avg_response_acceptable = avg_time < 1.0  # 1 second
        
        if success_rate > 0.95 and avg_response_acceptable:
            print("✅ Load test PASSED")
        else:
            print("❌ Load test FAILED")
            if success_rate <= 0.95:
                print(f"   - Success rate too low: {success_rate*100:.1f}%")
            if not avg_response_acceptable:
                print(f"   - Average response time too high: {avg_time:.3f}s")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test Beaver ARS API")
    parser.add_argument("--url", default="http://localhost:5000", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    tester = LoadTester(args.url)
    tester.run_concurrent_test(args.requests, args.workers)


if __name__ == "__main__":
    main()

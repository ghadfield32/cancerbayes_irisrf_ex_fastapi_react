#!/usr/bin/env python3
"""
Updated test script for rate limiting and concurrency protection.
Uses debug endpoints to reset rate limits between tests.
"""

import asyncio
import httpx
import time
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TestResult:
    """Test result data structure."""
    name: str
    success: bool
    expected_status: int
    actual_status: int
    response_time: float
    rate_limit_headers: Dict[str, str]
    error_message: str = ""

class ProtectionTester:
    """Test both rate limiting and concurrency protection systems."""
    
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:8000")
        self.auth_token = None
        self.results: List[TestResult] = []
        
    async def wait_until_ready(self, timeout: int = 30) -> bool:
        """Poll /api/v1/health until the backend responds 200 or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{self.base_url}/api/v1/health", timeout=3)
                    if r.status_code == 200:
                        return True
            except httpx.ConnectError:
                # Backend not up yet
                pass
            await asyncio.sleep(1)
        return False

    async def authenticate(self) -> bool:
        """Get JWT; waits for backend and surfaces clear errors."""
        if not await self.wait_until_ready():
            print(f"❌ Backend not reachable at {self.base_url}. "
                  f"Start it with 'uvicorn api.app.main:app --reload' or "
                  f"set BASE_URL to the correct host:port.")
            return False
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.base_url}/api/v1/token",
                    json={"username": "alice", "password": "supersecretvalue"},
                    headers={"Content-Type": "application/json"},
                    timeout=5.0,
                )
            if r.status_code == 200:
                self.auth_token = r.json()["access_token"]
                print("✅ Authentication successful")
                return True
            else:
                print(f"❌ Authentication failed: {r.status_code} {r.text}")
                return False
        except httpx.ConnectError as e:
            print(f"❌ Connection error: {e}. Is the API running on {self.base_url}?")
            return False
    
    async def reset_rate_limits(self) -> bool:
        """Reset rate limits for current user."""
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                response = await client.post(
                    f"{self.base_url}/api/v1/debug/ratelimit/reset",
                    headers=headers,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Rate limits reset: {result.get('reset', 0)} keys cleared")
                    return True
                else:
                    print(f"❌ Rate limit reset failed: {response.status_code}")
                    return False
        except httpx.ConnectError as e:
            print(f"❌ Rate limit reset connection error: {e}")
            return False
        except Exception as e:
            print(f"❌ Rate limit reset error: {e}")
            return False
    
    async def make_request(
        self, 
        endpoint: str, 
        method: str = "GET",
        data: Dict[str, Any] = None,
        expected_status: int = 200
    ) -> TestResult:
        """Make a single request and return detailed result."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                
                if method == "GET":
                    response = await client.get(
                        f"{self.base_url}{endpoint}", 
                        headers=headers,
                        timeout=10.0
                    )
                elif method == "POST":
                    headers["Content-Type"] = "application/json"
                    response = await client.post(
                        f"{self.base_url}{endpoint}", 
                        json=data, 
                        headers=headers,
                        timeout=10.0
                    )
                else:
                    return TestResult(
                        name=f"{method} {endpoint}",
                        success=False,
                        expected_status=expected_status,
                        actual_status=0,
                        response_time=0,
                        rate_limit_headers={},
                        error_message=f"Unsupported method: {method}"
                    )
                
                response_time = time.time() - start_time
                
                # Extract rate limit headers
                rate_limit_headers = {
                    k: v for k, v in response.headers.items() 
                    if k.lower().startswith('x-ratelimit') or k.lower() == 'retry-after'
                }
                
                success = response.status_code == expected_status
                
                return TestResult(
                    name=f"{method} {endpoint}",
                    success=success,
                    expected_status=expected_status,
                    actual_status=response.status_code,
                    response_time=response_time,
                    rate_limit_headers=rate_limit_headers,
                    error_message="" if success else f"Expected {expected_status}, got {response.status_code}"
                )
                
        except httpx.ConnectError as e:
            return TestResult(
                name=f"{method} {endpoint}",
                success=False,
                expected_status=expected_status,
                actual_status=0,
                response_time=time.time() - start_time,
                rate_limit_headers={},
                error_message=f"Connection error: {e}. Is the API running on {self.base_url}?"
            )
        except Exception as e:
            return TestResult(
                name=f"{method} {endpoint}",
                success=False,
                expected_status=expected_status,
                actual_status=0,
                response_time=time.time() - start_time,
                rate_limit_headers={},
                error_message=str(e)
            )
    
    async def test_rate_limiting(self) -> None:
        """Test rate limiting functionality."""
        print("\n🔍 Testing Rate Limiting...")
        
        # Test 1: Login rate limiting (IP-based)
        print("\n📊 Test 1: Login Rate Limiting (IP-based)")
        for i in range(5):  # Try 5 login attempts (limit is 3)
            result = await self.make_request(
                "/api/v1/token",
                method="POST",
                data={"username": "wrong", "password": "wrong"},
                expected_status=401 if i < 3 else 429
            )
            self.results.append(result)
            print(f"  Attempt {i+1}: {result.actual_status} (expected: {result.expected_status})")
            if result.rate_limit_headers:
                print(f"    Headers: {result.rate_limit_headers}")
        
        # Test 2: Iris prediction rate limiting
        print("\n🌺 Test 2: Iris Prediction Rate Limiting")
        iris_data = {
            "model_type": "rf",
            "samples": [{
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }]
        }
        
        # Make requests until we hit the limit
        for i in range(125):  # Light limit is 120, so 125th should fail
            result = await self.make_request(
                "/api/v1/iris/predict",
                method="POST",
                data=iris_data,
                expected_status=200 if i < 120 else 429
            )
            self.results.append(result)
            
            if i % 20 == 0:  # Progress indicator
                print(f"  Progress: {i+1}/125 requests")
            
            if result.actual_status == 429:
                print(f"  ✅ Rate limit hit at request {i+1}")
                if result.rate_limit_headers:
                    print(f"    Headers: {result.rate_limit_headers}")
                break
        
        # Test 3: Cancer prediction rate limiting (heavier limit)
        print("\n🔬 Test 3: Cancer Prediction Rate Limiting")
        cancer_data = {
            "model_type": "bayes",
            "samples": [{
                "mean_radius": 14.13,
                "mean_texture": 19.26,
                "mean_perimeter": 91.97,
                "mean_area": 654.89,
                "mean_smoothness": 0.096,
                "mean_compactness": 0.104,
                "mean_concavity": 0.089,
                "mean_concave_points": 0.048,
                "mean_symmetry": 0.181,
                "mean_fractal_dimension": 0.063,
                "se_radius": 0.406,
                "se_texture": 1.216,
                "se_perimeter": 2.866,
                "se_area": 40.34,
                "se_smoothness": 0.007,
                "se_compactness": 0.025,
                "se_concavity": 0.032,
                "se_concave_points": 0.012,
                "se_symmetry": 0.020,
                "se_fractal_dimension": 0.004,
                "worst_radius": 16.27,
                "worst_texture": 25.68,
                "worst_perimeter": 107.26,
                "worst_area": 880.58,
                "worst_smoothness": 0.132,
                "worst_compactness": 0.254,
                "worst_concavity": 0.273,
                "worst_concave_points": 0.114,
                "worst_symmetry": 0.290,
                "worst_fractal_dimension": 0.084
            }]
        }
        
        for i in range(35):  # Heavy limit is 30, so 35th should fail
            result = await self.make_request(
                "/api/v1/cancer/predict",
                method="POST",
                data=cancer_data,
                expected_status=200 if i < 30 else 429
            )
            self.results.append(result)
            
            if i % 10 == 0:
                print(f"  Progress: {i+1}/35 requests")
            
            if result.actual_status == 429:
                print(f"  ✅ Rate limit hit at request {i+1}")
                if result.rate_limit_headers:
                    print(f"    Headers: {result.rate_limit_headers}")
                break
    
    async def test_concurrency_limiting(self) -> None:
        """Test concurrency limiting functionality."""
        print("\n🔍 Testing Concurrency Limiting...")
        
        # Reset rate limits before concurrency test
        print("\n🔄 Resetting rate limits for concurrency test...")
        if not await self.reset_rate_limits():
            print("⚠️  Could not reset rate limits - test may fail")
        
        # Test concurrent cancer predictions (heavy endpoint)
        print("\n🏋️ Test: Concurrent Cancer Predictions")
        
        cancer_data = {
            "model_type": "bayes",
            "samples": [{
                "mean_radius": 14.13,
                "mean_texture": 19.26,
                "mean_perimeter": 91.97,
                "mean_area": 654.89,
                "mean_smoothness": 0.096,
                "mean_compactness": 0.104,
                "mean_concavity": 0.089,
                "mean_concave_points": 0.048,
                "mean_symmetry": 0.181,
                "mean_fractal_dimension": 0.063,
                "se_radius": 0.406,
                "se_texture": 1.216,
                "se_perimeter": 2.866,
                "se_area": 40.34,
                "se_smoothness": 0.007,
                "se_compactness": 0.025,
                "se_concavity": 0.032,
                "se_concave_points": 0.012,
                "se_symmetry": 0.020,
                "se_fractal_dimension": 0.004,
                "worst_radius": 16.27,
                "worst_texture": 25.68,
                "worst_perimeter": 107.26,
                "worst_area": 880.58,
                "worst_smoothness": 0.132,
                "worst_compactness": 0.254,
                "worst_concavity": 0.273,
                "worst_concave_points": 0.114,
                "worst_symmetry": 0.290,
                "worst_fractal_dimension": 0.084
            }]
        }
        
        async def make_concurrent_request(request_id: int) -> TestResult:
            """Make a single concurrent request."""
            return await self.make_request(
                "/api/v1/cancer/predict",
                method="POST",
                data=cancer_data,
                expected_status=200
            )
        
        # Launch 6 concurrent requests (max_concurrent=4)
        print("  Launching 6 concurrent requests (max_concurrent=4)...")
        start_time = time.time()
        
        tasks = [make_concurrent_request(i) for i in range(6)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful = 0
        failed = 0
        response_times = []
        
        for i, result in enumerate(results):
            if isinstance(result, TestResult):
                self.results.append(result)
                if result.success:
                    successful += 1
                    response_times.append(result.response_time)
                else:
                    failed += 1
                print(f"  Request {i+1}: {result.actual_status} ({result.response_time:.2f}s)")
            else:
                failed += 1
                print(f"  Request {i+1}: Exception - {result}")
        
        print(f"\n  📊 Concurrency Test Results:")
        print(f"    Successful: {successful}/6")
        print(f"    Failed: {failed}/6")
        print(f"    Total time: {total_time:.2f}s")
        if response_times:
            print(f"    Avg response time: {sum(response_times)/len(response_times):.2f}s")
    
    async def test_health_endpoints(self) -> None:
        """Test that health endpoints work without rate limiting."""
        print("\n🔍 Testing Health Endpoints (No Rate Limiting)...")
        
        # Test health endpoint
        result = await self.make_request("/api/v1/health")
        self.results.append(result)
        print(f"  Health check: {result.actual_status}")
        
        # Test ready endpoint
        result = await self.make_request("/api/v1/ready")
        self.results.append(result)
        print(f"  Ready check: {result.actual_status}")
    
    def print_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("📊 PROTECTION SYSTEMS TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Rate limiting analysis
        rate_limit_tests = [r for r in self.results if "predict" in r.name or "token" in r.name]
        rate_limit_429s = [r for r in rate_limit_tests if r.actual_status == 429]
        
        print(f"\n🔒 Rate Limiting:")
        print(f"  Rate-limited requests: {len(rate_limit_tests)}")
        print(f"  429 responses (expected): {len(rate_limit_429s)}")
        
        # Concurrency analysis
        concurrent_tests = [r for r in self.results if "cancer/predict" in r.name]
        concurrent_success = [r for r in concurrent_tests if r.success]
        
        print(f"\n🏋️ Concurrency Limiting:")
        print(f"  Concurrent requests: {len(concurrent_tests)}")
        print(f"  Successful concurrent: {len(concurrent_success)}")
        
        # Show failed tests
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.success:
                    print(f"  {result.name}: {result.error_message}")
        
        print("\n" + "="*60)

async def main():
    """Main test function."""
    print("🚀 Starting Updated Protection Systems Test Suite...")
    print("📝 This version uses debug endpoints to reset rate limits between tests")
    print(f"🔗 Using API base URL: {os.getenv('BASE_URL', 'http://localhost:8000')}")
    
    tester = ProtectionTester()          # now picks up BASE_URL env automatically
    
    # Authenticate first
    if not await tester.authenticate():
        print("❌ Cannot proceed without authentication")
        return 1
    
    # Run all tests
    await tester.test_health_endpoints()
    await tester.test_rate_limiting()
    await tester.test_concurrency_limiting()
    
    # Print summary
    tester.print_summary()
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main())) 
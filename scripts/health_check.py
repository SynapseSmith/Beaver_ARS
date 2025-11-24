#!/usr/bin/env python3
"""
Health Check Script for Beaver ARS
Tests all critical system components
"""

import sys
import time
import requests
import mysql.connector
import redis
from typing import Dict, List, Tuple


class HealthChecker:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        
    def check_api(self, url: str = "http://localhost:5000") -> bool:
        """Check API health endpoint"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            self.results.append(("API Health", success, message))
            return success
        except Exception as e:
            self.results.append(("API Health", False, str(e)))
            return False
    
    def check_database(self, host: str = "localhost", user: str = "beaver_user", 
                      password: str = "beaver_password", database: str = "beaver_ars") -> bool:
        """Check MySQL database connection"""
        try:
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                connect_timeout=5
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            self.results.append(("MySQL Database", True, "Connected"))
            return True
        except Exception as e:
            self.results.append(("MySQL Database", False, str(e)))
            return False
    
    def check_redis(self, host: str = "localhost", port: int = 6379, 
                   password: str = None) -> bool:
        """Check Redis connection"""
        try:
            client = redis.Redis(
                host=host,
                port=port,
                password=password,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            client.ping()
            self.results.append(("Redis Cache", True, "Connected"))
            return True
        except Exception as e:
            self.results.append(("Redis Cache", False, str(e)))
            return False
    
    def check_disk_space(self, threshold: int = 90) -> bool:
        """Check disk space usage"""
        try:
            import shutil
            usage = shutil.disk_usage("/")
            percent_used = (usage.used / usage.total) * 100
            success = percent_used < threshold
            message = f"Used: {percent_used:.1f}%"
            self.results.append(("Disk Space", success, message))
            return success
        except Exception as e:
            self.results.append(("Disk Space", False, str(e)))
            return False
    
    def check_memory(self, threshold: int = 90) -> bool:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            success = memory.percent < threshold
            message = f"Used: {memory.percent:.1f}%"
            self.results.append(("Memory Usage", success, message))
            return success
        except Exception as e:
            self.results.append(("Memory Usage", False, str(e)))
            return False
    
    def run_all_checks(self) -> bool:
        """Run all health checks"""
        print("=" * 60)
        print("Beaver ARS Health Check")
        print("=" * 60)
        print()
        
        # Run checks
        self.check_api()
        self.check_database()
        self.check_redis()
        self.check_disk_space()
        self.check_memory()
        
        # Print results
        all_passed = True
        for name, success, message in self.results:
            status = "✓ PASS" if success else "✗ FAIL"
            color = "\033[92m" if success else "\033[91m"
            reset = "\033[0m"
            print(f"{color}{status}{reset} {name:20s} - {message}")
            if not success:
                all_passed = False
        
        print()
        print("=" * 60)
        
        if all_passed:
            print("\033[92m✓ All health checks passed\033[0m")
            return True
        else:
            print("\033[91m✗ Some health checks failed\033[0m")
            return False


def main():
    checker = HealthChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

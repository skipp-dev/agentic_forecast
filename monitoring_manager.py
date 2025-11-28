#!/usr/bin/env python3
"""
Monitoring Stack Management Script
Helps manage the Prometheus + Grafana monitoring stack
"""

import subprocess
import sys
import time
import requests

def run_command(cmd, description):
    """Run a shell command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_service(url, name):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} is responding at {url}")
            return True
        else:
            print(f"⚠️  {name} responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} is not responding: {e}")
        return False

def main():
    print("🚀 Agentic Forecast Monitoring Stack Manager")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("Usage: python monitoring_manager.py <command>")
        print("Commands:")
        print("  start    - Start the monitoring stack")
        print("  stop     - Stop the monitoring stack")
        print("  restart  - Restart the monitoring stack")
        print("  status   - Check status of all services")
        print("  logs     - Show logs from monitoring containers")
        return

    command = sys.argv[1]

    if command == "start":
        print("Starting monitoring stack...")
        success = run_command("docker compose up -d prometheus grafana", "Starting Prometheus and Grafana")
        if success:
            print("\n⏳ Waiting for services to be ready...")
            time.sleep(10)
            check_service("http://localhost:9090", "Prometheus")
            check_service("http://localhost:3000", "Grafana")
            print("\n📊 Access your monitoring:")
            print("   Prometheus: http://localhost:9090")
            print("   Grafana:    http://localhost:3000 (admin/admin)")
            print("   API:        http://localhost:8002/metrics")

    elif command == "stop":
        run_command("docker compose down", "Stopping monitoring stack")

    elif command == "restart":
        print("Restarting monitoring stack...")
        run_command("docker compose down", "Stopping services")
        success = run_command("docker compose up -d prometheus grafana", "Starting services")
        if success:
            print("⏳ Waiting for services...")
            time.sleep(10)
            check_service("http://localhost:9090", "Prometheus")
            check_service("http://localhost:3000", "Grafana")

    elif command == "status":
        print("Checking service status...")
        check_service("http://localhost:8002/health", "API Health")
        check_service("http://localhost:8002/metrics", "API Metrics")
        check_service("http://localhost:9090", "Prometheus")
        check_service("http://localhost:3000", "Grafana")

        # Check Docker containers
        try:
            result = subprocess.run("docker ps --filter name=prometheus --filter name=grafana --format 'table {{.Names}}\t{{.Status}}'",
                                  shell=True, capture_output=True, text=True)
            print("\n🐳 Docker containers:")
            print(result.stdout)
        except Exception as e:
            print(f"Could not check Docker status: {e}")

    elif command == "logs":
        print("Showing recent logs...")
        run_command("docker compose logs --tail=50 prometheus", "Prometheus logs")
        run_command("docker compose logs --tail=50 grafana", "Grafana logs")

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()

echo "=== Testing FastAPI Endpoints ==="

echo "Testing root endpoint..."
curl -s http://127.0.0.1:8000/ | jq '.' || curl -s http://127.0.0.1:8000/

echo -e "\nTesting health endpoint..."
curl -s http://127.0.0.1:8000/health | jq '.' || curl -s http://127.0.0.1:8000/health

echo -e "\nTesting session creation..."
curl -X POST "http://127.0.0.1:8000/sessions/create" \
  -H "Content-Type: application/json" \
  -d '{
    "ticketKey": "TEST-123",
    "gitUrl": "https://github.com/example/repo.git",
    "baseBranch": "main",
    "prompt": "Test prompt for WSL"
  }' | jq '.' || curl -X POST "http://127.0.0.1:8000/sessions/create" \
  -H "Content-Type: application/json" \
  -d '{
    "ticketKey": "TEST-123",
    "gitUrl": "https://github.com/example/repo.git",
    "baseBranch": "main",
    "prompt": "Test prompt for WSL"
  }'

echo -e "\n PI testing complete!"

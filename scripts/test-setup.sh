#!/bin/bash

# Test script to verify FastAPI + React setup
# ==========================================

set -e

echo "🧪 Testing FastAPI + React Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test functions
test_backend() {
    echo -e "${YELLOW}Testing Backend...${NC}"
    
    # Check if backend is running
    if curl -s -f http://localhost:8000/api/health > /dev/null; then
        echo -e "${GREEN}✅ Backend health check passed${NC}"
    else
        echo -e "${RED}❌ Backend health check failed${NC}"
        return 1
    fi
    
    # Test hello endpoint
    if curl -s -f http://localhost:8000/api/hello > /dev/null; then
        echo -e "${GREEN}✅ Hello endpoint working${NC}"
    else
        echo -e "${RED}❌ Hello endpoint failed${NC}"
        return 1
    fi
    
    # Test info endpoint
    if curl -s -f http://localhost:8000/api/info > /dev/null; then
        echo -e "${GREEN}✅ Info endpoint working${NC}"
    else
        echo -e "${RED}❌ Info endpoint failed${NC}"
        return 1
    fi
    
    # Test prediction endpoint
    if curl -s -f -X POST http://localhost:8000/api/predict \
        -H "Content-Type: application/json" \
        -d '{"data": {"test": true}}' > /dev/null; then
        echo -e "${GREEN}✅ Prediction endpoint working${NC}"
    else
        echo -e "${RED}❌ Prediction endpoint failed${NC}"
        return 1
    fi
}

test_frontend() {
    echo -e "${YELLOW}Testing Frontend...${NC}"
    
    # Check if frontend is running
    if curl -s -f http://localhost:5173 > /dev/null; then
        echo -e "${GREEN}✅ Frontend is accessible${NC}"
    else
        echo -e "${RED}❌ Frontend is not accessible${NC}"
        return 1
    fi
}

test_api_docs() {
    echo -e "${YELLOW}Testing API Documentation...${NC}"
    
    # Check Swagger UI
    if curl -s -f http://localhost:8000/api/docs > /dev/null; then
        echo -e "${GREEN}✅ Swagger UI accessible${NC}"
    else
        echo -e "${RED}❌ Swagger UI not accessible${NC}"
        return 1
    fi
    
    # Check OpenAPI JSON
    if curl -s -f http://localhost:8000/api/openapi.json > /dev/null; then
        echo -e "${GREEN}✅ OpenAPI JSON accessible${NC}"
    else
        echo -e "${RED}❌ OpenAPI JSON not accessible${NC}"
        return 1
    fi
}

# Main test execution
main() {
    echo "Starting comprehensive test suite..."
    echo ""
    
    # Wait a moment for services to be ready
    echo "Waiting for services to be ready..."
    sleep 5
    
    # Run tests
    if test_backend; then
        echo -e "${GREEN}Backend tests passed!${NC}"
    else
        echo -e "${RED}Backend tests failed!${NC}"
        exit 1
    fi
    
    echo ""
    
    if test_frontend; then
        echo -e "${GREEN}Frontend tests passed!${NC}"
    else
        echo -e "${RED}Frontend tests failed!${NC}"
        exit 1
    fi
    
    echo ""
    
    if test_api_docs; then
        echo -e "${GREEN}API documentation tests passed!${NC}"
    else
        echo -e "${RED}API documentation tests failed!${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}🎉 All tests passed! Your setup is working correctly.${NC}"
    echo ""
    echo "You can now:"
    echo "- Visit the frontend: http://localhost:5173"
    echo "- Check the API docs: http://localhost:8000/api/docs"
    echo "- Test API endpoints directly"
    echo ""
    echo "Ready for development! 🚀"
}

# Run main function
main "$@" 
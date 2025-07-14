#!/bin/bash
# Railway deployment testing script
# ================================

set -e

echo "🚂 Testing Railway Deployment Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Railway CLI is installed
check_railway_cli() {
    print_status "Checking Railway CLI installation..."
    if command -v railway &> /dev/null; then
        print_success "Railway CLI is installed"
        railway --version
    else
        print_error "Railway CLI is not installed"
        echo "Install it with: npm install -g @railway/cli"
        exit 1
    fi
}

# Check if user is logged in
check_railway_auth() {
    print_status "Checking Railway authentication..."
    if railway whoami &> /dev/null; then
        print_success "Railway authentication verified"
        railway whoami
    else
        print_error "Not logged in to Railway"
        echo "Login with: railway login"
        exit 1
    fi
}

# Test backend configuration
test_backend_config() {
    print_status "Testing backend configuration..."
    
    if [ -f "backend/railway.json" ]; then
        print_success "Backend railway.json found"
        
        # Validate JSON
        if python3 -m json.tool backend/railway.json > /dev/null 2>&1; then
            print_success "Backend railway.json is valid JSON"
        else
            print_error "Backend railway.json is invalid JSON"
            return 1
        fi
        
        # Check for required fields
        if grep -q "uvicorn app.main:app" backend/railway.json; then
            print_success "Backend start command is correct"
        else
            print_error "Backend start command may be incorrect"
            return 1
        fi
    else
        print_error "Backend railway.json not found"
        return 1
    fi
}

# Test frontend configuration
test_frontend_config() {
    print_status "Testing frontend configuration..."
    
    if [ -f "frontend/railway.json" ]; then
        print_success "Frontend railway.json found"
        
        # Validate JSON
        if python3 -m json.tool frontend/railway.json > /dev/null 2>&1; then
            print_success "Frontend railway.json is valid JSON"
        else
            print_error "Frontend railway.json is invalid JSON"
            return 1
        fi
        
        # Check for serve command
        if grep -q "serve" frontend/railway.json; then
            print_success "Frontend serve command found"
        else
            print_error "Frontend serve command not found"
            return 1
        fi
    else
        print_error "Frontend railway.json not found"
        return 1
    fi
}

# Test backend locally with Railway environment
test_backend_locally() {
    print_status "Testing backend locally with Railway environment..."
    
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found, creating one..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Test import
    if python -c "from app.main import app; print('✅ FastAPI app imports successfully')" 2>/dev/null; then
        print_success "Backend app imports successfully"
    else
        print_error "Backend app import failed"
        cd ..
        return 1
    fi
    
    cd ..
}

# Test frontend build
test_frontend_build() {
    print_status "Testing frontend build..."
    
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        print_warning "Node modules not found, installing..."
        npm ci
    fi
    
    # Test build
    if npm run build; then
        print_success "Frontend builds successfully"
        
        # Check if dist directory exists
        if [ -d "dist" ]; then
            print_success "Frontend dist directory created"
        else
            print_error "Frontend dist directory not found after build"
            cd ..
            return 1
        fi
    else
        print_error "Frontend build failed"
        cd ..
        return 1
    fi
    
    cd ..
}

# Test with Railway CLI
test_with_railway_cli() {
    print_status "Testing with Railway CLI..."
    
    # Test backend
    print_status "Testing backend with railway run..."
    cd backend
    if timeout 10s railway run python -c "from app.main import app; print('✅ Backend works with Railway environment')" 2>/dev/null; then
        print_success "Backend works with Railway environment"
    else
        print_warning "Backend test with Railway environment timed out or failed (this might be normal if not linked to a project)"
    fi
    cd ..
    
    # Test frontend
    print_status "Testing frontend with railway run..."
    cd frontend
    if timeout 10s railway run npm run build 2>/dev/null; then
        print_success "Frontend builds with Railway environment"
    else
        print_warning "Frontend test with Railway environment timed out or failed (this might be normal if not linked to a project)"
    fi
    cd ..
}

# Main execution
main() {
    check_railway_cli
    check_railway_auth
    
    echo ""
    print_status "Testing configuration files..."
    test_backend_config
    test_frontend_config
    
    echo ""
    print_status "Testing local builds..."
    test_backend_locally
    test_frontend_build
    
    echo ""
    print_status "Testing with Railway CLI..."
    test_with_railway_cli
    
    echo ""
    print_success "🎉 All tests completed!"
    echo ""
    echo "Next steps for Railway deployment:"
    echo "1. Create two services in Railway:"
    echo "   - Backend service with root directory: backend/"
    echo "   - Frontend service with root directory: frontend/"
    echo "2. Set environment variables:"
    echo "   - Backend: PORT (automatically set by Railway)"
    echo "   - Frontend: Set VITE_API_URL to your backend service URL"
    echo "3. Deploy both services"
    echo ""
    echo "🔗 Useful Railway commands:"
    echo "   railway link    # Link to Railway project"
    echo "   railway up      # Deploy current directory"
    echo "   railway logs    # View deployment logs"
    echo "   railway shell   # Open shell with Railway environment"
}

# Run main function
main "$@" 
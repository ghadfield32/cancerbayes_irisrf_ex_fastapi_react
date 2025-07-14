import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import MLModelFrontend from './components/MLModelFrontend';
import Login from './components/Login';
import Layout from './components/Layout';
import { apiService } from './services/api';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [backendReady, setBackendReady] = useState(false);

  useEffect(() => {
    console.debug('[App] useEffect starting - checking auth and readiness');
    (async () => {
      try {
        console.debug('[App] Calling checkAuthStatus');
        await checkAuthStatus();
        console.debug('[App] checkAuthStatus completed successfully');
      } catch (error) {
        console.error('[App] checkAuthStatus failed:', error);
      }

      console.debug('[App] Starting waitUntilReady');
      waitUntilReady();
    })();
  }, []);

  const waitUntilReady = async (attempt = 0) => {
    const jitter = Math.random() * 250;  // ±250 ms jitter
    try {
      console.debug(`[ready] Polling backend readiness (attempt ${attempt + 1})`);
      const res = await apiService.getReady();
      console.debug('[ready] Backend response:', res);
      console.debug('[ready] res?.ready value:', res?.ready);
      console.debug('[ready] backendReady before set:', backendReady);

      setBackendReady(Boolean(res?.ready));
      console.debug('[ready] backendReady after set:', Boolean(res?.ready));

      if (!res?.ready) {
        const delay = Math.min(1000 * 2 ** attempt, 8000) + jitter;
        console.debug(`[ready] Backend not ready, retrying in ${delay}ms (attempt ${attempt + 1})`);
        setTimeout(() => waitUntilReady(attempt + 1), delay);
      } else {
        console.debug('[ready] Backend is ready!');
      }
    } catch (err) {
      console.error('[ready] Poll failed:', err);
      const delay = Math.min(1000 * 2 ** attempt, 8000) + jitter;
      console.debug(`[ready] Poll failed, retrying in ${delay}ms (attempt ${attempt + 1})`);
      setTimeout(() => waitUntilReady(attempt + 1), delay);
    }
  };

  const checkAuthStatus = async () => {
    console.debug('[auth] checkAuthStatus starting');
    const token = localStorage.getItem('jwt');
    console.debug('[auth] Token found:', !!token);

    if (!token) {
      console.debug('[auth] No token found');
      setIsLoading(false);
      return;
    }

    try {
      console.debug('[auth] Validating existing token');
      // Verify token is still valid by calling a protected endpoint
      const healthResponse = await apiService.getHealth();
      console.debug('[auth] Health check response:', healthResponse);
      setUser({ username: 'authenticated' });
      setIsAuthenticated(true);
      console.debug('[auth] Token is valid');
    } catch (error) {
      console.error('[auth] Token validation failed:', error);
      localStorage.removeItem('jwt');
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
      console.debug('[auth] checkAuthStatus completed');
    }
  };

  const handleLogin = async (credentials) => {
    try {
      console.debug('[auth] Attempting login for:', credentials.username);
      const response = await apiService.login(credentials);
      localStorage.setItem('jwt', response.access_token);
      setUser({ username: credentials.username });
      setIsAuthenticated(true);
      console.debug('[auth] Login successful');
      return { success: true };
    } catch (error) {
      console.error('[auth] Login failed:', error);
      return { success: false, error: error.message };
    }
  };

  const handleLogout = () => {
    console.debug('[auth] Logging out');
    localStorage.removeItem('jwt');
    setUser(null);
    setIsAuthenticated(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />

        <Routes>
          <Route 
            path="/login" 
            element={
              isAuthenticated ? (
                <Navigate to="/" replace />
              ) : (
                <Login onLogin={handleLogin} backendReady={backendReady} />
              )
            } 
          />

          <Route 
            path="/" 
            element={
              isAuthenticated ? (
                <Layout user={user} onLogout={handleLogout}>
                  <MLModelFrontend />
                </Layout>
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />

          <Route 
            path="/dashboard" 
            element={
              isAuthenticated ? (
                <Layout user={user} onLogout={handleLogout}>
                  <MLModelFrontend />
                </Layout>
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 





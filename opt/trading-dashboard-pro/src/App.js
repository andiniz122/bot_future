import React, { useState, useEffect } from 'react';

function App() {
  const [apiStatus, setApiStatus] = useState('Checking...');
  const [data, setData] = useState(null);

  useEffect(() => {
    const testAPI = async () => {
      try {
        const response = await fetch('http://62.72.1.122:8000/api/status');
        if (response.ok) {
          const data = await response.json();
          setApiStatus('Connected ✅');
          setData(data);
        } else {
          setApiStatus('API Error ❌');
        }
      } catch (error) {
        setApiStatus('Connection Failed ❌');
        console.error('API Error:', error);
      }
    };

    testAPI();
  }, []);

  return (
    <div className="container">
      <div className="header">
        <h1 className="title">📈 Trading Dashboard Pro</h1>
        <p>AI-Powered Trading Platform v6.0</p>
        <p>Node.js v20.19.2 - Ready!</p>
      </div>

      <div className="grid">
        <div className="card">
          <h3>🔗 Connection Status</h3>
          <div className={`status ${apiStatus.includes('✅') ? 'connected' : 'error'}`}>
            API Status: {apiStatus}
          </div>
          <p><strong>Frontend:</strong> http://62.72.1.122:5173</p>
          <p><strong>Backend:</strong> http://62.72.1.122:8000</p>
          {data && (
            <div>
              <p><strong>Version:</strong> {data.version}</p>
              <p><strong>Status:</strong> {data.status}</p>
            </div>
          )}
        </div>

        <div className="card">
          <h3>🚀 Quick Actions</h3>
          <button 
            className="button"
            onClick={() => window.open('http://62.72.1.122:8000/docs', '_blank')}
          >
            📚 API Documentation
          </button>
          <button 
            className="button"
            onClick={() => window.location.reload()}
          >
            🔄 Refresh Dashboard
          </button>
        </div>
      </div>

      <div className="footer">
        <p>🎉 Trading Dashboard Pro successfully deployed!</p>
      </div>
    </div>
  );
}

export default App;

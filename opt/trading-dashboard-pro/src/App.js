import React from 'react';
import './index.css';
import BloombergDashboard from './components/charts/BloombergDashboard'; // atualize o caminho se necessário

function App() {
  return (
    <React.StrictMode>
      <BloombergDashboard />
    </React.StrictMode>
  );
}

export default App;
// src/App.js
import React from 'react';
import './index.css'; // Importa estilos globais
// Importa o seu componente principal do Dashboard
import RSIMACDDashboard from './components/charts/RSIMACDDashboard'; 

function App() {
  return (
    <React.StrictMode>
      {/* Renderiza o componente principal do Dashboard diretamente */}
      <RSIMACDDashboard /> 
    </React.StrictMode>
  );
}

export default App;
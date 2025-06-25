import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Importa o seu componente App (removida a extensão .jsx para maior compatibilidade)
import './index.css'; // Importa o ficheiro CSS global, se existir

// Cria o ponto de montagem da aplicação React na sua página HTML
// O 'root' é geralmente um elemento <div id="root"> no seu ficheiro index.html
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* Renderiza o componente principal da sua aplicação (App) */}
    <App />
  </React.StrictMode>,
);

import React from 'react'
import ReactDOM from 'react-dom/client' // Importa a API moderna do React 18
import App from './App' // Componente principal da aplicação
import './index.css' // Estilos globais

// 1. Seleciona o elemento raiz do HTML
const rootElement = document.getElementById('root')

// 2. Cria uma raiz de renderização React
const root = ReactDOM.createRoot(rootElement)

// 3. Renderiza a aplicação dentro do StrictMode
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
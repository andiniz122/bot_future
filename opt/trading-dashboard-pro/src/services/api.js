// src/services/api.js

// URL base da sua API FastAPI
const API_BASE_URL = 'http://62.72.1.122:8000'; //

/**
 * Busca dados atuais do mercado da API.
 * @returns {Promise<Object|null>} Dados atuais do mercado ou null em caso de erro.
 */
export const fetchCurrentData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/current`); //
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`); //
    }
    const data = await response.json(); //
    return data;
  } catch (error) {
    console.error('Erro ao buscar dados atuais:', error); //
    throw error; // Propagar o erro para o componente que chamou
  }
};

/**
 * Busca dados históricos de preços e volume para um período.
 * @param {string} period Período desejado (ex: '1d', '1mo').
 * @returns {Promise<Object|null>} Dados históricos processados para gráficos ou null em caso de erro.
 */
export const fetchHistoricalData = async (period) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/precos/${period}`); //
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`); //
    }
    const data = await response.json(); //
    return data; // O componente App.js ainda fará o processamento para gráficos
  } catch (error) {
    console.error('Erro ao buscar dados históricos:', error); //
    throw error; // Propagar o erro para o componente que chamou
  }
};

/**
 * Busca o status detalhado do bot de trading.
 * @returns {Promise<Object|null>} Status do bot ou null em caso de erro.
 */
export const fetchBotStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trading-bot/status`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar status do bot:', error);
    throw error;
  }
};

/**
 * Envia comando para iniciar o bot de trading.
 * @returns {Promise<Object>} Resposta da API.
 */
export const startBot = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trading-bot/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}) // Envia corpo vazio conforme a API espera
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao iniciar o bot:', error);
    throw error;
  }
};

/**
 * Envia comando para parar o bot de trading.
 * @returns {Promise<Object>} Resposta da API.
 */
export const stopBot = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trading-bot/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}) // Envia corpo vazio
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao parar o bot:', error);
    throw error;
  }
};

/**
 * Busca as posições ativas do bot.
 * @returns {Promise<Object|null>} Posições ativas ou null em caso de erro.
 */
export const fetchActivePositions = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trading-bot/positions`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar posições ativas:', error);
    throw error;
  }
};

/**
 * Busca os sinais recentes do bot.
 * @returns {Promise<Object|null>} Sinais recentes ou null em caso de erro.
 */
export const fetchRecentSignals = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trading-bot/signals`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar sinais recentes:', error);
    throw error;
  }
};

/**
 * Busca as recomendações de backtest.
 * @returns {Promise<Object|null>} Recomendações ou null em caso de erro.
 */
export const fetchBacktestRecommendations = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/backtest-recommendations`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar recomendações de backtest:', error);
    throw error;
  }
};

/**
 * Busca os alertas do sistema.
 * @returns {Promise<Object|null>} Alertas ou null em caso de erro.
 */
export const fetchAlerts = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar alertas:', error);
    throw error;
  }
};

/**
 * Busca o calendário econômico.
 * @returns {Promise<Object|null>} Calendário econômico ou null em caso de erro.
 */
export const fetchEconomicCalendar = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/calendar`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar calendário econômico:', error);
    throw error;
  }
};

/**
 * Busca os eventos econômicos próximos.
 * @returns {Promise<Object|null>} Eventos próximos ou null em caso de erro.
 */
export const fetchUpcomingEvents = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/events`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar eventos próximos:', error);
    throw error;
  }
};

/**
 * Busca os dados de sentimento de mercado.
 * @returns {Promise<Object|null>} Dados de sentimento ou null em caso de erro.
 */
export const fetchMarketSentiment = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/sentiment`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar sentimento de mercado:', error);
    throw error;
  }
};

/**
 * Busca o histórico de sentimento de mercado.
 * @returns {Promise<Object|null>} Histórico de sentimento ou null em caso de erro.
 */
export const fetchSentimentHistory = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/sentiment/history`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar histórico de sentimento:', error);
    throw error;
  }
};

/**
 * Busca dados MACD em tempo real para um ativo.
 * @param {string} asset O ativo (ex: 'btc').
 * @returns {Promise<Object|null>} Dados MACD em tempo real ou null em caso de erro.
 */
export const fetchRealtimeMACD = async (asset) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/macd/realtime/${asset}`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Erro ao buscar MACD em tempo real para ${asset}:`, error);
    throw error;
  }
};

/**
 * Busca os cruzamentos MACD recentes.
 * @returns {Promise<Object|null>} Cruzamentos recentes ou null em caso de erro.
 */
export const fetchRecentCrossovers = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/crossovers/recent`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Erro ao buscar cruzamentos MACD recentes:', error);
    throw error;
  }
};
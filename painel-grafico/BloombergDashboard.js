import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ComposedChart } from 'recharts'; // Removido Area, AreaChart se não estiver usando
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Zap, Globe, Clock, AlertTriangle, Play, Pause, Settings, Wifi, WifiOff, Database, Brain, Calendar, Bell, Info, Compass } from 'lucide-react'; // Adicionado Compass para inclinação

const BloombergDashboard = () => {
  // Estado para armazenar os dados das APIs
  const [marketData, setMarketData] = useState(null); // Dados de /api/current
  const [chartData, setChartData] = useState({ combined: [], period: '1d', dataPoints: 0, lastUpdate: null, fallback: false }); // Dados de /api/precos/{period}
  const [selectedPeriod, setSelectedPeriod] = useState('1d'); // Período selecionado para o gráfico
  const [botStatus, setBotStatus] = useState(null); // Dados de /api/trading-bot/status
  const [sentimentData, setSentimentData] = useState(null); // Dados de /api/sentiment ou ws/sentiment
  const [alerts, setAlerts] = useState([]); // Dados de /api/alerts
  const [economicCalendar, setEconomicCalendar] = useState([]); // Dados de /api/calendar
  const [backtestRecommendations, setBacktestRecommendations] = useState([]); // Dados de /api/backtest-recommendations
  const [financialData, setFinancialData] = useState({ // Dados de /api/trading-bot/performance e /api/trading-bot/positions
    testnet: {
      balance: 0,
      startBalance: 10000,
      totalPnL: 0,
      dailyPnL: 0,
      totalTrades: 0,
      winningTrades: 0,
      winRate: 0,
      maxDrawdown: 0,
      roiPercent: 0,
      lastUpdate: null
    },
    live: {
      balance: 0,
      startBalance: 0,
      totalPnL: 0,
      dailyPnL: 0,
      totalTrades: 0,
      winningTrades: 0,
      winRate: 0,
      maxDrawdown: 0,
      roiPercent: 0,
      lastUpdate: null
    },
    recentTrades: [] // Posições ativas e sinais recentes
  });

  // NOVO ESTADO: Para indicadores em tempo real (RSI e Ângulos MACD)
  const [realtimeIndicators, setRealtimeIndicators] = useState({
    btc: { rsi: null, macd: null, macdAngle: null },
    eth: { rsi: null, macd: null, macdAngle: null }
  });

  // Estados da UI/UX
  const [isLoading, setIsLoading] = useState(true); // Indica se a carga inicial está em andamento
  const [connectionStatus, setConnectionStatus] = useState('connecting'); // Status geral da conexão com o backend
  const [environment, setEnvironment] = useState('testnet'); // Ambiente do bot (testnet ou live)
  const [showEnvironmentModal, setShowEnvironmentModal] = useState(false); // Exibir modal de seleção de ambiente
  const [selectedFinancialTab, setSelectedFinancialTab] = useState('testnet'); // Aba selecionada para dados financeiros
  const [wsConnections, setWsConnections] = useState({ // Status das conexões WebSocket
    sentiment: false,
    ohlcv: false,
    rsiMacd: false
  });

  // Backend API Base URL - ajuste conforme necessário
  const API_BASE = 'http://62.72.1.122:8000'; // Exemplo: 'http://localhost:8000' ou 'http://seu_ip_publico:8000'

  // --- Funções de Formatação ---
  const formatCurrency = useCallback((value) => {
    if (value === null || value === undefined) return '$0.00';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  }, []);

  const formatPercentage = useCallback((value) => {
    if (value === null || value === undefined) return '0.00%';
    const formatted = value.toFixed(2);
    return `${value > 0 ? '+' : ''}${formatted}%`;
  }, []);

  const formatPnL = useCallback((value, isPercentage = false) => {
    let colorClass = 'text-gray-400';
    if (value > 0) {
      colorClass = 'text-green-400';
    } else if (value < 0) {
      colorClass = 'text-red-400';
    }

    const formattedValue = isPercentage ? formatPercentage(value) : formatCurrency(value);
    return { formatted: formattedValue, colorClass };
  }, [formatCurrency, formatPercentage]);


  // --- Funções de Fetch de Dados (REST) ---

  // Busca dados de mercado em tempo real de /api/current
  const fetchMarketData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/current`);
      if (!response.ok) throw new Error(`Failed to fetch market data: ${response.statusText}`);
      const data = await response.json();
      setMarketData(data);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Error fetching market data:', error);
      setConnectionStatus('error');
    }
  }, []);

  // Fetch historical chart data from /api/precos/{period}
  const fetchChartData = useCallback(async (period) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/precos/${period}`);
      if (!response.ok) throw new Error(`Failed to fetch chart data for ${period}: ${response.statusText}`);
      const data = await response.json();

      const chartPoints = [];
      if (data.dates && data.assets) {
        data.dates.forEach((date, index) => {
          const point = {
            time: new Date(date).getTime(),
            date: new Date(date).toLocaleDateString(),
            timestamp: date
          };

          Object.keys(data.assets).forEach(asset => {
            const assetData = data.assets[asset];
            // Garante que os valores são numéricos ou null
            point[`${asset}_price`] = Number(assetData?.price_data?.[index]) || null;
            point[`${asset}_volume`] = Number(assetData?.volume_data?.[index]) || null;
            point[`${asset}_macd`] = Number(assetData?.macd_data?.[index]) || null;
            point[`${asset}_macd_signal`] = Number(assetData?.macd_signal_data?.[index]) || null;
            point[`${asset}_macd_hist`] = Number(assetData?.macd_hist_data?.[index]) || null;
            point[`${asset}_rsi`] = Number(assetData?.rsi_data?.[index]) || null; // NOVO: RSI histórico
          });
          chartPoints.push(point);
        });
      }

      setChartData({
        combined: chartPoints,
        period: period,
        dataPoints: chartPoints.length,
        lastUpdate: new Date().toISOString(),
        fallback: data.status === 'error' // Se o backend indicar fallback/erro
      });
    } catch (error) {
      console.error('Error fetching chart data:', error);
      setChartData(prev => ({
        ...prev,
        combined: [], // Limpa os dados em caso de erro
        period: period,
        dataPoints: 0,
        lastUpdate: new Date().toISOString(),
        fallback: true // Indica que está em modo de fallback
      }));
    } finally {
      setIsLoading(false); // Desativa loading do gráfico
    }
  }, []);

  // Fetch trading bot status from /api/trading-bot/status
  const fetchBotStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/trading-bot/status`);
      if (!response.ok) throw new Error(`Failed to fetch bot status: ${response.statusText}`);
      const data = await response.json();
      setBotStatus(data);
      // Atualiza o ambiente do frontend com o ambiente reportado pelo bot
      if (data.environment) {
        setEnvironment(data.environment.toLowerCase());
      }
    } catch (error) {
      console.error('Error fetching bot status:', error);
      // Pode definir um status de erro para o bot aqui se necessário
      setBotStatus(prev => prev ? { ...prev, status: 'error', error: error.message } : null);
    }
  }, []);

  // Fetch market sentiment from /api/sentiment (REST fallback para WS)
  const fetchSentiment = useCallback(async () => {
    // Esta função será usada como fallback se o WebSocket de sentimento não conectar
    try {
      const response = await fetch(`${API_BASE}/api/sentiment`);
      if (!response.ok) throw new Error(`Failed to fetch sentiment: ${response.statusText}`);
      const data = await response.json();
      setSentimentData(data);
    } catch (error) {
      console.error('Error fetching sentiment (REST):', error);
    }
  }, []);

  // Fetch alerts from /api/alerts
  const fetchAlerts = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/alerts`);
      if (!response.ok) throw new Error(`Failed to fetch alerts: ${response.statusText}`);
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  }, []);

  // Fetch economic calendar from /api/calendar
  const fetchEconomicCalendar = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/calendar`);
      if (!response.ok) throw new Error(`Failed to fetch economic calendar: ${response.statusText}`);
      const data = await response.json();
      // Filtra apenas eventos futuros para exibir os próximos
      const now = new Date();
      const upcoming = (data.upcoming_events || []).filter(event => new Date(event.date) >= now);
      setEconomicCalendar(upcoming);
    } catch (error) {
      console.error('Error fetching economic calendar:', error);
    }
  }, []);

  // Fetch trading performance data for both environments
  const fetchFinancialData = useCallback(async () => {
    try {
      // Get bot performance data
      const performanceResponse = await fetch(`${API_BASE}/api/trading-bot/performance`);
      if (performanceResponse.ok) {
        const performanceData = await performanceResponse.json();

        // Get recent trades/positions
        const positionsResponse = await fetch(`${API_BASE}/api/trading-bot/positions`);
        const signalsResponse = await fetch(`${API_BASE}/api/trading-bot/signals`);

        let positions = [];
        let signals = [];

        if (positionsResponse.ok) {
          const posData = await positionsResponse.json();
          positions = posData.positions || [];
        }

        if (signalsResponse.ok) {
          const sigData = await signalsResponse.json();
          signals = sigData.signals || [];
        }

        // Combina posições ativas e sinais recentes para 'recentTrades'
        const combinedRecentTrades = [...positions, ...signals]
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()) // Ordena por data mais recente
          .slice(0, 10); // Pega os 10 mais recentes

        const currentEnv = botStatus?.environment?.toLowerCase() || environment; // Usa o ambiente do bot se disponível

        setFinancialData(prev => ({
          ...prev,
          [currentEnv]: { // Atualiza apenas o ambiente correto
            balance: performanceData.current_balance || 0,
            startBalance: performanceData.start_balance || (currentEnv === 'testnet' ? 10000 : 0),
            totalPnL: performanceData.total_pnl || 0,
            dailyPnL: performanceData.daily_pnl || 0,
            totalTrades: performanceData.total_trades || 0,
            winningTrades: performanceData.winning_trades || 0,
            winRate: performanceData.win_rate || 0,
            maxDrawdown: performanceData.max_drawdown || 0,
            roiPercent: performanceData.roi_percentage || 0,
            lastUpdate: performanceData.last_update || new Date().toISOString()
          },
          recentTrades: combinedRecentTrades
        }));
      }
    } catch (error) {
      console.error('Error fetching financial data:', error);
    }
  }, [botStatus, environment]);

  // Fetch backtest recommendations from /api/backtest-recommendations
  const fetchBacktestRecommendations = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/backtest-recommendations`);
      if (!response.ok) throw new Error(`Failed to fetch backtest recommendations: ${response.statusText}`);
      const data = await response.json();
      setBacktestRecommendations(data.recommendations || []);
    } catch (error) {
      console.error('Error fetching backtest recommendations:', error);
    }
  }, []);


  // Setup WebSocket connections
  const setupWebSockets = useCallback(() => {
    // Sentiment WebSocket
    try {
      const sentimentWs = new WebSocket(`ws://${API_BASE.split('//')[1]}/ws/sentiment`);
      sentimentWs.onopen = () => {
        setWsConnections(prev => ({ ...prev, sentiment: true }));
        console.log('Sentiment WebSocket connected.');
      };
      sentimentWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setSentimentData(data);
      };
      sentimentWs.onclose = (e) => {
        setWsConnections(prev => ({ ...prev, sentiment: false }));
        console.warn('Sentiment WebSocket disconnected:', e.code, e.reason);
        setTimeout(() => setupWebSockets(), 5000); // Tenta reconectar
      };
      sentimentWs.onerror = (err) => {
        console.error('Sentiment WebSocket error:', err);
        sentimentWs.close();
      };
    } catch (error) {
      console.error('Sentiment WebSocket setup error:', error);
    }

    // OHLCV WebSocket
    try {
      const ohlcvWs = new WebSocket(`ws://${API_BASE.split('//')[1]}/ws/ohlcv`);
      ohlcvWs.onopen = () => {
        setWsConnections(prev => ({ ...prev, ohlcv: true }));
        console.log('OHLCV WebSocket connected.');
      };
      ohlcvWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'ohlcv_update') {
          const assetKey = data.asset.toLowerCase();
          setChartData(prev => {
            const newCombined = [...prev.combined];
            const newCandle = data.candle;

            // Encontre ou adicione o ponto de dados
            let existingPointIndex = newCombined.findIndex(p => p.time === newCandle.timestamp * 1000);
            if (existingPointIndex === -1) {
              // Adiciona um novo ponto se não for uma atualização da vela atual
              existingPointIndex = newCombined.length;
              newCombined.push({
                time: newCandle.timestamp * 1000,
                date: new Date(newCandle.timestamp * 1000).toLocaleDateString(),
                timestamp: newCandle.datetime,
              });
            }
            const currentPoint = newCombined[existingPointIndex];

            // Atualiza dados de preço e volume
            currentPoint[`${assetKey}_price`] = newCandle.close;
            currentPoint[`${assetKey}_volume`] = newCandle.volume;

            // Atualiza dados MACD se presentes
            if (data.macd_data) {
              currentPoint[`${assetKey}_macd`] = data.macd_data.macd?.[data.macd_data.macd.length - 1] || null;
              currentPoint[`${assetKey}_macd_signal`] = data.macd_data.signal?.[data.macd_data.signal.length - 1] || null;
              currentPoint[`${assetKey}_macd_hist`] = data.macd_data.histogram?.[data.macd_data.histogram.length - 1] || null;
            }

            // NOVO: Atualiza dados RSI se presentes
            if (data.rsi_data) {
              currentPoint[`${assetKey}_rsi`] = data.rsi_data.last_value || null;
            }

            // Mantenha apenas os últimos N pontos para evitar que o gráfico cresça indefinidamente
            return { ...prev, combined: newCombined.slice(-200) }; // Ex: Mantenha os últimos 200 pontos
          });

          // NOVO: Atualiza o estado de indicadores em tempo real para o card
          setRealtimeIndicators(prev => ({
            ...prev,
            [assetKey]: {
              rsi: data.rsi_data || null,
              macd: data.macd_data || null,
              macdAngle: data.macd_angle_data || null
            }
          }));
        }
      };
      ohlcvWs.onclose = (e) => {
        setWsConnections(prev => ({ ...prev, ohlcv: false }));
        console.warn('OHLCV WebSocket disconnected:', e.code, e.reason);
        setTimeout(() => setupWebSockets(), 5000); // Tenta reconectar
      };
      ohlcvWs.onerror = (err) => {
        console.error('OHLCV WebSocket error:', err);
        ohlcvWs.close();
      };
    } catch (error) {
      console.error('OHLCV WebSocket setup error:', error);
    }

    // RSI/MACD WebSocket (para dados específicos que não estão no OHLCV principal)
    try {
      const rsiMacdWs = new WebSocket(`ws://${API_BASE.split('//')[1]}/ws/rsi-macd`);
      rsiMacdWs.onopen = () => {
        setWsConnections(prev => ({ ...prev, rsiMacd: true }));
        console.log('RSI/MACD WebSocket connected.');
      };
      rsiMacdWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('RSI/MACD specific update:', data);
        // Exemplo: se este WebSocket envia um sinal combinado global
        // if (data.type === 'combined_signal') {
        //   setAlerts(prev => [...prev, { title: "Sinal Combinado!", message: data.message, severity: "MEDIUM", timestamp: new Date().toISOString() }]);
        // }
      };
      rsiMacdWs.onclose = (e) => {
        setWsConnections(prev => ({ ...prev, rsiMacd: false }));
        console.warn('RSI/MACD WebSocket disconnected:', e.code, e.reason);
        setTimeout(() => setupWebSockets(), 5000); // Tenta reconectar
      };
      rsiMacdWs.onerror = (err) => {
        console.error('RSI/MACD WebSocket error:', err);
        rsiMacdWs.close();
      };
    } catch (error) {
      console.error('RSI/MACD WebSocket setup error:', error);
    }
  }, [API_BASE]); // Adicionado API_BASE como dependência


  // Bot control functions with environment selection
  const sendBotCommand = useCallback(async (endpoint, method = 'POST', body = {}) => {
    try {
      const response = await fetch(`${API_BASE}/api/trading-bot/${endpoint}`, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to ${endpoint}: ${errorData.detail || response.statusText}`);
      }
      const result = await response.json();
      console.log(`Bot ${endpoint} successful:`, result);
      fetchBotStatus(); // Atualiza o status do bot após o comando
      return true;
    } catch (error) {
      console.error(`Error with bot ${endpoint}:`, error);
      alert(`Error: ${error.message}`);
      return false;
    }
  }, [fetchBotStatus, API_BASE]);

  const startBot = useCallback(() => {
    confirmStart(); // Usa a função de confirmação para o LIVE MODE
  }, []); // Removido sendBotCommand da dependência para evitar loop

  const stopBot = useCallback(() => {
    sendBotCommand('stop');
  }, [sendBotCommand]);

  const confirmStart = () => {
    if (environment === 'live') {
      const confirmed = window.confirm(
        '⚠️ AVISO: Você está prestes a iniciar o bot em modo LIVE com DINHEIRO REAL!\n\n' +
        'Isso executará trades reais na sua conta.\n\n' +
        'Tem certeza absoluta que deseja continuar?'
      );
      if (!confirmed) return;
    }
    sendBotCommand('start', 'POST', { environment: environment });
  };

  const handleEnvironmentChange = (newEnv) => {
    if (botStatus?.status === 'running') {
      alert('Por favor, pare o bot antes de mudar o ambiente.');
      return;
    }
    setEnvironment(newEnv);
    setShowEnvironmentModal(false);
  };

  // Initial data load and setup intervals
  useEffect(() => {
    const loadInitialData = async () => {
      // Todas as chamadas de fetch REST para dados iniciais
      await Promise.all([
        fetchMarketData(),
        fetchChartData(selectedPeriod),
        fetchBotStatus(),
        fetchSentiment(), // Será sobrescrito pelo WS se conectar
        fetchAlerts(),
        fetchEconomicCalendar(),
        fetchBacktestRecommendations(),
        fetchFinancialData()
      ]);
      setIsLoading(false); // Indica que a carga inicial está completa
    };

    loadInitialData(); // Carrega os dados REST iniciais
    setupWebSockets(); // Configura e tenta conectar os WebSockets

    // Setup intervals for periodic updates
    const intervals = {
      marketData: setInterval(fetchMarketData, 30000), // A cada 30 segundos
      botStatus: setInterval(fetchBotStatus, 15000),    // A cada 15 segundos
      alerts: setInterval(fetchAlerts, 60000),         // A cada 1 minuto
      calendar: setInterval(fetchEconomicCalendar, 300000), // A cada 5 minutos
      backtest: setInterval(fetchBacktestRecommendations, 1200000), // A cada 20 minutos
      financial: setInterval(fetchFinancialData, 10000) // A cada 10 segundos
    };

    // Função de limpeza para parar os intervalos ao desmontar o componente
    return () => {
      Object.values(intervals).forEach(clearInterval);
    };
  }, [
    fetchMarketData, fetchChartData, selectedPeriod, fetchBotStatus, fetchSentiment,
    fetchAlerts, fetchEconomicCalendar, fetchBacktestRecommendations, fetchFinancialData, setupWebSockets
  ]);

  // Update chart data when period changes
  useEffect(() => {
    if (selectedPeriod) {
      fetchChartData(selectedPeriod);
    }
  }, [selectedPeriod, fetchChartData]);

  // Tela de Carregamento Inicial
  if (isLoading && !marketData) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-xl">Carregando Dashboard de Trading...</p>
          <p className="text-sm text-gray-400 mt-2">Conectando a {API_BASE}</p>
          <p className="text-sm text-yellow-400 mt-2">Verifique se o seu backend está rodando no IP/Porta corretos.</p>
        </div>
      </div>
    );
  }

  // Dashboard Principal
  return (
    <div className="min-h-screen bg-black text-white font-mono">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <h1 className="text-2xl font-bold text-blue-400">TRADING PRO</h1>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
                  <span className="text-gray-300">{connectionStatus.toUpperCase()}</span>
                </div>
                {/* Indicadores de Conexão WebSocket */}
                <div className="flex items-center space-x-1" title="Sentiment WebSocket Status">
                  {wsConnections.sentiment ? <Wifi className="w-4 h-4 text-green-500" /> : <WifiOff className="w-4 h-4 text-red-500" />}
                  <span className="text-xs text-gray-400">SENTIMENTO</span>
                </div>
                <div className="flex items-center space-x-1" title="OHLCV WebSocket Status">
                  {wsConnections.ohlcv ? <Database className="w-4 h-4 text-green-500" /> : <Database className="w-4 h-4 text-red-500" />}
                  <span className="text-xs text-gray-400">OHLCV</span>
                </div>
                <div className="flex items-center space-x-1" title="Indicators WebSocket Status">
                  {wsConnections.rsiMacd ? <Activity className="w-4 h-4 text-green-500" /> : <Activity className="w-4 h-4 text-red-500" />}
                  <span className="text-xs text-gray-400">INDICADORES</span>
                </div>
              </div>
            </div>

            {/* Status do Mercado (BTC, GOLD, DXY) */}
            <div className="flex items-center space-x-6 text-sm">
              {marketData?.assets ? Object.entries(marketData.assets).map(([key, asset]) => (
                <div key={key} className="text-center">
                  <div className="text-xs text-gray-400 uppercase">{asset.name}</div>
                  <div className="font-bold">{formatCurrency(asset.current_price)}</div>
                  <div className={`text-xs ${asset.change > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatPercentage(asset.change_percent)}
                  </div>
                </div>
              )) : (
                <div className="text-sm text-yellow-400 flex items-center">
                  <Info className="w-4 h-4 mr-1" />
                  Carregando dados de mercado...
                </div>
              )}
            </div>

            {/* Bot Status & Controles (versão compacta) */}
            <div className="flex items-center space-x-3 bg-gray-800/50 px-4 py-2 rounded-lg border border-gray-700/50">
              {/* Status do Bot */}
              <div className="text-center">
                <div className="text-xs text-gray-400">BOT</div>
                <div className={`text-sm font-bold ${
                  botStatus?.status === 'running' ? 'text-green-400' :
                  botStatus?.status === 'stopped' ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  {botStatus?.status?.toUpperCase() || 'OFF'}
                </div>
              </div>

              {/* Seletor de Ambiente */}
              <div className="text-center border-l border-gray-600 pl-3">
                <div className="text-xs text-gray-400">ENV</div>
                <button
                  onClick={() => setShowEnvironmentModal(true)}
                  disabled={botStatus?.status === 'running'}
                  className={`px-2 py-1 text-xs font-bold rounded transition-all ${
                    environment === 'live'
                      ? 'bg-red-600 text-white hover:bg-red-700'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  } ${botStatus?.status === 'running' ? 'opacity-50 cursor-not-allowed' : 'hover:opacity-80 cursor-pointer'}`}
                  title="Clique para mudar o ambiente"
                >
                  {environment === 'live' ? '💰 LIVE' : '🧪 TEST'}
                </button>
              </div>

              {/* Botões de Controle do Bot */}
              <div className="flex space-x-1 border-l border-gray-600 pl-3">
                <button
                  onClick={confirmStart}
                  disabled={botStatus?.status === 'running'}
                  className={`p-2 rounded text-xs font-bold transition-all ${
                    environment === 'live'
                      ? 'bg-red-600 hover:bg-red-700 text-white'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  } ${botStatus?.status === 'running' ? 'bg-gray-600 cursor-not-allowed' : ''}`}
                  title={environment === 'live' ? 'Iniciar Bot - MODO LIVE' : 'Iniciar Bot - TESTNET'}
                >
                  <Play className="w-4 h-4" />
                </button>
                <button
                  onClick={stopBot}
                  disabled={botStatus?.status !== 'running'}
                  className={`p-2 rounded text-xs font-bold transition-all ${
                    botStatus?.status === 'running'
                      ? 'bg-red-600 hover:bg-red-700 text-white'
                      : 'bg-gray-600 cursor-not-allowed text-gray-400'
                  }`}
                  title="Parar Bot"
                >
                  <Pause className="w-4 h-4" />
                </button>
              </div>

              {/* P&L Display */}
              <div className="text-center border-l border-gray-600 pl-3">
                <div className="text-xs text-gray-400">P&L</div>
                <div className={`text-xs font-bold ${
                  (botStatus?.total_pnl || 0) > 0 ? 'text-green-400' :
                  (botStatus?.total_pnl || 0) < 0 ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {botStatus ? formatCurrency(botStatus.total_pnl) : '$0.00'}
                </div>
              </div>

              {/* Status Geral da Conexão */}
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-400' :
                connectionStatus === 'connecting' ? 'bg-yellow-400' :
                'bg-red-400'
              }`} title={`Status da API: ${connectionStatus}`}></div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-screen">
        {/* Sidebar Esquerda */}
        <div className="w-80 border-r border-gray-800 bg-gray-900 p-4 overflow-y-auto custom-scrollbar">
          {/* Sentimento do Mercado */}
          <div className="mb-6">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Brain className="w-4 h-4 mr-2" />
              SENTIMENTO DO MERCADO
            </h3>
            {sentimentData ? (
              <div className="space-y-3">
                <div className="bg-gray-800 p-3 rounded">
                  <div className="text-xs text-gray-400">Fear & Greed Index</div>
                  <div className="text-lg font-bold text-blue-400">{sentimentData.fear_greed_index || 'N/A'}</div>
                  <div className="text-xs text-gray-300">{sentimentData.market_mood || 'N/A'}</div>
                </div>
                {sentimentData.btc && (
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-xs text-gray-400">BTC Sentimento</div>
                    <div className="flex justify-between">
                      <span className="text-green-400">{sentimentData.btc.buyers?.toFixed(1) || '0.0'}% BULLS</span>
                      <span className="text-red-400">{sentimentData.btc.sellers?.toFixed(1) || '0.0'}% BEARS</span>
                    </div>
                    <div className="text-xs text-gray-300 mt-1">{sentimentData.btc.trend || 'N/A'}</div>
                  </div>
                )}
                {sentimentData.paxg && (
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-xs text-gray-400">PAXG Sentimento</div>
                    <div className="flex justify-between">
                      <span className="text-green-400">{sentimentData.paxg.buyers?.toFixed(1) || '0.0'}% BULLS</span>
                      <span className="text-red-400">{sentimentData.paxg.sellers?.toFixed(1) || '0.0'}% BEARS</span>
                    </div>
                    <div className="text-xs text-gray-300 mt-1">{sentimentData.paxg.trend || 'N/A'}</div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-gray-500 text-sm flex items-center">
                <Info className="w-4 h-4 mr-1" />
                Carregando sentimento...
              </div>
            )}
          </div>

          {/* Economic Calendar */}
          <div className="mb-6">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Calendar className="w-4 h-4 mr-2" />
              EVENTOS ECONÔMICOS
            </h3>
            <div className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar">
              {economicCalendar.length > 0 ? economicCalendar.slice(0, 5).map((event, index) => (
                <div key={index} className="bg-gray-800 p-2 rounded text-xs">
                  <div className="font-bold text-blue-400">{event.name}</div>
                  <div className="text-gray-400">{new Date(event.date).toLocaleDateString()} {event.time || ''}</div>
                  <div className={`text-xs ${event.importance === 'HIGH' ? 'text-red-400' : event.importance === 'MEDIUM' ? 'text-yellow-400' : 'text-green-400'}`}>
                    {event.importance} IMPACTO
                  </div>
                </div>
              )) : (
                <div className="text-gray-500 text-sm text-center">Nenhum evento futuro ou dados não disponíveis.</div>
              )}
            </div>
          </div>

          {/* Recent Alerts */}
          <div>
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Bell className="w-4 h-4 mr-2" />
              ALERTAS
            </h3>
            <div className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar">
              {alerts.length > 0 ? alerts.slice(0, 5).map((alert, index) => (
                <div key={index} className={`p-2 rounded text-xs border-l-2 ${
                  alert.severity === 'HIGH' ? 'border-red-500 bg-red-900/20' :
                  alert.severity === 'MEDIUM' ? 'border-yellow-500 bg-yellow-900/20' :
                  'border-green-500 bg-green-900/20'
                }`}>
                  <div className="font-bold">{alert.title}</div>
                  <div className="text-gray-400">{alert.message}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              )) : (
                <div className="text-gray-500 text-sm text-center">Nenhum alerta recente.</div>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Period Selector */}
          <div className="border-b border-gray-800 p-4">
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-400">TIMEFRAME:</span>
              {['5m', '15m', '1h', '1d', '5d', '1mo'].map(period => (
                <button
                  key={period}
                  onClick={() => setSelectedPeriod(period)}
                  className={`px-3 py-1 text-xs rounded ${
                    selectedPeriod === period
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {period.toUpperCase()}
                </button>
              ))}
              <div className="ml-auto text-xs text-gray-400 flex items-center space-x-4">
                <span>Pontos de Dados: {chartData.dataPoints || 0}</span>
                <span>Última Atualização: {chartData.lastUpdate ? new Date(chartData.lastUpdate).toLocaleTimeString() : 'N/A'}</span>
                {chartData.fallback && (
                  <span className="text-yellow-400 flex items-center" title="Dados do gráfico em modo de fallback (API pode ter falhado ou dados incompletos)">
                    <AlertTriangle className="w-3 h-3 mr-1" />
                    Modo Fallback
                  </span>
                )}
                {isLoading && (
                  <span className="text-blue-400 flex items-center">
                    <div className="animate-spin w-3 h-3 border border-blue-400 border-t-transparent rounded-full mr-1"></div>
                    Carregando...
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="flex-1 p-4">
            <div className="grid grid-cols-1 gap-4 h-full">
              {/* Main Price Chart */}
              <div className="bg-gray-900 border border-gray-800 rounded p-4 h-96">
                <h3 className="text-sm font-bold text-gray-400 mb-4">GRÁFICO DE PREÇO</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.combined}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="time"
                      type="number"
                      scale="time"
                      domain={['dataMin', 'dataMax']}
                      tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                      stroke="#9CA3AF"
                      fontSize={10}
                    />
                    <YAxis stroke="#9CA3AF" fontSize={10} domain={['dataMin', 'dataMax']} />
                    <Tooltip
                      labelFormatter={(time) => new Date(time).toLocaleString()}
                      formatter={(value, name) => [
                        typeof value === 'number' ? value.toFixed(2) : value,
                        name?.toUpperCase().replace('_PRICE', '') // Formata o nome da legenda
                      ]}
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '4px',
                        color: '#F9FAFB'
                      }}
                    />
                    {/* Linhas de Preço para BTC, ETH, GOLD - Certifique-se de que os nomes correspondem */}
                    <Line
                      type="monotone"
                      dataKey="btc_price"
                      stroke="#F59E0B"
                      strokeWidth={2}
                      dot={false}
                      name="BTC"
                    />
                    <Line
                      type="monotone"
                      dataKey="eth_price"
                      stroke="#10B981"
                      strokeWidth={2}
                      dot={false}
                      name="ETH"
                    />
                    <Line
                      type="monotone"
                      dataKey="gold_price"
                      stroke="#EF4444"
                      strokeWidth={2}
                      dot={false}
                      name="GOLD"
                    />
                     <Line
                      type="monotone"
                      dataKey="dxy_price"
                      stroke="#7D3C98" // Cor para DXY
                      strokeWidth={2}
                      dot={false}
                      name="DXY"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Volume and MACD Charts */}
              <div className="grid grid-cols-2 gap-4 h-64">
                {/* Volume Chart */}
                <div className="bg-gray-900 border border-gray-800 rounded p-4">
                  <h3 className="text-sm font-bold text-gray-400 mb-4 flex justify-between">
                    <span>VOLUME (BTC)</span> {/* Alterei para BTC para ser mais específico */}
                    <span className="text-xs">
                      {chartData.combined?.length > 0 && chartData.combined.some(d => d.btc_volume !== null) ?
                        `${chartData.combined.length} barras` :
                        'Sem dados'
                      }
                    </span>
                  </h3>
                  <ResponsiveContainer width="100%" height="100%">
                    {chartData.combined?.length > 0 && chartData.combined.some(d => d.btc_volume !== null) ? (
                      <BarChart data={chartData.combined}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="time"
                          type="number"
                          scale="time"
                          domain={['dataMin', 'dataMax']}
                          tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis stroke="#9CA3AF" fontSize={10} />
                        <Tooltip
                          labelFormatter={(time) => new Date(time).toLocaleString()}
                          formatter={(value, name) => [
                            typeof value === 'number' ? value.toLocaleString() : 'N/A',
                            name?.replace('_volume', '').toUpperCase() + ' Volume'
                          ]}
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            border: '1px solid #374151',
                            borderRadius: '4px',
                            color: '#F9FAFB'
                          }}
                        />
                        {/* Apenas BTC Volume para simplificar, se quiser todos, adicione mais <Bar> */}
                        <Bar
                          dataKey="btc_volume"
                          fill="#F59E0B"
                          name="BTC Volume"
                          radius={[2, 2, 0, 0]}
                        />
                      </BarChart>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500">
                        <div className="text-center">
                          {isLoading ? 'Carregando dados de volume...' : 'Nenhum dado de volume disponível'}
                          {chartData.fallback && (
                            <div className="text-xs text-yellow-400 mt-2">
                              Backend em modo fallback
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </ResponsiveContainer>
                </div>

                {/* MACD & RSI (BTC) Chart */}
                <div className="bg-gray-900 border border-gray-800 rounded p-4">
                  <h3 className="text-sm font-bold text-gray-400 mb-4 flex justify-between">
                    <span>MACD & RSI (BTC)</span> {/* Título atualizado */}
                    <span className="text-xs">
                      {chartData.combined?.some(d => d.btc_macd !== null || d.btc_rsi !== null) ? // Verifica se tem MACD ou RSI
                        'Sinal + Histograma + RSI' :
                        'Calculando...'
                      }
                    </span>
                  </h3>
                  <ResponsiveContainer width="100%" height="100%">
                    {chartData.combined?.length > 0 && chartData.combined.some(d => d.btc_macd !== null || d.btc_rsi !== null) ? (
                      <ComposedChart data={chartData.combined}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="time"
                          type="number"
                          scale="time"
                          domain={['dataMin', 'dataMax']}
                          tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        {/* Eixo Y para MACD */}
                        <YAxis yAxisId="left" stroke="#9CA3AF" fontSize={10} />
                        {/* Eixo Y para RSI, se a escala for muito diferente, pode precisar de um segundo eixo Y */}
                        <YAxis yAxisId="right" orientation="right" stroke="#FF4500" fontSize={10} domain={[0, 100]} /> {/* Eixo para RSI (0-100) */}
                        <Tooltip
                          labelFormatter={(time) => new Date(time).toLocaleString()}
                          formatter={(value, name) => [
                            typeof value === 'number' ? value.toFixed(4) : 'N/A',
                            name === 'btc_macd' ? 'MACD Linha' :
                            name === 'btc_macd_signal' ? 'Sinal Linha' :
                            name === 'btc_macd_hist' ? 'Histograma' :
                            name === 'btc_rsi' ? 'RSI' : name
                          ]}
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            border: '1px solid #374151',
                            borderRadius: '4px',
                            color: '#F9FAFB'
                          }}
                        />
                        {/* MACD Line */}
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="btc_macd"
                          stroke="#3B82F6"
                          strokeWidth={2}
                          dot={false}
                          name="MACD"
                        />
                        {/* Signal Line */}
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="btc_macd_signal"
                          stroke="#EF4444"
                          strokeWidth={2}
                          dot={false}
                          name="Sinal"
                        />
                        {/* Histogram */}
                        {chartData.combined.some(d => d.btc_macd_hist !== null) && (
                          <Bar
                            yAxisId="left"
                            dataKey="btc_macd_hist"
                            fill="#10B981"
                            name="Histograma"
                            opacity={0.6}
                          />
                        )}
                        {/* NOVO: RSI Line */}
                        <Line
                          yAxisId="right" // Atribua ao segundo eixo Y
                          type="monotone"
                          dataKey="btc_rsi"
                          stroke="#FF4500" // Cor para RSI
                          strokeWidth={1}
                          dot={false}
                          name="RSI"
                        />
                      </ComposedChart>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500">
                        <div className="text-center">
                          {isLoading ? 'Carregando dados MACD/RSI...' : 'Cálculo requer mais dados'}
                          {chartData.fallback && (
                            <div className="text-xs text-yellow-400 mt-2">
                              Backend processando indicadores
                            </div>
                          )}
                          <div className="text-xs text-gray-400 mt-1">
                            Tente mudar para um período de tempo maior (1h, 1d)
                          </div>
                        </div>
                      </div>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="w-80 border-l border-gray-800 bg-gray-900 p-4 overflow-y-auto custom-scrollbar">
          {/* Financial Monitoring Section */}
          <div className="mb-6">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <DollarSign className="w-4 h-4 mr-2" />
              MONITORAMENTO FINANCEIRO
            </h3>

            {/* Environment Tabs */}
            <div className="flex mb-4 bg-gray-800 rounded p-1">
              <button
                onClick={() => setSelectedFinancialTab('testnet')}
                className={`flex-1 py-2 px-3 text-xs font-bold rounded transition-all ${
                  selectedFinancialTab === 'testnet'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                🧪 TESTNET
              </button>
              <button
                onClick={() => setSelectedFinancialTab('live')}
                className={`flex-1 py-2 px-3 text-xs font-bold rounded transition-all ${
                  selectedFinancialTab === 'live'
                    ? 'bg-red-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                💰 LIVE
              </button>
            </div>

            {/* Financial Data Display */}
            {(() => {
              const currentFinData = financialData[selectedFinancialTab || 'testnet'];
              const totalPnLFormat = formatPnL(currentFinData.totalPnL);
              const dailyPnLFormat = formatPnL(currentFinData.dailyPnL);
              const roiFormat = formatPnL(currentFinData.roiPercent, true);

              return (
                <div className="space-y-3">
                  {/* Balance & P&L */}
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <div className="text-gray-400">Current Balance</div>
                        <div className="text-lg font-bold text-blue-400">
                          {formatCurrency(currentFinData.balance)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Start Balance</div>
                        <div className="text-sm text-gray-300">
                          {formatCurrency(currentFinData.startBalance)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Total P&L</div>
                        <div className={`text-lg font-bold ${totalPnLFormat.colorClass}`}>
                          {totalPnLFormat.formatted}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Daily P&L</div>
                        <div className={`text-sm font-bold ${dailyPnLFormat.colorClass}`}>
                          {dailyPnLFormat.formatted}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">ROI</div>
                        <div className={`text-sm font-bold ${roiFormat.colorClass}`}>
                          {roiFormat.formatted}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Max Drawdown</div>
                        <div className="text-sm font-bold text-red-400">
                          -{currentFinData.maxDrawdown.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Trading Statistics */}
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-xs text-gray-400 mb-2">TRADING STATS</div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="text-center">
                        <div className="text-lg font-bold text-blue-400">
                          {currentFinData.totalTrades}
                        </div>
                        <div className="text-gray-400">Total Trades</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-green-400">
                          {currentFinData.winningTrades}
                        </div>
                        <div className="text-gray-400">Winners</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-red-400">
                          {currentFinData.totalTrades - currentFinData.winningTrades}
                        </div>
                        <div className="text-gray-400">Losers</div>
                      </div>
                    </div>
                    <div className="mt-2 pt-2 border-t border-gray-700">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Win Rate:</span>
                        <span className={`font-bold ${
                          currentFinData.winRate > 60 ? 'text-green-400' :
                          currentFinData.winRate > 40 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {currentFinData.winRate.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span className="text-gray-400">Last Update:</span>
                        <span className="text-gray-300">
                          {currentFinData.lastUpdate ?
                            new Date(currentFinData.lastUpdate).toLocaleTimeString() :
                            'Never'
                          }
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Performance Chart (Simple) */}
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-xs text-gray-400 mb-2">PERFORMANCE OVERVIEW</div>
                    <div className="h-20 flex items-end space-x-1">
                      {/* Simple bar chart showing recent performance */}
                      {Array.from({length: 7}, (_, i) => {
                          const randomPnl = (Math.random() * 20 - 10); // PnL aleatório entre -10 e 10
                          const height = Math.abs(randomPnl) * 3 + 10; // Escala para o gráfico
                          const isPositive = randomPnl > 0;
                          return (
                            <div
                              key={i}
                              className={`flex-1 rounded-t ${isPositive ? 'bg-green-400' : 'bg-red-400'}`}
                              style={{ height: `${height}%` }}
                              title={`Day ${i + 1}: ${formatPnL(randomPnl).formatted}`}
                            />
                          );
                      })}
                    </div>
                    <div className="text-xs text-gray-500 mt-1 text-center">
                      Performance dos últimos 7 dias (simulado)
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Real-time Indicators Section - NOVO CARD */}
          <div className="mb-6">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              INDICADORES EM TEMPO REAL (BTC)
            </h3>
            <div className="space-y-3">
              {realtimeIndicators.btc.rsi && realtimeIndicators.btc.macd && realtimeIndicators.btc.macdAngle ? (
                <>
                  {/* RSI */}
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-xs text-gray-400">RSI (14)</div>
                    <div className="flex justify-between items-center mt-1">
                      <span className={`text-xl font-bold ${
                        realtimeIndicators.btc.rsi.last_value > 70 ? 'text-red-400' :
                        realtimeIndicators.btc.rsi.last_value < 30 ? 'text-green-400' :
                        'text-blue-400'
                      }`}>
                        {realtimeIndicators.btc.rsi.last_value.toFixed(2)}
                      </span>
                      {realtimeIndicators.btc.rsi.angle !== undefined && (
                        <span className="text-sm text-gray-300 flex items-center">
                          <Compass className="w-4 h-4 mr-1 text-gray-500" />
                          {realtimeIndicators.btc.rsi.angle.toFixed(1)}°
                          <span className={`ml-1 text-xs ${
                            realtimeIndicators.btc.rsi.trend === 'RISING' ? 'text-green-400' :
                            realtimeIndicators.btc.rsi.trend === 'FALLING' ? 'text-red-400' :
                            'text-gray-400'
                          }`}>
                            ({realtimeIndicators.btc.rsi.trend})
                          </span>
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                        {realtimeIndicators.btc.rsi.last_value > 70 ? 'SOBRECOMPRADO' :
                        realtimeIndicators.btc.rsi.last_value < 30 ? 'SOBREVENDIDO' :
                        'NEUTRO'}
                    </div>
                  </div>

                  {/* MACD */}
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-xs text-gray-400">MACD (12, 26, 9)</div>
                    <div className="flex justify-between items-center mt-1">
                        <span className={`text-xl font-bold ${
                            (realtimeIndicators.btc.macd.histogram?.[realtimeIndicators.btc.macd.histogram.length - 1] || 0) > 0 ? 'text-green-400' :
                            (realtimeIndicators.btc.macd.histogram?.[realtimeIndicators.btc.macd.histogram.length - 1] || 0) < 0 ? 'text-red-400' :
                            'text-blue-400'
                        }`}>
                            Hist: {(realtimeIndicators.btc.macd.histogram?.[realtimeIndicators.btc.macd.histogram.length - 1] || 0).toFixed(3)}
                        </span>
                        {realtimeIndicators.btc.macdAngle && (
                          <span className="text-sm text-gray-300 flex items-center">
                            <Compass className="w-4 h-4 mr-1 text-gray-500" />
                            MACD: {realtimeIndicators.btc.macdAngle.macd_angle.toFixed(1)}°
                            <span className={`ml-1 text-xs ${
                              realtimeIndicators.btc.macdAngle.macd_angle > 0 ? 'text-green-400' :
                              realtimeIndicators.btc.macdAngle.macd_angle < 0 ? 'text-red-400' :
                              'text-gray-400'
                            }`}>
                                ({realtimeIndicators.btc.macdAngle.macd_angle > 0 ? 'SUBINDO' : realtimeIndicators.btc.macdAngle.macd_angle < 0 ? 'CAINDO' : 'PLANO'})
                            </span>
                          </span>
                        )}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                        Linha: {(realtimeIndicators.btc.macd.macd?.[realtimeIndicators.btc.macd.macd.length - 1] || 0).toFixed(3)} | Sinal: {(realtimeIndicators.btc.macd.signal?.[realtimeIndicators.btc.macd.signal.length - 1] || 0).toFixed(3)}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-gray-500 text-sm text-center">Aguardando dados de indicadores em tempo real...</div>
              )}
            </div>
          </div>

          {/* Recent Trades */}
          <div className="mb-6">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              TRADES RECENTES
            </h3>
            <div className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar">
              {financialData.recentTrades.length > 0 ? financialData.recentTrades.map((trade, index) => (
                <div key={index} className="bg-gray-800 p-2 rounded text-xs">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-blue-400">
                      {trade.symbol || 'N/A'}
                    </span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      (trade.side || trade.action) === 'buy' || (trade.side || trade.action) === 'BUY'
                        ? 'bg-green-600' : 'bg-red-600'
                    }`}>
                      {(trade.side || trade.action || 'N/A').toUpperCase()}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-1 mt-1">
                    <div>
                      <div className="text-gray-400">Entrada</div>
                      <div className="font-bold">
                        {formatCurrency(trade.entry_price || trade.signal?.entry_price || 0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">P&L</div>
                      <div className={`font-bold ${
                        (trade.pnl || 0) > 0 ? 'text-green-400' :
                        (trade.pnl || 0) < 0 ? 'text-red-400' : 'text-gray-400'
                      }`}>
                        {formatCurrency(trade.pnl || 0)}
                      </div>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {trade.timestamp ? new Date(trade.timestamp).toLocaleTimeString() : 'Recentemente'}
                  </div>
                </div>
              )) : (
                <div className="text-gray-500 text-xs text-center py-4">
                  Nenhum trade recente.
                </div>
              )}
            </div>
          </div>

          {/* Bot Performance */}
          {botStatus && (
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                PERFORMANCE DO BOT AI
              </h3>
              <div className="space-y-3">
                <div className="bg-gray-800 p-3 rounded">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <div className="text-gray-400">Acurácia da IA</div>
                      <div className="font-bold text-purple-400">{botStatus.ai_accuracy?.toFixed(2) || '0.00'}%</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Amostras Treino</div>
                      <div className="font-bold text-blue-400">{botStatus.training_samples || 0}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Modelo Acurácia</div>
                      <div className="font-bold text-green-400">{botStatus.ml_model_accuracy?.toFixed(3) || '0.000'}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Predições IA</div>
                      <div className="font-bold">{botStatus.ai_predictions || 0}</div>
                    </div>
                  </div>
                </div>

                {botStatus.active_positions && botStatus.active_positions.length > 0 && (
                  <div>
                    <div className="text-xs text-gray-400 mb-2">POSIÇÕES ATIVAS</div>
                    {botStatus.active_positions.map((position, index) => (
                      <div key={index} className="bg-gray-800 p-2 rounded mb-2 text-xs">
                        <div className="flex justify-between items-center">
                          <span className="font-bold text-blue-400">{position.symbol}</span>
                          <span className={`px-2 py-1 rounded text-xs ${
                            position.side === 'buy' ? 'bg-green-600' : 'bg-red-600'
                          }`}>
                            {position.side.toUpperCase()}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-1 mt-1">
                          <div>
                            <div className="text-gray-400">Entry</div>
                            <div className="font-bold">{formatCurrency(position.entry_price)}</div>
                          </div>
                          <div>
                            <div className="text-gray-400">Current</div>
                            <div className="font-bold">{formatCurrency(position.current_price)}</div>
                          </div>
                          <div>
                            <div className="text-gray-400">P&L</div>
                            <div className={`font-bold ${position.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {formatCurrency(position.pnl)}
                            </div>
                          </div>
                          <div>
                            <div className="text-gray-400">AI Conf</div>
                            <div className="font-bold text-purple-400">{(position.ai_prediction * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Backtest Recommendations */}
          <div className="mb-6">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <BarChart3 className="w-4 h-4 mr-2" />
              SINAIS DE BACKTEST
            </h3>
            <div className="space-y-2 max-h-60 overflow-y-auto custom-scrollbar">
              {backtestRecommendations.length > 0 ? backtestRecommendations.slice(0, 5).map((rec, index) => (
                <div key={index} className="bg-gray-800 p-3 rounded text-xs">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-blue-400">{rec.pattern_name}</span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      rec.trade_type === 'LONG' ? 'bg-green-600' : 'bg-red-600'
                    }`}>
                      {rec.trade_type}
                    </span>
                  </div>
                  <div className="text-gray-300 mb-2">{rec.description}</div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <div className="text-gray-400">Confiança</div>
                      <div className="font-bold text-green-400">{(rec.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Taxa de Sucesso</div>
                      <div className="font-bold">{(rec.success_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Retorno Esp.</div>
                      <div className="font-bold text-blue-400">{rec.expected_return}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Max DD</div>
                      <div className="font-bold text-red-400">{rec.max_drawdown}</div>
                    </div>
                  </div>
                  {rec.backtest_details && (
                    <div className="mt-2 pt-2 border-t border-gray-700">
                      <div className="text-gray-400">Backtest: {rec.backtest_details.total_trades} trades, {rec.backtest_details.win_rate}</div>
                    </div>
                  )}
                </div>
              )) : (
                <div className="text-gray-500 text-xs text-center py-4">
                  Nenhuma recomendação de backtest disponível.
                </div>
              )}
            </div>
          </div>

          {/* System Status */}
          <div>
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Settings className="w-4 h-4 mr-2" />
              STATUS DO SISTEMA
            </h3>
            <div className="space-y-2">
              <div className="bg-gray-800 p-3 rounded text-xs">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="text-gray-400">API Status</div>
                    <div className={`font-bold ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
                      {connectionStatus.toUpperCase()}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Data Source</div>
                    <div className="font-bold text-blue-400">Gate.io + YFinance</div>
                  </div>
                  <div>
                    <div className="text-gray-400">WebSockets</div>
                    <div className="font-bold">
                      {Object.values(wsConnections).filter(Boolean).length}/3 ATIVOS
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Ambiente</div>
                    <div className={`font-bold ${environment === 'live' ? 'text-red-400' : 'text-blue-400'}`}>
                      {environment.toUpperCase()}
                    </div>
                  </div>
                </div>
              </div>

              {marketData?.system_health && (
                <div className="bg-gray-800 p-3 rounded text-xs">
                  <div className="text-gray-400 mb-2">Atualização dos Dados</div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span>Price Data:</span>
                      <span className="text-green-400">LIVE</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Sentiment:</span>
                      <span className={wsConnections.sentiment ? 'text-green-400' : 'text-red-400'}>
                        {wsConnections.sentiment ? 'LIVE' : 'OFFLINE'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>OHLCV:</span>
                      <span className={wsConnections.ohlcv ? 'text-green-400' : 'text-red-400'}>
                        {wsConnections.ohlcv ? 'LIVE' : 'OFFLINE'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Indicators:</span>
                      <span className={wsConnections.rsiMacd ? 'text-green-400' : 'text-red-400'}>
                        {wsConnections.rsiMacd ? 'LIVE' : 'OFFLINE'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="text-xs text-gray-500 text-center pt-4">
                Trading Dashboard Pro v6.0
                <br />
                Backend: {API_BASE}
                <br />
                {new Date().toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Environment Selection Modal */}
      {showEnvironmentModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-bold text-white mb-4">Selecionar Ambiente de Trading</h3>

            <div className="space-y-4">
              {/* Opção Testnet */}
              <div
                onClick={() => handleEnvironmentChange('testnet')}
                className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                  environment === 'testnet'
                    ? 'border-blue-500 bg-blue-900/20'
                    : 'border-gray-600 hover:border-blue-400'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-blue-400 font-bold">🧪 TESTNET</div>
                    <div className="text-sm text-gray-300">Trading de demonstração com dinheiro virtual</div>
                    <div className="text-xs text-gray-400 mt-1">
                      ✅ Seguro para testes<br />
                      ✅ Sem risco de dinheiro real<br />
                      ✅ Funcionalidade completa
                    </div>
                  </div>
                  <div className={`w-4 h-4 rounded-full border-2 ${
                    environment === 'testnet' ? 'bg-blue-500 border-blue-500' : 'border-gray-400'
                  }`}></div>
                </div>
              </div>

              {/* Opção Live */}
              <div
                onClick={() => handleEnvironmentChange('live')}
                className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                  environment === 'live'
                    ? 'border-red-500 bg-red-900/20'
                    : 'border-gray-600 hover:border-red-400'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-red-400 font-bold">💰 LIVE TRADING</div>
                    <div className="text-sm text-gray-300">Trading com dinheiro real</div>
                    <div className="text-xs text-gray-400 mt-1">
                      ⚠️ Usa dinheiro real<br />
                      ⚠️ Lucros e perdas reais<br />
                      ⚠️ Requer conta capitalizada
                    </div>
                  </div>
                  <div className={`w-4 h-4 rounded-full border-2 ${
                    environment === 'live' ? 'bg-red-500 border-red-500' : 'border-gray-400'
                  }`}></div>
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowEnvironmentModal(false)}
                className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              >
                Cancelar
              </button>
              <button
                onClick={() => setShowEnvironmentModal(false)} // A seleção já é feita no onClick do div
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded"
              >
                Confirmar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BloombergDashboard;
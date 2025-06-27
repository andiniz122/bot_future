import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ReferenceLine } from 'recharts'; // Adicionado ReferenceLine para o MACD
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Brain,
  Database,
  Wifi,
  WifiOff,
  AlertTriangle,
  Play,
  Pause,
  DollarSign
} from 'lucide-react';

const API_BASE_URL = 'http://62.72.1.122:8000'; // URL real da sua API

const BloombergDashboard = () => {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [marketData, setMarketData] = useState({
    assets: {
      btc: { name: 'BTC/USD', current_price: 0, change: 0 },
      gold: { name: 'GOLD', current_price: 0, change: 0 },
      dxy: { name: 'DXY', current_price: 0, change: 0 }
    }
  });
  
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    ai_accuracy: 0,
    training_samples: 0,
    ml_model_accuracy: 0,
    ai_predictions: 0,
    active_positions: [],
    total_pnl: 0,
    roi_percentage: 0,
    win_rate: 0,
    total_trades: 0,
    ai_system_status: {
      ml_available: false,
      xgb_available: false,
      sentiment_available: false,
      talib_available: false,
      model_trained: false,
      fred_calendar_active: false,
      cryptopanic_active: false,
    }
  });

  const [wsConnections, setWsConnections] = useState({
    sentiment: false,
    ohlcv: false,
    rsiMacd: false // Isso será atualizado com base no status do ML/TALIB
  });

  const [realtimeIndicators, setRealtimeIndicators] = useState({
    btc: {
      rsi: 0,
      rsi_angle: 0, // NOVO: Ângulo do RSI
      macd: 0,
      macd_signal: 0, // Adicionado para exibir a linha de sinal do MACD no realtime card
      macd_histogram: 0, // Adicionado para o histograma do MACD
      macd_angle: 0,
      supertrend_dir: 0,
      vwap_distance: 0,
      volume_ratio: 0,
      talib_entrada_score: 0,
      signal: 'N/A'
    }
  });

  const [selectedPeriod, setSelectedPeriod] = useState('5m');
  const [isLoading, setIsLoading] = useState(false);
  const [tradingEnvironment, setTradingEnvironment] = useState('testnet');
  const [showEnvironmentWarning, setShowEnvironmentWarning] = useState(false);

  const [chartData, setChartData] = useState({
    combined: [],
    dataPoints: 0,
    lastUpdate: null,
    fallback: false
  });

  const [recentTrades, setRecentTrades] = useState([]);

  const formatCurrency = (value) => {
    if (typeof value !== 'number') return '$0.00';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPnL = (pnl) => {
    const formatted = formatCurrency(Math.abs(pnl));
    return {
      formatted: pnl >= 0 ? `+${formatted}` : `-${formatted}`,
      isPositive: pnl >= 0
    };
  };

  const fetchBotStatus = useCallback(async () => {
    try {
      setConnectionStatus('connecting');
      const response = await fetch(`${API_BASE_URL}/api/trading-bot/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      setBotStatus({
        is_running: data.bot_status === 'running' || data.bot_status === 'active',
        ai_accuracy: data.performance?.ai_accuracy || 0,
        training_samples: data.performance?.training_samples || 0,
        ml_model_accuracy: data.performance?.ml_model_accuracy || 0,
        ai_predictions: data.performance?.ai_predictions || 0,
        active_positions: (data.active_positions || []).map(pos => ({
          symbol: pos.symbol?.replace('_USDT', '/USDT') || 'N/A',
          side: pos.side || 'unknown',
          size: pos.size || 0,
          entry_price: pos.entry_price || 0,
          current_pnl: pos.unrealized_pnl || pos.pnl || 0
        })),
        total_pnl: data.performance?.total_pnl || 0,
        roi_percentage: data.performance?.roi_percentage || 0,
        win_rate: data.performance?.win_rate || 0,
        total_trades: data.performance?.total_trades || 0,
        ai_system_status: data.ai_system_status || { // Garante que ai_system_status existe
          ml_available: false, xgb_available: false, sentiment_available: false,
          talib_available: false, model_trained: false, fred_calendar_active: false,
          cryptopanic_active: false,
        }
      });
      
      setRecentTrades((data.trade_history || []).map(trade => ({
        symbol: trade.symbol?.replace('_USDT', '/USDT') || 'N/A',
        side: trade.side || 'unknown',
        size: trade.size || 0,
        price: trade.entry_price || trade.price || 0,
        pnl: trade.realized_pnl || trade.pnl || 0,
        timestamp: trade.timestamp ? new Date(trade.timestamp).getTime() : Date.now()
      })).sort((a, b) => b.timestamp - a.timestamp));
      
      setConnectionStatus('connected');

      // Atualizar status das conexões baseado no ai_system_status do bot
      setWsConnections({
        sentiment: data.ai_system_status?.sentiment_available || false,
        ohlcv: data.ai_system_status?.ml_available || false, // 'ml_available' indica que o bot está vivo e processando dados
        rsiMacd: data.ai_system_status?.talib_available || false // 'talib_available' indica que talib está funcionando
      });

    } catch (error) {
      console.error("Erro ao buscar status do bot:", error);
      setConnectionStatus('disconnected');
      setBotStatus(prev => ({
        ...prev,
        is_running: false,
        total_pnl: 0,
        roi_percentage: 0,
        win_rate: 0,
        total_trades: 0,
        ai_system_status: { // Resetar para evitar erros em caso de falha da API
          ml_available: false, xgb_available: false, sentiment_available: false,
          talib_available: false, model_trained: false, fred_calendar_active: false,
          cryptopanic_active: false,
        }
      }));
    }
  }, []);

  const fetchMarketData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/current`);
      if (response.ok) {
        const data = await response.json();
        
        setMarketData({
          assets: {
            btc: {
              name: 'BTC/USD',
              current_price: data.assets?.btc?.current_price || 0,
              change: data.assets?.btc?.change_percent || 0
            },
            gold: {
              name: 'GOLD',
              current_price: data.assets?.paxg?.current_price || data.assets?.gold?.current_price || 0,
              change: data.assets?.paxg?.change_percent || data.assets?.gold?.change_percent || 0
            },
            dxy: {
              name: 'DXY',
              current_price: data.assets?.dxy?.current_price || 103.45,
              change: data.assets?.dxy?.change_percent || 0
            }
          }
        });
      }
    } catch (error) {
      console.error("Erro ao buscar preço de mercado:", error);
      setMarketData({ // Fallback para dados padrão em caso de erro
        assets: {
          btc: { name: 'BTC/USD', current_price: 0, change: 0 },
          gold: { name: 'GOLD', current_price: 0, change: 0 },
          dxy: { name: 'DXY', current_price: 0, change: 0 }
        }
      });
    }
  }, []);

  const fetchChartData = useCallback(async (period) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/precos/${period}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (data.dates && data.assets && data.assets.btc) {
        const btcData = data.assets.btc;
        
        const formattedData = data.dates.map((dateStr, index) => {
          const timeStamp = new Date(dateStr).getTime();
          return {
            time: timeStamp,
            price: btcData.price_data?.[index] || null,
            volume: btcData.volume_data?.[index] || null,
            rsi: btcData.rsi_data?.[index] || 50, // Default 50 se não houver RSI
            macd: btcData.macd_data?.[index] || null,
            macd_signal: btcData.macd_signal_data?.[index] || null,
            macd_hist: btcData.macd_hist_data?.[index] || null
          };
        }).filter(d => d.time && d.price !== null && d.volume !== null);

        const cleanedData = formattedData.map(d => ({
          ...d,
          price: Number(d.price) || 0,
          volume: Number(d.volume) || 0,
          rsi: Number(d.rsi) || 50,
          // MACD pode ser NaN no início, trate com 0 se preferir para gráficos
          macd: Number(d.macd) || 0,
          macd_signal: Number(d.macd_signal) || 0,
          macd_hist: Number(d.macd_hist) || 0
        }));

        setChartData({
          combined: cleanedData,
          dataPoints: cleanedData.length,
          lastUpdate: Date.now(),
          fallback: cleanedData.length === 0
        });
      } else {
        console.warn("Dados de gráfico para BTC não encontrados na resposta da API");
        throw new Error("Dados de gráfico não encontrados");
      }

    } catch (error) {
      console.error("Erro ao buscar dados do gráfico:", error);
      // Fallback de dados mais realistas em caso de erro da API
      const fallbackData = Array.from({ length: 30 }, (_, i) => {
        const baseTime = Date.now() - (30 - i) * 300000;
        const basePrice = 65000;
        const variation = Math.sin(i * 0.5) * 500 + Math.random() * 200;
        return {
          time: baseTime,
          price: basePrice + variation,
          volume: 1000000 + Math.random() * 1500000,
          rsi: 40 + Math.sin(i * 0.3) * 20 + Math.random() * 10,
          macd: Math.sin(i * 0.2) * 50 + Math.random() * 20,
          macd_signal: Math.sin(i * 0.2 - 0.1) * 50 + Math.random() * 20,
          macd_hist: Math.sin(i * 0.2) * 25 + Math.random() * 10
        };
      });
      
      setChartData({
        combined: fallbackData,
        dataPoints: fallbackData.length,
        lastUpdate: Date.now(),
        fallback: true
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchRealtimeIndicators = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/macd/realtime/BTC`);
      if (response.ok) {
        const data = await response.json();
        
        const macdData = data.macd_data || {};
        const rsiData = data.rsi_data || {}; // Agora rsi_data deve vir com last_value, angle, strength, trend
        const macdAngleData = data.macd_angle_data || {}; // Dados de ângulo do MACD

        const talibEntradaScore = data.talib_entrada_score || 0; 

        let currentSignal = 'N/A';
        if (macdData.histogram && macdData.histogram.length > 0 && rsiData.last_value !== undefined) {
          const lastHist = macdData.histogram[macdData.histogram.length - 1];
          const lastRsi = rsiData.last_value;

          if (lastHist > 0 && lastRsi < 70) {
            currentSignal = 'BUY';
          } else if (lastHist < 0 && lastRsi > 30) {
            currentSignal = 'SELL';
          } else {
            currentSignal = 'HOLD';
          }
        }

        setRealtimeIndicators(prev => ({
          ...prev,
          btc: {
            rsi: rsiData.last_value || 0,
            rsi_angle: rsiData.angle || 0, // NOVO: Ângulo do RSI
            macd: macdData.macd && macdData.macd.length > 0 ? macdData.macd[macdData.macd.length - 1] : 0,
            macd_signal: macdData.signal && macdData.signal.length > 0 ? macdData.signal[macdData.signal.length - 1] : 0, // Adicionado
            macd_histogram: macdData.histogram && macdData.histogram.length > 0 ? macdData.histogram[macdData.histogram.length - 1] : 0, // Adicionado
            macd_angle: macdAngleData.macd_angle || 0,
            supertrend_dir: 0, // Não retornado por este endpoint atualmente.
            vwap_distance: 0,  // Não retornado por este endpoint atualmente.
            volume_ratio: 0,   // Não retornado por este endpoint atualmente.
            signal: currentSignal,
            talib_entrada_score: talibEntradaScore
          }
        }));
      } else {
        // Fallback se /api/macd/realtime/BTC falhar
        const currentResponse = await fetch(`${API_BASE_URL}/api/current`); // Tentar buscar do /api/current como fallback
        if (currentResponse.ok) {
          const currentData = await currentResponse.json();
          const btcCurrent = currentData.assets?.btc || {};
          
          setRealtimeIndicators(prev => ({ // Atualizar com dados do /api/current se disponíveis
            ...prev,
            btc: {
              rsi: 50, // current_rsi não existe aqui, defina um padrão
              rsi_angle: 0, // Não existe no /api/current
              macd: 0, // current_macd não existe, defina um padrão
              macd_signal: 0, // Não existe no /api/current
              macd_histogram: 0, // Não existe no /api/current
              macd_angle: 0, // Não existe no /api/current
              supertrend_dir: 0, vwap_distance: 0, volume_ratio: 1, // Simular ou 0
              signal: 'N/A', // Não existe no /api/current
              talib_entrada_score: 0 // Não existe no /api/current
            }
          }));
        } else {
          // Fallback completo se todas as APIs falharem
          setRealtimeIndicators(prev => ({
            ...prev,
            btc: {
              rsi: 0, macd: 0, macd_signal: 0, macd_histogram: 0, macd_angle: 0,
              supertrend_dir: 0, vwap_distance: 0, volume_ratio: 0, talib_entrada_score: 0, signal: 'N/A'
            }
          }));
        }
      }
    } catch (error) {
      console.error("Erro ao buscar indicadores em tempo real:", error);
      setRealtimeIndicators(prev => ({ // Resetar para valores padrão em caso de erro
        ...prev,
        btc: {
          rsi: 0, macd: 0, macd_signal: 0, macd_histogram: 0, macd_angle: 0,
          supertrend_dir: 0, vwap_distance: 0, volume_ratio: 0, talib_entrada_score: 0, signal: 'N/A'
        }
      }));
    }
  }, []);

  const toggleBotStatus = useCallback(async () => {
    const action = botStatus.is_running ? 'stop' : 'start';
    try {
      const response = await fetch(`${API_BASE_URL}/api/trading-bot/${action}`, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // O backend não precisa do `environment` no corpo para start/stop.
        // O bot real é inicializado uma vez no startup do backend com o ENVIRONMENT global.
        // A requisição de start/stop simplesmente muda o estado de execução da instância já criada.
        // Por isso, removemos o body.
        // body: JSON.stringify({ environment: tradingEnvironment }) // REMOVIDO
      });
      if (response.ok) {
        fetchBotStatus();
      } else {
        console.error(`Failed to ${action} bot:`, await response.text());
      }
    } catch (error) {
      console.error(`Error toggling bot status:`, error);
    }
  }, [botStatus.is_running, fetchBotStatus]);

  const handleEnvironmentChange = (newEnvironment) => {
    // Se o bot estiver rodando, não permita a mudança de ambiente para evitar inconsistências.
    if (botStatus.is_running) {
        alert("Pare o bot antes de mudar o ambiente de trading.");
        return;
    }

    if (newEnvironment === 'live' && tradingEnvironment === 'testnet') {
      setShowEnvironmentWarning(true);
    } else {
      setTradingEnvironment(newEnvironment);
    }
  };

  const confirmLiveEnvironment = () => {
    setTradingEnvironment('live');
    setShowEnvironmentWarning(false);
  };

  const cancelLiveEnvironment = () => {
    setShowEnvironmentWarning(false);
  };

  useEffect(() => {
    fetchBotStatus();
    fetchMarketData();
    fetchChartData(selectedPeriod);
    fetchRealtimeIndicators();
  }, [fetchBotStatus, fetchMarketData, fetchChartData, fetchRealtimeIndicators, selectedPeriod]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchBotStatus();
      fetchMarketData();
      fetchRealtimeIndicators();
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchBotStatus, fetchMarketData, fetchRealtimeIndicators, selectedPeriod]);

  useEffect(() => {
    fetchChartData(selectedPeriod);
  }, [selectedPeriod, fetchChartData]);

  const periods = ['5m', '15m', '1h', '4h', '1d'];
  const hasValidChartData = chartData.combined && chartData.combined.length > 0;
  const hasValidMarketData = marketData.assets && Object.keys(marketData.assets).length > 0;

  return (
    <div className="min-h-screen bg-black text-white font-mono">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <h1 className="text-2xl font-bold text-blue-400">TRADING PRO AI</h1>
              {/* Indicador de Ambiente */}
              <div className={`px-3 py-1 rounded text-xs font-bold ${
                tradingEnvironment === 'live' 
                  ? 'bg-red-600 text-white animate-pulse' 
                  : 'bg-blue-600 text-white'
              }`}>
                {tradingEnvironment === 'live' ? '💰 LIVE MODE' : '🧪 TESTNET'}
              </div>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-500' : 
                    connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                  <span className="text-gray-300">{connectionStatus.toUpperCase()}</span>
                </div>
                
                {/* Indicadores de Conexão (corrigidos - apenas um conjunto) */}
                <div className="flex items-center space-x-1" title="Sentiment API Status">
                  {botStatus.ai_system_status?.sentiment_available ? <Wifi className="w-4 h-4 text-green-500" /> : <WifiOff className="w-4 h-4 text-red-500" />}
                  <span className="text-xs text-gray-400">SENTIMENTO</span>
                </div>
                <div className="flex items-center space-x-1" title="ML Data Status">
                  {botStatus.ai_system_status?.ml_available ? <Database className="w-4 h-4 text-green-500" /> : <Database className="w-4 h-4 text-red-500" />}
                  <span className="text-xs text-gray-400">ML DATA</span>
                </div>
                <div className="flex items-center space-x-1" title="TALIB Indicators Status">
                  {botStatus.ai_system_status?.talib_available ? <Activity className="w-4 h-4 text-green-500" /> : <Activity className="w-4 h-4 text-red-500" />}
                  <span className="text-xs text-gray-400">TALIB</span>
                </div>
              </div>
            </div>

            {/* Status do Mercado */}
            <div className="flex items-center space-x-6 text-sm">
              {hasValidMarketData ? Object.entries(marketData.assets).map(([key, asset]) => (
                <div key={key} className="text-center">
                  <div className="text-xs text-gray-400 uppercase">{asset.name}</div>
                  <div className="font-bold">{formatCurrency(asset.current_price)}</div>
                  <div className={`text-xs ${asset.change > 0 ? 'text-green-400' : asset.change < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                    {asset.change > 0 ? '+' : ''}{asset.change.toFixed(2)}%
                  </div>
                </div>
              )) : (
                <div className="text-gray-500 text-xs">Carregando dados do mercado...</div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex h-screen">
        {/* Left Sidebar */}
        <div className="w-80 bg-gray-900 border-r border-gray-800 overflow-y-auto">
          <div className="p-4">
            {/* Bot Controls */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                CONTROLE DO BOT
              </h3>
              <div className="space-y-3">
                {/* Trading Environment Selector */}
                <div className="space-y-2">
                  <span className="text-sm text-gray-400">Ambiente de Trading:</span>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleEnvironmentChange('testnet')}
                      className={`flex-1 px-3 py-2 rounded text-xs font-bold transition-colors ${
                        tradingEnvironment === 'testnet'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      🧪 TESTNET
                    </button>
                    <button
                      onClick={() => handleEnvironmentChange('live')}
                      className={`flex-1 px-3 py-2 rounded text-xs font-bold transition-colors ${
                        tradingEnvironment === 'live'
                          ? 'bg-red-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      💰 LIVE
                    </button>
                  </div>
                  {tradingEnvironment === 'live' && (
                    <div className="bg-red-900/30 border border-red-500 rounded p-2 text-xs">
                      <div className="text-red-400 font-bold">⚠️ MODO LIVE ATIVO</div>
                      <div className="text-red-300">Operando com dinheiro real!</div>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Status:</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${botStatus.is_running ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className="text-sm">{botStatus.is_running ? 'ATIVO' : 'PARADO'}</span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button 
                    className={`flex-1 px-3 py-2 rounded text-xs font-bold flex items-center justify-center space-x-1 ${
                      botStatus.is_running ? 'bg-red-600 hover:bg-red-700' : 
                      tradingEnvironment === 'live' ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                    }`}
                    onClick={toggleBotStatus}
                  >
                    {botStatus.is_running ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                    <span>{botStatus.is_running ? 'PARAR' : 'INICIAR'}</span>
                    {tradingEnvironment === 'live' && !botStatus.is_running && (
                      <span className="text-xs">💰</span>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Overall Performance */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <TrendingUp className="w-4 h-4 mr-2" />
                PERFORMANCE GERAL
              </h3>
              <div className="bg-gray-800 p-3 rounded">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="text-gray-400">PnL Total</div>
                    <div className={`font-bold ${formatPnL(botStatus.total_pnl).isPositive ? 'text-green-400' : 'text-red-400'}`}>
                      {formatPnL(botStatus.total_pnl).formatted}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">ROI (%)</div>
                    <div className={`font-bold ${botStatus.roi_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {botStatus.roi_percentage?.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Trades Totais</div>
                    <div className="font-bold text-blue-400">{botStatus.total_trades?.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Win Rate</div>
                    <div className="font-bold text-purple-400">{botStatus.win_rate?.toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Trades */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <DollarSign className="w-4 h-4 mr-2" />
                TRADES RECENTES
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900">
                {recentTrades.length === 0 && <div className="text-gray-500 text-xs">Nenhum trade recente.</div>}
                {recentTrades.map((trade, index) => (
                  <div key={index} className="bg-gray-800 p-3 rounded text-xs">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="font-bold text-blue-400">{trade.symbol}</div>
                        <div className={`text-xs px-2 py-1 rounded ${
                          trade.side === 'buy' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
                        }`}>
                          {trade.side.toUpperCase()}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-gray-300">{trade.size} @ {formatCurrency(trade.price)}</div>
                        <div className={`font-bold ${
                          trade.pnl > 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {formatPnL(trade.pnl).formatted}
                        </div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(trade.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Bot Performance */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                MÉTRICAS DO BOT AI
              </h3>
              <div className="space-y-3">
                <div className="bg-gray-800 p-3 rounded">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <div className="text-gray-400">Acurácia da IA</div>
                      <div className="font-bold text-purple-400">{botStatus.ai_accuracy?.toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Amostras Treino</div>
                      <div className="font-bold text-blue-400">{botStatus.training_samples?.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Modelo Acurácia</div>
                      <div className="font-bold text-green-400">{botStatus.ml_model_accuracy?.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Predições IA</div>
                      <div className="font-bold">{botStatus.ai_predictions?.toLocaleString()}</div>
                    </div>
                  </div>
                </div>

                {/* Posições Ativas */}
                {botStatus.active_positions && botStatus.active_positions.length > 0 && (
                  <div>
                    <div className="text-xs text-gray-400 mb-2">POSIÇÕES ATIVAS</div>
                    <div className="max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900">
                      {botStatus.active_positions.map((position, index) => (
                        <div key={index} className="bg-gray-800 p-2 rounded mb-2 text-xs">
                          <div className="flex justify-between items-center">
                            <span className="font-bold text-blue-400">{position.symbol}</span>
                            <span className={`px-2 py-1 rounded text-xs ${
                              position.side === 'buy' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
                            }`}>
                              {position.side.toUpperCase()}
                            </span>
                          </div>
                          <div className="flex justify-between mt-1">
                            <span className="text-gray-400">Size: {position.size}</span>
                            <span className="text-gray-400">Entry: {formatCurrency(position.entry_price)}</span>
                          </div>
                          <div className="text-center mt-1">
                            <span className={`font-bold ${
                              position.current_pnl > 0 ? 'text-green-400' : 'text-red-400'
                            }`}>
                              PnL: {formatPnL(position.current_pnl).formatted}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Indicadores em Tempo Real */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Activity className="w-4 h-4 mr-2" />
                INDICADORES EM TEMPO REAL (BTC)
              </h3>
              <div className="space-y-3">
                <div className="bg-gray-800 p-3 rounded">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <div className="text-gray-400">RSI</div>
                      <div className={`font-bold ${
                        realtimeIndicators.btc.rsi > 70 ? 'text-red-400' :
                        realtimeIndicators.btc.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                      }`}>
                        {realtimeIndicators.btc.rsi?.toFixed(1)}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">RSI Ângulo</div> {/* NOVO ELEMENTO */}
                      <div className={`font-bold ${
                        realtimeIndicators.btc.rsi_angle > 10 ? 'text-green-400' :
                        realtimeIndicators.btc.rsi_angle < -10 ? 'text-red-400' : 'text-gray-400'
                      }`}>
                        {realtimeIndicators.btc.rsi_angle?.toFixed(1)}°
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">MACD</div>
                      <div className={`font-bold ${
                        realtimeIndicators.btc.macd > 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {realtimeIndicators.btc.macd?.toFixed(4)}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">Signal (Base)</div>
                      <div className={`font-bold ${
                        realtimeIndicators.btc.signal === 'BUY' ? 'text-green-400' :
                        realtimeIndicators.btc.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                      }`}>
                        {realtimeIndicators.btc.signal}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">TALIB Score</div>
                      <div className={`font-bold ${
                        realtimeIndicators.btc.talib_entrada_score === 1.0 ? 'text-green-400' :
                        realtimeIndicators.btc.talib_entrada_score === 0.5 ? 'text-yellow-400' : 'text-gray-400'
                      }`}>
                        {realtimeIndicators.btc.talib_entrada_score?.toFixed(1)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Chart Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Chart Controls */}
          <div className="border-b border-gray-800 bg-gray-900">
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center space-x-2">
                {periods.map((period) => (
                  <button
                    key={period}
                    onClick={() => setSelectedPeriod(period)}
                    className={`px-3 py-1 rounded text-xs font-bold transition-colors ${
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
                    <span className="text-yellow-400 flex items-center" title="Dados do gráfico em modo de fallback">
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
          </div>

          <div className="flex-1 p-4 overflow-hidden" style={{ maxHeight: 'calc(100vh - 200px)' }}>
            <div className="h-full flex flex-col gap-4">
              {/* Main Price Chart */}
              <div className="bg-gray-900 border border-gray-800 rounded p-4 flex-shrink-0" style={{ height: '350px' }}>
                <h3 className="text-sm font-bold text-gray-400 mb-4">
                  GRÁFICO DE PREÇO BTC/USDT
                  {!hasValidChartData && (
                    <span className="ml-2 text-yellow-400 text-xs">
                      <AlertTriangle className="w-3 h-3 inline mr-1" />
                      Sem dados
                    </span>
                  )}
                </h3>
                <div className="w-full h-full" style={{ height: 'calc(100% - 32px)' }}>
                  {hasValidChartData ? (
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
                            typeof value === 'number' ? formatCurrency(value) : value,
                            name
                          ]}
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            border: '1px solid #374151',
                            borderRadius: '6px'
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="price"
                          stroke="#3B82F6"
                          strokeWidth={2}
                          dot={false}
                          name="Preço"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      <div className="text-center">
                        <div className="text-4xl mb-2">📊</div>
                        <div>Aguardando dados do gráfico...</div>
                        <div className="text-xs mt-1">Verifique a conexão com a API</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Volume and RSI Charts */}
              <div className="grid grid-cols-2 gap-4 flex-shrink-0" style={{ height: '200px' }}>
                <div className="bg-gray-900 border border-gray-800 rounded p-3 overflow-hidden">
                  <h3 className="text-sm font-bold text-gray-400 mb-2">VOLUME</h3>
                  <div className="w-full" style={{ height: 'calc(100% - 24px)' }}>
                    {hasValidChartData ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.combined} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={10}
                          />
                          <YAxis 
                            stroke="#9CA3AF" 
                            fontSize={10}
                            tickFormatter={(value) => `${Math.round(value / 1000000)}M`}
                          />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value) => [`${Math.round(value / 1000000)}M`, 'Volume']}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '6px',
                              color: '#fff'
                            }}
                          />
                          <Bar dataKey="volume" fill="#8B5CF6" />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500 text-xs">
                        Sem dados de volume
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-gray-900 border border-gray-800 rounded p-3 overflow-hidden">
                  <h3 className="text-sm font-bold text-gray-400 mb-2">RSI</h3>
                  <div className="w-full" style={{ height: 'calc(100% - 24px)' }}>
                    {hasValidChartData ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData.combined} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={10}
                          />
                          <YAxis stroke="#9CA3AF" fontSize={10} domain={[0, 100]} />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value) => [value?.toFixed(1), 'RSI']}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '6px',
                              color: '#fff'
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="rsi"
                            stroke="#F59E0B"
                            strokeWidth={2}
                            dot={false}
                          />
                          {/* Linha sobrecomprado */}
                          <Line
                            dataKey={() => 70}
                            stroke="#EF4444"
                            strokeWidth={1}
                            strokeDasharray="5 5"
                            dot={false}
                          />
                          {/* Linha sobrevendido */}
                          <Line
                            dataKey={() => 30}
                            stroke="#10B981"
                            strokeWidth={1}
                            strokeDasharray="5 5"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500 text-xs">
                        Sem dados de RSI
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* NEW: MACD Chart */}
              <div className="bg-gray-900 border border-gray-800 rounded p-3 flex-shrink-0" style={{ height: '200px' }}>
                <h3 className="text-sm font-bold text-gray-400 mb-2">MACD</h3>
                <div className="w-full" style={{ height: 'calc(100% - 24px)' }}>
                  {hasValidChartData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData.combined} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="time"
                          tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis stroke="#9CA3AF" fontSize={10} />
                        <Tooltip
                          labelFormatter={(time) => new Date(time).toLocaleString()}
                          formatter={(value, name) => [value?.toFixed(4), name]}
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            border: '1px solid #374151',
                            borderRadius: '6px',
                            color: '#fff'
                          }}
                        />
                        <ReferenceLine y={0} stroke="#6B7280" strokeWidth={1} strokeDasharray="3 3" />
                        <Line
                          type="monotone"
                          dataKey="macd"
                          stroke="#3B82F6"
                          strokeWidth={2}
                          dot={false}
                          name="MACD Line"
                        />
                        <Line
                          type="monotone"
                          dataKey="macd_signal"
                          stroke="#F59E0B"
                          strokeWidth={2}
                          dot={false}
                          name="Signal Line"
                        />
                        <Bar
                          dataKey="macd_hist"
                          fill={(data) => data.macd_hist > 0 ? '#10B981' : '#EF4444'} // Verde para positivo, Vermelho para negativo
                          opacity={0.8}
                          name="Histogram"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500 text-xs">
                      Sem dados de MACD
                    </div>
                  )}
                </div>
              </div>

            </div>
          </div>
        </div>
      </div>

      {showEnvironmentWarning && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-red-500 rounded-lg p-6 max-w-md mx-4">
            <div className="text-center space-y-4">
              <div className="text-4xl">⚠️</div>
              <h2 className="text-xl font-bold text-red-400">ATENÇÃO: MODO LIVE</h2>
              <div className="text-sm text-gray-300 space-y-2">
                <p>Você está prestes a ativar o <strong className="text-red-400">MODO LIVE</strong>.</p>
                <p>O bot irá operar com <strong className="text-yellow-400">DINHEIRO REAL</strong>.</p>
                <p>Certifique-se de que:</p>
                <ul className="text-left text-xs space-y-1 bg-gray-800 p-3 rounded">
                  <li>• As estratégias foram testadas no Testnet</li>
                  <li>• Os parâmetros de risco estão configurados</li>
                  <li>• Você tem experiência suficiente</li>
                  <li>• Está ciente dos riscos financeiros</li>
                </ul>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={cancelLiveEnvironment}
                  className="flex-1 px-4 py-2 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 text-sm"
                >
                  Cancelar
                </button>
                <button
                  onClick={confirmLiveEnvironment}
                  className="flex-1 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm font-bold"
                >
                  Confirmar LIVE
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BloombergDashboard;
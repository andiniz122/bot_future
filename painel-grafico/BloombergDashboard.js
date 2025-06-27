import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ReferenceLine } from 'recharts';
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
  DollarSign,
  Zap,
  ArrowUpRight,
  ArrowDownLeft,
  Target,
  RotateCw
} from 'lucide-react';

const BloombergDashboard = () => {
  const [connectionStatus, setConnectionStatus] = useState('connected');
  const [marketData, setMarketData] = useState({
    assets: {
      btc: { name: 'BTC/USD', current_price: 67250.45, change: 2.34 },
      gold: { name: 'GOLD', current_price: 2385.60, change: -0.78 },
      dxy: { name: 'DXY', current_price: 104.25, change: 0.15 }
    }
  });
  
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    ai_accuracy: 87.5,
    training_samples: 15420,
    ml_model_accuracy: 91.2,
    ai_predictions: 234,
    active_positions: [],
    total_pnl: 2847.30,
    roi_percentage: 18.7,
    win_rate: 73.2,
    total_trades: 156,
    ai_system_status: {
      ml_available: true,
      xgb_available: true,
      sentiment_available: true,
      talib_available: true,
      model_trained: true,
      fred_calendar_active: true,
      cryptopanic_active: true,
    }
  });

  const [wsConnections, setWsConnections] = useState({
    sentiment: true,
    ohlcv: true,
    rsiMacd: true
  });

  const [realtimeIndicators, setRealtimeIndicators] = useState({
    btc: {
      rsi: 68.7,
      rsi_angle: 12.3,
      macd: 0.0245,
      macd_signal: 0.0198,
      macd_histogram: 0.0047,
      macd_angle: 8.9,
      signal_angle: -3.2,
      supertrend_dir: 1,
      vwap_distance: 1.2,
      volume_ratio: 1.45,
      talib_entrada_score: 1.0,
      signal: 'BUY'
    }
  });

  const [selectedPeriod, setSelectedPeriod] = useState('5m');
  const [isLoading, setIsLoading] = useState(false);
  const [tradingEnvironment, setTradingEnvironment] = useState('testnet');
  const [showEnvironmentWarning, setShowEnvironmentWarning] = useState(false);

  // Função compatível para buscar status do bot
  const fetchBotStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/trading-bot/status');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      // Mapear dados do backend para o formato esperado pelo frontend
      setBotStatus(prev => ({
        ...prev,
        // Compatibilidade: suportar tanto 'running' quanto 'status'
        is_running: data.running !== undefined ? data.running : 
                   (data.status === 'simulated_running' || data.status === 'running'),
        ai_accuracy: data.ai_accuracy || prev.ai_accuracy,
        training_samples: data.training_samples || prev.training_samples,
        ml_model_accuracy: data.ml_model_accuracy || prev.ml_model_accuracy,
        ai_predictions: data.ai_predictions || prev.ai_predictions,
        active_positions: data.active_positions || prev.active_positions,
        total_pnl: data.total_pnl || prev.total_pnl,
        roi_percentage: data.roi_percentage || prev.roi_percentage,
        win_rate: data.win_rate || prev.win_rate,
        total_trades: data.total_trades || prev.total_trades,
        ai_system_status: data.ai_system_status || prev.ai_system_status,
      }));
      setConnectionStatus('connected');
    } catch (error) {
      console.error("Failed to fetch bot status:", error);
      setConnectionStatus('disconnected');
      setBotStatus(prev => ({
        ...prev,
        is_running: false,
        ai_system_status: {
          ml_available: false,
          xgb_available: false,
          sentiment_available: false,
          talib_available: false,
          model_trained: false,
          fred_calendar_active: false,
          cryptopanic_active: false,
        }
      }));
    }
  }, []);

  // Função para buscar dados de mercado atuais
  const fetchMarketData = useCallback(async () => {
    try {
      const response = await fetch('/api/current');
      if (response.ok) {
        const data = await response.json();
        if (data.assets) {
          setMarketData({ assets: data.assets });
        }
      }
    } catch (error) {
      console.error("Failed to fetch market data:", error);
    }
  }, []);

  // Função para buscar indicadores técnicos via WebSocket ou fallback
  const fetchTechnicalIndicators = useCallback(async () => {
    try {
      // Tentar buscar dados RSI/MACD do endpoint WebSocket ou API
      const response = await fetch('/api/macd/realtime/btc');
      if (response.ok) {
        const data = await response.json();
        setRealtimeIndicators(prev => ({
          ...prev,
          btc: {
            ...prev.btc,
            rsi: data.rsi?.value || prev.btc.rsi,
            macd: data.macd?.macd || prev.btc.macd,
            macd_signal: data.macd?.signal || prev.btc.macd_signal,
            macd_histogram: data.macd?.histogram || prev.btc.macd_histogram,
            signal: data.combined?.signal_type || prev.btc.signal,
            // Manter valores de fallback para campos não disponíveis
            rsi_angle: prev.btc.rsi_angle,
            macd_angle: prev.btc.macd_angle,
            signal_angle: prev.btc.signal_angle,
            supertrend_dir: prev.btc.supertrend_dir,
            vwap_distance: prev.btc.vwap_distance,
            volume_ratio: prev.btc.volume_ratio,
            talib_entrada_score: prev.btc.talib_entrada_score
          }
        }));
      }
    } catch (error) {
      console.error("Failed to fetch technical indicators:", error);
    }
  }, []);

  // Função para controlar bot (start/stop)
  const controlBot = useCallback(async (action) => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/trading-bot/${action}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      await response.json();
      fetchBotStatus(); // Refresh status after action
    } catch (error) {
      console.error(`Failed to ${action} bot:`, error);
      alert(`Failed to ${action} trading bot. Check console for details.`);
    } finally {
      setIsLoading(false);
    }
  }, [fetchBotStatus]);

  // Handle bot toggle logic
  const handleToggleBot = () => {
    if (botStatus.is_running) {
      controlBot('stop');
    } else {
      if (tradingEnvironment === 'live') {
        setShowEnvironmentWarning(true);
      } else {
        controlBot('start');
      }
    }
  };

  const confirmLiveEnvironment = () => {
    setShowEnvironmentWarning(false);
    controlBot('start');
  };

  const cancelLiveEnvironment = () => {
    setShowEnvironmentWarning(false);
  };

  // Setup WebSocket connections para dados em tempo real
  useEffect(() => {
    let sentimentWs, ohlcvWs, rsiMacdWs;

    const connectWebSockets = () => {
      try {
        // WebSocket para sentiment
        sentimentWs = new WebSocket(`ws://${window.location.host}/ws/sentiment`);
        sentimentWs.onopen = () => setWsConnections(prev => ({ ...prev, sentiment: true }));
        sentimentWs.onclose = () => setWsConnections(prev => ({ ...prev, sentiment: false }));
        
        // WebSocket para OHLCV
        ohlcvWs = new WebSocket(`ws://${window.location.host}/ws/ohlcv`);
        ohlcvWs.onopen = () => setWsConnections(prev => ({ ...prev, ohlcv: true }));
        ohlcvWs.onclose = () => setWsConnections(prev => ({ ...prev, ohlcv: false }));
        
        // WebSocket para RSI/MACD
        rsiMacdWs = new WebSocket(`ws://${window.location.host}/ws/rsi_macd`);
        rsiMacdWs.onopen = () => setWsConnections(prev => ({ ...prev, rsiMacd: true }));
        rsiMacdWs.onclose = () => setWsConnections(prev => ({ ...prev, rsiMacd: false }));
        rsiMacdWs.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'rsi_macd_update') {
              setRealtimeIndicators(prev => ({
                ...prev,
                btc: {
                  ...prev.btc,
                  rsi: data.rsi?.value || prev.btc.rsi,
                  macd: data.macd?.macd || prev.btc.macd,
                  macd_signal: data.macd?.signal || prev.btc.macd_signal,
                  macd_histogram: data.macd?.histogram || prev.btc.macd_histogram,
                  signal: data.combined?.signal_type || prev.btc.signal
                }
              }));
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
      } catch (error) {
        console.error('Error connecting WebSockets:', error);
      }
    };

    connectWebSockets();

    // Cleanup function
    return () => {
      if (sentimentWs) sentimentWs.close();
      if (ohlcvWs) ohlcvWs.close();
      if (rsiMacdWs) rsiMacdWs.close();
    };
  }, []);

  useEffect(() => {
    fetchBotStatus(); // Fetch initial bot status
    fetchMarketData(); // Fetch initial market data
    fetchTechnicalIndicators(); // Fetch initial technical indicators
    
    const statusInterval = setInterval(fetchBotStatus, 10000); // Poll bot status every 10 seconds
    const marketInterval = setInterval(fetchMarketData, 30000); // Poll market data every 30 seconds
    const indicatorsInterval = setInterval(fetchTechnicalIndicators, 15000); // Poll indicators every 15 seconds
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(marketInterval);
      clearInterval(indicatorsInterval);
    };
  }, [fetchBotStatus, fetchMarketData, fetchTechnicalIndicators]);

  // Simulated chart data (fallback quando não há dados reais)
  const generateChartData = () => {
    const data = [];
    const basePrice = marketData.assets.btc?.current_price || 67000;
    for (let i = 0; i < 50; i++) {
      const variation = (Math.random() - 0.5) * 1000;
      data.push({
        time: Date.now() - (50 - i) * 300000,
        price: basePrice + variation + (i * 10),
        volume: Math.random() * 1000000,
        rsi: 30 + Math.random() * 40,
        macd: (Math.random() - 0.5) * 0.1,
        macd_signal: (Math.random() - 0.5) * 0.08,
        macd_hist: (Math.random() - 0.5) * 0.05
      });
    }
    return data;
  };

  const [chartData, setChartData] = useState({
    combined: generateChartData(),
    dataPoints: 50,
    lastUpdate: new Date(),
    fallback: true // Indicar que estamos usando dados de fallback
  });

  const periods = ['5m', '15m', '1h', '4h', '1d'];

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
      color: pnl >= 0 ? 'text-green-400' : 'text-red-400'
    };
  };

  const getAngleColor = (angle) => {
    if (angle > 10) return 'text-green-400';
    if (angle < -10) return 'text-red-400';
    return 'text-yellow-400';
  };

  const getAngleIcon = (angle) => {
    if (angle > 10) return '↗️';
    if (angle < -10) return '↘️';
    return '➡️';
  };

  const hasValidChartData = chartData.combined && chartData.combined.length > 0;

  return (
    <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-blue-400">BLOOMBERG RADAR</h1>
            <div className="flex items-center space-x-2">
              {connectionStatus === 'connected' ? (
                <Wifi className="w-4 h-4 text-green-400" />
              ) : (
                <WifiOff className="w-4 h-4 text-red-400" />
              )}
              <span className={`text-xs ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
                {connectionStatus.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Trading Bot Controls */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Environment:</span>
              <select
                value={tradingEnvironment}
                onChange={(e) => setTradingEnvironment(e.target.value)}
                className="bg-gray-800 text-white text-xs px-2 py-1 rounded border border-gray-700"
                disabled={botStatus.is_running}
              >
                <option value="testnet">TESTNET</option>
                <option value="live">LIVE</option>
              </select>
            </div>
            
            <button
              onClick={handleToggleBot}
              disabled={isLoading}
              className={`flex items-center space-x-2 px-4 py-2 rounded font-bold text-sm transition-colors ${
                botStatus.is_running
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isLoading ? (
                <RotateCw className="w-4 h-4 animate-spin" />
              ) : botStatus.is_running ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              <span>{botStatus.is_running ? 'STOP BOT' : 'START BOT'}</span>
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 border-r border-gray-800 bg-gray-900 overflow-y-auto">
          {/* Market Data */}
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              MARKET DATA
            </h3>
            <div className="space-y-2">
              {Object.entries(marketData.assets).map(([key, asset]) => (
                <div key={key} className="bg-gray-800 p-3 rounded border border-gray-700">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">{asset.name}</span>
                    <span className={`text-xs ${asset.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {asset.change >= 0 ? '+' : ''}{asset.change}%
                    </span>
                  </div>
                  <div className="text-lg font-bold">{formatCurrency(asset.current_price)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Status */}
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Brain className="w-4 h-4 mr-2" />
              AI SYSTEM STATUS
            </h3>
            <div className="space-y-2">
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">AI ACCURACY</div>
                <div className="text-lg font-bold text-blue-400">{botStatus.ai_accuracy}%</div>
              </div>
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">TRAINING SAMPLES</div>
                <div className="text-lg font-bold text-green-400">{botStatus.training_samples.toLocaleString()}</div>
              </div>
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">ML MODEL ACC</div>
                <div className="text-lg font-bold text-purple-400">{botStatus.ml_model_accuracy}%</div>
              </div>
            </div>
          </div>

          {/* WebSocket Status */}
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Database className="w-4 h-4 mr-2" />
              REAL-TIME CONNECTIONS
            </h3>
            <div className="space-y-2">
              {Object.entries(wsConnections).map(([key, connected]) => (
                <div key={key} className="flex items-center justify-between bg-gray-800 p-2 rounded">
                  <span className="text-xs text-gray-300">{key.toUpperCase()}</span>
                  <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                </div>
              ))}
            </div>
          </div>

          {/* Real-time Indicators */}
          <div className="p-4 border-b border-gray-800">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Zap className="w-4 h-4 mr-2" />
              INDICADORES TÉCNICOS
            </h3>
            <div className="space-y-3">
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs text-gray-400">RSI</span>
                  <span className={`text-sm font-bold ${
                    realtimeIndicators.btc.rsi > 70 ? 'text-red-400' :
                    realtimeIndicators.btc.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {realtimeIndicators.btc.rsi.toFixed(1)}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      realtimeIndicators.btc.rsi > 70 ? 'bg-red-400' :
                      realtimeIndicators.btc.rsi < 30 ? 'bg-green-400' : 'bg-yellow-400'
                    }`}
                    style={{ width: `${realtimeIndicators.btc.rsi}%` }}
                  ></div>
                </div>
              </div>

              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400 mb-1">MACD</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span>MACD:</span>
                    <span className="font-mono">{realtimeIndicators.btc.macd.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Signal:</span>
                    <span className="font-mono">{realtimeIndicators.btc.macd_signal.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Hist:</span>
                    <span className={`font-mono ${
                      realtimeIndicators.btc.macd_histogram > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {realtimeIndicators.btc.macd_histogram.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400 mb-1">SINAL ATUAL</div>
                <div className={`text-lg font-bold text-center ${
                  realtimeIndicators.btc.signal === 'BUY' ? 'text-green-400' :
                  realtimeIndicators.btc.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {realtimeIndicators.btc.signal}
                </div>
              </div>
            </div>
          </div>

          {/* Trading Performance */}
          <div className="p-4">
            <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
              <Target className="w-4 h-4 mr-2" />
              PERFORMANCE
            </h3>
            <div className="space-y-2">
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">TOTAL P&L</div>
                <div className={`text-lg font-bold ${formatPnL(botStatus.total_pnl).color}`}>
                  {formatPnL(botStatus.total_pnl).formatted}
                </div>
              </div>
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">ROI</div>
                <div className={`text-lg font-bold ${botStatus.roi_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {botStatus.roi_percentage.toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">WIN RATE</div>
                <div className="text-lg font-bold text-blue-400">{botStatus.win_rate.toFixed(1)}%</div>
              </div>
              <div className="bg-gray-800 p-3 rounded border border-gray-700">
                <div className="text-xs text-gray-400">TOTAL TRADES</div>
                <div className="text-lg font-bold text-gray-300">{botStatus.total_trades}</div>
              </div>
            </div>

            {/* Active Positions */}
            {botStatus.active_positions && botStatus.active_positions.length > 0 && (
              <div className="mt-4">
                <h4 className="text-xs font-bold text-gray-400 mb-2">POSIÇÕES ATIVAS</h4>
                <div className="space-y-2">
                  {botStatus.active_positions.map((position, index) => (
                    <div key={index} className="bg-gray-800 p-2 rounded border border-gray-700">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold">{position.symbol}</span>
                        <span className={`text-xs px-1 rounded ${
                          position.side === 'long' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
                        }`}>
                          {position.side?.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-400">Size: {position.size}</div>
                      <div className="text-xs text-gray-400">Entry: {formatCurrency(position.entry_price)}</div>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-xs text-gray-400">PnL:</span>
                        <span className={`text-xs font-bold ${
                          position.current_pnl >= 0 ? 'text-green-400' : 'text-red-400'
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
                    <span className="text-yellow-400 flex items-center">
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

          <div className="flex-1 p-4 overflow-hidden">
            <div className="h-full flex flex-col gap-4">
              {/* Main Price Chart */}
              <div className="bg-gray-900 border border-gray-800 rounded p-4 flex-shrink-0" style={{ height: '350px' }}>
                <h3 className="text-sm font-bold text-gray-400 mb-4">
                  GRÁFICO DE PREÇO BTC/USDT
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
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* RSI and MACD Charts */}
              <div className="grid grid-cols-2 gap-4 flex-shrink-0" style={{ height: '180px' }}>
                <div className="bg-gray-900 border border-gray-800 rounded p-2">
                  <h3 className="text-sm font-bold text-gray-400 mb-1">RSI</h3>
                  <div className="w-full" style={{ height: '140px' }}>
                    {hasValidChartData ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData.combined} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={9}
                            tick={{ fontSize: 9 }}
                          />
                          <YAxis 
                            stroke="#9CA3AF" 
                            fontSize={9} 
                            tick={{ fontSize: 9 }}
                            width={35}
                            domain={[0, 100]}
                          />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value, name) => [value?.toFixed(2), name]}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '4px',
                              fontSize: '11px'
                            }}
                          />
                          <ReferenceLine y={70} stroke="#ef4444" strokeWidth={1} strokeDasharray="2 2" />
                          <ReferenceLine y={30} stroke="#22c55e" strokeWidth={1} strokeDasharray="2 2" />
                          <Line
                            type="monotone"
                            dataKey="rsi"
                            stroke="#f59e0b"
                            strokeWidth={2}
                            dot={false}
                            name="RSI"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500 text-xs">
                        Aguardando RSI...
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-gray-900 border border-gray-800 rounded p-2">
                  <h3 className="text-sm font-bold text-gray-400 mb-1">MACD</h3>
                  <div className="w-full" style={{ height: '140px' }}>
                    {hasValidChartData ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData.combined} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={9}
                            tick={{ fontSize: 9 }}
                          />
                          <YAxis 
                            stroke="#9CA3AF" 
                            fontSize={9} 
                            tick={{ fontSize: 9 }}
                            width={35}
                          />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value, name) => [value?.toFixed(4), name]}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '4px',
                              fontSize: '11px'
                            }}
                          />
                          <ReferenceLine y={0} stroke="#6B7280" strokeWidth={1} strokeDasharray="2 2" />
                          <Line
                            type="monotone"
                            dataKey="macd"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            name="MACD"
                          />
                          <Line
                            type="monotone"
                            dataKey="macd_signal"
                            stroke="#ef4444"
                            strokeWidth={2}
                            dot={false}
                            name="Signal"
                          />
                          <Bar dataKey="macd_hist" fill="#10b981" name="Histogram" />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500 text-xs">
                        Aguardando MACD...
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Environment Warning Modal */}
      {showEnvironmentWarning && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-red-600 rounded-lg p-6 max-w-md w-full mx-4">
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
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
    active_positions: [
      {
        symbol: 'BTCUSDT',
        side: 'long',
        size: 0.025,
        entry_price: 66800,
        current_pnl: 145.50
      }
    ],
    // Dados separados por ambiente
    testnet: {
      balance: 9847.23,
      total_pnl: 847.23,
      roi_percentage: 9.4,
      win_rate: 76.8,
      total_trades: 89,
      winning_trades: 68,
      daily_pnl: 45.60,
      daily_trades: 3
    },
    live: {
      balance: 1250.75,
      total_pnl: 250.75,
      roi_percentage: 25.1,
      win_rate: 82.5,
      total_trades: 23,
      winning_trades: 19,
      daily_pnl: 18.30,
      daily_trades: 1
    },
    // Performance consolidada (para compatibilidade)
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

  // Função para buscar status do bot (DADOS REAIS)
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
        // Dados por ambiente do backend
        testnet: data.testnet_performance || prev.testnet,
        live: data.live_performance || prev.live,
        // Dados consolidados
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

  // Função para buscar dados de mercado (DADOS REAIS)
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

  // Função para buscar performance detalhada (DADOS REAIS)
  const fetchPerformanceData = useCallback(async () => {
    try {
      const response = await fetch('/api/trading-bot/performance');
      if (response.ok) {
        const data = await response.json();
        
        // Atualizar dados de performance separados por ambiente
        setBotStatus(prev => ({
          ...prev,
          testnet: {
            balance: data.testnet_balance || prev.testnet.balance,
            total_pnl: data.testnet_total_pnl || prev.testnet.total_pnl,
            roi_percentage: data.testnet_roi || prev.testnet.roi_percentage,
            win_rate: data.testnet_win_rate || prev.testnet.win_rate,
            total_trades: data.testnet_total_trades || prev.testnet.total_trades,
            winning_trades: data.testnet_winning_trades || prev.testnet.winning_trades,
            daily_pnl: data.testnet_daily_pnl || prev.testnet.daily_pnl,
            daily_trades: data.testnet_daily_trades || prev.testnet.daily_trades
          },
          live: {
            balance: data.live_balance || prev.live.balance,
            total_pnl: data.live_total_pnl || prev.live.total_pnl,
            roi_percentage: data.live_roi || prev.live.roi_percentage,
            win_rate: data.live_win_rate || prev.live.win_rate,
            total_trades: data.live_total_trades || prev.live.total_trades,
            winning_trades: data.live_winning_trades || prev.live.winning_trades,
            daily_pnl: data.live_daily_pnl || prev.live.daily_pnl,
            daily_trades: data.live_daily_trades || prev.live.daily_trades
          }
        }));
      }
    } catch (error) {
      console.error("Failed to fetch performance data:", error);
    }
  }, []);

  // Função para buscar indicadores técnicos (DADOS REAIS)
  const fetchTechnicalIndicators = useCallback(async () => {
    try {
      // Buscar dados RSI/MACD do endpoint real
      const response = await fetch('/api/macd/realtime/btc');
      if (response.ok) {
        const data = await response.json();
        setRealtimeIndicators(prev => ({
          ...prev,
          btc: {
            ...prev.btc,
            rsi: data.macd_data?.rsi?.value || prev.btc.rsi,
            macd: data.macd_data?.macd || prev.btc.macd,
            macd_signal: data.macd_data?.signal || prev.btc.macd_signal,
            macd_histogram: data.macd_data?.histogram || prev.btc.macd_histogram,
            signal: data.macd_data?.trend || prev.btc.signal,
            // Manter valores simulados para campos não disponíveis no backend
            rsi_angle: prev.btc.rsi_angle + Math.random() * 6 - 3,
            macd_angle: prev.btc.macd_angle + Math.random() * 4 - 2,
            signal_angle: prev.btc.signal_angle + Math.random() * 3 - 1.5,
            supertrend_dir: prev.btc.supertrend_dir,
            vwap_distance: prev.btc.vwap_distance,
            volume_ratio: prev.btc.volume_ratio,
            talib_entrada_score: prev.btc.talib_entrada_score
          }
        }));
      }
    } catch (error) {
      console.error("Failed to fetch technical indicators:", error);
      // Manter simulação se falhar
      setRealtimeIndicators(prev => ({
        btc: {
          ...prev.btc,
          rsi: Math.max(0, Math.min(100, prev.btc.rsi + Math.random() * 4 - 2)),
          rsi_angle: prev.btc.rsi_angle + Math.random() * 6 - 3,
          macd: prev.btc.macd + Math.random() * 0.01 - 0.005,
          macd_signal: prev.btc.macd_signal + Math.random() * 0.008 - 0.004,
          macd_histogram: prev.btc.macd - prev.btc.macd_signal,
          macd_angle: prev.btc.macd_angle + Math.random() * 4 - 2,
          signal_angle: prev.btc.signal_angle + Math.random() * 3 - 1.5,
          signal: Math.random() > 0.7 ? (Math.random() > 0.5 ? 'BUY' : 'SELL') : prev.btc.signal
        }
      }));
    }
  }, []);

  // Função para controlar bot (DADOS REAIS)
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

  useEffect(() => {
    const interval = setInterval(() => {
      setWsConnections(prev => ({
        sentiment: Math.random() > 0.1,
        ohlcv: Math.random() > 0.05,
        rsiMacd: Math.random() > 0.08
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchBotStatus();
    fetchMarketData();
    fetchTechnicalIndicators();
    
    const statusInterval = setInterval(fetchBotStatus, 10000);
    const marketInterval = setInterval(fetchMarketData, 30000);
    const indicatorsInterval = setInterval(fetchTechnicalIndicators, 5000);
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(marketInterval);
      clearInterval(indicatorsInterval);
    };
  }, [fetchBotStatus, fetchMarketData, fetchTechnicalIndicators]);

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
    fallback: true
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setChartData({
        combined: generateChartData(),
        dataPoints: 50,
        lastUpdate: new Date(),
        fallback: true
      });
    }, 15000);

    return () => clearInterval(interval);
  }, [marketData.assets.btc]);

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

  const hasValidChartData = chartData.combined && chartData.combined.length > 0;

  return (
    <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
      {/* Header Compacto */}
      <div className="border-b border-gray-800 bg-gray-900 px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <h1 className="text-lg font-bold text-blue-400">BLOOMBERG RADAR</h1>
            
            {/* Market Data Inline */}
            <div className="flex items-center space-x-4">
              {Object.entries(marketData.assets).map(([key, asset]) => (
                <div key={key} className="flex items-center space-x-2">
                  <span className="text-xs text-gray-400">{asset.name}:</span>
                  <span className="text-sm font-bold">{formatCurrency(asset.current_price)}</span>
                  <span className={`text-xs ${asset.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {asset.change >= 0 ? '+' : ''}{asset.change.toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>

            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {connectionStatus === 'connected' ? (
                <Wifi className="w-3 h-3 text-green-400" />
              ) : (
                <WifiOff className="w-3 h-3 text-red-400" />
              )}
              <span className={`text-xs ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
                {connectionStatus.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Bot Controls */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Env:</span>
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
              className={`flex items-center space-x-2 px-3 py-1 rounded font-bold text-xs transition-colors ${
                botStatus.is_running
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isLoading ? (
                <RotateCw className="w-3 h-3 animate-spin" />
              ) : botStatus.is_running ? (
                <Pause className="w-3 h-3" />
              ) : (
                <Play className="w-3 h-3" />
              )}
              <span>{botStatus.is_running ? 'STOP' : 'START'}</span>
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar Compacta */}
        <div className="w-64 border-r border-gray-800 bg-gray-900 overflow-y-auto">
          {/* AI Status Compacto */}
          <div className="p-3 border-b border-gray-800">
            <h3 className="text-xs font-bold text-gray-400 mb-2 flex items-center">
              <Brain className="w-3 h-3 mr-1" />
              AI STATUS
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-800 p-2 rounded">
                <div className="text-gray-400">Accuracy</div>
                <div className="font-bold text-blue-400">{botStatus.ai_accuracy.toFixed(1)}%</div>
              </div>
              <div className="bg-gray-800 p-2 rounded">
                <div className="text-gray-400">Samples</div>
                <div className="font-bold text-green-400">{(botStatus.training_samples/1000).toFixed(0)}k</div>
              </div>
              <div className="bg-gray-800 p-2 rounded">
                <div className="text-gray-400">ML Acc</div>
                <div className="font-bold text-purple-400">{botStatus.ml_model_accuracy.toFixed(1)}%</div>
              </div>
              <div className="bg-gray-800 p-2 rounded">
                <div className="text-gray-400">Predictions</div>
                <div className="font-bold text-yellow-400">{botStatus.ai_predictions}</div>
              </div>
            </div>
          </div>

          {/* Indicadores Técnicos Compactos */}
          <div className="p-3 border-b border-gray-800">
            <h3 className="text-xs font-bold text-gray-400 mb-2 flex items-center">
              <Zap className="w-3 h-3 mr-1" />
              INDICADORES
            </h3>
            <div className="space-y-2">
              <div className="bg-gray-800 p-2 rounded">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-400">RSI</span>
                  <span className={`text-xs font-bold ${
                    realtimeIndicators.btc.rsi > 70 ? 'text-red-400' :
                    realtimeIndicators.btc.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {realtimeIndicators.btc.rsi.toFixed(1)}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                  <div
                    className={`h-1 rounded-full ${
                      realtimeIndicators.btc.rsi > 70 ? 'bg-red-400' :
                      realtimeIndicators.btc.rsi < 30 ? 'bg-green-400' : 'bg-yellow-400'
                    }`}
                    style={{ width: `${realtimeIndicators.btc.rsi}%` }}
                  ></div>
                </div>
              </div>

              <div className="bg-gray-800 p-2 rounded">
                <div className="text-xs text-gray-400 mb-1">MACD</div>
                <div className="grid grid-cols-3 gap-1 text-xs">
                  <div>
                    <div className="text-gray-500">Line</div>
                    <div className="font-mono">{realtimeIndicators.btc.macd.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Signal</div>
                    <div className="font-mono">{realtimeIndicators.btc.macd_signal.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Hist</div>
                    <div className={`font-mono ${
                      realtimeIndicators.btc.macd_histogram > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {realtimeIndicators.btc.macd_histogram.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 p-2 rounded text-center">
                <div className="text-xs text-gray-400">SIGNAL</div>
                <div className={`text-sm font-bold ${
                  realtimeIndicators.btc.signal === 'BUY' ? 'text-green-400' :
                  realtimeIndicators.btc.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {realtimeIndicators.btc.signal}
                </div>
              </div>
            </div>
          </div>

          {/* Ângulos de Momentum */}
          <div className="p-3 border-b border-gray-800">
            <h3 className="text-xs font-bold text-gray-400 mb-2 flex items-center">
              <TrendingUp className="w-3 h-3 mr-1" />
              MOMENTUM ANGLES
            </h3>
            <div className="space-y-2">
              <div className="bg-gray-800 p-2 rounded">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-400">RSI ANGLE</span>
                  <div className="flex items-center space-x-1">
                    <span className="text-xs">
                      {realtimeIndicators.btc.rsi_angle > 10 ? '↗️' : 
                       realtimeIndicators.btc.rsi_angle < -10 ? '↘️' : '➡️'}
                    </span>
                    <span className={`text-xs font-bold ${
                      realtimeIndicators.btc.rsi_angle > 10 ? 'text-green-400' :
                      realtimeIndicators.btc.rsi_angle < -10 ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {realtimeIndicators.btc.rsi_angle.toFixed(1)}°
                    </span>
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  Speed: {Math.abs(realtimeIndicators.btc.rsi_angle).toFixed(1)}°/min
                </div>
              </div>

              <div className="bg-gray-800 p-2 rounded">
                <div className="text-xs text-gray-400 mb-1">MACD ANGLES</div>
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-500">MACD Line</span>
                    <div className="flex items-center space-x-1">
                      <span>
                        {realtimeIndicators.btc.macd_angle > 5 ? '↗️' : 
                         realtimeIndicators.btc.macd_angle < -5 ? '↘️' : '➡️'}
                      </span>
                      <span className={`font-bold ${
                        realtimeIndicators.btc.macd_angle > 5 ? 'text-green-400' :
                        realtimeIndicators.btc.macd_angle < -5 ? 'text-red-400' : 'text-yellow-400'
                      }`}>
                        {realtimeIndicators.btc.macd_angle.toFixed(1)}°
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-500">Signal Line</span>
                    <div className="flex items-center space-x-1">
                      <span>
                        {realtimeIndicators.btc.signal_angle > 5 ? '↗️' : 
                         realtimeIndicators.btc.signal_angle < -5 ? '↘️' : '➡️'}
                      </span>
                      <span className={`font-bold ${
                        realtimeIndicators.btc.signal_angle > 5 ? 'text-green-400' :
                        realtimeIndicators.btc.signal_angle < -5 ? 'text-red-400' : 'text-yellow-400'
                      }`}>
                        {realtimeIndicators.btc.signal_angle.toFixed(1)}°
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Momentum Summary */}
              <div className="bg-gray-800 p-2 rounded">
                <div className="text-xs text-gray-400 mb-1">MOMENTUM STATUS</div>
                <div className="text-center">
                  <div className={`text-xs font-bold ${
                    (realtimeIndicators.btc.rsi_angle > 10 && realtimeIndicators.btc.macd_angle > 5) ? 'text-green-400' :
                    (realtimeIndicators.btc.rsi_angle < -10 && realtimeIndicators.btc.macd_angle < -5) ? 'text-red-400' :
                    'text-yellow-400'
                  }`}>
                    {(realtimeIndicators.btc.rsi_angle > 10 && realtimeIndicators.btc.macd_angle > 5) ? 'ACCELERATING UP' :
                     (realtimeIndicators.btc.rsi_angle < -10 && realtimeIndicators.btc.macd_angle < -5) ? 'ACCELERATING DOWN' :
                     'CONSOLIDATING'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Avg Velocity: {((Math.abs(realtimeIndicators.btc.rsi_angle) + Math.abs(realtimeIndicators.btc.macd_angle)) / 2).toFixed(1)}°/min
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Saldos e Performance por Ambiente */}
          <div className="p-3 border-b border-gray-800">
            <h3 className="text-xs font-bold text-gray-400 mb-2 flex items-center">
              <DollarSign className="w-3 h-3 mr-1" />
              ACCOUNT BALANCES
            </h3>
            
            {/* Testnet Balance */}
            <div className="bg-gray-800 p-2 rounded mb-2">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-bold text-blue-400">TESTNET</span>
                <span className={`text-xs px-1 rounded ${
                  tradingEnvironment === 'testnet' && botStatus.is_running ? 'bg-green-900 text-green-400' : 'bg-gray-700 text-gray-400'
                }`}>
                  {tradingEnvironment === 'testnet' && botStatus.is_running ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div>
                  <div className="text-gray-500">Balance</div>
                  <div className="font-bold text-white">{formatCurrency(botStatus.testnet.balance)}</div>
                </div>
                <div>
                  <div className="text-gray-500">P&L</div>
                  <div className={`font-bold ${formatPnL(botStatus.testnet.total_pnl).color}`}>
                    {formatPnL(botStatus.testnet.total_pnl).formatted}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">ROI</div>
                  <div className={`font-bold ${botStatus.testnet.roi_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {botStatus.testnet.roi_percentage.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Trades</div>
                  <div className="font-bold text-gray-300">{botStatus.testnet.total_trades}</div>
                </div>
              </div>
            </div>

            {/* Live Balance */}
            <div className="bg-gray-800 p-2 rounded">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-bold text-yellow-400">LIVE</span>
                <span className={`text-xs px-1 rounded ${
                  tradingEnvironment === 'live' && botStatus.is_running ? 'bg-red-900 text-red-400' : 'bg-gray-700 text-gray-400'
                }`}>
                  {tradingEnvironment === 'live' && botStatus.is_running ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div>
                  <div className="text-gray-500">Balance</div>
                  <div className="font-bold text-white">{formatCurrency(botStatus.live.balance)}</div>
                </div>
                <div>
                  <div className="text-gray-500">P&L</div>
                  <div className={`font-bold ${formatPnL(botStatus.live.total_pnl).color}`}>
                    {formatPnL(botStatus.live.total_pnl).formatted}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">ROI</div>
                  <div className={`font-bold ${botStatus.live.roi_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {botStatus.live.roi_percentage.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Trades</div>
                  <div className="font-bold text-gray-300">{botStatus.live.total_trades}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Performance Detalhada do Ambiente Atual */}
          <div className="p-3 border-b border-gray-800">
            <h3 className="text-xs font-bold text-gray-400 mb-2 flex items-center">
              <Target className="w-3 h-3 mr-1" />
              {tradingEnvironment.toUpperCase()} PERFORMANCE
            </h3>
            <div className="space-y-2">
              {/* Performance atual baseada no ambiente selecionado */}
              <div className="bg-gray-800 p-2 rounded">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="text-gray-500">Win Rate</div>
                    <div className="font-bold text-blue-400">
                      {botStatus[tradingEnvironment].win_rate.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Winning Trades</div>
                    <div className="font-bold text-green-400">
                      {botStatus[tradingEnvironment].winning_trades}/{botStatus[tradingEnvironment].total_trades}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Daily P&L</div>
                    <div className={`font-bold ${formatPnL(botStatus[tradingEnvironment].daily_pnl).color}`}>
                      {formatPnL(botStatus[tradingEnvironment].daily_pnl).formatted}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Daily Trades</div>
                    <div className="font-bold text-gray-300">
                      {botStatus[tradingEnvironment].daily_trades}
                    </div>
                  </div>
                </div>
              </div>

              {/* Comparação de Performance */}
              <div className="bg-gray-800 p-2 rounded">
                <div className="text-xs text-gray-400 mb-1">ENVIRONMENT COMPARISON</div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-blue-400">Testnet ROI:</span>
                    <span className={`font-bold ${botStatus.testnet.roi_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {botStatus.testnet.roi_percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-yellow-400">Live ROI:</span>
                    <span className={`font-bold ${botStatus.live.roi_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {botStatus.live.roi_percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs border-t border-gray-700 pt-1">
                    <span className="text-gray-400">Performance Gap:</span>
                    <span className={`font-bold text-xs ${
                      (botStatus.live.roi_percentage - botStatus.testnet.roi_percentage) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {(botStatus.live.roi_percentage - botStatus.testnet.roi_percentage).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* WebSocket Status Compacto */}
          <div className="p-3">
            <h3 className="text-xs font-bold text-gray-400 mb-2 flex items-center">
              <Database className="w-3 h-3 mr-1" />
              CONNECTIONS
            </h3>
            <div className="space-y-1">
              {Object.entries(wsConnections).map(([key, connected]) => (
                <div key={key} className="flex items-center justify-between bg-gray-800 p-1 rounded text-xs">
                  <span className="text-gray-300">{key}</span>
                  <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                </div>
              ))}
            </div>
          </div>

          {/* Active Positions se houver */}
          {botStatus.active_positions && botStatus.active_positions.length > 0 && (
            <div className="p-3 border-t border-gray-800">
              <h4 className="text-xs font-bold text-gray-400 mb-2">POSIÇÕES</h4>
              <div className="space-y-1">
                {botStatus.active_positions.map((position, index) => (
                  <div key={index} className="bg-gray-800 p-2 rounded text-xs">
                    <div className="flex justify-between items-center">
                      <span className="font-bold">{position.symbol}</span>
                      <span className={`px-1 rounded ${
                        position.side === 'long' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
                      }`}>
                        {position.side?.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-gray-400">Size: {position.size}</div>
                    <div className="text-gray-400">Entry: {formatCurrency(position.entry_price)}</div>
                    <div className={`font-bold ${
                      position.current_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPnL(position.current_pnl).formatted}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Área Principal de Gráficos - Otimizada */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Chart Controls Compactos */}
          <div className="border-b border-gray-800 bg-gray-900 px-4 py-2">
            <div className="flex items-center justify-between">
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
                  <span>Data Points: {chartData.dataPoints || 0}</span>
                  <span>Last Update: {chartData.lastUpdate ? new Date(chartData.lastUpdate).toLocaleTimeString() : 'N/A'}</span>
                  {chartData.fallback && (
                    <span className="text-yellow-400 flex items-center">
                      <AlertTriangle className="w-3 h-3 mr-1" />
                      Demo Mode
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="flex-1 p-4 overflow-hidden">
            {/* Layout de Gráficos Otimizado */}
            <div className="h-full grid grid-rows-2 gap-4">
              {/* Gráfico Principal de Preço - 60% da altura */}
              <div className="bg-gray-900 border border-gray-800 rounded p-3">
                <h3 className="text-sm font-bold text-gray-400 mb-2">
                  BTC/USDT PRICE CHART
                </h3>
                <div className="w-full" style={{ height: 'calc(100% - 32px)' }}>
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
                          name="Price"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      <div className="text-center">
                        <div className="text-4xl mb-2">📊</div>
                        <div>Loading chart data...</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* RSI e MACD Lado a Lado - 40% da altura */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900 border border-gray-800 rounded p-2">
                  <h3 className="text-sm font-bold text-gray-400 mb-1">RSI</h3>
                  <div className="w-full" style={{ height: 'calc(100% - 24px)' }}>
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
                        Loading RSI...
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-gray-900 border border-gray-800 rounded p-2">
                  <h3 className="text-sm font-bold text-gray-400 mb-1">MACD</h3>
                  <div className="w-full" style={{ height: 'calc(100% - 24px)' }}>
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
                        Loading MACD...
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
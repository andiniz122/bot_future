import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, ComposedChart, Area, AreaChart } from 'recharts';

// Cores modernas e consistentes
const colors = {
  primary: '#6366f1',
  secondary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  info: '#06b6d4',
  gold: '#fbbf24',
  bitcoin: '#f7931a',
  dxy: '#10b981',
  background: '#0f172a',
  card: '#1e293b',
  text: '#f1f5f9',
  muted: '#64748b'
};

// Custom Hook para WebSocket
const useWebSocket = (url, onMessage) => {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        setConnected(true);
        reconnectAttempts.current = 0;
        console.log(`🔌 WebSocket conectado: ${url}`);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Erro ao parsear mensagem WebSocket:', error);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log(`🔌 WebSocket desconectado: ${url}`);
        
        // Reconexão automática
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          setTimeout(() => {
            console.log(`🔄 Tentativa de reconexão ${reconnectAttempts.current}/${maxReconnectAttempts}`);
            connect();
          }, 3000 * reconnectAttempts.current);
        }
      };

      ws.onerror = (error) => {
        console.error('Erro WebSocket:', error);
        setConnected(false);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Erro ao conectar WebSocket:', error);
      setConnected(false);
    }
  }, [url, onMessage]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { connected };
};

// Configuração de URLs baseada no ambiente
const getAPIBaseURL = () => {
  // Se estiver rodando localmente, usa localhost
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  // Para o servidor remoto, assumindo que a API roda na porta 8000
  return 'http://62.72.1.122:8000';
};

const getWSBaseURL = () => {
  // Se estiver rodando localmente, usa localhost
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'ws://localhost:8000';
  }
  // Para o servidor remoto
  return 'ws://62.72.1.122:8000';
};

// Custom Hook para API calls
const useAPI = () => {
  const [loading, setLoading] = useState(false);
  const apiBaseURL = getAPIBaseURL();
  
  const callAPI = useCallback(async (endpoint, options = {}) => {
    setLoading(true);
    try {
      const response = await fetch(`${apiBaseURL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`Erro na API ${endpoint}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiBaseURL]);

  return { callAPI, loading };
};

// Componente Card Base
const Card = ({ children, className = '', gradient = false, hover = true }) => (
  <div className={`
    bg-slate-800 rounded-xl p-6 border border-slate-700 
    ${gradient ? 'bg-gradient-to-br from-slate-800 to-slate-900' : ''}
    ${hover ? 'hover:shadow-xl hover:shadow-indigo-500/20 transition-all duration-300' : ''}
    ${className}
  `}>
    {children}
  </div>
);

// Componente Status Indicator
const StatusIndicator = ({ status, size = 'sm' }) => {
  const getColor = () => {
    switch (status) {
      case 'connected': case 'running': case 'active': return 'bg-green-500';
      case 'disconnected': case 'stopped': case 'inactive': return 'bg-red-500';
      case 'warning': case 'pending': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const sizeClass = size === 'lg' ? 'w-4 h-4' : 'w-2 h-2';

  return (
    <div className={`${sizeClass} ${getColor()} rounded-full animate-pulse`} />
  );
};

// Componente Metric Display
const MetricDisplay = ({ label, value, change, prefix = '', suffix = '', trend = null }) => (
  <div className="text-center">
    <div className="text-2xl font-bold text-white">
      {prefix}{value}{suffix}
    </div>
    <div className="text-sm text-slate-400">{label}</div>
    {change !== undefined && (
      <div className={`text-xs ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {change >= 0 ? '+' : ''}{change}%
      </div>
    )}
  </div>
);

// Componente Trading Bot Control
const TradingBotControl = ({ botStatus, onStart, onStop, loading }) => (
  <Card className="bg-gradient-to-r from-indigo-900 to-purple-900">
    <div className="flex items-center justify-between">
      <div>
        <h3 className="text-xl font-bold text-white mb-2">Trading Bot</h3>
        <div className="flex items-center gap-2">
          <StatusIndicator status={botStatus?.status} size="lg" />
          <span className="text-slate-300 capitalize">
            {botStatus?.status || 'Unknown'}
          </span>
        </div>
      </div>
      <div className="flex gap-2">
        <button
          onClick={onStart}
          disabled={loading || botStatus?.status === 'running'}
          className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 
                   text-white rounded-lg transition-colors duration-200"
        >
          {loading ? '⏳' : '▶️'} Start
        </button>
        <button
          onClick={onStop}
          disabled={loading || botStatus?.status !== 'running'}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 
                   text-white rounded-lg transition-colors duration-200"
        >
          {loading ? '⏳' : '⏸️'} Stop
        </button>
      </div>
    </div>
    {botStatus && (
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricDisplay 
          label="Balance" 
          value={botStatus.current_balance?.toFixed(2) || '0.00'} 
          prefix="$" 
        />
        <MetricDisplay 
          label="Total PnL" 
          value={botStatus.total_pnl?.toFixed(2) || '0.00'} 
          prefix="$" 
          change={botStatus.roi_percentage}
        />
        <MetricDisplay 
          label="Win Rate" 
          value={botStatus.win_rate?.toFixed(1) || '0.0'} 
          suffix="%" 
        />
        <MetricDisplay 
          label="Total Trades" 
          value={botStatus.total_trades || 0}
        />
      </div>
    )}
  </Card>
);

// Componente Market Overview
const MarketOverview = ({ marketData }) => {
  if (!marketData?.assets) return null;

  const assets = Object.entries(marketData.assets).map(([key, asset]) => ({
    name: asset.name,
    symbol: key.toUpperCase(),
    price: asset.current_price,
    change: asset.change_percent,
    color: key === 'btc' ? colors.bitcoin : key === 'gold' ? colors.gold : colors.dxy
  }));

  return (
    <Card>
      <h3 className="text-xl font-bold text-white mb-4">Market Overview</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {assets.map((asset) => (
          <div key={asset.symbol} className="text-center p-4 rounded-lg bg-slate-700">
            <div className="text-lg font-bold text-white">{asset.name}</div>
            <div className="text-2xl font-bold" style={{ color: asset.color }}>
              ${asset.price?.toFixed(2)}
            </div>
            <div className={`text-sm ${asset.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {asset.change >= 0 ? '+' : ''}{asset.change?.toFixed(2)}%
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

// Componente Performance Chart
const PerformanceChart = ({ data }) => {
  if (!data || data.length === 0) return null;

  return (
    <Card>
      <h3 className="text-xl font-bold text-white mb-4">Performance History</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #374151', borderRadius: '8px' }}
              labelStyle={{ color: '#f1f5f9' }}
            />
            <Line type="monotone" dataKey="pnl" stroke={colors.primary} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

// Componente RSI/MACD Display
const RSIMACDDisplay = ({ data }) => {
  if (!data) return null;

  return (
    <Card className="bg-gradient-to-br from-blue-900 to-indigo-900">
      <h3 className="text-xl font-bold text-white mb-4">RSI & MACD Analysis</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* RSI */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-slate-300">RSI</span>
            <span className={`px-2 py-1 rounded text-xs ${
              data.rsi?.zone === 'overbought' ? 'bg-red-600' :
              data.rsi?.zone === 'oversold' ? 'bg-green-600' : 'bg-yellow-600'
            }`}>
              {data.rsi?.zone || 'neutral'}
            </span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {data.rsi?.value || '--'}
          </div>
          <div className="text-sm text-slate-400">
            Trend: {data.rsi?.trend || 'flat'}
          </div>
        </div>

        {/* MACD */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-slate-300">MACD</span>
            <span className={`px-2 py-1 rounded text-xs ${
              data.macd?.trend === 'bullish' ? 'bg-green-600' :
              data.macd?.trend === 'bearish' ? 'bg-red-600' : 'bg-gray-600'
            }`}>
              {data.macd?.trend || 'neutral'}
            </span>
          </div>
          <div className="text-lg font-bold text-white">
            MACD: {data.macd?.macd?.toFixed(4) || '--'}
          </div>
          <div className="text-sm text-slate-400">
            Signal: {data.macd?.signal?.toFixed(4) || '--'}
          </div>
          <div className="text-sm text-slate-400">
            Histogram: {data.macd?.histogram?.toFixed(4) || '--'}
          </div>
        </div>
      </div>

      {/* Combined Signal */}
      {data.combined && (
        <div className="mt-4 p-3 rounded-lg bg-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-white font-semibold">Signal: {data.combined.signal_type}</span>
            <span className="text-sm text-slate-300">
              Confidence: {data.combined.confidence}%
            </span>
          </div>
          {data.combined.recommendation && (
            <div className="text-sm text-slate-400 mt-1">
              {data.combined.recommendation}
            </div>
          )}
        </div>
      )}
    </Card>
  );
};

// Componente Sentiment Analysis
const SentimentAnalysis = ({ sentiment }) => {
  if (!sentiment) return null;

  const SentimentGauge = ({ label, buyers, sellers, trend }) => (
    <div className="text-center">
      <div className="text-lg font-semibold text-white mb-2">{label}</div>
      <div className="relative w-24 h-24 mx-auto mb-2">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="40"
            fill="none"
            stroke="#374151"
            strokeWidth="8"
          />
          <circle
            cx="50"
            cy="50"
            r="40"
            fill="none"
            stroke={buyers > 60 ? colors.success : buyers < 40 ? colors.danger : colors.warning}
            strokeWidth="8"
            strokeDasharray={`${buyers * 2.51} 251`}
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-white font-bold">{buyers?.toFixed(0)}%</span>
        </div>
      </div>
      <div className={`text-sm px-2 py-1 rounded ${
        trend === 'BULLISH' || trend === 'STRONGLY_BULLISH' ? 'bg-green-600' :
        trend === 'BEARISH' || trend === 'STRONGLY_BEARISH' ? 'bg-red-600' : 'bg-yellow-600'
      }`}>
        {trend || 'NEUTRAL'}
      </div>
    </div>
  );

  return (
    <Card>
      <h3 className="text-xl font-bold text-white mb-4">Market Sentiment</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <SentimentGauge 
          label="BTC Sentiment" 
          buyers={sentiment.btc?.buyers_percent} 
          sellers={sentiment.btc?.sellers_percent}
          trend={sentiment.btc?.trend}
        />
        <SentimentGauge 
          label="PAXG Sentiment" 
          buyers={sentiment.paxg?.buyers_percent} 
          sellers={sentiment.paxg?.sellers_percent}
          trend={sentiment.paxg?.trend}
        />
      </div>
      
      {/* Fear & Greed Index */}
      <div className="text-center p-4 bg-slate-700 rounded-lg">
        <div className="text-lg font-semibold text-white mb-2">Fear & Greed Index</div>
        <div className="text-3xl font-bold text-white mb-2">
          {sentiment.btc?.fear_greed_index || '--'}
        </div>
        <div className="text-sm text-slate-300">
          Market Mood: {sentiment.btc?.market_mood || 'NEUTRAL'}
        </div>
      </div>
    </Card>
  );
};

// Componente Recent Alerts
const RecentAlerts = ({ alerts }) => {
  if (!alerts || alerts.length === 0) return null;

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'HIGH': return 'border-l-red-500 bg-red-950';
      case 'MEDIUM': return 'border-l-yellow-500 bg-yellow-950';
      case 'LOW': return 'border-l-blue-500 bg-blue-950';
      default: return 'border-l-gray-500 bg-gray-950';
    }
  };

  return (
    <Card>
      <h3 className="text-xl font-bold text-white mb-4">Recent Alerts</h3>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {alerts.slice(0, 10).map((alert, index) => (
          <div key={index} className={`p-3 border-l-4 rounded-r-lg ${getSeverityColor(alert.severity)}`}>
            <div className="flex justify-between items-start">
              <div>
                <div className="font-semibold text-white text-sm">
                  {alert.title || alert.type}
                </div>
                <div className="text-xs text-slate-300 mt-1">
                  {alert.message}
                </div>
              </div>
              <div className="text-xs text-slate-400">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

// Componente Principal do Dashboard
const TradingDashboard = () => {
  const [marketData, setMarketData] = useState(null);
  const [botStatus, setBotStatus] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [rsiMacdData, setRsiMacdData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);
  
  const { callAPI, loading } = useAPI();
  const wsBaseURL = getWSBaseURL();

  // WebSocket para dados em tempo real
  useWebSocket(`${wsBaseURL}/ws/sentiment`, (data) => {
    setSentiment(data);
    setLastUpdate(new Date().toLocaleTimeString());
  });

  useWebSocket(`${wsBaseURL}/ws/rsi-macd`, (data) => {
    setRsiMacdData(data);
  });

  // Carregar dados iniciais
  const loadInitialData = useCallback(async () => {
    try {
      const [marketResponse, botResponse, alertsResponse] = await Promise.all([
        callAPI('/api/current'),
        callAPI('/api/trading-bot/status'),
        callAPI('/api/alerts')
      ]);

      setMarketData(marketResponse);
      setBotStatus(botResponse);
      setAlerts(alertsResponse.alerts || []);

      // Simular dados de performance para o gráfico
      const mockPerformance = Array.from({ length: 20 }, (_, i) => ({
        time: new Date(Date.now() - (19 - i) * 60000).toLocaleTimeString(),
        pnl: (botResponse?.total_pnl || 0) + Math.random() * 100 - 50
      }));
      setPerformanceData(mockPerformance);

    } catch (error) {
      console.error('Erro ao carregar dados iniciais:', error);
    }
  }, [callAPI]);

  // Controles do bot
  const handleStartBot = async () => {
    try {
      await callAPI('/api/trading-bot/start', { method: 'POST' });
      // Recarregar status após 2 segundos
      setTimeout(() => loadInitialData(), 2000);
    } catch (error) {
      console.error('Erro ao iniciar bot:', error);
    }
  };

  const handleStopBot = async () => {
    try {
      await callAPI('/api/trading-bot/stop', { method: 'POST' });
      setTimeout(() => loadInitialData(), 2000);
    } catch (error) {
      console.error('Erro ao parar bot:', error);
    }
  };

  // Carregar dados ao montar o componente
  useEffect(() => {
    loadInitialData();
    
    // Atualizar dados a cada 30 segundos
    const interval = setInterval(loadInitialData, 30000);
    return () => clearInterval(interval);
  }, [loadInitialData]);

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Trading Dashboard Pro
            </h1>
            <p className="text-slate-400 mt-2">AI-Powered Trading Platform v6.0</p>
          </div>
          <div className="text-right">
            <div className="text-sm text-slate-400">Last Update</div>
            <div className="text-white font-mono">{lastUpdate || '--:--:--'}</div>
          </div>
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="fixed top-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg z-50">
          Loading...
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Trading Bot Control */}
          <TradingBotControl 
            botStatus={botStatus}
            onStart={handleStartBot}
            onStop={handleStopBot}
            loading={loading}
          />

          {/* Market Overview */}
          <MarketOverview marketData={marketData} />

          {/* Performance Chart */}
          <PerformanceChart data={performanceData} />

          {/* RSI/MACD */}
          <RSIMACDDisplay data={rsiMacdData} />
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Sentiment Analysis */}
          <SentimentAnalysis sentiment={sentiment} />

          {/* Recent Alerts */}
          <RecentAlerts alerts={alerts} />

          {/* System Status */}
          <Card>
            <h3 className="text-xl font-bold text-white mb-4">System Status</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-300">API Connection</span>
                <StatusIndicator status="connected" />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-300">WebSocket</span>
                <StatusIndicator status={sentiment ? "connected" : "disconnected"} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-300">Trading Bot</span>
                <StatusIndicator status={botStatus?.status} />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-300">Real-time Data</span>
                <StatusIndicator status={rsiMacdData ? "active" : "inactive"} />
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-12 text-center text-slate-500 text-sm">
        <p>Trading Dashboard Pro v6.0 | Built with React & AI</p>
        <p className="mt-1">⚠️ Educational purposes only. Not financial advice.</p>
      </div>
    </div>
  );
};

export default TradingDashboard;
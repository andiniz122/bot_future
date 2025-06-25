import React, { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, DollarSign, AlertTriangle, 
  Bot, BarChart3, Calendar, Target, Zap, RefreshCw, Play, Pause,
  ArrowUp, ArrowDown, Clock, Shield, Gauge, Brain, Wifi, WifiOff,
  Settings, Info, CheckCircle, XCircle, AlertCircle
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// Componente de Card reutilizável
const Card = ({ children, className = "", ...props }) => (
  <div className={`bg-gray-800 border border-gray-700 rounded-lg shadow-lg ${className}`} {...props}>
    {children}
  </div>
);

// Componente de Badge para status
const Badge = ({ children, variant = "default", className = "" }) => {
  const variants = {
    default: "bg-gray-600 text-gray-100",
    success: "bg-green-600 text-green-100",
    danger: "bg-red-600 text-red-100", 
    warning: "bg-yellow-600 text-yellow-100",
    info: "bg-blue-600 text-blue-100"
  };
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]} ${className}`}>
      {children}
    </span>
  );
};

// Componente para exibir preços
const PriceDisplay = ({ price, change, changePercent, symbol }) => {
  const isPositive = change >= 0;
  
  return (
    <div className="text-right">
      <div className="text-lg font-bold text-white">
        ${typeof price === 'number' ? price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '0.00'}
      </div>
      <div className={`text-sm flex items-center justify-end ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
        {isPositive ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
        <span className="ml-1">
          {typeof changePercent === 'number' ? `${changePercent.toFixed(2)}%` : '0.00%'}
        </span>
      </div>
    </div>
  );
};

// Hook para WebSocket
const useWebSocket = (url, onMessage) => {
  const ws = useRef(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const connect = () => {
      try {
        ws.current = new WebSocket(url);
        
        ws.current.onopen = () => {
          setConnected(true);
          console.log('WebSocket conectado:', url);
        };
        
        ws.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            onMessage(data);
          } catch (e) {
            console.error('Erro ao parsear mensagem WebSocket:', e);
          }
        };
        
        ws.current.onclose = () => {
          setConnected(false);
          console.log('WebSocket desconectado');
          // Reconectar após 5 segundos
          setTimeout(connect, 5000);
        };
        
        ws.current.onerror = (error) => {
          console.error('Erro WebSocket:', error);
          setConnected(false);
        };
      } catch (error) {
        console.error('Erro ao conectar WebSocket:', error);
        setTimeout(connect, 5000);
      }
    };

    connect();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url, onMessage]);

  return connected;
};

// Hook para dados da API
const useApiData = (endpoint, interval = 30000) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error(`Erro ao buscar ${endpoint}:`, err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [endpoint, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Componente principal do Dashboard
const TradingDashboard = () => {
  const [currentData, setCurrentData] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [botStatus, setBotStatus] = useState(null);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [wsConnected, setWsConnected] = useState(false);

  // Dados da API
  const { data: apiCurrent, loading: loadingCurrent } = useApiData('/api/current', 10000);
  const { data: recommendations } = useApiData('/api/backtest-recommendations', 60000);
  const { data: alerts } = useApiData('/api/alerts', 30000);
  const { data: calendar } = useApiData('/api/calendar', 300000);
  const { data: patterns } = useApiData('/api/patterns', 30000);
  const { data: botStatusData } = useApiData('/api/trading-bot/status', 15000);
  const { data: positions } = useApiData('/api/trading-bot/positions', 15000);
  const { data: signals } = useApiData('/api/trading-bot/signals', 15000);

  // WebSocket para sentimento
  const sentimentConnected = useWebSocket(`${WS_BASE}/ws/sentiment`, (data) => {
    setSentimentData(data);
    setWsConnected(true);
  });

  useEffect(() => {
    if (apiCurrent) setCurrentData(apiCurrent);
    if (botStatusData) setBotStatus(botStatusData);
  }, [apiCurrent, botStatusData]);

  // Funções para controlar o bot
  const toggleBot = async (action) => {
    try {
      const response = await fetch(`${API_BASE}/api/trading-bot/${action}`, { method: 'POST' });
      if (response.ok) {
        // Recarrega status do bot
        window.location.reload();
      }
    } catch (error) {
      console.error(`Erro ao ${action} bot:`, error);
    }
  };

  // Tab do Overview
  const OverviewTab = () => (
    <div className="space-y-6">
      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Status da API</p>
              <p className="text-2xl font-bold text-green-400">Online</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">WebSocket</p>
              <p className={`text-2xl font-bold ${sentimentConnected ? 'text-green-400' : 'text-red-400'}`}>
                {sentimentConnected ? 'Conectado' : 'Offline'}
              </p>
            </div>
            {sentimentConnected ? <Wifi className="h-8 w-8 text-green-400" /> : <WifiOff className="h-8 w-8 text-red-400" />}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Bot Status</p>
              <p className={`text-2xl font-bold ${botStatus?.running ? 'text-green-400' : 'text-yellow-400'}`}>
                {botStatus?.running ? 'Ativo' : 'Parado'}
              </p>
            </div>
            <Bot className={`h-8 w-8 ${botStatus?.running ? 'text-green-400' : 'text-yellow-400'}`} />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Alertas Ativos</p>
              <p className="text-2xl font-bold text-blue-400">{alerts?.total_alerts || 0}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-blue-400" />
          </div>
        </Card>
      </div>

      {/* Preços em Tempo Real */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Activity className="mr-2" />
          Preços em Tempo Real
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {currentData?.assets && Object.entries(currentData.assets).map(([key, asset]) => (
            <Card key={key} className="p-4 bg-gray-900">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-white">{asset.name}</h4>
                  <p className="text-sm text-gray-400">{asset.symbol}</p>
                  <div className="mt-2">
                    <Badge variant={asset.angular?.trend === 'STRONG_UP' ? 'success' : asset.angular?.trend === 'STRONG_DOWN' ? 'danger' : 'default'}>
                      {asset.angular?.trend || 'NEUTRAL'}
                    </Badge>
                  </div>
                </div>
                <PriceDisplay 
                  price={asset.current_price}
                  change={asset.change}
                  changePercent={asset.change_percent}
                />
              </div>
            </Card>
          ))}
        </div>
      </Card>

      {/* Sentimento de Mercado */}
      {sentimentData && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Gauge className="mr-2" />
            Sentimento de Mercado (Gate.io)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-white mb-2">Bitcoin (BTC/USDT)</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Compradores:</span>
                  <span className="text-green-400 font-medium">{sentimentData.btc?.buyers.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Vendedores:</span>
                  <span className="text-red-400 font-medium">{sentimentData.btc?.sellers.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Tendência:</span>
                  <Badge variant={sentimentData.btc?.trend === 'BULLISH' ? 'success' : sentimentData.btc?.trend === 'BEARISH' ? 'danger' : 'default'}>
                    {sentimentData.btc?.trend}
                  </Badge>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-medium text-white mb-2">Medo & Ganância</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Índice:</span>
                  <span className="text-blue-400 font-medium">{sentimentData.fear_greed_index}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Humor:</span>
                  <Badge variant={sentimentData.market_mood === 'GREED' || sentimentData.market_mood === 'EXTREME_GREED' ? 'success' : sentimentData.market_mood === 'FEAR' || sentimentData.market_mood === 'EXTREME_FEAR' ? 'danger' : 'warning'}>
                    {sentimentData.market_mood}
                  </Badge>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );

  // Tab de Recomendações
  const RecommendationsTab = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Brain className="mr-2" />
          Recomendações de Backtest
        </h3>
        {recommendations?.recommendations?.length > 0 ? (
          <div className="space-y-4">
            {recommendations.recommendations.slice(0, 5).map((rec, index) => (
              <Card key={index} className="p-4 bg-gray-900">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">{rec.pattern_name}</h4>
                  <div className="flex items-center space-x-2">
                    <Badge variant={rec.trade_type === 'LONG' ? 'success' : 'danger'}>
                      {rec.trade_type}
                    </Badge>
                    <Badge variant="info">
                      {(rec.confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-gray-400 mb-2">{rec.description}</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Ativo:</span>
                    <span className="text-white ml-1">{rec.asset}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Entrada:</span>
                    <span className="text-white ml-1">${rec.entry_price?.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Retorno Esperado:</span>
                    <span className="text-green-400 ml-1">{rec.expected_return}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Timeframe:</span>
                    <span className="text-white ml-1">{rec.timeframe}</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <Brain className="mx-auto h-12 w-12 mb-4 opacity-50" />
            <p>Nenhuma recomendação disponível no momento</p>
          </div>
        )}
      </Card>

      {/* Sinais do Bot ao Vivo */}
      {signals?.signals?.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Zap className="mr-2" />
            Sinais ao Vivo do Bot
          </h3>
          <div className="space-y-4">
            {signals.signals.slice(0, 5).map((signal, index) => (
              <Card key={index} className="p-4 bg-gray-900">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">{signal.symbol}</h4>
                  <div className="flex items-center space-x-2">
                    <Badge variant={signal.signal_type === 'LONG' ? 'success' : 'danger'}>
                      {signal.signal_type}
                    </Badge>
                    <Badge variant="info">
                      {(signal.confidence * 100).toFixed(0)}%
                    </Badge>
                    <Badge variant={signal.status === 'ACTIVE' ? 'success' : 'warning'}>
                      {signal.status}
                    </Badge>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Estratégia:</span>
                    <span className="text-white ml-1">{signal.strategy_name}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Entrada:</span>
                    <span className="text-white ml-1">${signal.entry_price?.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">R/R:</span>
                    <span className="text-green-400 ml-1">{signal.risk_reward_ratio}:1</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Expira em:</span>
                    <span className="text-yellow-400 ml-1">{signal.expires_in_minutes?.toFixed(0)}min</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>
      )}
    </div>
  );

  // Tab do Bot
  const BotTab = () => (
    <div className="space-y-6">
      {/* Controles do Bot */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <Bot className="mr-2" />
            Controle do Trading Bot
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => toggleBot('start')}
              disabled={botStatus?.running}
              className={`flex items-center px-4 py-2 rounded-lg font-medium text-sm ${
                botStatus?.running 
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              <Play className="mr-2 h-4 w-4" />
              Iniciar
            </button>
            <button
              onClick={() => toggleBot('stop')}
              disabled={!botStatus?.running}
              className={`flex items-center px-4 py-2 rounded-lg font-medium text-sm ${
                !botStatus?.running 
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                  : 'bg-red-600 hover:bg-red-700 text-white'
              }`}
            >
              <Pause className="mr-2 h-4 w-4" />
              Parar
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-400">Status</p>
            <p className={`text-lg font-bold ${botStatus?.running ? 'text-green-400' : 'text-yellow-400'}`}>
              {botStatus?.running ? 'Ativo' : 'Parado'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Posições Ativas</p>
            <p className="text-lg font-bold text-blue-400">{positions?.total_positions || 0}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Sinais Ativos</p>
            <p className="text-lg font-bold text-purple-400">{signals?.active_signals || 0}</p>
          </div>
        </div>
      </Card>

      {/* Posições Ativas */}
      {positions?.positions?.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Target className="mr-2" />
            Posições Ativas
          </h3>
          <div className="space-y-4">
            {positions.positions.map((position, index) => (
              <Card key={index} className="p-4 bg-gray-900">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">{position.symbol}</h4>
                  <div className="flex items-center space-x-2">
                    <Badge variant={position.side === 'long' ? 'success' : 'danger'}>
                      {position.side.toUpperCase()}
                    </Badge>
                    <Badge variant={position.pnl >= 0 ? 'success' : 'danger'}>
                      {position.pnl >= 0 ? '+' : ''}{position.pnl?.toFixed(2)} USDT
                    </Badge>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Tamanho:</span>
                    <span className="text-white ml-1">{position.size}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Entrada:</span>
                    <span className="text-white ml-1">${position.entry_price?.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Atual:</span>
                    <span className="text-white ml-1">${position.current_price?.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">P&L %:</span>
                    <span className={position.pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {position.pnl_percent >= 0 ? '+' : ''}{position.pnl_percent?.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>
      )}
    </div>
  );

  // Tab de Alertas
  const AlertsTab = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <AlertTriangle className="mr-2" />
          Sistema de Alertas
        </h3>
        {alerts?.alerts?.length > 0 ? (
          <div className="space-y-3">
            {alerts.alerts.slice(0, 10).map((alert, index) => (
              <Card key={index} className="p-4 bg-gray-900">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center mb-1">
                      <h4 className="font-medium text-white mr-2">{alert.title}</h4>
                      <Badge variant={alert.severity === 'HIGH' ? 'danger' : alert.severity === 'MEDIUM' ? 'warning' : 'info'}>
                        {alert.severity}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-400">{alert.message}</p>
                  </div>
                  <span className="text-xs text-gray-500">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <AlertTriangle className="mx-auto h-12 w-12 mb-4 opacity-50" />
            <p>Nenhum alerta ativo no momento</p>
          </div>
        )}
      </Card>
    </div>
  );

  if (loadingCurrent) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-400 mx-auto mb-4" />
          <p className="text-white">Carregando dados do mercado...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center">
            <BarChart3 className="h-8 w-8 text-blue-400 mr-3" />
            <h1 className="text-2xl font-bold">Trading Dashboard</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${sentimentConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm text-gray-400">
                {sentimentConnected ? 'WebSocket Conectado' : 'WebSocket Offline'}
              </span>
            </div>
            <Badge variant="info">v6.0</Badge>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8">
            {[
              { id: 'overview', label: 'Visão Geral', icon: Activity },
              { id: 'recommendations', label: 'Recomendações', icon: Brain },
              { id: 'bot', label: 'Trading Bot', icon: Bot },
              { id: 'alerts', label: 'Alertas', icon: AlertTriangle }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id)}
                className={`flex items-center px-3 py-4 text-sm font-medium border-b-2 ${
                  selectedTab === tab.id
                    ? 'border-blue-400 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-white'
                }`}
              >
                <tab.icon className="mr-2 h-4 w-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-6">
        {selectedTab === 'overview' && <OverviewTab />}
        {selectedTab === 'recommendations' && <RecommendationsTab />}
        {selectedTab === 'bot' && <BotTab />}
        {selectedTab === 'alerts' && <AlertsTab />}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="max-w-7xl mx-auto text-center text-sm text-gray-400">
          Trading Dashboard v6.0 - Sistema de Trading Automatizado | 
          <span className="ml-2">
            Última atualização: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </footer>
    </div>
  );
};

export default TradingDashboard;
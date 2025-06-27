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
    is_running: true,
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

  // Dados simulados para o gráfico
  const generateChartData = () => {
    const data = [];
    const basePrice = 67000;
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
    fallback: false
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
      isPositive: pnl >= 0
    };
  };

  // Função para mapear talib_entrada_score para recomendação e cor
  const getRecommendationFromTalibScore = (score) => {
    if (score === 1.0) {
      return { 
        text: 'COMPRA FORTE', 
        subtext: 'LONG POSITION',
        color: 'text-green-400', 
        bgColor: 'bg-green-900',
        borderColor: 'border-green-400',
        icon: ArrowUpRight 
      };
    } else if (score === 0.5) {
      return { 
        text: 'AGUARDAR', 
        subtext: 'HOLD POSITION',
        color: 'text-yellow-400', 
        bgColor: 'bg-yellow-900',
        borderColor: 'border-yellow-400',
        icon: Zap 
      };
    } else if (score === -1.0) {
      return { 
        text: 'VENDA FORTE', 
        subtext: 'SHORT POSITION',
        color: 'text-red-400', 
        bgColor: 'bg-red-900',
        borderColor: 'border-red-400',
        icon: ArrowDownLeft 
      };
    }
    return { 
      text: 'SEM SINAL', 
      subtext: 'NEUTRAL',
      color: 'text-gray-400', 
      bgColor: 'bg-gray-800',
      borderColor: 'border-gray-600',
      icon: Target 
    };
  };

  const getAngleColor = (angle) => {
    if (angle > 15) return 'text-green-400';
    if (angle > 5) return 'text-green-300';
    if (angle > -5) return 'text-yellow-400';
    if (angle > -15) return 'text-orange-400';
    return 'text-red-400';
  };

  const getAngleIcon = (angle) => {
    if (Math.abs(angle) < 5) return '→';
    return angle > 0 ? '↗' : '↘';
  };

  const toggleBot = () => {
    if (tradingEnvironment === 'live' && !botStatus.is_running) {
      setShowEnvironmentWarning(true);
    } else {
      setBotStatus(prev => ({ ...prev, is_running: !prev.is_running }));
    }
  };

  const confirmLiveEnvironment = () => {
    setBotStatus(prev => ({ ...prev, is_running: true }));
    setShowEnvironmentWarning(false);
  };

  const cancelLiveEnvironment = () => {
    setShowEnvironmentWarning(false);
  };

  const hasValidChartData = chartData.combined && chartData.combined.length > 0;
  const hasValidMarketData = marketData.assets && Object.keys(marketData.assets).length > 0;

  // Obter a recomendação formatada
  const recommendation = getRecommendationFromTalibScore(realtimeIndicators.btc.talib_entrada_score);
  const RecommendationIcon = recommendation.icon;

  return (
    <div className="min-h-screen bg-black text-white font-mono">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <h1 className="text-2xl font-bold text-blue-400">TRADING PRO AI</h1>
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
            {/* DESTAQUE: Recomendação TALIB */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Target className="w-4 h-4 mr-2" />
                RECOMENDAÇÃO AI
              </h3>
              <div className={`${recommendation.bgColor} ${recommendation.borderColor} border-2 p-4 rounded-lg`}>
                <div className="flex items-center justify-between mb-2">
                  <RecommendationIcon className={`w-8 h-8 ${recommendation.color}`} />
                  <div className="text-right">
                    <div className={`text-lg font-bold ${recommendation.color}`}>
                      {recommendation.text}
                    </div>
                    <div className="text-xs text-gray-400">
                      {recommendation.subtext}
                    </div>
                  </div>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-400">Score TALIB:</span>
                  <span className={`font-bold ${recommendation.color}`}>
                    {realtimeIndicators.btc.talib_entrada_score?.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            {/* DESTAQUE: Ângulos dos Indicadores */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <RotateCw className="w-4 h-4 mr-2" />
                ÂNGULOS DE MOMENTUM
              </h3>
              <div className="space-y-3">
                {/* RSI Angle */}
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-gray-400">RSI ÂNGULO</div>
                      <div className="text-sm text-gray-300">Velocidade: {Math.abs(realtimeIndicators.btc.rsi_angle).toFixed(1)}°/min</div>
                    </div>
                    <div className="text-right">
                      <div className={`text-2xl font-bold ${getAngleColor(realtimeIndicators.btc.rsi_angle)}`}>
                        {getAngleIcon(realtimeIndicators.btc.rsi_angle)} {realtimeIndicators.btc.rsi_angle?.toFixed(1)}°
                      </div>
                      <div className="text-xs text-gray-400">
                        {realtimeIndicators.btc.rsi_angle > 10 ? 'ACELERANDO ALTA' :
                         realtimeIndicators.btc.rsi_angle < -10 ? 'ACELERANDO BAIXA' : 'ESTÁVEL'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* MACD Angles */}
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-xs text-gray-400">MACD ÂNGULO</div>
                        <div className="text-sm text-gray-300">Linha Principal</div>
                      </div>
                      <div className="text-right">
                        <div className={`text-xl font-bold ${getAngleColor(realtimeIndicators.btc.macd_angle)}`}>
                          {getAngleIcon(realtimeIndicators.btc.macd_angle)} {realtimeIndicators.btc.macd_angle?.toFixed(1)}°
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-xs text-gray-400">SIGNAL ÂNGULO</div>
                        <div className="text-sm text-gray-300">Linha de Sinal</div>
                      </div>
                      <div className="text-right">
                        <div className={`text-xl font-bold ${getAngleColor(realtimeIndicators.btc.signal_angle)}`}>
                          {getAngleIcon(realtimeIndicators.btc.signal_angle)} {realtimeIndicators.btc.signal_angle?.toFixed(1)}°
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Bot Controls */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                CONTROLE DO BOT
              </h3>
              <div className="space-y-3">
                <div className="space-y-2">
                  <span className="text-sm text-gray-400">Ambiente de Trading:</span>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setTradingEnvironment('testnet')}
                      className={`flex-1 px-3 py-2 rounded text-xs font-bold transition-colors ${
                        tradingEnvironment === 'testnet'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      🧪 TESTNET
                    </button>
                    <button
                      onClick={() => setTradingEnvironment('live')}
                      className={`flex-1 px-3 py-2 rounded text-xs font-bold transition-colors ${
                        tradingEnvironment === 'live'
                          ? 'bg-red-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      💰 LIVE
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Status:</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${botStatus.is_running ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className="text-sm">{botStatus.is_running ? 'ATIVO' : 'PARADO'}</span>
                  </div>
                </div>
                <button 
                  className={`w-full px-3 py-2 rounded text-xs font-bold flex items-center justify-center space-x-1 ${
                    botStatus.is_running ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                  }`}
                  onClick={toggleBot}
                >
                  {botStatus.is_running ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                  <span>{botStatus.is_running ? 'PARAR' : 'INICIAR'}</span>
                </button>
              </div>
            </div>

            {/* Performance Geral */}
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

            {/* Indicadores Técnicos Básicos */}
            <div className="mb-6">
              <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                <Activity className="w-4 h-4 mr-2" />
                INDICADORES TÉCNICOS
              </h3>
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
                    <div className="text-gray-400">MACD</div>
                    <div className={`font-bold ${
                      realtimeIndicators.btc.macd > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {realtimeIndicators.btc.macd?.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Signal Base</div>
                    <div className={`font-bold ${
                      realtimeIndicators.btc.signal === 'BUY' ? 'text-green-400' :
                      realtimeIndicators.btc.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {realtimeIndicators.btc.signal}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Histograma</div>
                    <div className={`font-bold ${
                      realtimeIndicators.btc.macd_histogram > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {realtimeIndicators.btc.macd_histogram?.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Posições Ativas */}
            {botStatus.active_positions && botStatus.active_positions.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-bold text-gray-400 mb-3 flex items-center">
                  <DollarSign className="w-4 h-4 mr-2" />
                  POSIÇÕES ATIVAS
                </h3>
                <div className="space-y-2">
                  {botStatus.active_positions.map((position, index) => (
                    <div key={index} className="bg-gray-800 p-3 rounded text-xs">
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-blue-400">{position.symbol}</span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          position.side === 'long' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
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
                            domain={[0, 100]} 
                            tick={{ fontSize: 9 }}
                            width={30}
                          />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value) => [value?.toFixed(1), 'RSI']}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '4px',
                              fontSize: '11px'
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="rsi"
                            stroke="#F59E0B"
                            strokeWidth={1.5}
                            dot={false}
                          />
                          <ReferenceLine y={70} stroke="#EF4444" strokeWidth={1} strokeDasharray="2 2" />
                          <ReferenceLine y={30} stroke="#10B981" strokeWidth={1} strokeDasharray="2 2" />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500 text-xs">
                        Sem dados de RSI
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
                            stroke="#3B82F6"
                            strokeWidth={1.5}
                            dot={false}
                            name="MACD"
                          />
                          <Line
                            type="monotone"
                            dataKey="macd_signal"
                            stroke="#F59E0B"
                            strokeWidth={1.5}
                            dot={false}
                            name="Signal"
                          />
                          <Bar
                            dataKey="macd_hist"
                            fill="#8B5CF6"
                            opacity={0.6}
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
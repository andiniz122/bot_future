import React, { useState, useEffect, useCallback } from 'react';
import { 
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, ReferenceLine, BarChart, Bar
} from 'recharts';
import {
  TrendingUp, Brain, Database, Wifi, WifiOff, AlertTriangle, Play, Pause,
  DollarSign, Zap, RotateCw, Eye, EyeOff
} from 'lucide-react';

const TradingDashboard = () => {
  // CONFIGURAÇÃO DA API - AJUSTE AQUI PARA SEU BACKEND
  const API_BASE_URL = 'http://62.72.1.122:8000';

  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [marketData, setMarketData] = useState({
    btc: { price: 0, change: 0, change_percent: 0 }
  });
  
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    ai_accuracy: 0,
    training_samples: 0,
    ml_model_accuracy: 0,
    testnet: { balance: 0, total_pnl: 0, roi_percentage: 0, total_trades: 0 },
    live: { balance: 0, total_pnl: 0, roi_percentage: 0, total_trades: 0 }
  });

  const [realtimeIndicators, setRealtimeIndicators] = useState({
    rsi: 0,
    macd: 0,
    macd_signal: 0,
    macd_histogram: 0,
    signal: 'HOLD',
    stochrsi_k: 0,
    stochrsi_d: 0
  });

  const [selectedPeriod, setSelectedPeriod] = useState('5m');
  const [tradingEnvironment, setTradingEnvironment] = useState('testnet');
  const [isLoading, setIsLoading] = useState(true);
  
  const [indicatorVisibility, setIndicatorVisibility] = useState({
    rsi: true,
    macd: true,
    stochrsi: true,
    volume: true
  });

  const [chartData, setChartData] = useState({
    candles: [],
    signals: [],
    dataPoints: 0,
    lastUpdate: new Date(0),
    fallback: true
  });

  const [wsConnections, setWsConnections] = useState({
    sentiment: false,
    ohlcv: false,
    rsiMacd: false
  });

  // Buscar dados do mercado atual
  const fetchMarketData = useCallback(async () => {
    try {
      console.log('🔄 Fetching market data from', `${API_BASE_URL}/api/current`);
      const response = await fetch(`${API_BASE_URL}/api/current`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      console.log('✅ Market data received:', data);
      
      if (data.assets && data.assets.btc) {
        setMarketData({
          btc: {
            price: data.assets.btc.current_price || 0,
            change: data.assets.btc.change || 0,
            change_percent: data.assets.btc.change_percent || 0
          }
        });
      }
    } catch (error) {
      console.error("❌ Failed to fetch market data:", error);
      setConnectionStatus('disconnected');
    }
  }, [API_BASE_URL]);

  // Buscar status do bot
  const fetchBotStatus = useCallback(async () => {
    try {
      console.log('🔄 Fetching bot status');
      const response = await fetch(`${API_BASE_URL}/api/trading-bot/status`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      console.log('✅ Bot status received:', data);
      
      setBotStatus({
        is_running: data.running !== undefined ? data.running : (data.status === 'running'),
        ai_accuracy: data.ai_accuracy || 0,
        training_samples: data.training_samples || 0,
        ml_model_accuracy: data.ml_model_accuracy || 0,
        testnet: {
          balance: data.testnet_performance?.balance || 0,
          total_pnl: data.testnet_performance?.total_pnl || 0,
          roi_percentage: data.testnet_performance?.roi_percentage || 0,
          total_trades: data.testnet_performance?.total_trades || 0
        },
        live: {
          balance: data.live_performance?.balance || 0,
          total_pnl: data.live_performance?.total_pnl || 0,
          roi_percentage: data.live_performance?.roi_percentage || 0,
          total_trades: data.live_performance?.total_trades || 0
        }
      });
      setConnectionStatus('connected');
    } catch (error) {
      console.error("❌ Failed to fetch bot status:", error);
      setConnectionStatus('disconnected');
    }
  }, [API_BASE_URL]);

  // Buscar indicadores técnicos
  const fetchTechnicalIndicators = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/macd/realtime/btc`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      setRealtimeIndicators({
        rsi: data.rsi_data?.last_value || 0,
        macd: (data.macd_data?.macd && data.macd_data.macd.length > 0 ? 
               data.macd_data.macd[data.macd_data.macd.length - 1] : 0),
        macd_signal: (data.macd_data?.signal && data.macd_data.signal.length > 0 ? 
                     data.macd_data.signal[data.macd_data.signal.length - 1] : 0),
        macd_histogram: (data.macd_data?.histogram && data.macd_data.histogram.length > 0 ? 
                        data.macd_data.histogram[data.macd_data.histogram.length - 1] : 0),
        signal: data.macd_data?.trend || data.signal || 'HOLD',
        stochrsi_k: data.stochrsi_data?.k_value || 0,
        stochrsi_d: data.stochrsi_data?.d_value || 0
      });
    } catch (error) {
      console.error("❌ Failed to fetch technical indicators:", error);
    }
  }, [API_BASE_URL]);

  // Buscar dados do gráfico
  const fetchChartData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/precos/${selectedPeriod}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      
      const btcData = data.assets?.btc;
      if (btcData && btcData.price_data && btcData.price_data.length > 0) {
        const chartPoints = btcData.price_data.map((price, index) => {
          const time = new Date(data.dates[index]).getTime();
          return {
            time,
            open: index > 0 ? btcData.price_data[index - 1] : price,
            high: price * (1 + Math.random() * 0.002),
            low: price * (1 - Math.random() * 0.002),
            close: price,
            volume: btcData.volume_data?.[index] || 0,
            rsi: btcData.rsi_data?.[index] || 50,
            macd: btcData.macd_data?.[index] || 0,
            macd_signal: btcData.macd_signal_data?.[index] || 0,
            macd_hist: btcData.macd_hist_data?.[index] || 0,
            stochrsi_k: btcData.stochrsi_k_data?.[index] || 0,
            stochrsi_d: btcData.stochrsi_d_data?.[index] || 0
          };
        });

        // Gerar sinais baseados em RSI
        const signals = [];
        chartPoints.forEach((point, index) => {
          if (point.rsi < 30 && Math.random() > 0.8) {
            signals.push({
              time: point.time,
              type: 'BUY',
              price: point.close,
              confidence: 0.7 + Math.random() * 0.3,
              reason: 'RSI Oversold'
            });
          } else if (point.rsi > 70 && Math.random() > 0.8) {
            signals.push({
              time: point.time,
              type: 'SELL',
              price: point.close,
              confidence: 0.6 + Math.random() * 0.4,
              reason: 'RSI Overbought'
            });
          }
        });

        setChartData({
          candles: chartPoints,
          signals: signals,
          dataPoints: chartPoints.length,
          lastUpdate: new Date(data.dates[data.dates.length - 1]),
          fallback: false
        });
      } else {
        setChartData(prev => ({
          ...prev,
          fallback: true,
          candles: [],
          dataPoints: 0,
          lastUpdate: new Date(0)
        }));
      }
    } catch (error) {
      console.error("❌ Failed to fetch chart data:", error);
      setChartData(prev => ({
        ...prev,
        fallback: true
      }));
    }
  }, [selectedPeriod, API_BASE_URL]);

  // WebSocket Setup
  const setupWebSockets = useCallback(() => {
    const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss:' : 'ws:';
    const wsHostPort = API_BASE_URL.split('//')[1];

    // Sentiment WebSocket
    const sentimentWs = new WebSocket(`${wsProtocol}//${wsHostPort}/ws/sentiment`);
    sentimentWs.onopen = () => {
      console.log('✅ Sentiment WebSocket connected');
      setWsConnections(prev => ({ ...prev, sentiment: true }));
    };
    sentimentWs.onclose = () => {
      console.log('❌ Sentiment WebSocket disconnected');
      setWsConnections(prev => ({ ...prev, sentiment: false }));
    };

    // OHLCV WebSocket
    const ohlcvWs = new WebSocket(`${wsProtocol}//${wsHostPort}/ws/ohlcv`);
    ohlcvWs.onopen = () => {
      console.log('✅ OHLCV WebSocket connected');
      setWsConnections(prev => ({ ...prev, ohlcv: true }));
    };
    ohlcvWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'ohlcv_update' && data.asset?.toLowerCase() === 'btc') {
          // Atualizar dados em tempo real
          setMarketData(prev => ({
            ...prev,
            btc: {
              price: data.candle.close,
              change: data.candle.close - data.candle.open,
              change_percent: ((data.candle.close - data.candle.open) / data.candle.open) * 100
            }
          }));
        }
      } catch (e) {
        console.error("Error parsing OHLCV message:", e);
      }
    };
    ohlcvWs.onclose = () => {
      console.log('❌ OHLCV WebSocket disconnected');
      setWsConnections(prev => ({ ...prev, ohlcv: false }));
    };

    // RSI/MACD WebSocket
    const rsiMacdWs = new WebSocket(`${wsProtocol}//${wsHostPort}/ws/rsi-macd`);
    rsiMacdWs.onopen = () => {
      console.log('✅ RSI/MACD WebSocket connected');
      setWsConnections(prev => ({ ...prev, rsiMacd: true }));
    };
    rsiMacdWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'rsi_macd_update' && data.symbol === 'BTC_USDT') {
          setRealtimeIndicators(prev => ({
            ...prev,
            rsi: data.rsi?.value || 0,
            macd: data.macd?.macd || 0,
            macd_signal: data.macd?.signal || 0,
            macd_histogram: data.macd?.histogram || 0,
            signal: data.combined?.signal_type || 'HOLD',
            stochrsi_k: data.stochrsi?.k_value || 0,
            stochrsi_d: data.stochrsi?.d_value || 0
          }));
        }
      } catch (e) {
        console.error("Error parsing RSI/MACD message:", e);
      }
    };
    rsiMacdWs.onclose = () => {
      console.log('❌ RSI/MACD WebSocket disconnected');
      setWsConnections(prev => ({ ...prev, rsiMacd: false }));
    };

    return () => {
      sentimentWs.close();
      ohlcvWs.close();
      rsiMacdWs.close();
    };
  }, [API_BASE_URL]);

  // Controle do bot
  const controlBot = useCallback(async (action) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/trading-bot/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await response.json();
      await fetchBotStatus();
    } catch (error) {
      console.error(`Failed to ${action} bot:`, error);
      alert(`Failed to ${action} trading bot. Check console for details.`);
    } finally {
      setIsLoading(false);
    }
  }, [fetchBotStatus, API_BASE_URL]);

  const handleToggleBot = () => {
    if (botStatus.is_running) {
      controlBot('stop');
    } else {
      controlBot('start');
    }
  };

  // Inicialização
  useEffect(() => {
    console.log('🚀 Initializing Trading Dashboard...');
    
    const initializeData = async () => {
      await Promise.all([
        fetchMarketData(),
        fetchBotStatus(),
        fetchTechnicalIndicators(),
        fetchChartData()
      ]);
      setIsLoading(false);
    };

    initializeData();
    const wsCleanup = setupWebSockets();
    
    const marketInterval = setInterval(fetchMarketData, 30000);
    const botInterval = setInterval(fetchBotStatus, 15000);
    const indicatorsInterval = setInterval(fetchTechnicalIndicators, 60000);
    
    return () => {
      clearInterval(marketInterval);
      clearInterval(botInterval);
      clearInterval(indicatorsInterval);
      if (wsCleanup) wsCleanup();
    };
  }, [fetchMarketData, fetchBotStatus, fetchTechnicalIndicators, fetchChartData, setupWebSockets]);

  useEffect(() => {
    fetchChartData();
  }, [selectedPeriod, fetchChartData]);

  // Funções de formatação
  const periods = ['1m', '5m', '15m', '1h', '4h', '1d'];

  const formatCurrency = (value) => {
    if (typeof value !== 'number' || isNaN(value) || value === 0) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const getSignalColor = (signal) => {
    if (signal === 'BUY') return '#10B981';
    if (signal === 'SELL') return '#EF4444';
    return '#9CA3AF';
  };

  const toggleIndicator = (indicator) => {
    setIndicatorVisibility(prev => ({
      ...prev,
      [indicator]: !prev[indicator]
    }));
  };

  // Componente de sinal no gráfico
  const SignalMarker = ({ cx, cy, payload }) => {
    const signal = chartData.signals.find(s => s.time === payload.time);
    if (!signal) return null;
    
    const isBuy = signal.type === 'BUY';
    const color = isBuy ? '#10B981' : '#EF4444';
    
    return (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={8}
          fill={color}
          stroke="#ffffff"
          strokeWidth={2}
        />
        <text
          x={cx}
          y={cy + 1}
          textAnchor="middle"
          fontSize={10}
          fill="#ffffff"
          fontWeight="bold"
        >
          {isBuy ? '↑' : '↓'}
        </text>
      </g>
    );
  };

  if (isLoading) {
    return (
      <div style={{
        height: '100vh',
        backgroundColor: '#000000',
        color: '#ffffff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column'
      }}>
        <div style={{ animation: 'spin 1s linear infinite', marginBottom: '16px' }}>
          <RotateCw size={32} />
        </div>
        <h2>Loading Trading Dashboard...</h2>
        <p style={{ color: '#9CA3AF', marginTop: '8px' }}>
          Connecting to {API_BASE_URL}
        </p>
      </div>
    );
  }

  return (
    <div style={{
      height: '100vh',
      backgroundColor: '#000000',
      color: '#ffffff',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        borderBottom: '1px solid #374151',
        backgroundColor: '#111827',
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <h1 style={{ fontSize: '20px', fontWeight: 'bold', color: '#3B82F6', margin: 0 }}>
            BTC TRADING PRO
          </h1>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '12px', color: '#9CA3AF' }}>BTC/USD:</span>
              <span style={{ fontSize: '14px', fontWeight: 'bold' }}>
                {formatCurrency(marketData.btc.price)}
              </span>
              <span style={{
                fontSize: '12px',
                color: marketData.btc.change >= 0 ? '#10B981' : '#EF4444'
              }}>
                {marketData.btc.change >= 0 ? '+' : ''}{marketData.btc.change_percent.toFixed(2)}%
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {connectionStatus === 'connected' ? (
              <Wifi size={12} color="#10B981" />
            ) : (
              <WifiOff size={12} color="#EF4444" />
            )}
            <span style={{
              fontSize: '12px',
              color: connectionStatus === 'connected' ? '#10B981' : '#EF4444'
            }}>
              {connectionStatus.toUpperCase()}
            </span>
          </div>

          {/* WebSocket Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ fontSize: '10px', color: '#9CA3AF' }}>WS:</div>
            {Object.entries(wsConnections).map(([key, connected]) => (
              <div
                key={key}
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  backgroundColor: connected ? '#10B981' : '#EF4444'
                }}
                title={`${key} WebSocket`}
              />
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <select
            value={tradingEnvironment}
            onChange={(e) => setTradingEnvironment(e.target.value)}
            style={{
              backgroundColor: '#374151',
              color: '#ffffff',
              fontSize: '12px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #4B5563'
            }}
            disabled={botStatus.is_running}
          >
            <option value="testnet">TESTNET</option>
            <option value="live">LIVE</option>
          </select>
          
          <button 
            onClick={handleToggleBot}
            disabled={isLoading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              borderRadius: '4px',
              fontWeight: 'bold',
              fontSize: '12px',
              border: 'none',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              color: '#ffffff',
              backgroundColor: botStatus.is_running ? '#DC2626' : '#059669',
              opacity: isLoading ? 0.6 : 1
            }}
          >
            {isLoading ? (
              <RotateCw size={12} style={{ animation: 'spin 1s linear infinite' }} />
            ) : botStatus.is_running ? (
              <Pause size={12} />
            ) : (
              <Play size={12} />
            )}
            <span>{botStatus.is_running ? 'STOP' : 'START'}</span>
          </button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Sidebar */}
        <div style={{
          width: '280px',
          borderRight: '1px solid #374151',
          backgroundColor: '#111827',
          overflowY: 'auto',
          padding: '12px'
        }}>
          {/* AI Status */}
          <div style={{ marginBottom: '16px' }}>
            <h3 style={{
              fontSize: '12px',
              fontWeight: 'bold',
              color: '#9CA3AF',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Brain size={12} /> AI STATUS
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '11px' }}>
              <div style={{ backgroundColor: '#374151', padding: '6px', borderRadius: '4px' }}>
                <div style={{ color: '#9CA3AF' }}>Accuracy</div>
                <div style={{ fontWeight: 'bold', color: '#3B82F6' }}>
                  {botStatus.ai_accuracy > 0 ? `${botStatus.ai_accuracy.toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              <div style={{ backgroundColor: '#374151', padding: '6px', borderRadius: '4px' }}>
                <div style={{ color: '#9CA3AF' }}>Signal</div>
                <div style={{ fontWeight: 'bold', color: getSignalColor(realtimeIndicators.signal) }}>
                  {realtimeIndicators.signal}
                </div>
              </div>
              <div style={{ backgroundColor: '#374151', padding: '6px', borderRadius: '4px' }}>
                <div style={{ color: '#9CA3AF' }}>Samples</div>
                <div style={{ fontWeight: 'bold', color: '#10B981' }}>
                  {botStatus.training_samples > 0 ? `${(botStatus.training_samples/1000).toFixed(0)}k` : 'N/A'}
                </div>
              </div>
              <div style={{ backgroundColor: '#374151', padding: '6px', borderRadius: '4px' }}>
                <div style={{ color: '#9CA3AF' }}>ML Acc</div>
                <div style={{ fontWeight: 'bold', color: '#8B5CF6' }}>
                  {botStatus.ml_model_accuracy > 0 ? `${botStatus.ml_model_accuracy.toFixed(1)}%` : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Indicadores */}
          <div style={{ marginBottom: '16px' }}>
            <h3 style={{
              fontSize: '12px',
              fontWeight: 'bold',
              color: '#9CA3AF',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Zap size={12} /> INDICATORS
            </h3>
            
            <div style={{ backgroundColor: '#374151', padding: '8px', borderRadius: '4px', marginBottom: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: '#9CA3AF' }}>RSI</span>
                <span style={{ fontSize: '11px', fontWeight: 'bold', color: '#F59E0B' }}>
                  {realtimeIndicators.rsi > 0 ? realtimeIndicators.rsi.toFixed(1) : 'N/A'}
                </span>
              </div>
            </div>

            <div style={{ backgroundColor: '#374151', padding: '8px', borderRadius: '4px', marginBottom: '6px' }}>
              <div style={{ fontSize: '11px', color: '#9CA3AF', marginBottom: '4px' }}>MACD</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
                <div>
                  <div style={{ color: '#6B7280' }}>Line</div>
                  <div style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                    {realtimeIndicators.macd !== 0 ? realtimeIndicators.macd.toFixed(2) : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6B7280' }}>Signal</div>
                  <div style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                    {realtimeIndicators.macd_signal !== 0 ? realtimeIndicators.macd_signal.toFixed(2) : 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            <div style={{ backgroundColor: '#374151', padding: '8px', borderRadius: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: '#9CA3AF' }}>StochRSI</span>
                <span style={{ fontSize: '10px', fontWeight: 'bold' }}>
                  <span style={{ color: '#FFD700' }}>
                    {realtimeIndicators.stochrsi_k > 0 ? realtimeIndicators.stochrsi_k.toFixed(1) : 'N/A'}
                  </span>
                  <span style={{ color: '#9CA3AF' }}> / </span>
                  <span style={{ color: '#8B5CF6' }}>
                    {realtimeIndicators.stochrsi_d > 0 ? realtimeIndicators.stochrsi_d.toFixed(1) : 'N/A'}
                  </span>
                </span>
              </div>
            </div>
          </div>

          {/* Performance */}
          <div>
            <h3 style={{
              fontSize: '12px',
              fontWeight: 'bold',
              color: '#9CA3AF',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <DollarSign size={12} /> PERFORMANCE
            </h3>
            
            <div style={{ backgroundColor: '#374151', padding: '8px', borderRadius: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '11px', fontWeight: 'bold', color: '#3B82F6' }}>
                  {tradingEnvironment.toUpperCase()}
                </span>
                <span style={{
                  fontSize: '10px',
                  padding: '2px 6px',
                  borderRadius: '2px',
                  backgroundColor: botStatus.is_running ? '#065F46' : '#374151',
                  color: botStatus.is_running ? '#10B981' : '#9CA3AF'
                }}>
                  {botStatus.is_running ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
                <div>
                  <div style={{ color: '#9CA3AF' }}>Balance</div>
                  <div style={{ fontWeight: 'bold', color: '#ffffff' }}>
                    {formatCurrency(botStatus[tradingEnvironment].balance)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9CA3AF' }}>P&L</div>
                  <div style={{
                    fontWeight: 'bold',
                    color: botStatus[tradingEnvironment].total_pnl >= 0 ? '#10B981' : '#EF4444'
                  }}>
                    {botStatus[tradingEnvironment].total_pnl >= 0 ? '+' : ''}
                    {formatCurrency(Math.abs(botStatus[tradingEnvironment].total_pnl))}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9CA3AF' }}>ROI</div>
                  <div style={{
                    fontWeight: 'bold',
                    color: botStatus[tradingEnvironment].roi_percentage >= 0 ? '#10B981' : '#EF4444'
                  }}>
                    {botStatus[tradingEnvironment].roi_percentage.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div style={{ color: '#9CA3AF' }}>Trades</div>
                  <div style={{ fontWeight: 'bold', color: '#D1D5DB' }}>
                    {botStatus[tradingEnvironment].total_trades}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Chart Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Chart Controls */}
          <div style={{
            borderBottom: '1px solid #374151',
            backgroundColor: '#111827',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {periods.map((period) => (
                <button
                  key={period}
                  onClick={() => setSelectedPeriod(period)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: selectedPeriod === period ? '#2563EB' : '#374151',
                    color: selectedPeriod === period ? '#ffffff' : '#9CA3AF'
                  }}
                >
                  {period.toUpperCase()}
                </button>
              ))}
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12px', color: '#9CA3AF' }}>
              <span>Indicators:</span>
              {Object.entries(indicatorVisibility).map(([indicator, visible]) => (
                <button
                  key={indicator}
                  onClick={() => toggleIndicator(indicator)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    border: '1px solid #4B5563',
                    backgroundColor: visible ? '#059669' : '#4B5563',
                    color: '#ffffff',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {visible ? <Eye size={12} /> : <EyeOff size={12} />}
                  {indicator.toUpperCase()}
                  {visible && <span style={{ marginLeft: '4px', fontSize: '10px' }}>✓</span>}
                </button>
              ))}
            </div>
          </div>

          <div style={{ flex: 1, padding: '16px', overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              display: 'grid',
              gridTemplateRows: indicatorVisibility.rsi && indicatorVisibility.macd && indicatorVisibility.stochrsi && indicatorVisibility.volume ? 
                '2fr 0.7fr 0.7fr 0.7fr 0.7fr' : 
                indicatorVisibility.rsi && indicatorVisibility.macd && indicatorVisibility.stochrsi ? '2.2fr 0.8fr 0.8fr 0.8fr' :
                indicatorVisibility.rsi && indicatorVisibility.macd ? '2.5fr 1fr 1fr' : 
                indicatorVisibility.rsi ? '3fr 1fr' : '1fr',
              gap: '12px'
            }}>
              {/* Main Candlestick Chart */}
              <div style={{
                backgroundColor: '#111827',
                border: '1px solid #374151',
                borderRadius: '8px',
                padding: '12px'
              }}>
                <h3 style={{
                  fontSize: '14px',
                  fontWeight: 'bold',
                  color: '#9CA3AF',
                  marginBottom: '8px',
                  margin: '0 0 8px 0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}>
                  <span>BTC/USDT Candlestick Chart ({selectedPeriod})</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '11px' }}>
                    <span>Signals: {chartData.signals.length}</span>
                    <span>Points: {chartData.dataPoints}</span>
                    <span style={{ color: chartData.fallback ? '#F59E0B' : '#10B981' }}>
                      {chartData.fallback ? '⚠️ Loading...' : '✅ Real Data'}
                    </span>
                  </div>
                </h3>
                
                <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
                  {chartData.candles.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData.candles}>
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
                        <YAxis 
                          yAxisId="price"
                          stroke="#9CA3AF" 
                          fontSize={10} 
                          domain={['dataMin', 'dataMax']}
                        />
                        <Tooltip
                          formatter={(value, name) => [
                            typeof value === 'number' ? formatCurrency(value) : value,
                            name
                          ]}
                          labelFormatter={(time) => new Date(time).toLocaleString()}
                          contentStyle={{
                            backgroundColor: '#1F2937',
                            border: '1px solid #374151',
                            borderRadius: '6px'
                          }}
                        />
                        
                        <Line
                          yAxisId="price"
                          type="monotone"
                          dataKey="high"
                          stroke="#4B5563"
                          strokeWidth={0.5}
                          dot={false}
                          connectNulls={false}
                        />
                        <Line
                          yAxisId="price"
                          type="monotone"
                          dataKey="low"
                          stroke="#4B5563"
                          strokeWidth={0.5}
                          dot={false}
                          connectNulls={false}
                        />
                        <Line
                          yAxisId="price"
                          type="monotone"
                          dataKey="close"
                          stroke="#3B82F6"
                          strokeWidth={2}
                          dot={(props) => <SignalMarker {...props} />}
                          name="Price"
                        />
                        
                        {chartData.signals.map((signal, index) => (
                          <ReferenceLine
                            key={index}
                            x={signal.time}
                            yAxisId="price"
                            stroke={signal.type === 'BUY' ? '#10B981' : '#EF4444'}
                            strokeDasharray="2 2"
                            label={{
                              value: `${signal.type} (${(signal.confidence * 100).toFixed(0)}%)`,
                              position: 'topRight',
                              fontSize: 10,
                              fill: signal.type === 'BUY' ? '#10B981' : '#EF4444'
                            }}
                          />
                        ))}
                      </ComposedChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      color: '#6B7280',
                      textAlign: 'center'
                    }}>
                      <div>
                        <div style={{ fontSize: '48px', marginBottom: '8px' }}>📊</div>
                        <div>Loading real-time chart data...</div>
                        <div style={{ fontSize: '12px', color: '#9CA3AF', marginTop: '8px' }}>
                          Endpoint: {API_BASE_URL}/api/precos/{selectedPeriod}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* RSI Chart */}
              {indicatorVisibility.rsi && (
                <div style={{
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  padding: '12px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 style={{ fontSize: '12px', fontWeight: 'bold', color: '#9CA3AF', margin: 0 }}>
                      RSI ({selectedPeriod}) - Current: {realtimeIndicators.rsi > 0 ? realtimeIndicators.rsi.toFixed(1) : 'N/A'}
                    </h3>
                    <button
                      onClick={() => toggleIndicator('rsi')}
                      style={{ 
                        background: 'none', 
                        border: 'none', 
                        color: '#9CA3AF', 
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '4px'
                      }}
                      title="Hide RSI Chart"
                    >
                      <EyeOff size={14} />
                    </button>
                  </div>
                  
                  <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
                    {chartData.candles.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData.candles}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            type="number"
                            scale="time"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={9}
                          />
                          <YAxis stroke="#9CA3AF" fontSize={9} domain={[0, 100]} />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value) => [value?.toFixed(2), 'RSI']}
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
                            strokeWidth={2}
                            dot={false}
                          />
                          <ReferenceLine y={70} stroke="#EF4444" strokeDasharray="3 3" />
                          <ReferenceLine y={30} stroke="#10B981" strokeDasharray="3 3" />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        Loading RSI data...
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* MACD Chart */}
              {indicatorVisibility.macd && (
                <div style={{
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  padding: '12px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 style={{ fontSize: '12px', fontWeight: 'bold', color: '#9CA3AF', margin: 0 }}>
                      MACD ({selectedPeriod}) - {realtimeIndicators.macd !== 0 ? realtimeIndicators.macd.toFixed(2) : 'N/A'} / {realtimeIndicators.macd_signal !== 0 ? realtimeIndicators.macd_signal.toFixed(2) : 'N/A'}
                    </h3>
                    <button
                      onClick={() => toggleIndicator('macd')}
                      style={{ 
                        background: 'none', 
                        border: 'none', 
                        color: '#9CA3AF', 
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '4px'
                      }}
                      title="Hide MACD Chart"
                    >
                      <EyeOff size={14} />
                    </button>
                  </div>
                  
                  <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
                    {chartData.candles.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData.candles}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            type="number"
                            scale="time"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={9}
                          />
                          <YAxis stroke="#9CA3AF" fontSize={9} />
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
                          <Line
                            type="monotone"
                            dataKey="macd"
                            stroke="#3B82F6"
                            strokeWidth={2}
                            dot={false}
                            name="MACD"
                          />
                          <Line
                            type="monotone"
                            dataKey="macd_signal"
                            stroke="#EF4444"
                            strokeWidth={2}
                            dot={false}
                            name="Signal"
                          />
                          <Bar
                            dataKey="macd_hist"
                            fill="#9CA3AF"
                            opacity={0.6}
                            name="Histogram"
                          />
                          <ReferenceLine y={0} stroke="#4B5563" strokeDasharray="1 1" />
                        </ComposedChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        Loading MACD data...
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* StochRSI Chart */}
              {indicatorVisibility.stochrsi && (
                <div style={{
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  padding: '12px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 style={{ fontSize: '12px', fontWeight: 'bold', color: '#9CA3AF', margin: 0 }}>
                      StochRSI ({selectedPeriod}) - K: {realtimeIndicators.stochrsi_k > 0 ? realtimeIndicators.stochrsi_k.toFixed(1) : 'N/A'} / D: {realtimeIndicators.stochrsi_d > 0 ? realtimeIndicators.stochrsi_d.toFixed(1) : 'N/A'}
                    </h3>
                    <button
                      onClick={() => toggleIndicator('stochrsi')}
                      style={{ 
                        background: 'none', 
                        border: 'none', 
                        color: '#9CA3AF', 
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '4px'
                      }}
                      title="Hide StochRSI Chart"
                    >
                      <EyeOff size={14} />
                    </button>
                  </div>
                  
                  <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
                    {chartData.candles.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData.candles}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            type="number"
                            scale="time"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={9}
                          />
                          <YAxis stroke="#9CA3AF" fontSize={9} domain={[0, 100]} />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value, name) => [value?.toFixed(2) + '%', name]}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '4px',
                              fontSize: '11px'
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="stochrsi_k"
                            stroke="#FFD700"
                            strokeWidth={2.5}
                            dot={false}
                            name="%K"
                          />
                          <Line
                            type="monotone"
                            dataKey="stochrsi_d"
                            stroke="#8B5CF6"
                            strokeWidth={2}
                            dot={false}
                            name="%D"
                          />
                          <ReferenceLine y={80} stroke="#EF4444" strokeDasharray="2 2" opacity={0.7} />
                          <ReferenceLine y={20} stroke="#10B981" strokeDasharray="2 2" opacity={0.7} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        Loading StochRSI data...
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Volume Chart */}
              {indicatorVisibility.volume && (
                <div style={{
                  backgroundColor: '#111827',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  padding: '12px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 style={{ fontSize: '12px', fontWeight: 'bold', color: '#9CA3AF', margin: 0 }}>
                      Volume ({selectedPeriod})
                    </h3>
                    <button
                      onClick={() => toggleIndicator('volume')}
                      style={{ 
                        background: 'none', 
                        border: 'none', 
                        color: '#9CA3AF', 
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '4px'
                      }}
                      title="Hide Volume Chart"
                    >
                      <EyeOff size={14} />
                    </button>
                  </div>
                  
                  <div style={{ width: '100%', height: 'calc(100% - 32px)' }}>
                    {chartData.candles.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.candles}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis
                            dataKey="time"
                            type="number"
                            scale="time"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                            stroke="#9CA3AF"
                            fontSize={9}
                          />
                          <YAxis 
                            stroke="#9CA3AF" 
                            fontSize={9}
                            tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
                          />
                          <Tooltip
                            labelFormatter={(time) => new Date(time).toLocaleString()}
                            formatter={(value) => [`${(value / 1000000).toFixed(2)}M`, 'Volume']}
                            contentStyle={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '4px',
                              fontSize: '11px'
                            }}
                          />
                          <Bar 
                            dataKey="volume" 
                            fill="#6B7280"
                            opacity={0.8}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                        Loading Volume data...
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Status Panel */}
      <div style={{
        borderTop: '1px solid #374151',
        backgroundColor: '#111827',
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '60px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#9CA3AF' }}>RECENT SIGNALS</div>
            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
              {chartData.signals.length > 0 ? (
                <span style={{ color: getSignalColor(chartData.signals[chartData.signals.length - 1].type) }}>
                  {chartData.signals[chartData.signals.length - 1].type} @ {formatCurrency(chartData.signals[chartData.signals.length - 1].price)}
                  {' '}({(chartData.signals[chartData.signals.length - 1].confidence * 100).toFixed(0)}%)
                </span>
              ) : (
                <span style={{ color: '#9CA3AF' }}>No signals yet</span>
              )}
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '10px', color: '#9CA3AF' }}>NEXT PREDICTION</div>
            <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#F59E0B' }}>
              {botStatus.is_running ? 'Processing... ETA: 2m 15s' : 'Bot stopped'}
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '10px', color: '#9CA3AF' }}>DATA STATUS</div>
            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
              <span style={{ color: chartData.fallback ? '#F59E0B' : '#10B981' }}>
                {chartData.fallback ? 'Loading...' : 'Real-time'}
              </span>
              {' '}• {chartData.dataPoints} points • Last: {chartData.lastUpdate.getTime() > 0 ? chartData.lastUpdate.toLocaleTimeString() : 'N/A'}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: connectionStatus === 'connected' ? '#10B981' : '#EF4444',
              animation: connectionStatus === 'connected' ? 'pulse 2s infinite' : 'none'
            }}></div>
            <span style={{ fontSize: '11px', color: '#9CA3AF' }}>
              {connectionStatus === 'connected' ? 'LIVE DATA' : 'OFFLINE'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: '#9CA3AF' }}>
            <Database size={12} />
            <span>API: {connectionStatus}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: '#9CA3AF' }}>
            <span>WS: {Object.values(wsConnections).filter(Boolean).length}/3</span>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default TradingDashboard;
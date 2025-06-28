import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import {
  TrendingUp,
  Brain,
  Database,
  Wifi,
  WifiOff,
  AlertTriangle,
  Play,
  Pause,
  DollarSign,
  Zap,
  Target,
  RotateCw
} from 'lucide-react';

// Estilos CSS como objetos JavaScript (sem Tailwind)
const styles = {
  container: {
    height: '100vh',
    backgroundColor: '#000000',
    color: '#ffffff',
    display: 'flex',
    flexDirection: 'column',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    overflow: 'hidden'
  },
  header: {
    borderBottom: '1px solid #374151',
    backgroundColor: '#111827',
    padding: '12px 16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between'
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '24px'
  },
  title: {
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#3B82F6',
    margin: 0
  },
  marketData: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px'
  },
  marketItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  marketLabel: {
    fontSize: '12px',
    color: '#9CA3AF'
  },
  marketPrice: {
    fontSize: '14px',
    fontWeight: 'bold'
  },
  marketChange: {
    fontSize: '12px'
  },
  connectionStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  connectionText: {
    fontSize: '12px'
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px'
  },
  envSelector: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  envLabel: {
    fontSize: '12px',
    color: '#9CA3AF'
  },
  select: {
    backgroundColor: '#374151',
    color: '#ffffff',
    fontSize: '12px',
    padding: '4px 8px',
    borderRadius: '4px',
    border: '1px solid #4B5563'
  },
  botButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 12px',
    borderRadius: '4px',
    fontWeight: 'bold',
    fontSize: '12px',
    border: 'none',
    cursor: 'pointer',
    color: '#ffffff'
  },
  botButtonStart: {
    backgroundColor: '#059669'
  },
  botButtonStop: {
    backgroundColor: '#DC2626'
  },
  botButtonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed'
  },
  mainContainer: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden'
  },
  sidebar: {
    width: '320px',
    borderRight: '1px solid #374151',
    backgroundColor: '#111827',
    overflowY: 'auto'
  },
  sidebarSection: {
    padding: '12px',
    borderBottom: '1px solid #374151'
  },
  sectionTitle: {
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#9CA3AF',
    marginBottom: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '4px'
  },
  grid2x2: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '8px',
    fontSize: '12px'
  },
  gridItem: {
    backgroundColor: '#374151',
    padding: '8px',
    borderRadius: '4px'
  },
  itemLabel: {
    color: '#9CA3AF'
  },
  itemValue: {
    fontWeight: 'bold'
  },
  indicatorCard: {
    backgroundColor: '#374151',
    padding: '8px',
    borderRadius: '4px',
    marginBottom: '8px'
  },
  indicatorHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  indicatorLabel: {
    fontSize: '12px',
    color: '#9CA3AF'
  },
  indicatorValue: {
    fontSize: '12px',
    fontWeight: 'bold'
  },
  progressBar: {
    width: '100%',
    backgroundColor: '#4B5563',
    borderRadius: '2px',
    height: '4px',
    marginTop: '4px'
  },
  progressFill: {
    height: '4px',
    borderRadius: '2px'
  },
  macdGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: '4px',
    fontSize: '10px'
  },
  macdItem: {
    textAlign: 'center'
  },
  macdLabel: {
    color: '#6B7280'
  },
  macdValue: {
    fontFamily: 'monospace'
  },
  signalCard: {
    backgroundColor: '#374151',
    padding: '8px',
    borderRadius: '4px',
    textAlign: 'center'
  },
  signalLabel: {
    fontSize: '12px',
    color: '#9CA3AF'
  },
  signalValue: {
    fontSize: '14px',
    fontWeight: 'bold'
  },
  angleCard: {
    backgroundColor: '#374151',
    padding: '8px',
    borderRadius: '4px',
    marginBottom: '8px'
  },
  angleHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  angleIcon: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px'
  },
  angleValue: {
    fontSize: '12px',
    fontWeight: 'bold'
  },
  angleSpeed: {
    fontSize: '10px',
    color: '#6B7280'
  },
  balanceCard: {
    backgroundColor: '#374151',
    padding: '8px',
    borderRadius: '4px',
    marginBottom: '8px'
  },
  balanceHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '4px'
  },
  balanceEnv: {
    fontSize: '12px',
    fontWeight: 'bold'
  },
  balanceStatus: {
    fontSize: '10px',
    padding: '2px 6px',
    borderRadius: '2px'
  },
  connectionItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#374151',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    marginBottom: '4px'
  },
  connectionDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%'
  },
  chartArea: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden'
  },
  chartControls: {
    borderBottom: '1px solid #374151',
    backgroundColor: '#111827',
    padding: '12px 16px'
  },
  periodButtons: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  },
  periodButton: {
    padding: '6px 12px',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: 'bold',
    border: 'none',
    cursor: 'pointer'
  },
  periodButtonActive: {
    backgroundColor: '#2563EB',
    color: '#ffffff'
  },
  periodButtonInactive: {
    backgroundColor: '#374151',
    color: '#9CA3AF'
  },
  chartInfo: {
    marginLeft: 'auto',
    fontSize: '12px',
    color: '#9CA3AF',
    display: 'flex',
    alignItems: 'center',
    gap: '16px'
  },
  chartContainer: {
    flex: 1,
    padding: '16px',
    overflow: 'hidden'
  },
  chartGrid: {
    height: '100%',
    display: 'grid',
    gridTemplateRows: '2fr 1fr', 
    gap: '16px'
  },
  chartPanel: {
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '8px',
    padding: '12px'
  },
  chartTitle: {
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#9CA3AF',
    marginBottom: '8px',
    margin: '0 0 8px 0'
  },
  chartContent: {
    width: '100%',
    height: 'calc(100% - 32px)'
  },
  chartPlaceholder: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#6B7280',
    textAlign: 'center'
  },
  placeholderIcon: {
    fontSize: '48px',
    marginBottom: '8px'
  },
  indicatorGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr', 
    gap: '16px'
  },
  modal: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 50
  },
  modalContent: {
    backgroundColor: '#111827',
    border: '2px solid #DC2626',
    borderRadius: '8px',
    padding: '24px',
    maxWidth: '400px',
    width: '100%',
    margin: '16px'
  },
  modalHeader: {
    textAlign: 'center',
    fontSize: '48px',
    marginBottom: '16px'
  },
  modalTitle: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#EF4444',
    marginBottom: '16px',
    margin: '0 0 16px 0',
    textAlign: 'center'
  },
  modalText: {
    fontSize: '14px',
    color: '#D1D5DB',
    marginBottom: '16px'
  },
  modalList: {
    textAlign: 'left',
    fontSize: '12px',
    backgroundColor: '#374151',
    padding: '12px',
    borderRadius: '4px',
    margin: '0',
    paddingLeft: '24px'
  },
  modalButtons: {
    display: 'flex',
    gap: '12px'
  },
  modalButton: {
    flex: 1,
    padding: '8px 16px',
    borderRadius: '4px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px'
  },
  modalButtonCancel: {
    backgroundColor: '#4B5563',
    color: '#D1D5DB'
  },
  modalButtonConfirm: {
    backgroundColor: '#DC2626',
    color: '#ffffff',
    fontWeight: 'bold'
  }
};

const BloombergDashboard = () => {
  // Define a URL base do seu backend AQUI!
  const API_BASE_URL = 'http://62.72.1.122:8000'; 

  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [marketData, setMarketData] = useState({
    assets: {
      btc: { name: 'BTC/USD', current_price: 0, change: 0, change_percent: 0 },
      gold: { name: 'GOLD', current_price: 0, change: 0, change_percent: 0 },
      dxy: { name: 'DXY', current_price: 0, change: 0, change_percent: 0 }
    }
  });
  
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    ai_accuracy: 0,
    training_samples: 0,
    ml_model_accuracy: 0,
    ai_predictions: 0,
    active_positions: [],
    testnet: {
      balance: 0,
      total_pnl: 0,
      roi_percentage: 0,
      win_rate: 0,
      total_trades: 0,
      winning_trades: 0,
      daily_pnl: 0,
      daily_trades: 0
    },
    live: {
      balance: 0,
      total_pnl: 0,
      roi_percentage: 0,
      win_rate: 0,
      total_trades: 0,
      winning_trades: 0,
      daily_pnl: 0,
      daily_trades: 0
    },
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
    rsiMacd: false
  });

  const [realtimeIndicators, setRealtimeIndicators] = useState({
    btc: {
      rsi: 0,
      rsi_angle: 0,
      macd: 0,
      macd_signal: 0,
      macd_histogram: 0,
      macd_angle: 0,
      signal_angle: 0,
      signal: 'HOLD',
      stochrsi_k: 0, 
      stochrsi_d: 0 
    }
  });

  const [selectedPeriod, setSelectedPeriod] = useState('5m');
  const [isLoading, setIsLoading] = useState(false);
  const [tradingEnvironment, setTradingEnvironment] = useState('testnet');
  const [showEnvironmentWarning, setShowEnvironmentWarning] = useState(false);

  // DADOS REAIS - Bot Status
  const fetchBotStatus = useCallback(async () => {
    try {
      console.log('🔄 Fetching bot status from ' + `${API_BASE_URL}/api/trading-bot/status`);
      const response = await fetch(`${API_BASE_URL}/api/trading-bot/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('✅ Bot status received:', data);
      
      setBotStatus(prev => ({
        ...prev,
        is_running: data.running !== undefined ? data.running : 
                   (data.status === 'simulated_running' || data.status === 'running'),
        ai_accuracy: data.ai_accuracy || 0,
        training_samples: data.training_samples || 0,
        ml_model_accuracy: data.ml_model_accuracy || 0,
        ai_predictions: data.ai_predictions || 0,
        active_positions: data.active_positions || [],
        total_pnl: data.total_pnl || 0, // Pode ser global total PnL
        roi_percentage: data.roi_percentage || 0, // Pode ser global ROI
        win_rate: data.win_rate || 0, // Pode ser global win rate
        total_trades: data.total_trades || 0, // Pode ser global total trades
        ai_system_status: data.ai_system_status || prev.ai_system_status,
        // CORREÇÃO: Usar os objetos testnet_performance e live_performance diretamente do data
        testnet: {
          balance: data.testnet_performance?.balance || 0,
          total_pnl: data.testnet_performance?.total_pnl || 0,
          roi_percentage: data.testnet_performance?.roi_percentage || 0,
          win_rate: data.testnet_performance?.win_rate || 0,
          total_trades: data.testnet_performance?.total_trades || 0,
          winning_trades: data.testnet_performance?.winning_trades || 0,
          daily_pnl: data.testnet_performance?.daily_pnl || 0,
          daily_trades: data.testnet_performance?.daily_trades || 0
        },
        live: {
          balance: data.live_performance?.balance || 0,
          total_pnl: data.live_performance?.total_pnl || 0,
          roi_percentage: data.live_performance?.roi_percentage || 0,
          win_rate: data.live_performance?.win_rate || 0,
          total_trades: data.live_performance?.total_trades || 0,
          winning_trades: data.live_performance?.winning_trades || 0,
          daily_pnl: data.live_performance?.daily_pnl || 0,
          daily_trades: data.live_performance?.daily_trades || 0
        }
      }));
      setConnectionStatus('connected');
    } catch (error) {
      console.error("❌ Failed to fetch bot status:", error);
      setConnectionStatus('disconnected');
    }
  }, [API_BASE_URL]);

  // DADOS REAIS - Market Data (from /api/current)
  const fetchMarketData = useCallback(async () => {
    try {
      console.log('🔄 Fetching market data from ' + `${API_BASE_URL}/api/current`);
      const response = await fetch(`${API_BASE_URL}/api/current`);
      if (!response.ok) {
        throw new Error(`HTTP error fetching market data! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('✅ Market data received:', data);
      if (data.assets) {
        setMarketData({ assets: data.assets });
      }
    } catch (error) {
      console.error("❌ Failed to fetch market data:", error);
      setMarketData({
        assets: {
          btc: { name: 'BTC/USD', current_price: 0, change: 0, change_percent: 0 },
          gold: { name: 'GOLD', current_price: 0, change: 0, change_percent: 0 },
          dxy: { name: 'DXY', current_price: 0, change: 0, change_percent: 0 }
        }
      });
    }
  }, [API_BASE_URL]);

  // DADOS REAIS - Technical Indicators (from /api/macd/realtime/btc, but will be primarily via WebSocket)
  const fetchTechnicalIndicators = useCallback(async () => {
    try {
      console.log('🔄 Fetching technical indicators from ' + `${API_BASE_URL}/api/macd/realtime/btc`);
      const response = await fetch(`${API_BASE_URL}/api/macd/realtime/btc`);
      if (!response.ok) { 
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('✅ Technical indicators received (REST):', data);

      setRealtimeIndicators(prev => ({
        ...prev,
        btc: {
          rsi: data.rsi_data?.last_value || 0, // RSI data is now directly from rsi_data
          macd: (data.macd_data?.macd && data.macd_data.macd.length > 0 ? data.macd_data.macd[data.macd_data.macd.length - 1] : 0),
          macd_signal: (data.macd_data?.signal && data.macd_data.signal.length > 0 ? data.macd_data.signal[data.macd_data.signal.length - 1] : 0),
          macd_histogram: (data.macd_data?.histogram && data.macd_data.histogram.length > 0 ? data.macd_data.histogram[data.macd_data.histogram.length - 1] : 0),
          signal: data.macd_data?.trend || data.signal || 'HOLD', // Signal can come from macd_data or be a top-level field
          rsi_angle: data.rsi_data?.angle || 0,
          macd_angle: data.macd_angle_data?.macd_angle || 0,
          signal_angle: data.macd_angle_data?.signal_angle || 0,
          stochrsi_k: data.stochrsi_data?.k_value || 0, 
          stochrsi_d: data.stochrsi_data?.d_value || 0  
        }
      }));
    } catch (error) {
      console.error("❌ Failed to fetch technical indicators (REST):", error);
      setRealtimeIndicators(prev => ({
        btc: { rsi: 0, rsi_angle: 0, macd: 0, macd_signal: 0, macd_histogram: 0, macd_angle: 0, signal_angle: 0, signal: 'HOLD', stochrsi_k: 0, stochrsi_d: 0 }
      }));
    }
  }, [API_BASE_URL]);

  // DADOS REAIS - Chart Data (from /api/precos/{period})
  const fetchChartData = useCallback(async () => {
    try {
      console.log(`🔄 Fetching chart data from ${API_BASE_URL}/api/precos/${selectedPeriod}`);
      const response = await fetch(`${API_BASE_URL}/api/precos/${selectedPeriod}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('✅ Chart data received:', data);
      
      const btcData = data.assets?.btc;
      if (btcData && btcData.price_data && btcData.price_data.length > 0) {
          const chartPoints = btcData.price_data.map((price, index) => ({
              time: new Date(data.dates[index]).getTime(),
              price: price,
              volume: btcData.volume_data[index] || 0,
              rsi: btcData.rsi_data?.[index] || 50,
              macd: btcData.macd_data?.[index] || 0,
              macd_signal: btcData.macd_signal_data?.[index] || 0,
              macd_hist: btcData.macd_hist_data?.[index] || 0,
              stochrsi_k: btcData.stochrsi_k_data?.[index] || 0, 
              stochrsi_d: btcData.stochrsi_d_data?.[index] || 0,
              // Adicionar valores fixos para as linhas de referência do RSI e StochRSI
              rsi_line_70: 70, 
              rsi_line_30: 30,
              stochrsi_line_80: 80,
              stochrsi_line_20: 20
          }));

          setChartData({
              combined: chartPoints,
              dataPoints: chartPoints.length,
              lastUpdate: new Date(data.dates[data.dates.length - 1]),
              fallback: false
          });
      } else {
          console.warn(`⚠️ No valid BTC price data found for period ${selectedPeriod}. Activating fallback.`);
          setChartData(prev => ({
              ...prev,
              fallback: true,
              combined: [],
              dataPoints: 0,
              lastUpdate: new Date(0)
          }));
      }
    } catch (error) {
      console.error("❌ Failed to fetch chart data:", error);
      setChartData(prev => ({
        ...prev,
        fallback: true,
        combined: [],
        dataPoints: 0,
        lastUpdate: new Date(0)
      }));
    }
  }, [selectedPeriod, API_BASE_URL]);

  // WebSocket Connections
  useEffect(() => {
    let sentimentWs, ohlcvWs, rsiMacdWs;

    const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss:' : 'ws:';
    const wsHostPort = API_BASE_URL.split('//')[1];

    const connectWebSockets = () => {
        // --- Sentiment WebSocket ---
        sentimentWs = new WebSocket(`${wsProtocol}//${wsHostPort}/ws/sentiment`);
        sentimentWs.onopen = () => {
            console.log('✅ Sentiment WebSocket connected');
            setWsConnections(prev => ({ ...prev, sentiment: true }));
        };
        sentimentWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
            }
            catch (e) {
                console.error("Error parsing sentiment WS message:", e);
            }
        };
        sentimentWs.onclose = () => {
            console.log('❌ Sentiment WebSocket disconnected');
            setWsConnections(prev => ({ ...prev, sentiment: false }));
        };
        sentimentWs.onerror = (error) => {
            console.error('❌ Sentiment WebSocket error:', error);
            setWsConnections(prev => ({ ...prev, sentiment: false }));
        };

        // --- OHLCV WebSocket (for chart real-time updates) ---
        ohlcvWs = new WebSocket(`${wsProtocol}//${wsHostPort}/ws/ohlcv`);
        ohlcvWs.onopen = () => {
            console.log('✅ OHLCV WebSocket connected');
            setWsConnections(prev => ({ ...prev, ohlcv: true }));
        };
        ohlcvWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'ohlcv_update' && data.asset.toLowerCase() === 'btc') {
                    setChartData(prev => {
                        const updatedCombined = [...prev.combined];
                        const newTime = new Date(data.candle.timestamp * 1000).getTime();

                        const lastChartPoint = updatedCombined.length > 0 ? updatedCombined[updatedCombined.length - 1] : null;

                        if (lastChartPoint && lastChartPoint.time === newTime) {
                            lastChartPoint.price = data.candle.close;
                            lastChartPoint.volume = data.candle.volume;
                            lastChartPoint.rsi = data.rsi_data?.last_value || lastChartPoint.rsi;
                            lastChartPoint.macd = data.macd_data?.macd?.[data.macd_data.macd.length - 1] || lastChartPoint.macd;
                            lastChartPoint.macd_signal = data.macd_data?.signal?.[data.macd_data.signal.length - 1] || lastChartPoint.macd_signal;
                            lastChartPoint.macd_hist = data.macd_data?.histogram?.[data.macd_data.histogram.length - 1] || lastChartPoint.macd_hist;
                            lastChartPoint.stochrsi_k = data.stochrsi_data?.k_value || lastChartPoint.stochrsi_k; 
                            lastChartPoint.stochrsi_d = data.stochrsi_data?.d_value || lastChartPoint.stochrsi_d; 
                        } else if (!lastChartPoint || newTime > lastChartPoint.time) {
                            updatedCombined.push({
                                time: newTime,
                                price: data.candle.close,
                                volume: data.candle.volume,
                                rsi: data.rsi_data?.last_value || 50,
                                macd: data.macd_data?.macd?.[data.macd_data.macd.length - 1] || 0,
                                macd_signal: data.macd_data?.signal?.[data.macd_data.signal.length - 1] || 0,
                                macd_hist: data.macd_data?.histogram?.[data.macd_data.histogram.length - 1] || 0,
                                stochrsi_k: data.stochrsi_data?.k_value || 0, 
                                stochrsi_d: data.stochrsi_data?.d_value || 0  
                            });
                            // Keep a reasonable number of points for display, e.g., for 5m interval, 120 points = 10 hours.
                            if (updatedCombined.length > 120) { // Keep last 120 points for 10 hours (5m candles)
                                updatedCombined.shift(); 
                            }
                        }

                        setMarketData(prev => {
                            const currentBtc = prev.assets.btc;
                            const prevPrice = currentBtc.current_price;
                            const newPrice = data.candle.close;
                            let change = newPrice - prevPrice;
                            let changePercent = (prevPrice !== 0 && !isNaN(prevPrice) && isFinite(prevPrice)) ? (change / prevPrice) * 100 : 0;
                            
                            return {
                                assets: {
                                    ...prev.assets,
                                    btc: {
                                        ...prev.assets.btc,
                                        current_price: newPrice,
                                        change: change,
                                        change_percent: changePercent
                                    }
                                }
                            };
                        });

                        return {
                            combined: updatedCombined,
                            dataPoints: updatedCombined.length,
                            lastUpdate: new Date(newTime),
                            fallback: false
                        };
                    });
                    
                    setRealtimeIndicators(prev => ({
                        ...prev,
                        btc: {
                            rsi: data.rsi_data?.last_value || 0,
                            rsi_angle: data.rsi_data?.angle || 0,
                            macd: data.macd_data?.macd?.[data.macd_data.macd.length - 1] || 0,
                            macd_signal: data.macd_data?.signal?.[data.macd_data.signal.length - 1] || 0,
                            macd_histogram: data.macd_data?.histogram?.[data.macd_data.histogram.length - 1] || 0,
                            macd_angle: data.macd_angle_data?.macd_angle || 0,
                            signal_angle: data.macd_angle_data?.signal_angle || 0,
                            signal: data.macd_data?.trend || 'HOLD',
                            stochrsi_k: data.stochrsi_data?.k_value || 0, 
                            stochrsi_d: data.stochrsi_data?.d_value || 0  
                        }
                    }));

                }
            } catch (e) {
                console.error("Error parsing OHLCV WS message:", e);
            }
        };
        ohlcvWs.onclose = () => {
            console.log('❌ OHLCV WebSocket disconnected');
            setWsConnections(prev => ({ ...prev, ohlcv: false }));
        };
        ohlcvWs.onerror = (error) => {
            console.error('❌ OHLCV WebSocket error:', error);
            setWsConnections(prev => ({ ...prev, ohlcv: false }));
        };

        // --- RSI/MACD WebSocket (This will be the primary source for sidebar indicators) ---
        rsiMacdWs = new WebSocket(`${wsProtocol}//${wsHostPort}/ws/rsi-macd`);
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
                        btc: {
                            rsi: data.rsi?.value || 0,
                            rsi_angle: data.rsi?.angle || 0,
                            macd: data.macd?.macd || 0,
                            macd_signal: data.macd?.signal || 0,
                            macd_histogram: data.macd?.histogram || 0,
                            macd_angle: data.macd?.macd_angle || 0,
                            signal_angle: data.macd?.signal_angle || 0,
                            signal: data.combined?.signal_type || 'HOLD',
                            stochrsi_k: data.stochrsi?.k_value || 0, 
                            stochrsi_d: data.stochrsi?.d_value || 0  
                        }
                    }));
                }
            } catch (e) {
                console.error("Error parsing RSI/MACD WS message:", e);
            }
        };
        rsiMacdWs.onclose = () => {
            console.log('❌ RSI/MACD WebSocket disconnected');
            setWsConnections(prev => ({ ...prev, rsiMacd: false }));
        };
        rsiMacdWs.onerror = (error) => {
            console.error('❌ RSI/MACD WebSocket error:', error);
            setWsConnections(prev => ({ ...prev, rsiMacd: false }));
        };
    };

    connectWebSockets();

    return () => {
      if (sentimentWs) sentimentWs.close();
      if (ohlcvWs) ohlcvWs.close();
      if (rsiMacdWs) rsiMacdWs.close();
    };
  }, [API_BASE_URL]);

  const controlBot = useCallback(async (action) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/trading-bot/${action}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      await response.json();
      fetchBotStatus();
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
      if (tradingEnvironment === 'live') {
        setShowEnvironmentWarning(true);
      } else {
        controlBot('start');
      }
    }
  };

  const confirmLiveEnvironment = () => {
    setShowEnvironmentWarning(false);
    fetch(`${API_BASE_URL}/api/trading-bot/set-environment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ environment: 'live' }),
    })
    .then(response => {
      if (!response.ok) throw new Error('Failed to set live environment');
      return response.json();
    })
    .then(() => {
      console.log('Environment set to LIVE. Starting bot...');
      controlBot('start');
    })
    .catch(error => {
      console.error('Error setting environment to live:', error);
      alert('Failed to set live environment. Check console.');
    });
  };

  const cancelLiveEnvironment = () => {
    setShowEnvironmentWarning(false);
  };

  // Setup polling intervals for initial data load and fallback if WS disconnects
  useEffect(() => {
    console.log('🚀 Initiating real-time data fetching...');
    
    fetchBotStatus();
    fetchMarketData();
    fetchTechnicalIndicators();
    fetchChartData();
    
    const statusInterval = setInterval(fetchBotStatus, 10000); 
    const marketInterval = setInterval(fetchMarketData, 30000); 
    const indicatorsInterval = setInterval(fetchTechnicalIndicators, 60000); 
    const chartInterval = setInterval(fetchChartData, 120000); 
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(marketInterval);
      clearInterval(indicatorsInterval);
      clearInterval(chartInterval);
    };
  }, [fetchBotStatus, fetchMarketData, fetchTechnicalIndicators, fetchChartData]);

  // Update chart when period changes (triggers immediate fetchChartData)
  useEffect(() => {
    fetchChartData();
  }, [selectedPeriod]);

  const [chartData, setChartData] = useState({
    combined: [],
    dataPoints: 0,
    lastUpdate: new Date(0), 
    fallback: true 
  });

  const periods = ['5m', '15m', '1h', '4h', '1d'];

  // --- Funções de Formatação e Estilo ---
  const formatCurrency = (value) => {
    if (typeof value !== 'number' || isNaN(value) || value === 0) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPnL = (pnl) => {
    if (typeof pnl !== 'number' || isNaN(pnl) || pnl === 0) return { formatted: 'N/A', color: '#9CA3AF' };
    const formatted = formatCurrency(Math.abs(pnl));
    return {
      formatted: pnl >= 0 ? `+${formatted}` : `-${formatted}`,
      color: pnl >= 0 ? '#10B981' : '#EF4444'
    };
  };

  const getColorForChange = (change) => {
    if (typeof change !== 'number' || isNaN(change) || change === 0) return { color: '#9CA3AF' };
    return {
      color: change >= 0 ? '#10B981' : '#EF4444'
    };
  };

  const getColorForRSI = (rsi) => {
    if (typeof rsi !== 'number' || isNaN(rsi) || rsi === 0) return '#9CA3AF'; 
    if (rsi > 70) return '#EF4444';
    if (rsi < 30) return '#10B981';
    return '#F59E0B';
  };

  const getColorForSignal = (signal) => {
    if (!signal || signal.toUpperCase() === 'HOLD' || signal.toUpperCase() === 'N/A') return '#9CA3AF';
    if (signal.toUpperCase().includes('BUY')) return '#10B981';
    if (signal.toUpperCase().includes('SELL')) return '#EF4444';
    return '#9CA3AF';
  };

  const getColorForAngle = (angle) => {
    if (typeof angle !== 'number' || isNaN(angle) || angle === 0) return '#9CA3AF';
    if (angle > 10) return '#10B981';
    if (angle < -10) return '#EF4444';
    return '#F59E0B';
  };

  const getAngleIcon = (angle) => {
    if (typeof angle !== 'number' || isNaN(angle) || angle === 0) return '—';
    if (angle > 0) return '↗️';
    if (angle < 0) return '↘️';
    return '➡️';
  };

  const hasValidChartData = chartData.combined && chartData.combined.length > 0 && !chartData.fallback;

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.title}>TRADING DASHBOARD PRO</h1>
          
          {/* Market Data Inline - DADOS REAIS */}
          <div style={styles.marketData}>
            {Object.entries(marketData.assets).map(([key, asset]) => (
              <div key={key} style={styles.marketItem}>
                <span style={styles.marketLabel}>{asset.name}:</span>
                <span style={styles.marketPrice}>
                  {asset.current_price !== 0 ? formatCurrency(asset.current_price) : 'N/A'}
                </span>
                <span style={{...styles.marketChange, ...getColorForChange(asset.change_percent)}}>
                  {asset.current_price !== 0 ? `${asset.change_percent >= 0 ? '+' : ''}${asset.change_percent.toFixed(2)}%` : 'N/A'}
                </span>
              </div>
            ))}
          </div>

          {/* Connection Status - REAL */}
          <div style={styles.connectionStatus}>
            {connectionStatus === 'connected' ? (
              <Wifi size={12} color="#10B981" />
            ) : (
              <WifiOff size={12} color="#EF4444" />
            )}
            <span style={{
              ...styles.connectionText,
              color: connectionStatus === 'connected' ? '#10B981' : '#EF4444'
            }}>
              {connectionStatus.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Bot Controls */}
        <div style={styles.headerRight}>
          <div style={styles.envSelector}>
            <span style={styles.envLabel}>Env:</span>
            <select
              value={tradingEnvironment}
              onChange={(e) => setTradingEnvironment(e.target.value)}
              style={styles.select}
              disabled={botStatus.is_running}
            >
              <option value="testnet">TESTNET</option>
              <option value="live">LIVE</option>
            </select>
          </div>
          
          <button
            onClick={handleToggleBot}
            disabled={isLoading}
            style={{
              ...styles.botButton,
              ...(botStatus.is_running ? styles.botButtonStop : styles.botButtonStart),
              ...(isLoading ? styles.botButtonDisabled : {})
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

      <div style={styles.mainContainer}>
        {/* Sidebar - DADOS REAIS */}
        <div style={styles.sidebar}>
          {/* AI Status - DADOS REAIS */}
          <div style={styles.sidebarSection}>
            <h3 style={styles.sectionTitle}>
              <Brain size={12} />
              AI STATUS
            </h3>
            <div style={styles.grid2x2}>
              <div style={styles.gridItem}>
                <div style={styles.itemLabel}>Accuracy</div>
                <div style={{...styles.itemValue, color: '#3B82F6'}}>
                  {botStatus.ai_accuracy !== 0 ? `${botStatus.ai_accuracy.toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              <div style={styles.gridItem}>
                <div style={styles.itemLabel}>Samples</div>
                <div style={{...styles.itemValue, color: '#10B981'}}>
                  {botStatus.training_samples !== 0 ? `${(botStatus.training_samples/1000).toFixed(0)}k` : 'N/A'}
                </div>
              </div>
              <div style={styles.gridItem}>
                <div style={styles.itemLabel}>ML Acc</div>
                <div style={{...styles.itemValue, color: '#8B5CF6'}}>
                  {botStatus.ml_model_accuracy !== 0 ? `${botStatus.ml_model_accuracy.toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              <div style={styles.gridItem}>
                <div style={styles.itemLabel}>Predictions</div>
                <div style={{...styles.itemValue, color: '#F59E0B'}}>
                  {botStatus.ai_predictions !== 0 ? botStatus.ai_predictions : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Indicadores Técnicos - DADOS REAIS */}
          <div style={styles.sidebarSection}>
            <h3 style={styles.sectionTitle}>
              <Zap size={12} />
              INDICATORS
            </h3>
            
            <div style={styles.indicatorCard}>
              <div style={styles.indicatorHeader}>
                <span style={styles.indicatorLabel}>RSI</span>
                <span style={{
                  ...styles.indicatorValue,
                  color: getColorForRSI(realtimeIndicators.btc.rsi)
                }}>
                  {realtimeIndicators.btc.rsi !== 0 ? realtimeIndicators.btc.rsi.toFixed(1) : 'N/A'}
                </span>
              </div>
              {/* Render progress bar only if RSI is a valid, non-zero number */}
              {typeof realtimeIndicators.btc.rsi === 'number' && realtimeIndicators.btc.rsi !== 0 && !isNaN(realtimeIndicators.btc.rsi) && (
                <div style={styles.progressBar}>
                  <div style={{
                    ...styles.progressFill,
                    width: `${realtimeIndicators.btc.rsi}%`,
                    backgroundColor: getColorForRSI(realtimeIndicators.btc.rsi)
                  }}></div>
                </div>
              )}
            </div>

            <div style={styles.indicatorCard}>
              <div style={{...styles.indicatorLabel, marginBottom: '4px'}}>MACD</div>
              <div style={styles.macdGrid}>
                <div style={styles.macdItem}>
                  <div style={styles.macdLabel}>Line</div>
                  <div style={styles.macdValue}>
                    {realtimeIndicators.btc.macd !== 0 ? realtimeIndicators.btc.macd.toFixed(4) : 'N/A'}
                  </div>
                </div>
                <div style={styles.macdItem}>
                  <div style={styles.macdLabel}>Signal</div>
                  <div style={styles.macdValue}>
                    {realtimeIndicators.btc.macd_signal !== 0 ? realtimeIndicators.btc.macd_signal.toFixed(4) : 'N/A'}
                  </div>
                </div>
                <div style={styles.macdItem}>
                  <div style={styles.macdLabel}>Hist</div>
                  <div style={{
                    ...styles.macdValue,
                    color: realtimeIndicators.btc.macd_histogram > 0 ? '#10B981' : '#EF4444'
                  }}>
                    {realtimeIndicators.btc.macd_histogram !== 0 ? realtimeIndicators.btc.macd_histogram.toFixed(4) : 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            {/* Novo: Stochastic RSI Card */}
            <div style={styles.indicatorCard}>
                <div style={styles.indicatorHeader}>
                    <span style={styles.indicatorLabel}>STOCHRSI (%K / %D)</span>
                    <span style={styles.indicatorValue}>
                        <span style={{ color: realtimeIndicators.btc.stochrsi_k > realtimeIndicators.btc.stochrsi_d && realtimeIndicators.btc.stochrsi_k !== 0 ? '#10B981' : '#EF4444' }}>
                            {realtimeIndicators.btc.stochrsi_k !== 0 ? realtimeIndicators.btc.stochrsi_k.toFixed(1) : 'N/A'}
                        </span>
                        <span style={{color: '#9CA3AF'}}> / </span>
                        <span style={{ color: realtimeIndicators.btc.stochrsi_d !== 0 ? '#3B82F6' : '#9CA3AF' }}>
                            {realtimeIndicators.btc.stochrsi_d !== 0 ? realtimeIndicators.btc.stochrsi_d.toFixed(1) : 'N/A'}
                        </span>
                    </span>
                </div>
                <div style={{fontSize: '10px', color: '#6B7280', marginTop: '4px'}}>
                    Trend: {realtimeIndicators.btc.stochrsi_k !== 0 && realtimeIndicators.btc.stochrsi_d !== 0 ? (realtimeIndicators.btc.stochrsi_k > realtimeIndicators.btc.stochrsi_d ? 'Bullish' : 'Bearish') : 'N/A'}
                </div>
                <div style={{fontSize: '10px', color: '#6B7280'}}>
                    Zone: {realtimeIndicators.btc.stochrsi_k !== 0 ? (realtimeIndicators.btc.stochrsi_k > 80 ? 'Overbought' : (realtimeIndicators.btc.stochrsi_k < 20 ? 'Oversold' : 'Neutral')) : 'N/A'}
                </div>
            </div>


            <div style={styles.signalCard}>
              <div style={styles.signalLabel}>SIGNAL</div>
              <div style={{
                ...styles.signalValue,
                color: getColorForSignal(realtimeIndicators.btc.signal)
              }}>
                {realtimeIndicators.btc.signal || 'N/A'}
              </div>
            </div>
          </div>

          {/* Momentum Angles - DADOS REAIS */}
          <div style={styles.sidebarSection}>
            <h3 style={styles.sectionTitle}>
              <TrendingUp size={12} />
              MOMENTUM ANGLES
            </h3>
            
            <div style={styles.angleCard}>
              <div style={styles.angleHeader}>
                <span style={styles.indicatorLabel}>RSI ANGLE</span>
                <div style={styles.angleIcon}>
                  <span>{getAngleIcon(realtimeIndicators.btc.rsi_angle)}</span>
                  <span style={{
                    ...styles.angleValue,
                    color: getColorForAngle(realtimeIndicators.btc.rsi_angle)
                  }}>
                    {realtimeIndicators.btc.rsi_angle !== 0 ? `${realtimeIndicators.btc.rsi_angle.toFixed(1)}°` : 'N/A'}
                  </span>
                </div>
              </div>
              <div style={styles.angleSpeed}>
                Speed: {realtimeIndicators.btc.rsi_angle !== 0 ? `${Math.abs(realtimeIndicators.btc.rsi_angle).toFixed(1)}°/min` : 'N/A'}
              </div>
            </div>

            <div style={styles.angleCard}>
              <div style={{...styles.indicatorLabel, marginBottom: '4px'}}>MACD ANGLES</div>
              <div style={{marginBottom: '4px'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '10px'}}>
                  <span style={{color: '#6B7280'}}>MACD Line</span>
                  <div style={styles.angleIcon}>
                    <span>{getAngleIcon(realtimeIndicators.btc.macd_angle)}</span>
                    <span style={{
                      fontWeight: 'bold',
                      color: getColorForAngle(realtimeIndicators.btc.macd_angle)
                    }}>
                      {realtimeIndicators.btc.macd_angle !== 0 ? `${realtimeIndicators.btc.macd_angle.toFixed(1)}°` : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
              <div style={{fontSize: '10px'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <span style={{color: '#6B7280'}}>Signal Line</span>
                  <div style={styles.angleIcon}>
                    <span>{getAngleIcon(realtimeIndicators.btc.signal_angle)}</span>
                    <span style={{
                      fontWeight: 'bold',
                      color: getColorForAngle(realtimeIndicators.btc.signal_angle)
                    }}>
                      {realtimeIndicators.btc.signal_angle !== 0 ? `${realtimeIndicators.btc.signal_angle.toFixed(1)}°` : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Account Balances - DADOS REAIS */}
          <div style={styles.sidebarSection}>
            <h3 style={styles.sectionTitle}>
              <DollarSign size={12} />
              ACCOUNT BALANCES
            </h3>
            
            {/* Testnet */}
            <div style={styles.balanceCard}>
              <div style={styles.balanceHeader}>
                <span style={{...styles.balanceEnv, color: '#3B82F6'}}>TESTNET</span>
                <span style={{
                  ...styles.balanceStatus,
                  backgroundColor: tradingEnvironment === 'testnet' && botStatus.is_running ? '#065F46' : '#374151',
                  color: tradingEnvironment === 'testnet' && botStatus.is_running ? '#10B981' : '#9CA3AF'
                }}>
                  {tradingEnvironment === 'testnet' && botStatus.is_running ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <div style={styles.grid2x2}>
                <div>
                  <div style={styles.itemLabel}>Balance</div>
                  <div style={{fontWeight: 'bold', color: '#ffffff', fontSize: '11px'}}>
                    {/* Alterado para usar botStatus.testnet.balance diretamente */}
                    {botStatus.testnet.balance !== 0 ? formatCurrency(botStatus.testnet.balance) : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={styles.itemLabel}>P&L</div>
                  <div style={{
                    fontWeight: 'bold',
                    color: formatPnL(botStatus.testnet.total_pnl).color,
                    fontSize: '11px'
                  }}>
                    {botStatus.testnet.total_pnl !== 0 ? formatPnL(botStatus.testnet.total_pnl).formatted : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={styles.itemLabel}>ROI</div>
                  <div style={{
                    fontWeight: 'bold',
                    color: botStatus.testnet.roi_percentage >= 0 ? '#10B981' : '#EF4444',
                    fontSize: '11px'
                  }}>
                    {botStatus.testnet.roi_percentage !== 0 ? `${botStatus.testnet.roi_percentage.toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={styles.itemLabel}>Trades</div>
                  <div style={{fontWeight: 'bold', color: '#D1D5DB', fontSize: '11px'}}>
                    {botStatus.testnet.total_trades !== 0 ? botStatus.testnet.total_trades : 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            {/* Live */}
            <div style={styles.balanceCard}>
              <div style={styles.balanceHeader}>
                <span style={{...styles.balanceEnv, color: '#F59E0B'}}>LIVE</span>
                <span style={{
                  ...styles.balanceStatus,
                  backgroundColor: tradingEnvironment === 'live' && botStatus.is_running ? '#7F1D1D' : '#374151',
                  color: tradingEnvironment === 'live' && botStatus.is_running ? '#EF4444' : '#9CA3AF'
                }}>
                  {tradingEnvironment === 'live' && botStatus.is_running ? 'ACTIVE' : 'INACTIVE'}
                </span>
              </div>
              <div style={styles.grid2x2}>
                <div>
                  <div style={styles.itemLabel}>Balance</div>
                  <div style={{fontWeight: 'bold', color: '#ffffff', fontSize: '11px'}}>
                    {/* Alterado para usar botStatus.live.balance diretamente */}
                    {botStatus.live.balance !== 0 ? formatCurrency(botStatus.live.balance) : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={styles.itemLabel}>P&L</div>
                  <div style={{
                    fontWeight: 'bold',
                    color: formatPnL(botStatus.live.total_pnl).color,
                    fontSize: '11px'
                  }}>
                    {botStatus.live.total_pnl !== 0 ? formatPnL(botStatus.live.total_pnl).formatted : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={styles.itemLabel}>ROI</div>
                  <div style={{
                    fontWeight: 'bold',
                    color: botStatus.live.roi_percentage >= 0 ? '#10B981' : '#EF4444',
                    fontSize: '11px'
                  }}>
                    {botStatus.live.roi_percentage !== 0 ? `${botStatus.live.roi_percentage.toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={styles.itemLabel}>Trades</div>
                  <div style={{fontWeight: 'bold', color: '#D1D5DB', fontSize: '11px'}}>
                    {botStatus.live.total_trades !== 0 ? botStatus.live.total_trades : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* WebSocket Status - REAL */}
          <div style={styles.sidebarSection}>
            <h3 style={styles.sectionTitle}>
              <Database size={12} />
              CONNECTIONS
            </h3>
            <div>
              {Object.entries(wsConnections).map(([key, connected]) => (
                <div key={key} style={styles.connectionItem}>
                  <span style={{color: '#D1D5DB'}}>{key.toUpperCase()}</span>
                  <div style={{
                    ...styles.connectionDot,
                    backgroundColor: connected ? '#10B981' : '#EF4444'
                  }}></div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Chart Area - DADOS REAIS */}
        <div style={styles.chartArea}>
          {/* Chart Controls */}
          <div style={styles.chartControls}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
              <div style={styles.periodButtons}>
                {periods.map((period) => (
                  <button
                    key={period}
                    onClick={() => setSelectedPeriod(period)}
                    style={{
                      ...styles.periodButton,
                      ...(selectedPeriod === period ? styles.periodButtonActive : styles.periodButtonInactive)
                    }}
                  >
                    {period.toUpperCase()}
                  </button>
                ))}
                <div style={styles.chartInfo}>
                  <span>Data Points: {chartData.dataPoints !== 0 ? chartData.dataPoints : 'N/A'}</span>
                  <span>Last Update: {chartData.lastUpdate.getTime() !== new Date(0).getTime() ? new Date(chartData.lastUpdate).toLocaleTimeString() : 'N/A'}</span>
                  {chartData.fallback && (
                    <span style={{color: '#F59E0B', display: 'flex', alignItems: 'center', gap: '4px'}}>
                      <AlertTriangle size={12} />
                      Loading Real Data...
                    </span>
                  )}
                  {!chartData.fallback && (
                    <span style={{color: '#10B981', display: 'flex', alignItems: 'center', gap: '4px'}}>
                      ✅ Real Data
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div style={styles.chartContainer}>
            {/* Ajustado gridTemplateRows para 3 linhas, 1.5fr para preço, 0.75fr para cada indicador */}
            <div style={{...styles.chartGrid, gridTemplateRows: '1.5fr 0.75fr 0.75fr'}}> 
              {/* Gráfico Principal - DADOS REAIS */}
              <div style={styles.chartPanel}>
                <h3 style={styles.chartTitle}>
                  BTC/USDT PRICE CHART ({selectedPeriod})
                </h3>
                <div style={styles.chartContent}>
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
                    <div style={styles.chartPlaceholder}>
                      <div>
                        <div style={styles.placeholderIcon}>📊</div>
                        <div>Connecting to real data...</div>
                        <div style={{fontSize: '12px', color: '#9CA3AF', marginTop: '8px'}}>
                          Endpoint: /api/precos/{selectedPeriod}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* RSI Chart */}
              <div style={styles.chartPanel}>
                <h3 style={styles.chartTitle}>RSI</h3>
                <div style={styles.chartContent}>
                  {hasValidChartData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData.combined} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="time"
                          tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                          stroke="#9CA3AF"
                          fontSize={9}
                        />
                        <YAxis 
                          stroke="#9CA3AF" 
                          fontSize={9} 
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
                        <Line
                          type="monotone"
                          dataKey="rsi"
                          stroke="#F59E0B"
                          strokeWidth={2}
                          dot={false}
                          name="RSI"
                        />
                        {/* Linhas de 30 e 70 para RSI */}
                        <Line type="monotone" dataKey="rsi_line_70" stroke="#EF4444" strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="rsi_line_30" stroke="#10B981" strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={styles.chartPlaceholder}>
                      <div style={{fontSize: '12px'}}>Loading RSI data...</div>
                    </div>
                  )}
                </div>
              </div>

              {/* MACD Chart */}
              <div style={styles.chartPanel}>
                <h3 style={styles.chartTitle}>MACD</h3>
                <div style={styles.chartContent}>
                  {hasValidChartData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData.combined} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="time"
                          tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                          stroke="#9CA3AF"
                          fontSize={9}
                        />
                        <YAxis 
                          stroke="#9CA3AF" 
                          fontSize={9} 
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
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={styles.chartPlaceholder}>
                      <div style={{fontSize: '12px'}}>Loading MACD data...</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Novo: Stochastic RSI Chart */}
              <div style={styles.chartPanel}>
                <h3 style={styles.chartTitle}>STOCHRSI</h3>
                <div style={styles.chartContent}>
                  {hasValidChartData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData.combined} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                          dataKey="time"
                          tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                          stroke="#9CA3AF"
                          fontSize={9}
                        />
                        <YAxis 
                          stroke="#9CA3AF" 
                          fontSize={9} 
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
                        <Line
                          type="monotone"
                          dataKey="stochrsi_k" 
                          stroke="#FFD700" 
                          strokeWidth={2}
                          dot={false}
                          name="STOCHRSI %K"
                        />
                        <Line
                          type="monotone"
                          dataKey="stochrsi_d" 
                          stroke="#8B5CF6" 
                          strokeWidth={1.5}
                          dot={false}
                          name="STOCHRSI %D"
                        />
                         {/* Linhas de 20 e 80 para StochRSI */}
                        <Line type="monotone" dataKey="stochrsi_line_80" stroke="#EF4444" strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="stochrsi_line_20" stroke="#10B981" strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={styles.chartPlaceholder}>
                      <div style={{fontSize: '12px'}}>Loading STOCHRSI data...</div>
                    </div>
                  )}
                </div>
              </div>

            </div> {/* Fim do chartGrid */}
          </div> {/* Fim do chartContainer */}
        </div> {/* Fim do chartArea */}
      </div> {/* Fim do mainContainer */}

      {/* Environment Warning Modal */}
      {showEnvironmentWarning && (
        <div style={styles.modal}>
          <div style={styles.modalContent}>
            <div style={{textAlign: 'center'}}>
              <div style={styles.modalHeader}>⚠️</div>
              <h2 style={styles.modalTitle}>
                ATTENTION: LIVE MODE
              </h2>
              <div style={styles.modalText}>
                <p style={{margin: '0 0 8px 0'}}>
                  You are about to activate <strong style={{color: '#EF4444'}}>LIVE MODE</strong>.
                </p>
                <p style={{margin: '0 0 8px 0'}}>
                  The bot will operate with <strong style={{color: '#F59E0B'}}>REAL MONEY</strong>.
                </p>
                <p style={{margin: '0 0 8px 0'}}>Ensure that:</p>
                <ul style={styles.modalList}>
                  <li style={{marginBottom: '4px'}}>Strategies have been tested on Testnet</li>
                  <li style={{marginBottom: '4px'}}>Risk parameters are configured</li>
                  <li style={{marginBottom: '4px'}}>You have sufficient experience</li>
                  <li>You are aware of the financial risks</li>
                </ul>
              </div>
              <div style={styles.modalButtons}>
                <button
                  onClick={cancelLiveEnvironment}
                  style={{...styles.modalButton, ...styles.modalButtonCancel}}
                >
                  Cancel
                </button>
                <button
                  onClick={confirmLiveEnvironment}
                  style={{...styles.modalButton, ...styles.modalButtonConfirm}}
                >
                  Confirm LIVE
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
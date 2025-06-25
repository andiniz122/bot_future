import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ComposedChart } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Target, Brain, AlertTriangle, CheckCircle, Triangle, RotateCcw, Zap, ArrowLeft } from 'lucide-react';
import * as math from 'mathjs';
import useWebSocket from '../../hooks/useWebSocket';
import { calculateRSI, calculateMACD, calculateRSIAngle, detectDivergence } from '../../utils/technicalIndicators';

const RSIMACDDashboard = ({ onBackToDashboard }) => {
  // Estados principais
  const [isLive, setIsLive] = useState(false);
  const [botActive, setBotActive] = useState(false);
  const [selectedPair, setSelectedPair] = useState('BTC/USDT');
  
  // Dados de mercado e indicadores
  const [marketData, setMarketData] = useState([]);
  const [rsiData, setRsiData] = useState([]);
  const [macdData, setMacdData] = useState([]);
  const [signals, setSignals] = useState([]);
  
  // Análise técnica avançada
  const [rsiAnalysis, setRsiAnalysis] = useState({
    current: 50,
    angle: 0,
    trend: 'neutral',
    momentum: 'stable',
    divergence: false,
    zone: 'neutral'
  });
  
  const [macdAnalysis, setMacdAnalysis] = useState({
    macd: 0,
    signal: 0,
    histogram: 0,
    trend: 'neutral',
    crossover: null,
    strength: 0
  });

  const [combinedSignal, setCombinedSignal] = useState({
    action: 'HOLD',
    confidence: 0,
    strength: 'weak',
    recommendation: '',
    riskReward: 0
  });

  // Configurações dos indicadores
  const [config, setConfig] = useState({
    rsi: {
      period: 14,
      overbought: 70,
      oversold: 30,
      detectDivergences: true,
      angleAnalysis: true
    },
    macd: {
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      detectCrossovers: true,
      strengthAnalysis: true
    },
    signals: {
      minConfidence: 60,
      minRSIAngle: 5,
      minMACDStrength: 0.05
    }
  });

  // WebSocket para dados RSI/MACD em tempo real
  const { data: rsiMacdWsData, isConnected: rsiMacdWsConnected } = useWebSocket('ws://62.72.1.122:8000/ws/rsi-macd');

  // Geração de dados em tempo real (simulação melhorada)
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const basePrice = selectedPair === 'BTC/USDT' ? 47000 : 
                       selectedPair === 'ETH/USDT' ? 2800 : 
                       selectedPair === 'BNB/USDT' ? 320 : 1.2;
      
      const volatility = selectedPair === 'BTC/USDT' ? 1000 : 
                        selectedPair === 'ETH/USDT' ? 100 : 
                        selectedPair === 'BNB/USDT' ? 20 : 0.1;

      // Simular movimento de preço mais realista
      const trend = Math.sin(now / 300000); // Tendência de longo prazo
      const noise = (Math.random() - 0.5) * volatility; // Ruído
      const newPrice = basePrice + (trend * volatility * 0.5) + noise;
      
      setMarketData(prev => {
        const newData = [...prev.slice(-99), {
          time: new Date(now).toLocaleTimeString(),
          timestamp: now,
          price: newPrice,
          volume: Math.random() * 1000000,
          high: newPrice + Math.random() * 50,
          low: newPrice - Math.random() * 50,
          open: prev.length > 0 ? prev[prev.length - 1].price : newPrice
        }];
        
        // Calcular indicadores com os novos dados
        const prices = newData.map(d => d.price);
        
        if (prices.length >= config.rsi.period + 1) {
          const rsiValues = calculateRSI(prices, config.rsi.period);
          const macdData = calculateMACD(prices, config.macd.fastPeriod, config.macd.slowPeriod, config.macd.signalPeriod);
          
          // Atualizar RSI data
          const newRsiData = newData.map((d, i) => ({
            time: d.time,
            timestamp: d.timestamp,
            rsi: rsiValues[i] || 50,
            price: d.price
          }));
          setRsiData(newRsiData);
          
          // Atualizar MACD data
          const newMacdData = newData.map((d, i) => ({
            time: d.time,
            timestamp: d.timestamp,
            macd: macdData.macd[i] || 0,
            signal: macdData.signal[i] || 0,
            histogram: macdData.histogram[i] || 0
          }));
          setMacdData(newMacdData);
          
          // Análise avançada do RSI
          if (rsiValues.length > 5) {
            const currentRSI = rsiValues[rsiValues.length - 1];
            const angle = calculateRSIAngle(rsiValues);
            const divergence = config.rsi.detectDivergences ? detectDivergence(prices, rsiValues) : false;
            
            setRsiAnalysis({
              current: currentRSI,
              angle: angle,
              trend: angle > config.signals.minRSIAngle ? 'bullish' : 
                     angle < -config.signals.minRSIAngle ? 'bearish' : 'neutral',
              momentum: Math.abs(angle) > 15 ? 'strong' : 
                       Math.abs(angle) > 5 ? 'moderate' : 'weak',
              divergence: divergence,
              zone: currentRSI > config.rsi.overbought ? 'overbought' : 
                   currentRSI < config.rsi.oversold ? 'oversold' : 'neutral'
            });
          }
          
          // Análise do MACD
          if (macdData.macd.length > 1 && macdData.signal.length > 1) {
            const currentMACD = macdData.macd[macdData.macd.length - 1];
            const currentSignal = macdData.signal[macdData.signal.length - 1];
            const currentHistogram = macdData.histogram[macdData.histogram.length - 1];
            const prevHistogram = macdData.histogram[macdData.histogram.length - 2];
            
            let crossover = null;
            if (config.macd.detectCrossovers) {
              if (prevHistogram < 0 && currentHistogram > 0) crossover = 'bullish';
              if (prevHistogram > 0 && currentHistogram < 0) crossover = 'bearish';
            }
            
            setMacdAnalysis({
              macd: currentMACD,
              signal: currentSignal,
              histogram: currentHistogram,
              trend: currentHistogram > 0 ? 'bullish' : 'bearish',
              crossover: crossover,
              strength: Math.abs(currentHistogram)
            });
          }
        }
        
        return newData;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [selectedPair, config]);

  // Análise combinada e geração de sinais
  useEffect(() => {
    if (rsiAnalysis.current && macdAnalysis.histogram !== undefined) {
      let action = 'HOLD';
      let confidence = 0;
      let strength = 'weak';
      
      // Sistema de pontuação aprimorado
      let score = 0;
      
      // RSI Score (peso: 40%)
      if (rsiAnalysis.zone === 'oversold') score += 30;
      if (rsiAnalysis.zone === 'overbought') score -= 30;
      if (rsiAnalysis.trend === 'bullish') score += 20;
      if (rsiAnalysis.trend === 'bearish') score -= 20;
      
      // RSI Angle Score (peso: 20%)
      const angleScore = Math.max(-25, Math.min(25, rsiAnalysis.angle * 0.5));
      score += angleScore;
      
      // MACD Score (peso: 30%)
      if (macdAnalysis.histogram > 0) score += 25;
      if (macdAnalysis.histogram < 0) score -= 25;
      if (macdAnalysis.crossover === 'bullish') score += 30;
      if (macdAnalysis.crossover === 'bearish') score -= 30;
      
      // MACD Strength Score (peso: 10%)
      if (macdAnalysis.strength > config.signals.minMACDStrength) {
        score += macdAnalysis.histogram > 0 ? 10 : -10;
      }
      
      // Divergence penalty/bonus
      if (rsiAnalysis.divergence === 'bearish') score -= 20;
      if (rsiAnalysis.divergence === 'bullish') score += 20;
      
      confidence = Math.min(100, Math.abs(score));
      
      if (score > 40) {
        action = 'BUY';
        strength = confidence > 70 ? 'strong' : confidence > 50 ? 'moderate' : 'weak';
      } else if (score < -40) {
        action = 'SELL';
        strength = confidence > 70 ? 'strong' : confidence > 50 ? 'moderate' : 'weak';
      }
      
      const riskReward = strength === 'strong' ? 3.0 : strength === 'moderate' ? 2.5 : 2.0;
      
      setCombinedSignal({
        action,
        confidence,
        strength,
        recommendation: `${action} - ${strength.toUpperCase()} (${confidence.toFixed(0)}%)`,
        riskReward
      });
      
      // Gerar sinal se atender aos critérios
      if (confidence >= config.signals.minConfidence && action !== 'HOLD') {
        const newSignal = {
          id: Date.now(),
          action,
          confidence,
          strength,
          rsi: rsiAnalysis.current,
          macd: macdAnalysis.histogram,
          time: new Date().toLocaleTimeString(),
          executed: botActive,
          pair: selectedPair,
          entry_price: marketData.length > 0 ? marketData[marketData.length - 1].price : 0,
          score: score
        };
        
        setSignals(prev => [newSignal, ...prev.slice(0, 19)]);
      }
    }
  }, [rsiAnalysis, macdAnalysis, botActive, config.signals, selectedPair, marketData]);

  // Atualizar com dados do WebSocket se disponíveis
  useEffect(() => {
    if (rsiMacdWsData && rsiMacdWsConnected) {
      // Atualizar análises com dados reais do backend
      if (rsiMacdWsData.rsi) {
        setRsiAnalysis(prev => ({
          ...prev,
          current: rsiMacdWsData.rsi.value || prev.current,
          zone: rsiMacdWsData.rsi.zone || prev.zone,
          trend: rsiMacdWsData.rsi.trend || prev.trend
        }));
      }
      
      if (rsiMacdWsData.macd) {
        setMacdAnalysis(prev => ({
          ...prev,
          macd: rsiMacdWsData.macd.macd || prev.macd,
          signal: rsiMacdWsData.macd.signal || prev.signal,
          histogram: rsiMacdWsData.macd.histogram || prev.histogram,
          trend: rsiMacdWsData.macd.trend || prev.trend
        }));
      }
    }
  }, [rsiMacdWsData, rsiMacdWsConnected]);

  const currentPrice = marketData.length > 0 ? marketData[marketData.length - 1].price : 47000;

  const handleConfigChange = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            RSI + MACD Analysis Dashboard
          </h1>
          <p className="text-gray-400 mt-2">Análise Técnica Avançada com Detecção de Tendências</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={onBackToDashboard}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            <ArrowLeft size={20} />
            <span>Dashboard Principal</span>
          </button>
          
          <div className="flex items-center space-x-3 bg-gray-800 rounded-lg p-3">
            <span className="text-sm">Testnet</span>
            <div 
              className={`relative w-12 h-6 rounded-full cursor-pointer transition-colors ${isLive ? 'bg-red-500' : 'bg-gray-600'}`}
              onClick={() => setIsLive(!isLive)}
            >
              <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${isLive ? 'translate-x-6' : ''}`}></div>
            </div>
            <span className="text-sm">Live</span>
          </div>
          
          <button
            onClick={() => setBotActive(!botActive)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              botActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            <Brain size={20} />
            <span>{botActive ? 'Bot Ativo' : 'Bot Inativo'}</span>
          </button>
        </div>
      </div>

      {/* Status WebSocket */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
              <span className="text-sm">{isLive ? 'LIVE TRADING' : 'TESTNET'}</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${botActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="text-sm">{botActive ? 'BOT ATIVO' : 'BOT INATIVO'}</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${rsiMacdWsConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">WebSocket: {rsiMacdWsConnected ? 'Conectado' : 'Desconectado'}</span>
            </div>
          </div>
          <div className="text-sm text-gray-400">
            Par: {selectedPair} | Preço: ${currentPrice.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Sinais Recentes */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Activity className="mr-2" size={20} />
          Sinais de Trading Recentes ({signals.length})
        </h3>
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {signals.length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              <Activity size={48} className="mx-auto mb-2 opacity-50" />
              <p>Aguardando sinais...</p>
              <p className="text-sm mt-1">Sinais serão gerados quando confiança ≥ {config.signals.minConfidence}%</p>
            </div>
          ) : (
            signals.map(signal => (
              <div key={signal.id} className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className={`w-3 h-3 rounded-full ${
                    signal.action === 'BUY' ? 'bg-green-500' : 'bg-red-500'
                  }`}></div>
                  <div>
                    <div className="font-medium text-white">
                      {signal.action} {signal.pair}
                    </div>
                    <div className="text-sm text-gray-400">
                      {signal.time} | Entry: ${signal.entry_price?.toFixed(2)} | Score: {signal.score?.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-500">
                      RSI: {signal.rsi.toFixed(1)} | MACD: {signal.macd.toFixed(4)}
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    signal.strength === 'strong' ? 'bg-blue-900 text-blue-200' :
                    signal.strength === 'moderate' ? 'bg-yellow-900 text-yellow-200' :
                    'bg-gray-600 text-gray-300'
                  }`}>
                    {signal.strength.toUpperCase()}
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-white">
                    {signal.confidence.toFixed(0)}% confiança
                  </div>
                  <div className={`text-sm ${signal.executed ? 'text-green-400' : 'text-gray-400'}`}>
                    {signal.executed ? '✅ Executado' : '⏳ Aguardando'}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Price Chart com overlays */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Preço do {selectedPair} com Sinais</h3>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={marketData.slice(-50)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151',
                borderRadius: '8px'
              }} 
            />
            <Area 
              type="monotone" 
              dataKey="price" 
              stroke="#3B82F6" 
              fill="url(#colorPrice)" 
              strokeWidth={2}
            />
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
          </ComposedChart>
        </ResponsiveContainer>
        <div className="mt-2 text-sm text-gray-400 text-center">
          Preço atual: ${currentPrice.toFixed(2)} | 
          Última atualização: {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* Configurações de Indicadores */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <RotateCcw className="mr-2" size={20} />
          Configurações dos Indicadores
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium mb-3 text-gray-300">RSI Settings</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Período</span>
                <input 
                  type="number" 
                  value={config.rsi.period}
                  onChange={(e) => handleConfigChange('rsi', 'period', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="5" max="50"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Overbought</span>
                <input 
                  type="number" 
                  value={config.rsi.overbought}
                  onChange={(e) => handleConfigChange('rsi', 'overbought', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="60" max="90"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Oversold</span>
                <input 
                  type="number" 
                  value={config.rsi.oversold}
                  onChange={(e) => handleConfigChange('rsi', 'oversold', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="10" max="40"
                />
              </div>
              <div className="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  checked={config.rsi.detectDivergences}
                  onChange={(e) => handleConfigChange('rsi', 'detectDivergences', e.target.checked)}
                  className="rounded" 
                />
                <span className="text-gray-400">Detectar Divergências</span>
              </div>
              <div className="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  checked={config.rsi.angleAnalysis}
                  onChange={(e) => handleConfigChange('rsi', 'angleAnalysis', e.target.checked)}
                  className="rounded" 
                />
                <span className="text-gray-400">Análise de Ângulo</span>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-3 text-gray-300">MACD Settings</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Fast Period</span>
                <input 
                  type="number" 
                  value={config.macd.fastPeriod}
                  onChange={(e) => handleConfigChange('macd', 'fastPeriod', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="5" max="20"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Slow Period</span>
                <input 
                  type="number" 
                  value={config.macd.slowPeriod}
                  onChange={(e) => handleConfigChange('macd', 'slowPeriod', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="20" max="50"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Signal Period</span>
                <input 
                  type="number" 
                  value={config.macd.signalPeriod}
                  onChange={(e) => handleConfigChange('macd', 'signalPeriod', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="5" max="20"
                />
              </div>
              <div className="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  checked={config.macd.detectCrossovers}
                  onChange={(e) => handleConfigChange('macd', 'detectCrossovers', e.target.checked)}
                  className="rounded" 
                />
                <span className="text-gray-400">Detectar Crossovers</span>
              </div>
              <div className="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  checked={config.macd.strengthAnalysis}
                  onChange={(e) => handleConfigChange('macd', 'strengthAnalysis', e.target.checked)}
                  className="rounded" 
                />
                <span className="text-gray-400">Análise de Força</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-3 text-gray-300">Configurações de Sinal</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Confiança Mínima (%)</span>
                <input 
                  type="number" 
                  value={config.signals.minConfidence}
                  onChange={(e) => handleConfigChange('signals', 'minConfidence', parseInt(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="30" max="95"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Ângulo RSI Mínimo (°)</span>
                <input 
                  type="number" 
                  value={config.signals.minRSIAngle}
                  onChange={(e) => handleConfigChange('signals', 'minRSIAngle', parseFloat(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="1" max="20" step="0.1"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Força MACD Mínima</span>
                <input 
                  type="number" 
                  value={config.signals.minMACDStrength}
                  onChange={(e) => handleConfigChange('signals', 'minMACDStrength', parseFloat(e.target.value))}
                  className="bg-gray-700 text-white px-2 py-1 rounded w-16 text-center"
                  min="0.01" max="0.5" step="0.01"
                />
              </div>
              
              <div className="mt-4 p-3 bg-gray-700 rounded-lg">
                <h5 className="font-medium text-gray-300 mb-2">Estatísticas de Performance</h5>
                <div className="text-sm text-gray-400 space-y-1">
                  <div>Total de Sinais: {signals.length}</div>
                  <div>Sinais BUY: {signals.filter(s => s.action === 'BUY').length}</div>
                  <div>Sinais SELL: {signals.filter(s => s.action === 'SELL').length}</div>
                  <div>Força Média: {signals.length > 0 ? (signals.filter(s => s.strength === 'strong').length / signals.length * 100).toFixed(1) : 0}% Strong</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer com estatísticas */}
      <div className="mt-8 text-center text-gray-400 space-y-2">
        <div className="flex justify-center space-x-8 text-sm">
          <span>RSI: {rsiAnalysis.current.toFixed(1)} ({rsiAnalysis.zone})</span>
          <span>MACD Histogram: {macdAnalysis.histogram.toFixed(4)}</span>
          <span>WebSocket: {rsiMacdWsConnected ? 'Conectado' : 'Desconectado'}</span>
          <span>Última atualização: {new Date().toLocaleTimeString()}</span>
        </div>
        <div className="text-xs">
          🧠 Sistema de Análise Técnica Avançada | ⚡ Detecção de Tendências em Tempo Real | 🎯 Precisão Algorítmica
        </div>
      </div>
    </div>
  );
};

export default RSIMACDDashboard;

      {/* Seletor de Par */}
      <div className="mb-6">
        <div className="flex space-x-2">
          {['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'].map(pair => (
            <button
              key={pair}
              onClick={() => setSelectedPair(pair)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedPair === pair 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {pair}
            </button>
          ))}
        </div>
      </div>

      {/* Indicadores Principais */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* RSI Analysis */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Target className="mr-2" size={20} />
            Análise RSI
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Valor Atual</span>
              <span className={`font-bold text-lg ${
                rsiAnalysis.zone === 'overbought' ? 'text-red-400' :
                rsiAnalysis.zone === 'oversold' ? 'text-green-400' : 'text-yellow-400'
              }`}>
                {rsiAnalysis.current.toFixed(1)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Ângulo da Linha</span>
              <span className={`font-semibold flex items-center ${
                rsiAnalysis.angle > 5 ? 'text-green-400' :
                rsiAnalysis.angle < -5 ? 'text-red-400' : 'text-gray-400'
              }`}>
                <Triangle size={16} className={`mr-1 ${
                  rsiAnalysis.angle > 0 ? 'rotate-0' : 'rotate-180'
                }`} />
                {Math.abs(rsiAnalysis.angle).toFixed(1)}°
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Tendência</span>
              <span className={`font-semibold ${
                rsiAnalysis.trend === 'bullish' ? 'text-green-400' :
                rsiAnalysis.trend === 'bearish' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {rsiAnalysis.trend === 'bullish' ? '↗ Alta' :
                 rsiAnalysis.trend === 'bearish' ? '↘ Baixa' : '→ Lateral'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Momentum</span>
              <span className={`font-semibold ${
                rsiAnalysis.momentum === 'strong' ? 'text-blue-400' :
                rsiAnalysis.momentum === 'moderate' ? 'text-yellow-400' : 'text-gray-400'
              }`}>
                {rsiAnalysis.momentum === 'strong' ? 'Forte' :
                 rsiAnalysis.momentum === 'moderate' ? 'Moderado' : 'Fraco'}
              </span>
            </div>
            {rsiAnalysis.divergence && (
              <div className="bg-orange-900 p-2 rounded-lg">
                <span className="text-orange-200 text-sm font-medium">
                  ⚠️ Divergência {rsiAnalysis.divergence} detectada
                </span>
              </div>
            )}
          </div>
        </div>

        {/* MACD Analysis */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Activity className="mr-2" size={20} />
            Análise MACD
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">MACD</span>
              <span className="font-semibold">{macdAnalysis.macd.toFixed(4)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Signal</span>
              <span className="font-semibold">{macdAnalysis.signal.toFixed(4)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Histograma</span>
              <span className={`font-bold ${
                macdAnalysis.histogram > 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {macdAnalysis.histogram.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Tendência</span>
              <span className={`font-semibold ${
                macdAnalysis.trend === 'bullish' ? 'text-green-400' : 'text-red-400'
              }`}>
                {macdAnalysis.trend === 'bullish' ? '↗ Bullish' : '↘ Bearish'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Força</span>
              <div className="w-20 bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    macdAnalysis.strength > 0.1 ? 'bg-green-500' : 'bg-yellow-500'
                  }`}
                  style={{ width: `${Math.min(100, macdAnalysis.strength * 1000)}%` }}
                ></div>
              </div>
            </div>
            {macdAnalysis.crossover && (
              <div className={`p-2 rounded-lg ${
                macdAnalysis.crossover === 'bullish' ? 'bg-green-900' : 'bg-red-900'
              }`}>
                <span className={`text-sm font-medium ${
                  macdAnalysis.crossover === 'bullish' ? 'text-green-200' : 'text-red-200'
                }`}>
                  ⚡ Crossover {macdAnalysis.crossover}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Combined Signal */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Zap className="mr-2" size={20} />
            Sinal Combinado
          </h3>
          <div className="space-y-3">
            <div className="text-center">
              <div className={`text-3xl font-bold mb-2 ${
                combinedSignal.action === 'BUY' ? 'text-green-400' :
                combinedSignal.action === 'SELL' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {combinedSignal.action}
              </div>
              <div className="text-gray-400">
                Confiança: {combinedSignal.confidence.toFixed(0)}%
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-500 ${
                  combinedSignal.action === 'BUY' ? 'bg-green-500' :
                  combinedSignal.action === 'SELL' ? 'bg-red-500' : 'bg-gray-500'
                }`}
                style={{ width: `${combinedSignal.confidence}%` }}
              ></div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Força</span>
              <span className={`font-semibold ${
                combinedSignal.strength === 'strong' ? 'text-blue-400' :
                combinedSignal.strength === 'moderate' ? 'text-yellow-400' : 'text-gray-400'
              }`}>
                {combinedSignal.strength.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">R/R Ratio</span>
              <span className="font-semibold text-blue-400">
                1:{combinedSignal.riskReward.toFixed(1)}
              </span>
            </div>
            <div className="bg-gray-700 p-2 rounded-lg text-center">
              <span className="text-sm text-gray-300">
                {combinedSignal.recommendation}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Gráficos */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* RSI Chart */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">RSI ({config.rsi.period}) com Análise de Ângulo</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={rsiData.slice(-50)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis domain={[0, 100]} stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }} 
              />
              <Line 
                type="monotone" 
                dataKey="rsi" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={false}
              />
              {/* Linhas de referência */}
              <Line 
                y={config.rsi.overbought}
                stroke="#EF4444"
                strokeDasharray="5 5"
                strokeWidth={1}
              />
              <Line 
                y={config.rsi.oversold}
                stroke="#10B981"
                strokeDasharray="5 5"
                strokeWidth={1}
              />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-400 text-center">
            Ângulo atual: {rsiAnalysis.angle.toFixed(1)}° | 
            Zona: {rsiAnalysis.zone} | 
            Tendência: {rsiAnalysis.trend}
          </div>
        </div>

        {/* MACD Chart */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">MACD ({config.macd.fastPeriod},{config.macd.slowPeriod},{config.macd.signalPeriod})</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={macdData.slice(-50)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }} 
              />
              <Bar 
                dataKey="histogram" 
                fill="#8B5CF6"
                opacity={0.7}
              />
              <Line 
                type="monotone" 
                dataKey="macd" 
                stroke="#3B82F6" 
                strokeWidth={2}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="signal" 
                stroke="#EF4444" 
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-400 text-center">
            MACD: {macdAnalysis.macd.toFixed(4)} | 
            Signal: {macdAnalysis.signal.toFixed(4)} | 
            Hist: {macdAnalysis.histogram.toFixed(4)}
          </div>
        </div>
      </div>
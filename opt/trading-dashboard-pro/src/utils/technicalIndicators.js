/**
 * Utilities para cálculos de indicadores técnicos
 * Implementações otimizadas para RSI, MACD e análises avançadas
 */

/**
 * Calcula o RSI (Relative Strength Index)
 * @param {Array<number>} prices - Array de preços
 * @param {number} period - Período para o cálculo (padrão: 14)
 * @returns {Array<number>} Array com valores RSI
 */
export const calculateRSI = (prices, period = 14) => {
  if (prices.length < period + 1) {
    return Array(prices.length).fill(50);
  }

  const gains = [];
  const losses = [];
  
  // Calcular ganhos e perdas
  for (let i = 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }
  
  const rsiValues = [];
  
  // Calcular média inicial
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  
  // Primeiro valor RSI
  if (avgLoss === 0) {
    rsiValues.push(100);
  } else {
    const rs = avgGain / avgLoss;
    rsiValues.push(100 - (100 / (1 + rs)));
  }
  
  // Valores subsequentes usando suavização de Wilder
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    
    if (avgLoss === 0) {
      rsiValues.push(100);
    } else {
      const rs = avgGain / avgLoss;
      rsiValues.push(100 - (100 / (1 + rs)));
    }
  }
  
  // Preencher início com valor neutro
  return Array(period).fill(50).concat(rsiValues);
};

/**
 * Calcula a EMA (Exponential Moving Average)
 * @param {Array<number>} data - Array de dados
 * @param {number} period - Período da EMA
 * @returns {Array<number>} Array com valores EMA
 */
export const calculateEMA = (data, period) => {
  if (data.length === 0) return [];
  
  const multiplier = 2 / (period + 1);
  const ema = [data[0]];
  
  for (let i = 1; i < data.length; i++) {
    ema.push((data[i] * multiplier) + (ema[i - 1] * (1 - multiplier)));
  }
  
  return ema;
};

/**
 * Calcula o MACD (Moving Average Convergence Divergence)
 * @param {Array<number>} prices - Array de preços
 * @param {number} fastPeriod - Período da EMA rápida (padrão: 12)
 * @param {number} slowPeriod - Período da EMA lenta (padrão: 26)
 * @param {number} signalPeriod - Período da linha de sinal (padrão: 9)
 * @returns {Object} Objeto com arrays macd, signal e histogram
 */
export const calculateMACD = (prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) => {
  if (prices.length < slowPeriod) {
    return { 
      macd: Array(prices.length).fill(0), 
      signal: Array(prices.length).fill(0), 
      histogram: Array(prices.length).fill(0) 
    };
  }
  
  // Calcular EMAs
  const fastEMA = calculateEMA(prices, fastPeriod);
  const slowEMA = calculateEMA(prices, slowPeriod);
  
  // Linha MACD
  const macdLine = fastEMA.map((fast, i) => fast - slowEMA[i]);
  
  // Linha de Sinal (EMA da linha MACD)
  const validMacdValues = macdLine.slice(slowPeriod - 1);
  const signalLine = calculateEMA(validMacdValues, signalPeriod);
  
  // Histograma
  const histogram = signalLine.map((signal, i) => 
    validMacdValues[i] - signal
  );
  
  // Preencher arrays com zeros no início
  const paddedSignal = Array(slowPeriod - 1 + signalPeriod - 1).fill(0).concat(signalLine);
  const paddedHistogram = Array(slowPeriod - 1 + signalPeriod - 1).fill(0).concat(histogram);
  
  return {
    macd: Array(slowPeriod - 1).fill(0).concat(validMacdValues),
    signal: paddedSignal,
    histogram: paddedHistogram
  };
};

/**
 * Calcula o ângulo da linha RSI usando regressão linear
 * @param {Array<number>} rsiValues - Array de valores RSI
 * @param {number} lookback - Número de pontos para análise (padrão: 5)
 * @returns {number} Ângulo em graus
 */
export const calculateRSIAngle = (rsiValues, lookback = 5) => {
  if (rsiValues.length < lookback) return 0;
  
  const recentRSI = rsiValues.slice(-lookback);
  const x = Array.from({length: lookback}, (_, i) => i);
  const y = recentRSI;
  
  // Regressão linear simples
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
  
  // Calcular inclinação (slope)
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  
  // Converter para ângulo em graus
  const angle = Math.atan(slope) * (180 / Math.PI);
  
  return isNaN(angle) ? 0 : angle;
};

/**
 * Detecta divergências entre preço e indicador
 * @param {Array<number>} prices - Array de preços
 * @param {Array<number>} indicator - Array do indicador (RSI, MACD, etc)
 * @param {number} lookback - Período de análise (padrão: 10)
 * @returns {string|false} Tipo de divergência ou false
 */
export const detectDivergence = (prices, indicator, lookback = 10) => {
  if (prices.length < lookback || indicator.length < lookback) return false;
  
  const recentPrices = prices.slice(-lookback);
  const recentIndicator = indicator.slice(-lookback);
  
  // Encontrar picos e vales
  const findPeaks = (data, minDistance = 3) => {
    const peaks = [];
    for (let i = minDistance; i < data.length - minDistance; i++) {
      let isPeak = true;
      for (let j = i - minDistance; j <= i + minDistance; j++) {
        if (j !== i && data[j] >= data[i]) {
          isPeak = false;
          break;
        }
      }
      if (isPeak) peaks.push({ index: i, value: data[i] });
    }
    return peaks;
  };
  
  const findValleys = (data, minDistance = 3) => {
    const valleys = [];
    for (let i = minDistance; i < data.length - minDistance; i++) {
      let isValley = true;
      for (let j = i - minDistance; j <= i + minDistance; j++) {
        if (j !== i && data[j] <= data[i]) {
          isValley = false;
          break;
        }
      }
      if (isValley) valleys.push({ index: i, value: data[i] });
    }
    return valleys;
  };
  
  const priceHighs = findPeaks(recentPrices);
  const priceLows = findValleys(recentPrices);
  const indicatorHighs = findPeaks(recentIndicator);
  const indicatorLows = findValleys(recentIndicator);
  
  // Verificar divergência bearish (preço sobe, indicador desce)
  if (priceHighs.length >= 2 && indicatorHighs.length >= 2) {
    const lastPriceHigh = priceHighs[priceHighs.length - 1];
    const prevPriceHigh = priceHighs[priceHighs.length - 2];
    const lastIndicatorHigh = indicatorHighs[indicatorHighs.length - 1];
    const prevIndicatorHigh = indicatorHighs[indicatorHighs.length - 2];
    
    if (lastPriceHigh.value > prevPriceHigh.value && 
        lastIndicatorHigh.value < prevIndicatorHigh.value) {
      return 'bearish';
    }
  }
  
  // Verificar divergência bullish (preço desce, indicador sobe)
  if (priceLows.length >= 2 && indicatorLows.length >= 2) {
    const lastPriceLow = priceLows[priceLows.length - 1];
    const prevPriceLow = priceLows[priceLows.length - 2];
    const lastIndicatorLow = indicatorLows[indicatorLows.length - 1];
    const prevIndicatorLow = indicatorLows[indicatorLows.length - 2];
    
    if (lastPriceLow.value < prevPriceLow.value && 
        lastIndicatorLow.value > prevIndicatorLow.value) {
      return 'bullish';
    }
  }
  
  return false;
};

/**
 * Calcula a força do momentum
 * @param {Array<number>} values - Array de valores
 * @param {number} period - Período para análise
 * @returns {Object} Objeto com força e direção do momentum
 */
export const calculateMomentum = (values, period = 5) => {
  if (values.length < period + 1) {
    return { strength: 0, direction: 'neutral' };
  }
  
  const recent = values.slice(-period);
  const change = recent[recent.length - 1] - recent[0];
  const avgChange = change / period;
  
  // Calcular volatilidade como medida de força
  const variations = [];
  for (let i = 1; i < recent.length; i++) {
    variations.push(Math.abs(recent[i] - recent[i - 1]));
  }
  const avgVariation = variations.reduce((a, b) => a + b, 0) / variations.length;
  
  const strength = Math.min(100, (avgVariation / Math.abs(avgChange || 1)) * 100);
  const direction = avgChange > 0.1 ? 'bullish' : avgChange < -0.1 ? 'bearish' : 'neutral';
  
  return {
    strength: isNaN(strength) ? 0 : strength,
    direction,
    change: avgChange
  };
};

/**
 * Detecta crossovers entre duas linhas
 * @param {Array<number>} line1 - Primeira linha
 * @param {Array<number>} line2 - Segunda linha
 * @param {number} lookback - Período de análise
 * @returns {Object|null} Informação sobre o crossover
 */
export const detectCrossover = (line1, line2, lookback = 3) => {
  if (line1.length < lookback || line2.length < lookback) return null;
  
  const recent1 = line1.slice(-lookback);
  const recent2 = line2.slice(-lookback);
  
  // Verificar se houve cruzamento
  const current1 = recent1[recent1.length - 1];
  const current2 = recent2[recent2.length - 1];
  const prev1 = recent1[recent1.length - 2];
  const prev2 = recent2[recent2.length - 2];
  
  if (prev1 <= prev2 && current1 > current2) {
    return {
      type: 'bullish',
      strength: Math.abs(current1 - current2),
      confidence: ((current1 - current2) / Math.max(current1, current2)) * 100
    };
  }
  
  if (prev1 >= prev2 && current1 < current2) {
    return {
      type: 'bearish',
      strength: Math.abs(current1 - current2),
      confidence: ((current2 - current1) / Math.max(current1, current2)) * 100
    };
  }
  
  return null;
};

/**
 * Calcula níveis de suporte e resistência
 * @param {Array<number>} prices - Array de preços
 * @param {number} sensitivity - Sensibilidade para detecção (padrão: 0.02 = 2%)
 * @returns {Object} Níveis de suporte e resistência
 */
export const calculateSupportResistance = (prices, sensitivity = 0.02) => {
  if (prices.length < 20) return { support: [], resistance: [] };
  
  const peaks = [];
  const valleys = [];
  
  // Encontrar picos e vales significativos
  for (let i = 5; i < prices.length - 5; i++) {
    let isPeak = true;
    let isValley = true;
    
    for (let j = i - 5; j <= i + 5; j++) {
      if (j !== i) {
        if (prices[j] >= prices[i]) isPeak = false;
        if (prices[j] <= prices[i]) isValley = false;
      }
    }
    
    if (isPeak) peaks.push(prices[i]);
    if (isValley) valleys.push(prices[i]);
  }
  
  // Agrupar níveis próximos
  const groupLevels = (levels) => {
    const grouped = [];
    const sorted = [...levels].sort((a, b) => a - b);
    
    let currentGroup = [sorted[0]];
    for (let i = 1; i < sorted.length; i++) {
      if (Math.abs(sorted[i] - sorted[i-1]) / sorted[i-1] < sensitivity) {
        currentGroup.push(sorted[i]);
      } else {
        grouped.push(currentGroup.reduce((a, b) => a + b) / currentGroup.length);
        currentGroup = [sorted[i]];
      }
    }
    if (currentGroup.length > 0) {
      grouped.push(currentGroup.reduce((a, b) => a + b) / currentGroup.length);
    }
    
    return grouped;
  };
  
  return {
    support: groupLevels(valleys),
    resistance: groupLevels(peaks)
  };
};

/**
 * Calcula o score de um sinal baseado em múltiplos indicadores
 * @param {Object} indicators - Objeto com valores dos indicadores
 * @param {Object} config - Configurações dos thresholds
 * @returns {Object} Score e análise do sinal
 */
export const calculateSignalScore = (indicators, config = {}) => {
  const {
    rsi = 50,
    rsiAngle = 0,
    macdHistogram = 0,
    macdCrossover = null,
    volume = 1,
    trend = 'neutral',
    divergence = false
  } = indicators;

  const {
    rsiOverbought = 70,
    rsiOversold = 30,
    minAngle = 5,
    minMacdStrength = 0.05
  } = config;

  let score = 0;
  const factors = [];

  // RSI Zone Analysis (peso: 30%)
  if (rsi < rsiOversold) {
    score += 30;
    factors.push({ factor: 'RSI Oversold', weight: 30, contribution: 30 });
  } else if (rsi > rsiOverbought) {
    score -= 30;
    factors.push({ factor: 'RSI Overbought', weight: 30, contribution: -30 });
  }

  // RSI Angle Analysis (peso: 25%)
  const angleScore = Math.max(-25, Math.min(25, rsiAngle));
  score += angleScore;
  factors.push({ factor: 'RSI Angle', weight: 25, contribution: angleScore });

  // MACD Analysis (peso: 35%)
  const macdScore = macdHistogram > 0 ? 20 : -20;
  score += macdScore;
  factors.push({ factor: 'MACD Histogram', weight: 20, contribution: macdScore });

  if (macdCrossover) {
    const crossoverScore = macdCrossover === 'bullish' ? 15 : -15;
    score += crossoverScore;
    factors.push({ factor: 'MACD Crossover', weight: 15, contribution: crossoverScore });
  }

  // Volume Confirmation (peso: 10%)
  if (volume > 1.5) {
    const volumeScore = score > 0 ? 10 : -10;
    score += volumeScore;
    factors.push({ factor: 'Volume Confirmation', weight: 10, contribution: volumeScore });
  }

  // Divergence Analysis
  if (divergence) {
    const divScore = divergence === 'bullish' ? 15 : -15;
    score += divScore;
    factors.push({ factor: 'Divergence', weight: 15, contribution: divScore });
  }

  const confidence = Math.min(100, Math.abs(score));
  const action = score > 40 ? 'BUY' : score < -40 ? 'SELL' : 'HOLD';
  const strength = confidence > 70 ? 'strong' : confidence > 50 ? 'moderate' : 'weak';

  return {
    score,
    confidence,
    action,
    strength,
    factors,
    recommendation: `${action} - ${strength.toUpperCase()} (${confidence.toFixed(0)}%)`
  };
};

/**
 * Valida a qualidade dos dados de entrada
 * @param {Array} data - Array de dados para validar
 * @returns {Object} Resultado da validação
 */
export const validateDataQuality = (data) => {
  if (!Array.isArray(data) || data.length === 0) {
    return { isValid: false, issues: ['Data is empty or not an array'] };
  }

  const issues = [];
  let validPoints = 0;

  for (let i = 0; i < data.length; i++) {
    const point = data[i];
    
    if (typeof point !== 'number' || isNaN(point) || !isFinite(point)) {
      issues.push(`Invalid data point at index ${i}: ${point}`);
    } else {
      validPoints++;
    }
  }

  const validPercentage = (validPoints / data.length) * 100;
  
  return {
    isValid: validPercentage >= 90,
    validPercentage,
    totalPoints: data.length,
    validPoints,
    issues: issues.slice(0, 5) // Limitar a 5 primeiros erros
  };
};

/**
 * Normaliza dados para um range específico
 * @param {Array<number>} data - Dados para normalizar
 * @param {number} min - Valor mínimo do range
 * @param {number} max - Valor máximo do range
 * @returns {Array<number>} Dados normalizados
 */
export const normalizeData = (data, min = 0, max = 100) => {
  if (data.length === 0) return [];
  
  const dataMin = Math.min(...data);
  const dataMax = Math.max(...data);
  const range = dataMax - dataMin;
  
  if (range === 0) return data.map(() => (min + max) / 2);
  
  return data.map(value => 
    min + ((value - dataMin) / range) * (max - min)
  );
};

/**
 * Calcula médias móveis múltiplas
 * @param {Array<number>} data - Dados de entrada
 * @param {Array<number>} periods - Períodos das médias
 * @returns {Object} Objeto com as médias calculadas
 */
export const calculateMultipleMA = (data, periods = [5, 10, 20, 50]) => {
  const result = {};
  
  periods.forEach(period => {
    const ma = calculateSMA(data, period);
    result[`ma${period}`] = ma;
  });
  
  return result;
};

/**
 * Calcula SMA (Simple Moving Average)
 * @param {Array<number>} data - Dados de entrada
 * @param {number} period - Período da média
 * @returns {Array<number>} Array com SMA
 */
export const calculateSMA = (data, period) => {
  if (data.length < period) return Array(data.length).fill(null);
  
  const sma = [];
  
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      sma.push(null);
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      sma.push(sum / period);
    }
  }
  
  return sma;
};

/**
 * Calcula Bollinger Bands
 * @param {Array<number>} data - Dados de entrada
 * @param {number} period - Período da média (padrão: 20)
 * @param {number} stdDev - Múltiplo do desvio padrão (padrão: 2)
 * @returns {Object} Objeto com upper, middle e lower bands
 */
export const calculateBollingerBands = (data, period = 20, stdDev = 2) => {
  const sma = calculateSMA(data, period);
  const upper = [];
  const lower = [];
  
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      upper.push(null);
      lower.push(null);
    } else {
      const slice = data.slice(i - period + 1, i + 1);
      const mean = sma[i];
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
      const std = Math.sqrt(variance);
      
      upper.push(mean + (std * stdDev));
      lower.push(mean - (std * stdDev));
    }
  }
  
  return {
    upper,
    middle: sma,
    lower
  };
};

/**
 * Calcula ATR (Average True Range)
 * @param {Array<Object>} ohlcData - Dados OHLC {high, low, close}
 * @param {number} period - Período do ATR (padrão: 14)
 * @returns {Array<number>} Array com valores ATR
 */
export const calculateATR = (ohlcData, period = 14) => {
  if (ohlcData.length < 2) return Array(ohlcData.length).fill(0);
  
  const trueRanges = [];
  
  for (let i = 1; i < ohlcData.length; i++) {
    const current = ohlcData[i];
    const previous = ohlcData[i - 1];
    
    const tr1 = current.high - current.low;
    const tr2 = Math.abs(current.high - previous.close);
    const tr3 = Math.abs(current.low - previous.close);
    
    trueRanges.push(Math.max(tr1, tr2, tr3));
  }
  
  // Calcular ATR usando EMA do True Range
  const atr = [0]; // Primeiro valor é 0
  
  if (trueRanges.length >= period) {
    // Primeira ATR é a média simples
    const firstATR = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
    atr.push(firstATR);
    
    // ATRs subsequentes usando suavização
    for (let i = period; i < trueRanges.length; i++) {
      const newATR = (atr[atr.length - 1] * (period - 1) + trueRanges[i]) / period;
      atr.push(newATR);
    }
  }
  
  // Preencher valores restantes
  while (atr.length < ohlcData.length) {
    atr.push(atr[atr.length - 1] || 0);
  }
  
  return atr;
};

/**
 * Calcula Stochastic Oscillator
 * @param {Array<Object>} ohlcData - Dados OHLC {high, low, close}
 * @param {number} period - Período do %K (padrão: 14)
 * @param {number} smoothK - Suavização do %K (padrão: 3)
 * @param {number} smoothD - Suavização do %D (padrão: 3)
 * @returns {Object} Objeto com %K e %D
 */
export const calculateStochastic = (ohlcData, period = 14, smoothK = 3, smoothD = 3) => {
  if (ohlcData.length < period) {
    return {
      k: Array(ohlcData.length).fill(50),
      d: Array(ohlcData.length).fill(50)
    };
  }
  
  const rawK = [];
  
  for (let i = period - 1; i < ohlcData.length; i++) {
    const slice = ohlcData.slice(i - period + 1, i + 1);
    const highest = Math.max(...slice.map(d => d.high));
    const lowest = Math.min(...slice.map(d => d.low));
    const current = ohlcData[i].close;
    
    const k = ((current - lowest) / (highest - lowest)) * 100;
    rawK.push(isNaN(k) ? 50 : k);
  }
  
  // Suavizar %K
  const smoothedK = calculateSMA([...Array(period - 1).fill(50), ...rawK], smoothK);
  
  // Calcular %D (SMA do %K suavizado)
  const d = calculateSMA(smoothedK, smoothD);
  
  return {
    k: smoothedK,
    d: d
  };
};

/**
 * Utilitário para performance profiling
 * @param {Function} fn - Função para medir
 * @param {Array} args - Argumentos da função
 * @returns {Object} Resultado e tempo de execução
 */
export const profileFunction = (fn, ...args) => {
  const startTime = performance.now();
  const result = fn(...args);
  const endTime = performance.now();
  
  return {
    result,
    executionTime: endTime - startTime,
    timestamp: new Date().toISOString()
  };
};

// Exportar todas as funções como default também para compatibilidade
export default {
  calculateRSI,
  calculateEMA,
  calculateMACD,
  calculateRSIAngle,
  detectDivergence,
  calculateMomentum,
  detectCrossover,
  calculateSupportResistance,
  calculateSignalScore,
  validateDataQuality,
  normalizeData,
  calculateMultipleMA,
  calculateSMA,
  calculateBollingerBands,
  calculateATR,
  calculateStochastic,
  profileFunction
};
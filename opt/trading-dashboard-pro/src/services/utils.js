// src/services/utils.js

/**
 * Formata um valor numérico para exibição de tooltip (ex: volume, preço, MACD).
 * @param {number} value O valor a ser formatado.
 * @param {string} name O nome da chave do dado (ex: 'btc_volume', 'gold_price', 'btc_macd').
 * @returns {Array<string|number>} Um array com o valor formatado e o nome.
 */
export const formatTooltip = (value, name) => {
  if (name.includes('volume')) {
    return [formatVolume(value), name.replace('_', ' ').toUpperCase()];
  }
  if (name.includes('price')) {
    return [`$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, name.replace('_', ' ').toUpperCase()];
  }
  if (name.includes('macd')) {
    return [Number(value).toFixed(4), name.replace('_', ' ').toUpperCase()];
  }
  // Adiciona formatação para RSI
  if (name.includes('rsi')) {
    return [Number(value).toFixed(2), name.replace('_', ' ').toUpperCase()];
  }
  return [value, name];
};

/**
 * Formata um valor de volume para exibição (K, M, B).
 * @param {number} volume O volume a ser formatado.
 * @returns {string} O volume formatado como string.
 */
export const formatVolume = (volume) => {
  if (volume === undefined || volume === null) return '0';
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
  return volume?.toString() || '0';
};

/**
 * Formata um tick do eixo X (timestamp) baseado no período selecionado.
 * @param {number} tickItem O valor do tick (timestamp em milissegundos).
 * @param {string} selectedPeriod O período selecionado (ex: '1d', '1mo').
 * @returns {string} A string formatada para o tick.
 */
export const formatXAxisTick = (tickItem, selectedPeriod) => {
  const date = new Date(tickItem);
  if (selectedPeriod === '1d' || selectedPeriod === '5m' || selectedPeriod === '15m' || selectedPeriod === '30m' || selectedPeriod === '1h' || selectedPeriod === '4h') {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  return date.toLocaleDateString();
};

/**
 * Gera dados de fallback para gráficos quando a API falha.
 * Mantido no frontend para agilizar o desenvolvimento, mas pode ser removido em produção.
 * @param {string} selectedPeriod O período atual.
 * @returns {Object} Dados de gráfico simulados.
 */
export const generateFallbackData = (selectedPeriod) => {
  const now = new Date();
  const points = 50;
  const interval = selectedPeriod === '1d' ? 5 : selectedPeriod === '1w' ? 60 : 240; // minutos

  const data = Array.from({ length: points }, (_, i) => {
    const time = new Date(now.getTime() - (points - i) * interval * 60000);
    const btcBase = 45000 + Math.sin(i * 0.1) * 5000 + Math.random() * 1000;
    const goldBase = 2000 + Math.sin(i * 0.08) * 100 + Math.random() * 50;
    const dxyBase = 104 + Math.sin(i * 0.06) * 2 + Math.random() * 0.5;

    // Simulação básica de RSI para fallback
    const rsiSim = 50 + Math.sin(i * 0.3) * 25; // RSI entre 25 e 75

    return {
      time: time.getTime(),
      date: time.toLocaleDateString(),
      timestamp: time.toISOString(),
      btc_price: Number(btcBase.toFixed(2)),
      gold_price: Number(goldBase.toFixed(2)),
      dxy_price: Number(dxyBase.toFixed(2)),
      btc_volume: Math.floor(Math.random() * 1000000),
      gold_volume: Math.floor(Math.random() * 500000),
      dxy_volume: Math.floor(Math.random() * 100000),
      btc_macd: Math.sin(i * 0.2) * 100,
      btc_macd_signal: Math.sin(i * 0.2 - 0.5) * 100,
      btc_macd_hist: Math.sin(i * 0.2) * 50 - Math.sin(i * 0.2 - 0.5) * 50,
      btc_rsi: Number(rsiSim.toFixed(2)) // Adicionado RSI simulado
    };
  });

  return {
    combined: data,
    period: selectedPeriod,
    dataPoints: data.length,
    lastUpdate: new Date().toISOString(),
    fallback: true
  };
};

/**
 * Processa dados do backend para o formato do gráfico recharts.
 * @param {Object} backendData Dados crus da API /api/precos/{period}.
 * @param {string} selectedPeriod O período selecionado.
 * @returns {Object} Dados formatados para o gráfico.
 */
export const processChartData = (backendData, selectedPeriod) => {
  if (!backendData || !backendData.dates || !backendData.assets) {
    console.warn('Dados do backend incompletos:', backendData);
    return generateFallbackData(selectedPeriod);
  }

  const { dates, assets } = backendData;
  
  if (!dates.length || !assets.gold?.price_data?.length) {
    console.warn('Arrays de dados vazios');
    return generateFallbackData(selectedPeriod);
  }

  const chartData = dates.map((date, index) => {
    const dataPoint = {
      time: new Date(date).getTime(),
      date: new Date(date).toLocaleDateString(),
      timestamp: date
    };

    Object.keys(assets).forEach(assetKey => {
      const asset = assets[assetKey];
      if (asset && asset.price_data && asset.price_data[index] !== undefined) {
        dataPoint[`${assetKey}_price`] = Number(asset.price_data[index]) || 0;
        dataPoint[`${assetKey}_volume`] = Number(asset.volume_data?.[index]) || 0;
        
        if (asset.open_data && asset.open_data[index] !== undefined) {
          dataPoint[`${assetKey}_open`] = Number(asset.open_data[index]) || 0;
          dataPoint[`${assetKey}_high`] = Number(asset.high_data[index]) || 0;
          dataPoint[`${assetKey}_low`] = Number(asset.low_data[index]) || 0;
        }

        // CORREÇÃO: Adicionando RSI aos dados do gráfico
        if (asset.rsi_data && asset.rsi_data[index] !== undefined) {
          dataPoint[`${assetKey}_rsi`] = Number(asset.rsi_data[index]) || 0;
        }

        if (asset.macd_data && asset.macd_data[index] !== undefined) {
          dataPoint[`${assetKey}_macd`] = Number(asset.macd_data[index]) || 0;
          dataPoint[`${assetKey}_macd_signal`] = Number(asset.macd_signal_data?.[index]) || 0;
          dataPoint[`${assetKey}_macd_hist`] = Number(asset.macd_hist_data?.[index]) || 0;
        }
      }
    });

    return dataPoint;
  }).filter(point => point.time && !isNaN(point.time));

  console.log('Dados processados para gráficos:', {
    totalPoints: chartData.length,
    samplePoint: chartData[0],
    assets: Object.keys(assets)
  });

  return {
    combined: chartData,
    period: backendData.period || selectedPeriod,
    dataPoints: chartData.length,
    lastUpdate: new Date().toISOString()
  };
};
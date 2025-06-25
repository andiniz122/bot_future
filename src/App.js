import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Area, AreaChart } from 'recharts';

function App() {
  const [apiStatus, setApiStatus] = useState('Checking...');
  const [marketData, setMarketData] = useState(null);
  const [chartData, setChartData] = useState({});
  const [selectedPeriod, setSelectedPeriod] = useState('1d');
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);

  // Função para buscar dados atuais do mercado
  const fetchCurrentData = async () => {
    try {
      const response = await fetch('http://62.72.1.122:8000/api/current');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setMarketData(data);
      setApiStatus('Connected ✅');
      setLastUpdate(new Date().toLocaleTimeString());
      setError(null);
      return data;
    } catch (error) {
      console.error('Erro ao buscar dados atuais:', error);
      setApiStatus('API Error ❌');
      setError(error.message);
      return null;
    }
  };

  // Função para buscar dados históricos
  const fetchHistoricalData = async (period) => {
    try {
      setIsLoading(true);
      const response = await fetch(`http://62.72.1.122:8000/api/precos/${period}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      
      // Processar dados para os gráficos
      const processedData = processChartData(data);
      setChartData(processedData);
      setError(null);
    } catch (error) {
      console.error('Erro ao buscar dados históricos:', error);
      setError(`Erro nos gráficos: ${error.message}`);
      // Dados de fallback para demonstração
      setChartData(generateFallbackData());
    } finally {
      setIsLoading(false);
    }
  };

  // Processar dados do backend para formato do gráfico
  const processChartData = (backendData) => {
    if (!backendData || !backendData.dates || !backendData.assets) {
      console.warn('Dados do backend incompletos:', backendData);
      return generateFallbackData();
    }

    const { dates, assets } = backendData;
    
    // Verificar se temos dados válidos
    if (!dates.length || !assets.gold?.price_data?.length) {
      console.warn('Arrays de dados vazios');
      return generateFallbackData();
    }

    // Combinar dados de todos os ativos por timestamp
    const chartData = dates.map((date, index) => {
      const dataPoint = {
        time: new Date(date).getTime(),
        date: new Date(date).toLocaleDateString(),
        timestamp: date
      };

      // Adicionar dados de cada ativo se disponível
      Object.keys(assets).forEach(assetKey => {
        const asset = assets[assetKey];
        if (asset && asset.price_data && asset.price_data[index] !== undefined) {
          dataPoint[`${assetKey}_price`] = Number(asset.price_data[index]) || 0;
          dataPoint[`${assetKey}_volume`] = Number(asset.volume_data?.[index]) || 0;
          
          // OHLC data se disponível
          if (asset.open_data && asset.open_data[index] !== undefined) {
            dataPoint[`${assetKey}_open`] = Number(asset.open_data[index]) || 0;
            dataPoint[`${assetKey}_high`] = Number(asset.high_data[index]) || 0;
            dataPoint[`${assetKey}_low`] = Number(asset.low_data[index]) || 0;
          }

          // MACD data se disponível
          if (asset.macd_data && asset.macd_data[index] !== undefined) {
            dataPoint[`${assetKey}_macd`] = Number(asset.macd_data[index]) || 0;
            dataPoint[`${assetKey}_macd_signal`] = Number(asset.macd_signal_data?.[index]) || 0;
            dataPoint[`${assetKey}_macd_hist`] = Number(asset.macd_hist_data?.[index]) || 0;
          }
        }
      });

      return dataPoint;
    }).filter(point => point.time && !isNaN(point.time)); // Filtrar pontos inválidos

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

  // Dados de fallback para quando a API falha
  const generateFallbackData = () => {
    const now = new Date();
    const points = 50;
    const interval = selectedPeriod === '1d' ? 5 : selectedPeriod === '1w' ? 60 : 240; // minutos
    
    const data = Array.from({ length: points }, (_, i) => {
      const time = new Date(now.getTime() - (points - i) * interval * 60000);
      const btcBase = 45000 + Math.sin(i * 0.1) * 5000 + Math.random() * 1000;
      const goldBase = 2000 + Math.sin(i * 0.08) * 100 + Math.random() * 50;
      const dxyBase = 104 + Math.sin(i * 0.06) * 2 + Math.random() * 0.5;

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
        btc_macd_hist: Math.sin(i * 0.2) * 50 - Math.sin(i * 0.2 - 0.5) * 50
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

  // Efeito para carregar dados iniciais
  useEffect(() => {
    const loadInitialData = async () => {
      await fetchCurrentData();
      await fetchHistoricalData(selectedPeriod);
    };
    
    loadInitialData();

    // Atualizar dados atuais a cada 30 segundos
    const currentDataInterval = setInterval(fetchCurrentData, 30000);
    
    // Atualizar dados históricos a cada 2 minutos
    const historicalDataInterval = setInterval(() => {
      fetchHistoricalData(selectedPeriod);
    }, 120000);

    return () => {
      clearInterval(currentDataInterval);
      clearInterval(historicalDataInterval);
    };
  }, []);

  // Efeito para mudança de período
  useEffect(() => {
    if (selectedPeriod) {
      fetchHistoricalData(selectedPeriod);
    }
  }, [selectedPeriod]);

  // Função para formatar tooltips
  const formatTooltip = (value, name) => {
    if (name.includes('volume')) {
      return [formatVolume(value), name.replace('_', ' ').toUpperCase()];
    }
    if (name.includes('price')) {
      return [`$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, name.replace('_', ' ').toUpperCase()];
    }
    if (name.includes('macd')) {
      return [Number(value).toFixed(4), name.replace('_', ' ').toUpperCase()];
    }
    return [value, name];
  };

  const formatVolume = (volume) => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume?.toString() || '0';
  };

  const formatXAxisTick = (tickItem) => {
    const date = new Date(tickItem);
    if (selectedPeriod === '1d') {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString();
  };

  // Componente de gráfico principal
  const PriceChart = () => {
    if (!chartData.combined || chartData.combined.length === 0) {
      return (
        <div className="chart-container">
          <div className="chart-loading">
            <p>📊 Carregando dados dos gráficos...</p>
            {error && <p className="error-text">Erro: {error}</p>}
          </div>
        </div>
      );
    }

    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>📈 Preços dos Ativos</h3>
          <p className="chart-info">
            {chartData.dataPoints} pontos de dados • 
            Período: {chartData.period} • 
            {chartData.fallback ? '⚠️ Dados simulados' : '✅ Dados reais'}
          </p>
        </div>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData.combined}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxisTick}
              stroke="#9CA3AF"
            />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              formatter={formatTooltip}
              labelFormatter={(value) => new Date(value).toLocaleString()}
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F3F4F6'
              }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="btc_price" 
              stroke="#F59E0B" 
              strokeWidth={2}
              name="BTC Price"
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="gold_price" 
              stroke="#EAB308" 
              strokeWidth={2}
              name="Gold Price"
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="dxy_price" 
              stroke="#10B981" 
              strokeWidth={2}
              name="DXY Price"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Componente de gráfico de volume
  const VolumeChart = () => {
    if (!chartData.combined || chartData.combined.length === 0) {
      return null;
    }

    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>📊 Volume de Negociação</h3>
        </div>
        
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData.combined}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxisTick}
              stroke="#9CA3AF"
            />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              formatter={formatTooltip}
              labelFormatter={(value) => new Date(value).toLocaleString()}
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F3F4F6'
              }}
            />
            <Legend />
            <Bar dataKey="btc_volume" fill="#F59E0B" name="BTC Volume" />
            <Bar dataKey="gold_volume" fill="#EAB308" name="Gold Volume" />
            <Bar dataKey="dxy_volume" fill="#10B981" name="DXY Volume" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Componente MACD Chart
  const MACDChart = () => {
    if (!chartData.combined || chartData.combined.length === 0) {
      return null;
    }

    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>📈 MACD - BTC</h3>
        </div>
        
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData.combined}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxisTick}
              stroke="#9CA3AF"
            />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              formatter={formatTooltip}
              labelFormatter={(value) => new Date(value).toLocaleString()}
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F3F4F6'
              }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="btc_macd" 
              stroke="#3B82F6" 
              strokeWidth={2}
              name="MACD"
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="btc_macd_signal" 
              stroke="#EF4444" 
              strokeWidth={2}
              name="Signal"
              dot={false}
            />
            <Bar dataKey="btc_macd_hist" fill="#8B5CF6" name="Histogram" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="container">
      <div className="header">
        <h1 className="title">📈 Trading Dashboard Pro v6.0</h1>
        <p>AI-Powered Trading Platform - Gráficos Corrigidos</p>
        <div className="status-bar">
          <div className={`status ${apiStatus.includes('✅') ? 'connected' : 'error'}`}>
            API Status: {apiStatus}
          </div>
          {lastUpdate && <span className="last-update">Última atualização: {lastUpdate}</span>}
        </div>
      </div>

      {/* Cards de Dados Atuais */}
      {marketData && (
        <div className="grid">
          <div className="card">
            <h3>🔗 Status de Conexão</h3>
            <div className="connection-info">
              <p><strong>Frontend:</strong> http://62.72.1.122:5173</p>
              <p><strong>Backend:</strong> http://62.72.1.122:8000</p>
              <p><strong>Qualidade:</strong> {marketData.data_quality?.price_data_valid ? '✅ Dados válidos' : '⚠️ Dados limitados'}</p>
              {error && <p className="error-text">⚠️ {error}</p>}
            </div>
          </div>

          <div className="card">
            <h3>💰 Preços Atuais</h3>
            <div className="prices-grid">
              {marketData.assets && Object.entries(marketData.assets).map(([key, asset]) => (
                <div key={key} className="price-item">
                  <span className="asset-name">{asset.name}:</span>
                  <span className="asset-price">${Number(asset.current_price).toLocaleString()}</span>
                  <span className={`price-change ${asset.change >= 0 ? 'positive' : 'negative'}`}>
                    {asset.change >= 0 ? '+' : ''}{asset.change_percent?.toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h3>📊 Análise Técnica</h3>
            <div className="analysis-info">
              <p><strong>Total de Alertas:</strong> {marketData.analysis?.total_alerts || 0}</p>
              <p><strong>Pontos Angulares:</strong> {marketData.analysis?.angular_history_points || 0}</p>
              <p><strong>Pontos de Preço:</strong> {marketData.analysis?.price_history_points || 0}</p>
              <p><strong>Status MACD:</strong> {marketData.analysis?.realtime_macd_status?.btc_websocket_connected ? '🟢 Conectado' : '🔴 Desconectado'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Controles de Período */}
      <div className="controls">
        <div className="period-selector">
          <h4>📅 Período dos Gráficos:</h4>
          <div className="button-group">
            {['5m', '15m', '1h', '1d', '1w', '1mo'].map(period => (
              <button
                key={period}
                className={`period-button ${selectedPeriod === period ? 'active' : ''}`}
                onClick={() => setSelectedPeriod(period)}
                disabled={isLoading}
              >
                {period.toUpperCase()}
              </button>
            ))}
          </div>
          {isLoading && <p className="loading-text">⏳ Carregando dados...</p>}
        </div>
      </div>

      {/* Gráficos */}
      <div className="charts-section">
        <PriceChart />
        <div className="charts-row">
          <VolumeChart />
          <MACDChart />
        </div>
      </div>

      <div className="footer">
        <p>🎉 Trading Dashboard Pro v6.0 - Gráficos integrados com backend real!</p>
        <p className="debug-info">
          Debug: {chartData.combined?.length || 0} pontos • 
          Período: {selectedPeriod} • 
          Backend: {apiStatus}
        </p>
      </div>

      <style jsx>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
          background-color: #0f172a;
          color: #f1f5f9;
        }

        .container {
          min-height: 100vh;
          padding: 2rem;
          max-width: 1400px;
          margin: 0 auto;
        }

        .header {
          text-align: center;
          margin-bottom: 2rem;
        }

        .title {
          font-size: 3rem;
          font-weight: bold;
          background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 1rem;
        }

        .status-bar {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 1rem;
          margin-top: 1rem;
        }

        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 2rem;
          margin: 2rem 0;
        }

        .card {
          background: #1e293b;
          border-radius: 1rem;
          padding: 2rem;
          border: 1px solid #374151;
        }

        .card h3 {
          margin-bottom: 1rem;
          color: #e2e8f0;
        }

        .status {
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          margin: 1rem 0;
          font-weight: 600;
        }

        .status.connected {
          background-color: rgba(16, 185, 129, 0.1);
          color: #10b981;
          border: 1px solid #10b981;
        }

        .status.error {
          background-color: rgba(239, 68, 68, 0.1);
          color: #ef4444;
          border: 1px solid #ef4444;
        }

        .connection-info p, .analysis-info p {
          margin: 0.5rem 0;
          font-size: 0.9rem;
        }

        .prices-grid {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .price-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem;
          background: rgba(15, 23, 42, 0.5);
          border-radius: 0.5rem;
        }

        .asset-name {
          font-weight: 500;
          color: #94a3b8;
        }

        .asset-price {
          font-weight: 700;
          color: #f1f5f9;
        }

        .price-change {
          font-size: 0.9rem;
          font-weight: 600;
          padding: 0.2rem 0.5rem;
          border-radius: 0.3rem;
        }

        .price-change.positive {
          color: #10b981;
          background: rgba(16, 185, 129, 0.1);
        }

        .price-change.negative {
          color: #ef4444;
          background: rgba(239, 68, 68, 0.1);
        }

        .controls {
          margin: 2rem 0;
          padding: 1.5rem;
          background: #1e293b;
          border-radius: 1rem;
          border: 1px solid #374151;
        }

        .period-selector h4 {
          margin-bottom: 1rem;
          color: #e2e8f0;
        }

        .button-group {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
        }

        .period-button {
          background: #374151;
          color: #f1f5f9;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.2s;
        }

        .period-button:hover {
          background: #4b5563;
        }

        .period-button.active {
          background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          color: white;
        }

        .period-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .charts-section {
          margin: 2rem 0;
        }

        .charts-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
          margin-top: 2rem;
        }

        .chart-container {
          background: #1e293b;
          border-radius: 1rem;
          padding: 1.5rem;
          border: 1px solid #374151;
          margin-bottom: 2rem;
        }

        .chart-header {
          margin-bottom: 1rem;
        }

        .chart-header h3 {
          color: #e2e8f0;
          margin-bottom: 0.5rem;
        }

        .chart-info {
          font-size: 0.9rem;
          color: #94a3b8;
        }

        .chart-loading {
          height: 300px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          color: #94a3b8;
        }

        .loading-text {
          color: #8b5cf6;
          font-size: 0.9rem;
          margin-top: 0.5rem;
        }

        .error-text {
          color: #ef4444;
          font-size: 0.9rem;
          margin-top: 0.5rem;
        }

        .last-update {
          font-size: 0.8rem;
          color: #94a3b8;
        }

        .footer {
          text-align: center;
          margin-top: 3rem;
          padding: 2rem;
          border-top: 1px solid #374151;
        }

        .debug-info {
          font-size: 0.8rem;
          color: #6b7280;
          margin-top: 0.5rem;
        }

        @media (max-width: 768px) {
          .charts-row {
            grid-template-columns: 1fr;
          }
          
          .title {
            font-size: 2rem;
          }
          
          .container {
            padding: 1rem;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
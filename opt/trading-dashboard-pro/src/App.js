import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'; // Removido Area/AreaChart pois não estão sendo usados
import useWebSocket from './hooks/useWebSocket'; //
import { 
  fetchCurrentData, 
  fetchHistoricalData, 
  fetchBotStatus, 
  startBot, // Importar funções do bot
  stopBot,  // Importar funções do bot
  fetchActivePositions, //
  fetchRecentSignals, //
  fetchEconomicCalendar, //
  fetchMarketSentiment, //
  fetchAlerts, //
  fetchBacktestRecommendations //
} from './services/api'; 
import { formatTooltip, formatVolume, formatXAxisTick, generateFallbackData, processChartData } from './services/utils'; //

// NOTE: Para não dividir demais os códigos, vamos manter os componentes aninhados
// como funções dentro do App.js, e o CSS direto no arquivo por enquanto.
// Em um projeto maior, seriam extraídos para arquivos separados em `src/components/`.

function App() {
  const [apiStatus, setApiStatus] = useState('Checking...'); //
  const [marketData, setMarketData] = useState(null); //
  const [chartData, setChartData] = useState({}); //
  const [selectedPeriod, setSelectedPeriod] = useState('1d'); //
  const [isLoading, setIsLoading] = useState(true); //
  const [lastUpdate, setLastUpdate] = useState(null); //
  const [error, setError] = useState(null); //

  // Estados para dados do Bot
  const [botStatusData, setBotStatusData] = useState(null);
  const [activePositions, setActivePositions] = useState([]);
  const [recentSignals, setRecentSignals] = useState([]);
  const [botLoading, setBotLoading] = useState(false);

  // Estados para novas seções
  const [economicCalendar, setEconomicCalendar] = useState(null);
  const [marketSentiment, setMarketSentiment] = useState(null);
  const [systemAlerts, setSystemAlerts] = useState([]);
  const [backtestRecommendations, setBacktestRecommendations] = useState([]);

  // Conexões WebSocket
  const { data: sentimentWsData, isConnected: sentimentWsConnected, error: sentimentWsError } = useWebSocket('ws://62.72.1.122:8000/ws/sentiment'); //
  const { data: ohlcvWsData, isConnected: ohlcvWsConnected, error: ohlcvWsError } = useWebSocket('ws://62.72.1.122:8000/ws/ohlcv'); //
  const { data: rsiMacdWsData, isConnected: rsiMacdWsConnected, error: rsiMacdWsError } = useWebSocket('ws://62.72.1.122:8000/ws/rsi-macd'); //

  // --- Funções de Busca de Dados ---

  const fetchAllData = useCallback(async () => {
    // Current Market Data
    try {
      const data = await fetchCurrentData(); //
      setMarketData(data); //
      setApiStatus('Connected ✅'); //
      setLastUpdate(new Date().toLocaleTimeString()); //
      setError(null); //
    } catch (err) {
      console.error('Erro ao buscar dados atuais:', err); //
      setApiStatus('API Error ❌'); //
      setError(err.message); //
    }

    // Bot Status, Positions, Signals
    try {
      const status = await fetchBotStatus(); //
      setBotStatusData(status);
      const positions = await fetchActivePositions(); //
      setActivePositions(positions?.positions || []);
      const signals = await fetchRecentSignals(); //
      setRecentSignals(signals?.signals || []);
    } catch (err) {
      console.error('Erro ao buscar dados do bot:', err);
    }

    // Economic Calendar
    try {
      const calendar = await fetchEconomicCalendar(); //
      setEconomicCalendar(calendar);
    } catch (err) {
      console.error('Erro ao buscar calendário econômico:', err);
    }

    // Market Sentiment (REST fallback, prefer WS)
    if (!sentimentWsConnected) { // Fetch via REST only if WS is not connected
      try {
        const sentiment = await fetchMarketSentiment(); //
        setMarketSentiment(sentiment);
      } catch (err) {
        console.error('Erro ao buscar sentimento de mercado (REST):', err);
      }
    }

    // Alerts
    try {
      const alerts = await fetchAlerts(); //
      setSystemAlerts(alerts?.alerts || []);
    } catch (err) {
      console.error('Erro ao buscar alertas:', err);
    }

    // Backtest Recommendations
    try {
      const recommendations = await fetchBacktestRecommendations(); //
      setBacktestRecommendations(recommendations?.recommendations || []);
    } catch (err) {
      console.error('Erro ao buscar recomendações de backtest:', err);
    }

  }, [sentimentWsConnected]); // Dependência para evitar loop infinito na chamada fetchMarketSentiment

  const fetchHistoricalChartData = useCallback(async (period) => {
    setIsLoading(true); //
    try {
      const data = await fetchHistoricalData(period); //
      const processedData = processChartData(data, period); //
      setChartData(processedData); //
      setError(null); //
    } catch (err) {
      console.error('Erro ao buscar dados históricos:', err); //
      setError(`Erro nos gráficos: ${err.message}`); //
      setChartData(generateFallbackData(period)); //
    } finally {
      setIsLoading(false); //
    }
  }, []);

  // --- Efeitos ---

  // Efeito para carregar dados iniciais e configurar intervalos de atualização
  useEffect(() => {
    // Carregar todos os dados REST iniciais
    fetchAllData();

    // Carregar dados históricos para o gráfico
    fetchHistoricalChartData(selectedPeriod);

    // Configurar intervalos de atualização
    const intervals = {
      allData: setInterval(fetchAllData, 30000), // Atualiza todos os dados REST a cada 30 segundos
      historicalCharts: setInterval(() => fetchHistoricalChartData(selectedPeriod), 120000), // Atualiza gráficos históricos a cada 2 minutos
    };

    return () => {
      // Limpar intervalos ao desmontar o componente
      Object.values(intervals).forEach(clearInterval);
    };
  }, [fetchAllData, fetchHistoricalChartData, selectedPeriod]);

  // Efeito para reagir aos dados do WebSocket de Sentimento
  useEffect(() => {
    if (sentimentWsData) {
      console.log('Dados de Sentimento WS recebidos:', sentimentWsData);
      setMarketSentiment(sentimentWsData); // Atualiza o estado de sentimento com dados WS
      setLastUpdate(new Date().toLocaleTimeString()); // Atualiza o timestamp da última atualização
    }
    if (sentimentWsError) {
      console.error('Erro no WebSocket de Sentimento:', sentimentWsError);
      setError(`Erro no WS de Sentimento: ${sentimentWsError}`);
    }
  }, [sentimentWsData, sentimentWsError]);

  // Efeito para reagir aos dados do WebSocket de OHLCV (para gráficos em tempo real)
  useEffect(() => {
    if (ohlcvWsData && ohlcvWsData.type === 'ohlcv_update') { //
      // Lógica para adicionar/atualizar o novo candle ao chartData.combined
      setChartData(prevChartData => { //
        const newCombined = [...(prevChartData.combined || [])]; //
        const newCandle = ohlcvWsData.candle; //
        const assetKey = ohlcvWsData.asset.toLowerCase(); //

        // Ajustar timestamp para milissegundos, se necessário
        const timestampMs = newCandle.timestamp * 1000;

        // Encontrar o último ponto do mesmo timestamp, se houver, para atualizar (candle incompleto)
        // ou adicionar um novo ponto (candle fechado)
        const existingIndex = newCombined.findIndex(p => p.time === timestampMs); //
        if (existingIndex !== -1) { //
          // Atualiza candle existente (ainda em formação)
          newCombined[existingIndex][`${assetKey}_price`] = newCandle.close; //
          newCombined[existingIndex][`${assetKey}_volume`] = newCandle.volume; //
          newCombined[existingIndex][`${assetKey}_open`] = newCandle.open; //
          newCombined[existingIndex][`${assetKey}_high`] = newCandle.high; //
          newCombined[existingIndex][`${assetKey}_low`] = newCandle.low; //
          // MACD e outros dados também podem vir no OHLCV update se o backend os envia
          if (ohlcvWsData.macd_data) { //
            newCombined[existingIndex][`${assetKey}_macd`] = ohlcvWsData.macd_data?.macd?.slice(-1)[0] || 0; //
            newCombined[existingIndex][`${assetKey}_macd_signal`] = ohlcvWsData.macd_data?.signal?.slice(-1)[0] || 0; //
            newCombined[existingIndex][`${assetKey}_macd_hist`] = ohlcvWsData.macd_data?.histogram?.slice(-1)[0] || 0; //
          }
        } else { //
          // Adiciona novo candle (provavelmente um candle fechado)
          const dataPoint = { //
            time: timestampMs, //
            date: new Date(timestampMs).toLocaleDateString(), //
            timestamp: new Date(timestampMs).toISOString(), //
            [`${assetKey}_price`]: newCandle.close, //
            [`${assetKey}_volume`]: newCandle.volume, //
            [`${assetKey}_open`]: newCandle.open, //
            [`${assetKey}_high`]: newCandle.high, //
            [`${assetKey}_low`]: newCandle.low, //
          };
          if (ohlcvWsData.macd_data) { //
            dataPoint[`${assetKey}_macd`] = ohlcvWsData.macd_data?.macd?.slice(-1)[0] || 0; //
            dataPoint[`${assetKey}_macd_signal`] = ohlcvWsData.macd_data?.signal?.slice(-1)[0] || 0; //
            dataPoint[`${assetKey}_macd_hist`] = ohlcvWsData.macd_data?.histogram?.slice(-1)[0] || 0; //
          }
          newCombined.push(dataPoint); //
          // Opcional: limitar o número de pontos para não sobrecarregar
          if (newCombined.length > 500) { //
              newCombined.shift(); //
          }
        }
        return { //
          ...prevChartData, //
          combined: newCombined, //
          dataPoints: newCombined.length, //
          lastUpdate: new Date().toISOString() //
        };
      });
    }
    if (ohlcvWsError) {
      console.error('Erro no WebSocket OHLCV:', ohlcvWsError);
      setError(`Erro no WS OHLCV: ${ohlcvWsError}`);
    }
  }, [ohlcvWsData, ohlcvWsError]);

  // Efeito para reagir aos dados do WebSocket de RSI+MACD
  useEffect(() => {
    if (rsiMacdWsData) {
      console.log('Dados RSI+MACD WS recebidos:', rsiMacdWsData);
      // Aqui você pode querer armazenar esses dados em um estado separado
      // ou atualizar partes de `botStatusData` ou `marketData` se aplicável
      // Por enquanto, vamos apenas logar e considerar que o card específico o consumirá diretamente
    }
    if (rsiMacdWsError) {
      console.error('Erro no WebSocket RSI+MACD:', rsiMacdWsError);
      setError(`Erro no WS RSI+MACD: ${rsiMacdWsError}`);
    }
  }, [rsiMacdWsData, rsiMacdWsError]);


  // --- Funções de Controle do Bot ---

  const handleStartBot = useCallback(async () => {
    setBotLoading(true);
    try {
      const result = await startBot(); //
      console.log('Bot iniciado:', result);
      await fetchAllData(); // Re-fetch all data to update bot status immediately
    } catch (err) {
      console.error('Falha ao iniciar o bot:', err);
      alert(`Falha ao iniciar o bot: ${err.message}`);
    } finally {
      setBotLoading(false);
    }
  }, [fetchAllData]);

  const handleStopBot = useCallback(async () => {
    setBotLoading(true);
    try {
      const result = await stopBot(); //
      console.log('Bot parado:', result);
      await fetchAllData(); // Re-fetch all data to update bot status immediately
    } catch (err) {
      console.error('Falha ao parar o bot:', err);
      alert(`Falha ao parar o bot: ${err.message}`);
    } finally {
      setBotLoading(false);
    }
  }, [fetchAllData]);

  // --- Componentes Aninhados (Mantendo o App.js coeso) ---

  const Header = () => (
    <div className="header">
      <h1 className="title">📈 Trading Dashboard Pro v6.0</h1> {/* */}
      <p>AI-Powered Trading Platform - Gráficos Corrigidos</p> {/* */}
    </div>
  );

  const StatusBar = ({ apiStatus, lastUpdate }) => (
    <div className="status-bar"> {/* */}
      <div className={`status ${apiStatus.includes('✅') ? 'connected' : 'error'}`}> {/* */}
        API Status: {apiStatus} {/* */}
      </div>
      {lastUpdate && <span className="last-update">Última atualização: {lastUpdate}</span>} {/* */}
      {/* Status dos WebSockets */}
      <div className={`status ${sentimentWsConnected ? 'connected' : 'error'}`}>
        Sentiment WS: {sentimentWsConnected ? 'Online ✅' : 'Offline ❌'}
      </div>
      <div className={`status ${ohlcvWsConnected ? 'connected' : 'error'}`}>
        OHLCV WS: {ohlcvWsConnected ? 'Online ✅' : 'Offline ❌'}
      </div>
      <div className={`status ${rsiMacdWsConnected ? 'connected' : 'error'}`}>
        RSI+MACD WS: {rsiMacdWsConnected ? 'Online ✅' : 'Offline ❌'}
      </div>
    </div>
  );

  const ConnectionStatusCard = ({ marketData, error, sentimentWsConnected, ohlcvWsConnected, rsiMacdWsConnected }) => (
    <div className="card"> {/* */}
      <h3>🔗 Status de Conexão</h3> {/* */}
      <div className="connection-info"> {/* */}
        <p><strong>Frontend:</strong> http://62.72.1.122:5173</p> {/* */}
        <p><strong>Backend:</strong> http://62.72.1.122:8000</p> {/* */}
        <p><strong>Qualidade:</strong> {marketData.data_quality?.price_data_valid ? '✅ Dados válidos' : '⚠️ Dados limitados'}</p> {/* */}
        <p><strong>WS Sentimento:</strong> {sentimentWsConnected ? 'Online ✅' : 'Offline ❌'}</p>
        <p><strong>WS OHLCV:</strong> {ohlcvWsConnected ? 'Online ✅' : 'Offline ❌'}</p>
        <p><strong>WS RSI+MACD:</strong> {rsiMacdWsConnected ? 'Online ✅' : 'Offline ❌'}</p>
        {error && <p className="error-text">⚠️ {error}</p>} {/* */}
      </div>
    </div>
  );

  const CurrentPricesCard = ({ assets }) => (
    <div className="card"> {/* */}
      <h3>💰 Preços Atuais</h3> {/* */}
      <div className="prices-grid"> {/* */}
        {assets && Object.entries(assets).map(([key, asset]) => ( //
          <div key={key} className="price-item"> {/* */}
            <span className="asset-name">{asset.name}:</span> {/* */}
            <span className="asset-price">${Number(asset.current_price).toLocaleString()}</span> {/* */}
            <span className={`price-change ${asset.change >= 0 ? 'positive' : 'negative'}`}> {/* */}
              {asset.change >= 0 ? '+' : ''}{asset.change_percent?.toFixed(2)}% {/* */}
            </span>
          </div>
        ))}
      </div>
    </div>
  );

  const TechnicalAnalysisCard = ({ analysis }) => (
    <div className="card"> {/* */}
      <h3>📊 Análise Técnica</h3> {/* */}
      <div className="analysis-info"> {/* */}
        <p><strong>Total de Alertas:</strong> {analysis?.total_alerts || 0}</p> {/* */}
        <p><strong>Pontos Angulares:</strong> {analysis?.angular_history_points || 0}</p> {/* */}
        <p><strong>Pontos de Preço:</strong> {analysis?.price_history_points || 0}</p> {/* */}
        <p><strong>Status MACD:</strong> {analysis?.realtime_macd_status?.btc_websocket_connected ? '🟢 Conectado' : '🔴 Desconectado'}</p> {/* */}
      </div>
    </div>
  );

  const PeriodSelector = ({ selectedPeriod, setSelectedPeriod, isLoading }) => (
    <div className="controls"> {/* */}
      <div className="period-selector"> {/* */}
        <h4>📅 Período dos Gráficos:</h4> {/* */}
        <div className="button-group"> {/* */}
          {['5m', '15m', '1h', '1d', '1mo', '3mo', '6mo', '1y'].map(period => ( // Expandindo períodos
            <button
              key={period}
              className={`period-button ${selectedPeriod === period ? 'active' : ''}`} //
              onClick={() => setSelectedPeriod(period)} //
              disabled={isLoading} //
            >
              {period.toUpperCase()} {/* */}
            </button>
          ))}
        </div>
        {isLoading && <p className="loading-text">⏳ Carregando dados...</p>} {/* */}
      </div>
    </div>
  );

  const PriceChartComponent = ({ chartData, selectedPeriod, error }) => { // Renomeado para evitar conflito com componente interno do App.js
    if (!chartData.combined || chartData.combined.length === 0) { //
      return (
        <div className="chart-container"> {/* */}
          <div className="chart-loading"> {/* */}
            <p>📊 Carregando dados dos gráficos...</p> {/* */}
            {error && <p className="error-text">Erro: {error}</p>} {/* */}
          </div>
        </div>
      );
    }

    return (
      <div className="chart-container"> {/* */}
        <div className="chart-header"> {/* */}
          <h3>📈 Preços dos Ativos</h3> {/* */}
          <p className="chart-info"> {/* */}
            {chartData.dataPoints} pontos de dados • {/* */}
            Período: {chartData.period} • {/* */}
            {chartData.fallback ? '⚠️ Dados simulados' : '✅ Dados reais'} {/* */}
          </p>
        </div>
        
        <ResponsiveContainer width="100%" height={300}> {/* */}
          <LineChart data={chartData.combined}> {/* */}
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" /> {/* */}
            <XAxis //
              dataKey="time" //
              type="number" //
              scale="time" //
              domain={['dataMin', 'dataMax']} //
              tickFormatter={(tick) => formatXAxisTick(tick, selectedPeriod)} //
              stroke="#9CA3AF" //
            />
            <YAxis stroke="#9CA3AF" /> {/* */}
            <Tooltip //
              formatter={formatTooltip} //
              labelFormatter={(value) => new Date(value).toLocaleString()} //
              contentStyle={{ //
                backgroundColor: '#1F2937', //
                border: '1px solid #374151', //
                borderRadius: '8px', //
                color: '#F3F4F6' //
              }}
            />
            <Legend /> {/* */}
            <Line //
              type="monotone" //
              dataKey="btc_price" //
              stroke="#F59E0B" //
              strokeWidth={2} //
              name="BTC Price" //
              dot={false} //
            />
            <Line //
              type="monotone" //
              dataKey="gold_price" //
              stroke="#EAB308" //
              strokeWidth={2} //
              name="Gold Price" //
              dot={false} //
            />
            <Line //
              type="monotone" //
              dataKey="dxy_price" //
              stroke="#10B981" //
              strokeWidth={2} //
              name="DXY Price" //
              dot={false} //
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const VolumeChartComponent = ({ chartData, selectedPeriod }) => { // Renomeado
    if (!chartData.combined || chartData.combined.length === 0) { //
      return null; //
    }

    return (
      <div className="chart-container"> {/* */}
        <div className="chart-header"> {/* */}
          <h3>📊 Volume de Negociação</h3> {/* */}
        </div>
        
        <ResponsiveContainer width="100%" height={200}> {/* */}
          <BarChart data={chartData.combined}> {/* */}
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" /> {/* */}
            <XAxis //
              dataKey="time" //
              type="number" //
              scale="time" //
              domain={['dataMin', 'dataMax']} //
              tickFormatter={(tick) => formatXAxisTick(tick, selectedPeriod)} //
              stroke="#9CA3AF" //
            />
            <YAxis stroke="#9CA3AF" /> {/* */}
            <Tooltip //
              formatter={formatTooltip} //
              labelFormatter={(value) => new Date(value).toLocaleString()} //
              contentStyle={{ //
                backgroundColor: '#1F2937', //
                border: '1px solid #374151', //
                borderRadius: '8px', //
                color: '#F3F4F6' //
              }}
            />
            <Legend /> {/* */}
            <Bar dataKey="btc_volume" fill="#F59E0B" name="BTC Volume" /> {/* */}
            <Bar dataKey="gold_volume" fill="#EAB308" name="Gold Volume" /> {/* */}
            <Bar dataKey="dxy_volume" fill="#10B981" name="DXY Volume" /> {/* */}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const MACDChartComponent = ({ chartData, selectedPeriod }) => { // Renomeado
    if (!chartData.combined || chartData.combined.length === 0) { //
      return null; //
    }

    // Filtrar apenas dados de BTC MACD
    const btcMacdData = chartData.combined.filter(d => d.btc_macd !== undefined);

    return (
      <div className="chart-container"> {/* */}
        <div className="chart-header"> {/* */}
          <h3>📈 MACD - BTC</h3> {/* */}
        </div>
        
        <ResponsiveContainer width="100%" height={250}> {/* */}
          <LineChart data={btcMacdData}> {/* */}
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" /> {/* */}
            <XAxis //
              dataKey="time" //
              type="number" //
              scale="time" //
              domain={['dataMin', 'dataMax']} //
              tickFormatter={(tick) => formatXAxisTick(tick, selectedPeriod)} //
              stroke="#9CA3AF" //
            />
            <YAxis stroke="#9CA3AF" /> {/* */}
            <Tooltip //
              formatter={formatTooltip} //
              labelFormatter={(value) => new Date(value).toLocaleString()} //
              contentStyle={{ //
                backgroundColor: '#1F2937', //
                border: '1px solid #374151', //
                borderRadius: '8px', //
                color: '#F3F4F6' //
              }}
            />
            <Legend /> {/* */}
            <Line //
              type="monotone" //
              dataKey="btc_macd" //
              stroke="#3B82F6" //
              strokeWidth={2} //
              name="MACD" //
              dot={false} //
            />
            <Line //
              type="monotone" //
              dataKey="btc_macd_signal" //
              stroke="#EF4444" //
              strokeWidth={2} //
              name="Signal" //
              dot={false} //
            />
            <Bar dataKey="btc_macd_hist" fill="#8B5CF6" name="Histogram" /> {/* */}
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const RSIMACDSignalCard = ({ rsiMacdData }) => {
    if (!rsiMacdData) {
      return (
        <div className="card">
          <h3>Signals (RSI+MACD)</h3>
          <p>Carregando dados de sinal...</p>
        </div>
      );
    }

    const { rsi, macd, combined, additional, symbol, environment } = rsiMacdData;

    return (
      <div className="card rsi-macd-card">
        <h3>Signals (RSI+MACD) - {symbol}</h3>
        <p className="signal-info">
          <strong>Environment:</strong> {environment}
        </p>
        <p className={`signal-type ${combined.signal_type === 'STRONG_BUY' || combined.signal_type === 'BUY' ? 'positive' : combined.signal_type === 'STRONG_SELL' || combined.signal_type === 'SELL' ? 'negative' : 'neutral'}`}>
          <strong>Signal:</strong> {combined.recommendation}
        </p>
        <div className="signal-metrics">
          <p><strong>RSI:</strong> {rsi.value} ({rsi.zone})</p>
          <p><strong>MACD:</strong> {macd.macd.toFixed(2)} / Signal: {macd.signal.toFixed(2)} (Hist: {macd.histogram.toFixed(2)})</p>
          <p><strong>Trend:</strong> {macd.trend.toUpperCase()}</p>
          <p><strong>Confidence:</strong> {combined.confidence}%</p>
          <p><strong>Risk/Reward:</strong> {additional.risk_reward}:1</p>
        </div>
        <p className="last-update">Last Update: {additional.last_update}</p>
      </div>
    );
  };

  const BotControlButtons = ({ botRunning, botLoading, handleStartBot, handleStopBot }) => (
    <div className="bot-controls">
      <h4>Bot Controls:</h4>
      <button 
        className="button" 
        onClick={handleStartBot} 
        disabled={botRunning || botLoading}
      >
        {botLoading && botRunning ? 'Starting...' : 'Start Bot'}
      </button>
      <button 
        className="button button-red" 
        onClick={handleStopBot} 
        disabled={!botRunning || botLoading}
      >
        {botLoading && !botRunning ? 'Stopping...' : 'Stop Bot'}
      </button>
      <p className="bot-status-text">Status: {botRunning ? 'Running ✅' : 'Stopped 🛑'}</p>
      {botLoading && <p className="loading-text">Sending command...</p>}
    </div>
  );

  const BotPerformanceCard = ({ performance }) => {
    if (!performance) return null;
    
    // Assegura que os valores numéricos são válidos
    const currentBalance = performance.current_balance ?? 0;
    const totalPnl = performance.total_pnl ?? 0;
    const roiPercentage = performance.roi_percentage ?? 0;
    const winRate = performance.win_rate ?? 0;
    const aiAccuracy = performance.ai_accuracy ?? 0;

    return (
      <div className="card bot-performance-card">
        <h3>🤖 Bot Performance ({performance.environment})</h3>
        <div className="performance-metrics">
          <p><strong>Balance:</strong> ${currentBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
          <p><strong>Total PnL:</strong> ${totalPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
          <p><strong>ROI:</strong> {roiPercentage.toFixed(2)}%</p>
          <p><strong>Win Rate:</strong> {winRate.toFixed(2)}% ({performance.winning_trades}/{performance.total_trades} wins)</p>
          <p><strong>AI Accuracy:</strong> {aiAccuracy.toFixed(2)}% ({performance.ai_correct_predictions}/{performance.ai_predictions} correct)</p>
          <p><strong>Daily PnL:</strong> ${performance.daily_pnl?.toFixed(2) || '0.00'} | Trades: {performance.daily_trades || 0}</p>
        </div>
      </div>
    );
  };

  const ActivePositionsTable = ({ positions }) => {
    if (!positions || positions.length === 0) return null;
    return (
      <div className="card active-positions-card">
        <h3>Current Positions ({positions.length})</h3>
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side</th>
              <th>Size</th>
              <th>Entry Price</th>
              <th>Current Price</th>
              <th>PnL (%)</th>
              <th>AI Pred</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos, index) => (
              <tr key={index}>
                <td>{pos.symbol}</td>
                <td className={pos.side === 'buy' ? 'positive' : 'negative'}>{pos.side.toUpperCase()}</td>
                <td>{pos.size}</td>
                <td>${pos.entry_price?.toFixed(2)}</td>
                <td>${pos.current_price?.toFixed(2)}</td>
                <td className={pos.pnl_percent >= 0 ? 'positive' : 'negative'}>{pos.pnl_percent?.toFixed(2)}%</td>
                <td>{(pos.ai_prediction * 100)?.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const RecentSignalsTable = ({ signals }) => {
    if (!signals || signals.length === 0) return null;
    return (
      <div className="card recent-signals-card">
        <h3>Recent Signals ({signals.length})</h3>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Symbol</th>
              <th>Action</th>
              <th>Confidence</th>
              <th>AI Prob</th>
              <th>Entry Price</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((sig, index) => (
              <tr key={index}>
                <td>{new Date(sig.timestamp).toLocaleTimeString()}</td>
                <td>{sig.symbol}</td>
                <td className={sig.action === 'BUY' ? 'positive' : 'negative'}>{sig.action}</td>
                <td>{(sig.final_confidence * 100)?.toFixed(1)}%</td>
                <td>{(sig.ai_probability * 100)?.toFixed(1)}%</td>
                <td>${sig.entry_price?.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  const MarketSentimentCard = ({ sentimentData, wsConnected }) => {
    if (!sentimentData) {
      return (
        <div className="card">
          <h3>🎭 Market Sentiment</h3>
          <p>Carregando dados de sentimento...</p>
          <p className={`status ${wsConnected ? 'connected' : 'error'}`}>
            WS Status: {wsConnected ? 'Online ✅' : 'Offline ❌'}
          </p>
        </div>
      );
    }

    const { btc, paxg, market_mood, fear_greed_index, volume_24h, last_update } = sentimentData;

    return (
      <div className="card sentiment-card">
        <h3>🎭 Market Sentiment</h3>
        <p className={`status ${wsConnected ? 'connected' : 'error'}`}>
          WS Status: {wsConnected ? 'Online ✅' : 'Offline ❌'}
        </p>
        <div className="sentiment-metrics">
          <p><strong>Overall Mood:</strong> <span className={market_mood.includes('GREED') ? 'positive' : market_mood.includes('FEAR') ? 'negative' : 'neutral'}>{market_mood.replace('_', ' ')}</span></p>
          <p><strong>Fear & Greed Index:</strong> {fear_greed_index}</p>
          <p><strong>BTC Buyers:</strong> {btc.buyers?.toFixed(2)}% | Sellers: {btc.sellers?.toFixed(2)}%</p>
          <p><strong>BTC Trend:</strong> <span className={btc.trend.includes('BULLISH') ? 'positive' : btc.trend.includes('BEARISH') ? 'negative' : 'neutral'}>{btc.trend.replace('_', ' ')}</span></p>
          <p><strong>PAXG Buyers:</strong> {paxg.buyers?.toFixed(2)}% | Sellers: {paxg.sellers?.toFixed(2)}%</p>
          <p><strong>PAXG Trend:</strong> <span className={paxg.trend.includes('BULLISH') ? 'positive' : paxg.trend.includes('BEARISH') ? 'negative' : 'neutral'}>{paxg.trend.replace('_', ' ')}</span></p>
          <p><strong>BTC 24h Volume:</strong> {volume_24h}</p>
        </div>
        <p className="last-update">Last Update: {new Date(last_update).toLocaleTimeString()}</p>
      </div>
    );
  };

  const EconomicCalendarCard = ({ calendarData }) => {
    if (!calendarData) {
      return (
        <div className="card">
          <h3>📅 Economic Calendar</h3>
          <p>Carregando calendário econômico...</p>
        </div>
      );
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0); // Para comparar apenas a data

    const highImpactToday = calendarData.today_events?.filter(
      event => event.importance === 'HIGH'
    ) || [];

    const nextCriticalEvent = calendarData.next_critical_event;

    return (
      <div className="card economic-calendar-card">
        <h3>📅 Economic Calendar</h3>
        <div className="calendar-summary">
          <p><strong>API Status:</strong> {calendarData.fred_api_status === 'active' ? 'Online ✅' : 'Offline ❌'}</p>
          <p><strong>Total Upcoming Events:</strong> {calendarData.total_events || 0}</p>
          <p><strong>High Impact Today:</strong> {highImpactToday.length} event(s)</p>
          {highImpactToday.length > 0 && (
            <ul>
              {highImpactToday.map((event, index) => (
                <li key={index} className="high-impact-event">
                  <strong>{event.name}</strong> ({event.time}) - {event.importance}
                </li>
              ))}
            </ul>
          )}
          {nextCriticalEvent && (
            <p>
              <strong>Next Critical:</strong> {nextCriticalEvent.name} on {new Date(nextCriticalEvent.date).toLocaleDateString()}
            </p>
          )}
        </div>
        {calendarData.upcoming_events?.length > 0 && (
          <div className="upcoming-events-list">
            <h4>Upcoming Events (Next 20):</h4>
            <ul>
              {calendarData.upcoming_events.slice(0, 5).map((event, index) => ( // Mostra top 5
                <li key={index} className={event.importance === 'HIGH' ? 'high-impact' : ''}>
                  {new Date(event.date).toLocaleDateString()} {event.time} - {event.name} ({event.importance})
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  const AlertsDisplay = ({ alerts }) => {
    if (!alerts || alerts.length === 0) {
      return (
        <div className="card alerts-card">
          <h3>🚨 System Alerts</h3>
          <p>No new alerts.</p>
        </div>
      );
    }

    return (
      <div className="card alerts-card">
        <h3>🚨 System Alerts ({alerts.length})</h3>
        <div className="alerts-list">
          {alerts.slice(0, 5).map((alert, index) => ( // Show top 5 recent alerts
            <div key={index} className={`alert-item alert-${alert.severity?.toLowerCase()}`}>
              <p className="alert-title"><strong>{alert.title}</strong></p>
              <p className="alert-message">{alert.message}</p>
              <span className="alert-timestamp">{new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
        {alerts.length > 5 && <p className="more-alerts">... and {alerts.length - 5} more alerts</p>}
      </div>
    );
  };

  const BacktestRecommendationsCard = ({ recommendations }) => {
    if (!recommendations || recommendations.length === 0) {
      return (
        <div className="card">
          <h3>💰 Backtest Recommendations</h3>
          <p>No recommendations available.</p>
        </div>
      );
    }

    return (
      <div className="card backtest-card">
        <h3>💰 Backtest Recommendations ({recommendations.length})</h3>
        <div className="recommendations-list">
          {recommendations.slice(0, 3).map((rec, index) => ( // Show top 3
            <div key={index} className="recommendation-item">
              <p><strong>{rec.pattern_name}</strong> - {rec.asset} ({rec.timeframe})</p>
              <p className={`rec-action ${rec.trade_type === 'LONG' ? 'positive' : 'negative'}`}>
                {rec.trade_type} (Conf: {(rec.confidence * 100)?.toFixed(1)}%)
              </p>
              <p>Exp. Return: {rec.expected_return} | Drawdown: {rec.max_drawdown}</p>
            </div>
          ))}
        </div>
        {recommendations.length > 3 && <p className="more-recommendations">... and {recommendations.length - 3} more</p>}
      </div>
    );
  };

  const Footer = ({ chartDataLength, selectedPeriod, apiStatus }) => (
    <div className="footer"> {/* */}
      <p>🎉 Trading Dashboard Pro v6.0 - Gráficos integrados com backend real!</p> {/* */}
      <p className="debug-info"> {/* */}
        Debug: {chartDataLength} pontos • {/* */}
        Período: {selectedPeriod} • {/* */}
        Backend: {apiStatus} {/* */}
      </p>
    </div>
  );

  // --- Renderização Principal ---

  if (isLoading && !marketData) { //
    return (
      <div className="loading"> {/* */}
        📈 Loading Trading Dashboard... {/* */}
        <p className="loading-text">Conectando ao backend e carregando dados iniciais...</p>
      </div>
    );
  }

  return (
    <div className="container">
      <Header />
      <StatusBar 
        apiStatus={apiStatus} 
        lastUpdate={lastUpdate} 
        sentimentWsConnected={sentimentWsConnected}
        ohlcvWsConnected={ohlcvWsConnected}
        rsiMacdWsConnected={rsiMacdWsConnected}
      />

      <div className="grid">
        <ConnectionStatusCard 
          marketData={marketData} 
          error={error} 
          sentimentWsConnected={sentimentWsConnected}
          ohlcvWsConnected={ohlcvWsConnected}
          rsiMacdWsConnected={rsiMacdWsConnected}
        />
        <CurrentPricesCard assets={marketData?.assets} />
        <TechnicalAnalysisCard analysis={marketData?.analysis} />
        {/* Card de Sentimento de Mercado */}
        <MarketSentimentCard sentimentData={marketSentiment} wsConnected={sentimentWsConnected} />
      </div>

      {/* Seção do Trading Bot */}
      <div className="card-section bot-section">
        <h2>🤖 AI Trading Bot Status & Controls</h2>
        <BotControlButtons 
          botRunning={botStatusData?.status === 'running'} // Verifica o status diretamente
          botLoading={botLoading} 
          handleStartBot={handleStartBot} 
          handleStopBot={handleStopBot} 
        />
        {botStatusData && (
          <div className="grid bot-details-grid">
            <BotPerformanceCard performance={botStatusData} />
            <ActivePositionsTable positions={activePositions} />
            <RecentSignalsTable signals={recentSignals} />
             {/* Card RSI+MACD para o bot, para ser exibido ao lado */}
            <RSIMACDSignalCard rsiMacdData={rsiMacdWsData} />
          </div>
        )}
      </div>

      {/* Controles de Período dos Gráficos */}
      <PeriodSelector 
        selectedPeriod={selectedPeriod} 
        setSelectedPeriod={setSelectedPeriod} 
        isLoading={isLoading} 
      />

      {/* Gráficos */}
      <div className="charts-section">
        <PriceChartComponent chartData={chartData} selectedPeriod={selectedPeriod} error={error} />
        <div className="charts-row">
          <VolumeChartComponent chartData={chartData} selectedPeriod={selectedPeriod} />
          <MACDChartComponent chartData={chartData} selectedPeriod={selectedPeriod} />
        </div>
      </div>
      
      {/* Novas Seções de Dados */}
      <div className="grid data-insights-grid">
        <EconomicCalendarCard calendarData={economicCalendar} />
        <AlertsDisplay alerts={systemAlerts} />
        <BacktestRecommendationsCard recommendations={backtestRecommendations} />
      </div>

      <Footer 
        chartDataLength={chartData.combined?.length || 0} 
        selectedPeriod={selectedPeriod} 
        apiStatus={apiStatus} 
      />

      {/* Estilos CSS embutidos (mantidos aqui conforme sua preferência) */}
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
          flex-wrap: wrap; /* Adicionado para responsividade */
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
          margin: 0; /* Ajustado para não ter margem extra no status-bar */
          font-weight: 600;
          font-size: 0.9rem;
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
          margin-bottom: 2rem; /* Adicionado para espaçamento entre PriceChart e a linha abaixo */
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

        /* Novas Regras para o Dashboard Moderno */

        .card-section {
          margin-top: 3rem;
          margin-bottom: 3rem;
        }

        .card-section h2 {
          font-size: 2rem;
          font-weight: bold;
          color: #e2e8f0;
          margin-bottom: 1.5rem;
          border-bottom: 2px solid #374151;
          padding-bottom: 0.5rem;
        }

        .bot-controls {
          background: #1e293b;
          border-radius: 1rem;
          padding: 1.5rem;
          border: 1px solid #374151;
          margin-bottom: 2rem;
          display: flex;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .bot-controls h4 {
          margin: 0;
          color: #e2e8f0;
        }

        .bot-controls .button {
          padding: 0.6rem 1.2rem;
          font-size: 0.9rem;
          min-width: 100px;
        }

        .bot-controls .button-red {
          background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        }

        .bot-controls .button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .bot-status-text {
          font-weight: 600;
          margin-left: 1rem;
          color: #94a3b8;
        }

        .bot-details-grid {
          grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }

        .bot-performance-card .performance-metrics p {
          margin: 0.5rem 0;
          font-size: 0.95rem;
        }

        .bot-performance-card .performance-metrics p strong {
          color: #e2e8f0;
          min-width: 120px;
          display: inline-block;
        }

        .active-positions-card table, .recent-signals-card table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 1rem;
          font-size: 0.85rem;
        }

        .active-positions-card th, .recent-signals-card th {
          background: #374151;
          padding: 0.75rem;
          text-align: left;
          color: #f1f5f9;
          border-bottom: 1px solid #4b5563;
        }

        .active-positions-card td, .recent-signals-card td {
          padding: 0.75rem;
          border-bottom: 1px solid #1f2937;
        }

        .active-positions-card tbody tr:hover, .recent-signals-card tbody tr:hover {
          background: #2a3547;
        }

        /* RSI+MACD Card */
        .rsi-macd-card {
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          min-height: 200px; /* Garante altura mínima para alinhamento no grid */
        }

        .rsi-macd-card h3 {
          margin-bottom: 0.75rem;
        }
        
        .rsi-macd-card .signal-info {
          font-size: 0.9rem;
          color: #94a3b8;
          margin-bottom: 0.5rem;
        }

        .rsi-macd-card .signal-type {
          font-size: 1.2rem;
          font-weight: bold;
          margin-bottom: 1rem;
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          display: inline-block;
        }

        .rsi-macd-card .signal-type.positive {
          background-color: rgba(16, 185, 129, 0.2);
          color: #10b981;
        }

        .rsi-macd-card .signal-type.negative {
          background-color: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }
        
        .rsi-macd-card .signal-type.neutral {
          background-color: rgba(255, 215, 0, 0.2);
          color: #FFD700;
        }

        .rsi-macd-card .signal-metrics p {
          margin: 0.3rem 0;
          font-size: 0.9rem;
        }

        /* Economic Calendar Card */
        .economic-calendar-card ul {
          list-style: none;
          padding: 0;
          margin: 0.75rem 0;
        }
        .economic-calendar-card li {
          font-size: 0.9rem;
          margin-bottom: 0.4rem;
          color: #cbd5e1;
        }
        .economic-calendar-card li.high-impact {
          font-weight: bold;
          color: #eab308;
        }

        /* Alerts Display Card */
        .alerts-card .alerts-list {
          margin-top: 1rem;
        }
        .alerts-card .alert-item {
          background: rgba(25, 35, 48, 0.7); /* Cor um pouco mais clara que o card */
          border-radius: 0.5rem;
          padding: 0.75rem;
          margin-bottom: 0.75rem;
          border-left: 4px solid;
        }
        .alerts-card .alert-item.alert-high {
          border-color: #ef4444;
        }
        .alerts-card .alert-item.alert-medium {
          border-color: #f59e0b;
        }
        .alerts-card .alert-item.alert-low {
          border-color: #3b82f6;
        }
        .alerts-card .alert-title {
          font-size: 1rem;
          margin-bottom: 0.25rem;
        }
        .alerts-card .alert-message {
          font-size: 0.85rem;
          color: #a0aec0;
        }
        .alerts-card .alert-timestamp {
          font-size: 0.75rem;
          color: #6b7280;
          display: block;
          text-align: right;
          margin-top: 0.5rem;
        }
        .alerts-card .more-alerts {
          font-size: 0.8rem;
          color: #94a3b8;
          text-align: center;
          margin-top: 1rem;
        }

        /* Backtest Recommendations Card */
        .backtest-card .recommendation-item {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 0.5rem;
          padding: 0.75rem;
          margin-bottom: 0.75rem;
        }
        .backtest-card .recommendation-item p {
          margin: 0.2rem 0;
          font-size: 0.9rem;
        }
        .backtest-card .rec-action {
          font-weight: bold;
        }
        .backtest-card .more-recommendations {
          font-size: 0.8rem;
          color: #94a3b8;
          text-align: center;
          margin-top: 1rem;
        }

        .data-insights-grid {
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }

        /* Responsividade */
        @media (max-width: 1024px) {
          .charts-row, .bot-details-grid, .data-insights-grid {
            grid-template-columns: 1fr; /* Coluna única em telas médias */
          }
        }

        @media (max-width: 768px) {
          .container {
            padding: 1rem;
          }
          .title {
            font-size: 2.5rem;
          }
          .status-bar {
            flex-direction: column;
            gap: 0.5rem;
          }
          .bot-controls {
            flex-direction: column;
            align-items: stretch;
          }
          .bot-controls .button {
            width: 100%;
          }
          .bot-status-text {
            margin-left: 0;
            margin-top: 0.5rem;
            text-align: center;
          }
        }

        @media (max-width: 480px) {
          .title {
            font-size: 2rem;
          }
          .card {
            padding: 1.5rem;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
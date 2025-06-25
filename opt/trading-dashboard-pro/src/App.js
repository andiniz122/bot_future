import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import useWebSocket from './hooks/useWebSocket';
import { 
  fetchCurrentData, 
  fetchHistoricalData, 
  fetchBotStatus, 
  startBot,
  stopBot,
  fetchActivePositions,
  fetchRecentSignals,
  fetchEconomicCalendar,
  fetchMarketSentiment,
  fetchAlerts,
  fetchBacktestRecommendations
} from './services/api'; 
import { formatTooltip, formatVolume, formatXAxisTick, generateFallbackData, processChartData } from './services/utils';

// Importar o novo componente RSI/MACD
import RSIMACDDashboard from './components/charts/RSIMACDDashboard';

function App() {
  const [apiStatus, setApiStatus] = useState('Checking...');
  const [marketData, setMarketData] = useState(null);
  const [chartData, setChartData] = useState({});
  const [selectedPeriod, setSelectedPeriod] = useState('1d');
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);

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

  // NOVO: Estado para controle de visualização
  const [currentView, setCurrentView] = useState('dashboard'); // 'dashboard' ou 'rsi-macd'

  // Conexões WebSocket
  const { data: sentimentWsData, isConnected: sentimentWsConnected, error: sentimentWsError } = useWebSocket('ws://62.72.1.122:8000/ws/sentiment');
  const { data: ohlcvWsData, isConnected: ohlcvWsConnected, error: ohlcvWsError } = useWebSocket('ws://62.72.1.122:8000/ws/ohlcv');
  const { data: rsiMacdWsData, isConnected: rsiMacdWsConnected, error: rsiMacdWsError } = useWebSocket('ws://62.72.1.122:8000/ws/rsi-macd');

  // --- Funções de Busca de Dados ---
  const fetchAllData = useCallback(async () => {
    try {
      const data = await fetchCurrentData();
      setMarketData(data);
      setApiStatus('Connected ✅');
      setLastUpdate(new Date().toLocaleTimeString());
      setError(null);
    } catch (err) {
      console.error('Erro ao buscar dados atuais:', err);
      setApiStatus('API Error ❌');
      setError(err.message);
    }

    try {
      const status = await fetchBotStatus();
      setBotStatusData(status);
      const positions = await fetchActivePositions();
      setActivePositions(positions?.positions || []);
      const signals = await fetchRecentSignals();
      setRecentSignals(signals?.signals || []);
    } catch (err) {
      console.error('Erro ao buscar dados do bot:', err);
    }

    try {
      const calendar = await fetchEconomicCalendar();
      setEconomicCalendar(calendar);
    } catch (err) {
      console.error('Erro ao buscar calendário econômico:', err);
    }

    if (!sentimentWsConnected) {
      try {
        const sentiment = await fetchMarketSentiment();
        setMarketSentiment(sentiment);
      } catch (err) {
        console.error('Erro ao buscar sentimento de mercado (REST):', err);
      }
    }

    try {
      const alerts = await fetchAlerts();
      setSystemAlerts(alerts?.alerts || []);
    } catch (err) {
      console.error('Erro ao buscar alertas:', err);
    }

    try {
      const recommendations = await fetchBacktestRecommendations();
      setBacktestRecommendations(recommendations?.recommendations || []);
    } catch (err) {
      console.error('Erro ao buscar recomendações de backtest:', err);
    }
  }, [sentimentWsConnected]);

  const fetchHistoricalChartData = useCallback(async (period) => {
    setIsLoading(true);
    try {
      const data = await fetchHistoricalData(period);
      const processedData = processChartData(data, period);
      setChartData(processedData);
      setError(null);
    } catch (err) {
      console.error('Erro ao buscar dados históricos:', err);
      setError(`Erro nos gráficos: ${err.message}`);
      setChartData(generateFallbackData(period));
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Efeitos para carregar dados
  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 30000);
    return () => clearInterval(interval);
  }, [fetchAllData]);

  useEffect(() => {
    fetchHistoricalChartData(selectedPeriod);
  }, [selectedPeriod, fetchHistoricalChartData]);

  // Atualizar sentimento via WebSocket
  useEffect(() => {
    if (sentimentWsData && sentimentWsConnected) {
      setMarketSentiment(sentimentWsData);
    }
  }, [sentimentWsData, sentimentWsConnected]);

  // Funções do bot
  const handleStartBot = async () => {
    setBotLoading(true);
    try {
      await startBot();
      await fetchAllData();
    } catch (error) {
      console.error('Erro ao iniciar bot:', error);
    } finally {
      setBotLoading(false);
    }
  };

  const handleStopBot = async () => {
    setBotLoading(true);
    try {
      await stopBot();
      await fetchAllData();
    } catch (error) {
      console.error('Erro ao parar bot:', error);
    } finally {
      setBotLoading(false);
    }
  };

  // --- Componentes do Dashboard ---
  const Header = () => (
    <div className="header">
      <h1 className="title">🚀 Trading Dashboard Pro</h1>
      <p className="subtitle">Sistema Avançado de Trading com RSI/MACD Analysis</p>
    </div>
  );

  const StatusBar = ({ apiStatus, lastUpdate, sentimentWsConnected, ohlcvWsConnected, rsiMacdWsConnected }) => (
    <div className="status-bar">
      <div className="status-item">
        <span className="status-label">API Status:</span>
        <span className={`status-value ${apiStatus.includes('✅') ? 'connected' : 'error'}`}>
          {apiStatus}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">Última Atualização:</span>
        <span className="status-value">{lastUpdate || 'Carregando...'}</span>
      </div>
      <div className="status-item">
        <span className="status-label">WebSockets:</span>
        <span className={`status-value ${sentimentWsConnected && ohlcvWsConnected && rsiMacdWsConnected ? 'connected' : 'error'}`}>
          RSI/MACD: {rsiMacdWsConnected ? '✅' : '❌'} | 
          Sentiment: {sentimentWsConnected ? '✅' : '❌'} | 
          OHLCV: {ohlcvWsConnected ? '✅' : '❌'}
        </span>
      </div>
    </div>
  );

  const NavigationButtons = () => (
    <div className="flex justify-center space-x-4 mb-6">
      <button
        onClick={() => setCurrentView('dashboard')}
        className={`px-6 py-3 rounded-lg font-medium transition-colors ${
          currentView === 'dashboard' 
            ? 'bg-blue-600 text-white shadow-lg' 
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        }`}
      >
        📊 Dashboard Principal
      </button>
      <button
        onClick={() => setCurrentView('rsi-macd')}
        className={`px-6 py-3 rounded-lg font-medium transition-colors ${
          currentView === 'rsi-macd' 
            ? 'bg-purple-600 text-white shadow-lg' 
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
        }`}
      >
        🎯 RSI + MACD Analysis
      </button>
    </div>
  );

  const ConnectionStatusCard = ({ marketData, error, sentimentWsConnected, ohlcvWsConnected, rsiMacdWsConnected }) => (
    <div className="card">
      <h3>🔗 Status de Conexão</h3>
      <div className="connection-info">
        <p><strong>Frontend:</strong> http://62.72.1.122:5173</p>
        <p><strong>Backend:</strong> http://62.72.1.122:8000</p>
        <p><strong>Qualidade dos Dados:</strong> {marketData?.data_quality?.price_data_valid ? '✅ Válidos' : '⚠️ Limitados'}</p>
        <div className="websocket-status">
          <div className={`ws-indicator ${rsiMacdWsConnected ? 'connected' : 'disconnected'}`}>
            RSI+MACD: {rsiMacdWsConnected ? 'Online' : 'Offline'}
          </div>
          <div className={`ws-indicator ${sentimentWsConnected ? 'connected' : 'disconnected'}`}>
            Sentiment: {sentimentWsConnected ? 'Online' : 'Offline'}
          </div>
          <div className={`ws-indicator ${ohlcvWsConnected ? 'connected' : 'disconnected'}`}>
            OHLCV: {ohlcvWsConnected ? 'Online' : 'Offline'}
          </div>
        </div>
        {error && <p className="error-text">⚠️ {error}</p>}
      </div>
    </div>
  );

  const QuickRSIMACDCard = () => (
    <div className="card rsi-macd-quick-card">
      <h3>🎯 RSI + MACD Quick View</h3>
      <div className="quick-analysis">
        {rsiMacdWsData ? (
          <div className="analysis-content">
            <div className="indicator-row">
              <span className="indicator-label">RSI:</span>
              <span className={`indicator-value ${
                rsiMacdWsData.rsi?.zone === 'overbought' ? 'overbought' :
                rsiMacdWsData.rsi?.zone === 'oversold' ? 'oversold' : 'neutral'
              }`}>
                {rsiMacdWsData.rsi?.value?.toFixed(1) || 'N/A'}
              </span>
              <span className="indicator-zone">({rsiMacdWsData.rsi?.zone || 'neutral'})</span>
            </div>
            
            <div className="indicator-row">
              <span className="indicator-label">MACD:</span>
              <span className={`indicator-value ${
                rsiMacdWsData.macd?.histogram > 0 ? 'positive' : 'negative'
              }`}>
                {rsiMacdWsData.macd?.histogram?.toFixed(4) || 'N/A'}
              </span>
              <span className="indicator-trend">({rsiMacdWsData.macd?.trend || 'neutral'})</span>
            </div>
            
            <div className="signal-recommendation">
              <span className={`signal-action ${
                rsiMacdWsData.combined?.signal_type === 'buy_signal' ? 'buy' :
                rsiMacdWsData.combined?.signal_type === 'sell_signal' ? 'sell' : 'hold'
              }`}>
                {rsiMacdWsData.combined?.signal_type === 'buy_signal' ? 'BUY' :
                 rsiMacdWsData.combined?.signal_type === 'sell_signal' ? 'SELL' : 'HOLD'}
              </span>
              {rsiMacdWsData.combined?.confidence && (
                <span className="signal-confidence">
                  ({rsiMacdWsData.combined.confidence}% conf.)
                </span>
              )}
            </div>
            
            <button
              onClick={() => setCurrentView('rsi-macd')}
              className="analysis-button"
            >
              Ver Análise Completa →
            </button>
          </div>
        ) : (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Aguardando dados RSI/MACD...</p>
            <button
              onClick={() => setCurrentView('rsi-macd')}
              className="analysis-button secondary"
            >
              Ir para Dashboard RSI/MACD
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const CurrentPricesCard = ({ assets }) => (
    <div className="card">
      <h3>💰 Preços Atuais</h3>
      <div className="prices-grid">
        {assets && Object.entries(assets).map(([key, asset]) => (
          <div key={key} className="price-item">
            <div className="asset-info">
              <span className="asset-name">{asset.name}:</span>
              <span className="asset-symbol">({asset.symbol})</span>
            </div>
            <div className="price-info">
              <span className="asset-price">${Number(asset.current_price).toLocaleString()}</span>
              <span className={`price-change ${asset.change >= 0 ? 'positive' : 'negative'}`}>
                {asset.change >= 0 ? '+' : ''}{asset.change_percent?.toFixed(2)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const BotStatusCard = () => (
    <div className="card bot-status-card">
      <h3>🤖 Trading Bot Status</h3>
      {botStatusData ? (
        <div className="bot-info">
          <div className="status-row">
            <span>Status:</span>
            <span className={`status-badge ${botStatusData.status === 'running' ? 'running' : 'stopped'}`}>
              {botStatusData.status}
            </span>
          </div>
          <div className="status-row">
            <span>Environment:</span>
            <span className="environment">{botStatusData.environment || 'Unknown'}</span>
          </div>
          <div className="status-row">
            <span>Uptime:</span>
            <span>{botStatusData.uptime || 'N/A'}</span>
          </div>
          
          {botStatusData.performance && (
            <div className="performance-summary">
              <div className="perf-item">
                <span>Balance:</span>
                <span className="balance">${botStatusData.performance.current_balance?.toLocaleString() || '0'}</span>
              </div>
              <div className="perf-item">
                <span>Total Trades:</span>
                <span>{botStatusData.performance.total_trades || 0}</span>
              </div>
              <div className="perf-item">
                <span>Win Rate:</span>
                <span className={`win-rate ${(botStatusData.performance.win_rate || 0) > 50 ? 'positive' : 'negative'}`}>
                  {(botStatusData.performance.win_rate || 0).toFixed(1)}%
                </span>
              </div>
            </div>
          )}
          
          <div className="bot-controls">
            <button
              onClick={handleStartBot}
              disabled={botLoading || botStatusData.status === 'running'}
              className="control-button start"
            >
              {botLoading ? 'Processando...' : '▶️ Start'}
            </button>
            <button
              onClick={handleStopBot}
              disabled={botLoading || botStatusData.status === 'stopped'}
              className="control-button stop"
            >
              {botLoading ? 'Processando...' : '⏸️ Stop'}
            </button>
          </div>
        </div>
      ) : (
        <div className="bot-loading">
          <p>Carregando status do bot...</p>
        </div>
      )}
    </div>
  );

  // Renderização condicional baseada na view atual
  if (currentView === 'rsi-macd') {
    return <RSIMACDDashboard onBackToDashboard={() => setCurrentView('dashboard')} />;
  }

  // Dashboard Principal
  if (isLoading && !marketData) {
    return (
      <div className="loading">
        📈 Loading Trading Dashboard...
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
      
      <NavigationButtons />

      <div className="grid main-grid">
        <ConnectionStatusCard 
          marketData={marketData} 
          error={error} 
          sentimentWsConnected={sentimentWsConnected}
          ohlcvWsConnected={ohlcvWsConnected}
          rsiMacdWsConnected={rsiMacdWsConnected}
        />
        <QuickRSIMACDCard />
        <BotStatusCard />
      </div>

      {/* Preços Atuais */}
      {marketData?.assets && (
        <CurrentPricesCard assets={marketData.assets} />
      )}

      {/* Gráficos Rápidos */}
      {chartData?.combined?.length > 0 && (
        <div className="charts-preview">
          <h2>📈 Preview dos Gráficos</h2>
          <div className="chart-grid">
            <div className="chart-container">
              <h3>Preços Históricos</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData.combined.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }} 
                  />
                  <Line type="monotone" dataKey="btc_price" stroke="#F59E0B" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="gold_price" stroke="#EF4444" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="chart-container">
              <h3>Volume</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData.combined.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }} 
                  />
                  <Bar dataKey="btc_volume" fill="#3B82F6" opacity={0.7} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div className="chart-actions">
            <button
              onClick={() => setCurrentView('rsi-macd')}
              className="chart-action-button primary"
            >
              🎯 Ver Análise RSI/MACD Completa
            </button>
          </div>
        </div>
      )}

      {/* Resumo de Alertas e Recomendações */}
      <div className="summary-grid">
        {systemAlerts?.length > 0 && (
          <div className="card">
            <h3>🚨 Alertas Recentes ({systemAlerts.length})</h3>
            <div className="alerts-summary">
              {systemAlerts.slice(0, 3).map((alert, index) => (
                <div key={index} className={`alert-item ${alert.severity?.toLowerCase()}`}>
                  <span className="alert-title">{alert.title}</span>
                  <span className="alert-time">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                </div>
              ))}
              {systemAlerts.length > 3 && (
                <p className="more-items">+{systemAlerts.length - 3} mais alertas</p>
              )}
            </div>
          </div>
        )}

        {backtestRecommendations?.length > 0 && (
          <div className="card">
            <h3>💡 Recomendações de Backtest ({backtestRecommendations.length})</h3>
            <div className="recommendations-summary">
              {backtestRecommendations.slice(0, 3).map((rec, index) => (
                <div key={index} className="recommendation-item">
                  <div className="rec-header">
                    <span className="rec-name">{rec.pattern_name}</span>
                    <span className={`rec-action ${rec.trade_type?.toLowerCase()}`}>
                      {rec.trade_type}
                    </span>
                  </div>
                  <div className="rec-details">
                    <span>Confidence: {(rec.confidence * 100).toFixed(1)}%</span>
                    <span>Asset: {rec.asset}</span>
                  </div>
                </div>
              ))}
              {backtestRecommendations.length > 3 && (
                <p className="more-items">+{backtestRecommendations.length - 3} mais recomendações</p>
              )}
            </div>
          </div>
        )}

        {economicCalendar && (
          <div className="card">
            <h3>📅 Calendário Econômico</h3>
            <div className="calendar-summary">
              {economicCalendar.events?.slice(0, 3).map((event, index) => (
                <div key={index} className="calendar-event">
                  <div className="event-date">{new Date(event.date).toLocaleDateString()}</div>
                  <div className="event-name">{event.name}</div>
                  <div className={`event-importance ${event.importance}`}>
                    {event.importance}
                  </div>
                </div>
              )) || <p>Nenhum evento encontrado</p>}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="footer">
        <p>🎉 Trading Dashboard Pro v6.1 - RSI + MACD Analysis Integrado!</p>
        <div className="footer-stats">
          <span>Backend: {apiStatus}</span>
          <span>•</span>
          <span>WebSocket RSI+MACD: {rsiMacdWsConnected ? 'Conectado' : 'Desconectado'}</span>
          <span>•</span>
          <span>Dados: {chartData.combined?.length || 0} pontos</span>
          <span>•</span>
          <span>Período: {selectedPeriod}</span>
        </div>
      </div>

      {/* Estilos CSS */}
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

        .subtitle {
          font-size: 1.2rem;
          color: #64748b;
          margin-bottom: 1rem;
        }

        .status-bar {
          display: flex;
          justify-content: space-around;
          background: #1e293b;
          padding: 1rem;
          border-radius: 0.5rem;
          margin-bottom: 2rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .status-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          min-width: 150px;
        }

        .status-label {
          font-size: 0.9rem;
          color: #64748b;
          margin-bottom: 0.25rem;
        }

        .status-value {
          font-weight: bold;
          font-size: 1rem;
        }

        .status-value.connected {
          color: #10b981;
        }

        .status-value.error {
          color: #ef4444;
        }

        .grid {
          display: grid;
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .main-grid {
          grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        }

        .card {
          background: #1e293b;
          padding: 1.5rem;
          border-radius: 0.75rem;
          border: 1px solid #334155;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .card h3 {
          margin-bottom: 1rem;
          color: #f1f5f9;
          font-size: 1.25rem;
          font-weight: 600;
        }

        .connection-info p {
          margin-bottom: 0.5rem;
          color: #cbd5e1;
        }

        .websocket-status {
          margin-top: 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .ws-indicator {
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.875rem;
          font-weight: 500;
        }

        .ws-indicator.connected {
          background-color: #065f46;
          color: #10b981;
        }

        .ws-indicator.disconnected {
          background-color: #7f1d1d;
          color: #ef4444;
        }

        .error-text {
          color: #ef4444;
          font-weight: 500;
        }

        .rsi-macd-quick-card {
          background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
          border: 1px solid #8b5cf6;
        }

        .quick-analysis {
          min-height: 200px;
        }

        .analysis-content {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .indicator-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem;
          background: #334155;
          border-radius: 0.5rem;
        }

        .indicator-label {
          font-weight: 600;
          color: #e2e8f0;
        }

        .indicator-value {
          font-weight: bold;
          font-size: 1.1rem;
        }

        .indicator-value.overbought {
          color: #ef4444;
        }

        .indicator-value.oversold {
          color: #10b981;
        }

        .indicator-value.neutral {
          color: #f59e0b;
        }

        .indicator-value.positive {
          color: #10b981;
        }

        .indicator-value.negative {
          color: #ef4444;
        }

        .indicator-zone, .indicator-trend {
          font-size: 0.875rem;
          color: #94a3b8;
        }

        .signal-recommendation {
          text-align: center;
          padding: 1rem;
          background: #475569;
          border-radius: 0.5rem;
        }

        .signal-action {
          font-size: 1.5rem;
          font-weight: bold;
          margin-right: 0.5rem;
        }

        .signal-action.buy {
          color: #10b981;
        }

        .signal-action.sell {
          color: #ef4444;
        }

        .signal-action.hold {
          color: #f59e0b;
        }

        .signal-confidence {
          font-size: 0.9rem;
          color: #94a3b8;
        }

        .analysis-button {
          width: 100%;
          padding: 0.75rem;
          background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
          color: white;
          border: none;
          border-radius: 0.5rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .analysis-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        .analysis-button.secondary {
          background: #374151;
          color: #d1d5db;
        }

        .analysis-button.secondary:hover {
          background: #4b5563;
        }

        .loading-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
          padding: 2rem 0;
        }

        .loading-spinner {
          width: 32px;
          height: 32px;
          border: 3px solid #374151;
          border-top: 3px solid #8b5cf6;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .prices-grid {
          display: grid;
          gap: 0.75rem;
        }

        .price-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem;
          background: #334155;
          border-radius: 0.5rem;
        }

        .asset-info {
          display: flex;
          flex-direction: column;
        }

        .asset-name {
          font-weight: 500;
          color: #e2e8f0;
        }

        .asset-symbol {
          font-size: 0.875rem;
          color: #94a3b8;
        }

        .price-info {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
        }

        .asset-price {
          font-weight: bold;
          color: #f1f5f9;
          font-size: 1.1rem;
        }

        .price-change.positive {
          color: #10b981;
        }

        .price-change.negative {
          color: #ef4444;
        }

        .bot-status-card {
          background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
          border: 1px solid #22d3ee;
        }

        .bot-info {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .status-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .status-badge {
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          font-size: 0.875rem;
          font-weight: 600;
        }

        .status-badge.running {
          background-color: #065f46;
          color: #10b981;
        }

        .status-badge.stopped {
          background-color: #7f1d1d;
          color: #ef4444;
        }

        .environment {
          font-weight: 600;
          color: #f59e0b;
        }

        .performance-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 0.5rem;
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #334155;
        }

        .perf-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
        }

        .balance {
          font-weight: bold;
          color: #22d3ee;
        }

        .win-rate.positive {
          color: #10b981;
        }

        .win-rate.negative {
          color: #ef4444;
        }

        .bot-controls {
          display: flex;
          gap: 0.5rem;
          margin-top: 1rem;
        }

        .control-button {
          flex: 1;
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 0.5rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .control-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .control-button.start {
          background-color: #059669;
          color: white;
        }

        .control-button.start:hover:not(:disabled) {
          background-color: #047857;
        }

        .control-button.stop {
          background-color: #dc2626;
          color: white;
        }

        .control-button.stop:hover:not(:disabled) {
          background-color: #b91c1c;
        }

        .charts-preview {
          background: #1e293b;
          padding: 2rem;
          border-radius: 0.75rem;
          margin-bottom: 2rem;
          border: 1px solid #334155;
        }

        .charts-preview h2 {
          margin-bottom: 1.5rem;
          color: #f1f5f9;
          text-align: center;
        }

        .chart-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 2rem;
          margin-bottom: 2rem;
        }

        .chart-container {
          background: #334155;
          padding: 1rem;
          border-radius: 0.5rem;
        }

        .chart-container h3 {
          margin-bottom: 1rem;
          color: #e2e8f0;
          font-size: 1.1rem;
        }

        .chart-actions {
          text-align: center;
        }

        .chart-action-button {
          padding: 1rem 2rem;
          border: none;
          border-radius: 0.5rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .chart-action-button.primary {
          background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
          color: white;
        }

        .chart-action-button.primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        .summary-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .alerts-summary {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .alert-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.875rem;
        }

        .alert-item.high {
          background-color: #7f1d1d;
          color: #fecaca;
        }

        .alert-item.medium {
          background-color: #92400e;
          color: #fed7aa;
        }

        .alert-item.low {
          background-color: #365314;
          color: #d9f99d;
        }

        .alert-title {
          font-weight: 500;
        }

        .alert-time {
          font-size: 0.75rem;
          opacity: 0.8;
        }

        .recommendations-summary {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .recommendation-item {
          padding: 0.75rem;
          background: #334155;
          border-radius: 0.5rem;
        }

        .rec-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .rec-name {
          font-weight: 600;
          color: #e2e8f0;
        }

        .rec-action {
          padding: 0.125rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .rec-action.long {
          background-color: #065f46;
          color: #10b981;
        }

        .rec-action.short {
          background-color: #7f1d1d;
          color: #ef4444;
        }

        .rec-details {
          display: flex;
          justify-content: space-between;
          font-size: 0.875rem;
          color: #94a3b8;
        }

        .calendar-summary {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .calendar-event {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem;
          background: #334155;
          border-radius: 0.25rem;
          font-size: 0.875rem;
        }

        .event-date {
          font-weight: 500;
          color: #cbd5e1;
        }

        .event-name {
          flex: 1;
          margin: 0 0.5rem;
          color: #e2e8f0;
        }

        .event-importance {
          padding: 0.125rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .event-importance.high {
          background-color: #7f1d1d;
          color: #ef4444;
        }

        .event-importance.medium {
          background-color: #92400e;
          color: #f59e0b;
        }

        .event-importance.low {
          background-color: #365314;
          color: #84cc16;
        }

        .more-items {
          text-align: center;
          color: #64748b;
          font-style: italic;
          margin-top: 0.5rem;
        }

        .loading {
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          min-height: 50vh;
          font-size: 1.5rem;
          color: #64748b;
        }

        .loading-text {
          margin-top: 1rem;
          font-size: 1rem;
          color: #94a3b8;
        }

        .bot-loading {
          text-align: center;
          padding: 2rem;
          color: #64748b;
        }

        .footer {
          text-align: center;
          margin-top: 3rem;
          padding: 2rem;
          color: #64748b;
          border-top: 1px solid #334155;
        }

        .footer-stats {
          margin-top: 0.5rem;
          font-size: 0.9rem;
          color: #475569;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        @media (max-width: 768px) {
          .container {
            padding: 1rem;
          }
          
          .title {
            font-size: 2rem;
          }
          
          .status-bar {
            flex-direction: column;
            gap: 1rem;
          }
          
          .chart-grid {
            grid-template-columns: 1fr;
          }
          
          .main-grid {
            grid-template-columns: 1fr;
          }
          
          .footer-stats {
            flex-direction: column;
            gap: 0.5rem;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
// src/components/ConnectionMonitor.js
import React, { useState, useEffect } from 'react';

const ConnectionMonitor = ({ 
    sentimentWs, 
    ohlcvWs, 
    rsiMacdWs 
}) => {
    const [backendHealth, setBackendHealth] = useState(null);
    const [isChecking, setIsChecking] = useState(false);

    const checkBackendHealth = async () => {
        setIsChecking(true);
        try {
            const response = await fetch('http://62.72.1.122:8000/health');
            const health = await response.json();
            setBackendHealth(health);
        } catch (error) {
            console.error('❌ Erro ao verificar saúde do backend:', error);
            setBackendHealth({ status: 'unhealthy', error: error.message });
        } finally {
            setIsChecking(false);
        }
    };

    useEffect(() => {
        checkBackendHealth();
        const interval = setInterval(checkBackendHealth, 30000); // Check every 30s
        return () => clearInterval(interval);
    }, []);

    const reconnectAllWebSockets = () => {
        sentimentWs.reconnect();
        ohlcvWs.reconnect();
        rsiMacdWs.reconnect();
    };

    const getConnectionStatus = (ws) => {
        if (ws.isConnected) {
            return { status: 'connected', color: 'green', text: 'Online ✅' };
        } else if (ws.reconnectAttempts > 0) {
            return { status: 'reconnecting', color: 'orange', text: `Reconectando... (${ws.reconnectAttempts})` };
        } else {
            return { status: 'disconnected', color: 'red', text: 'Offline ❌' };
        }
    };

    return (
        <div className="connection-monitor">
            <h3>🔗 Monitor de Conexões</h3>
            
            {/* Backend Health */}
            <div className="health-section">
                <h4>Backend Status</h4>
                <div style={{ 
                    color: backendHealth?.status === 'healthy' ? 'green' : 'red',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px'
                }}>
                    <span>{backendHealth?.status === 'healthy' ? '✅' : '❌'} Backend</span>
                    {isChecking && <span>🔄</span>}
                    <button onClick={checkBackendHealth} disabled={isChecking}>
                        Verificar
                    </button>
                </div>
                {backendHealth?.websockets && (
                    <div className="websocket-stats">
                        <small>
                            Sentiment: {backendHealth.websockets.sentiment_connections} | 
                            OHLCV: {backendHealth.websockets.ohlcv_connections} | 
                            RSI/MACD: {backendHealth.websockets.rsi_macd_connections}
                        </small>
                    </div>
                )}
            </div>

            {/* WebSocket Status */}
            <div className="websocket-section">
                <h4>WebSocket Status</h4>
                <div className="ws-status-grid">
                    {[
                        { name: 'Sentiment', ws: sentimentWs },
                        { name: 'OHLCV', ws: ohlcvWs },
                        { name: 'RSI/MACD', ws: rsiMacdWs }
                    ].map(({ name, ws }) => {
                        const status = getConnectionStatus(ws);
                        return (
                            <div key={name} className="ws-status-item">
                                <span style={{ color: status.color }}>
                                    {name}: {status.text}
                                </span>
                                {ws.error && (
                                    <small style={{ color: 'red', display: 'block' }}>
                                        {ws.error}
                                    </small>
                                )}
                            </div>
                        );
                    })}
                </div>
                
                <button 
                    onClick={reconnectAllWebSockets}
                    className="reconnect-button"
                    style={{
                        marginTop: '10px',
                        padding: '8px 16px',
                        backgroundColor: '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                    }}
                >
                    🔄 Reconectar Todos WebSockets
                </button>
            </div>

            <style jsx>{`
                .connection-monitor {
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                }
                
                .health-section, .websocket-section {
                    margin-bottom: 15px;
                }
                
                .ws-status-grid {
                    display: grid;
                    gap: 8px;
                }
                
                .ws-status-item {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                
                .websocket-stats {
                    margin-top: 5px;
                    font-family: monospace;
                    background: #e9ecef;
                    padding: 5px;
                    border-radius: 4px;
                }
                
                .reconnect-button:hover {
                    background-color: #0056b3;
                }
            `}</style>
        </div>
    );
};

export default ConnectionMonitor;
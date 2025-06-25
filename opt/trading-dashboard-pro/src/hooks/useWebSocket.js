import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook personalizado para gerenciar conexões WebSocket
 * @param {string} url - URL do WebSocket
 * @param {Object} options - Opções de configuração
 * @returns {Object} Estado e métodos do WebSocket
 */
const useWebSocket = (url, options = {}) => {
  const {
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    shouldReconnect = true,
    onOpen = () => {},
    onClose = () => {},
    onError = () => {},
    onMessage = () => {},
    protocols = []
  } = options;

  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [connectionState, setConnectionState] = useState('DISCONNECTED'); // CONNECTING, CONNECTED, DISCONNECTED, ERROR

  const ws = useRef(null);
  const reconnectTimer = useRef(null);
  const mountedRef = useRef(true);

  // Limpar timer ao desmontar
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
    };
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current || !url) return;

    try {
      setConnectionState('CONNECTING');
      setError(null);

      // Fechar conexão existente se houver
      if (ws.current) {
        ws.current.close();
      }

      // Criar nova conexão WebSocket
      ws.current = new WebSocket(url, protocols);

      ws.current.onopen = (event) => {
        if (!mountedRef.current) return;
        
        console.log(`✅ WebSocket conectado: ${url}`);
        setIsConnected(true);
        setConnectionState('CONNECTED');
        setError(null);
        setReconnectAttempts(0);
        onOpen(event);
      };

      ws.current.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const parsedData = JSON.parse(event.data);
          setData(parsedData);
          onMessage(parsedData, event);
        } catch (err) {
          console.warn('Erro ao parsear dados WebSocket:', err);
          setData(event.data); // Fallback para dados não-JSON
          onMessage(event.data, event);
        }
      };

      ws.current.onclose = (event) => {
        if (!mountedRef.current) return;

        console.log(`🔌 WebSocket fechado: ${url}`, event.code, event.reason);
        setIsConnected(false);
        setConnectionState('DISCONNECTED');
        onClose(event);

        // Tentar reconectar se necessário
        if (shouldReconnect && reconnectAttempts < maxReconnectAttempts && event.code !== 1000) {
          scheduleReconnect();
        }
      };

      ws.current.onerror = (event) => {
        if (!mountedRef.current) return;

        console.error(`❌ Erro WebSocket: ${url}`, event);
        setError(`Erro de conexão WebSocket: ${event.type}`);
        setConnectionState('ERROR');
        onError(event);
      };

    } catch (err) {
      console.error(`❌ Erro ao criar WebSocket: ${url}`, err);
      setError(`Erro ao criar WebSocket: ${err.message}`);
      setConnectionState('ERROR');
      
      if (shouldReconnect && reconnectAttempts < maxReconnectAttempts) {
        scheduleReconnect();
      }
    }
  }, [url, protocols, shouldReconnect, reconnectAttempts, maxReconnectAttempts, onOpen, onMessage, onClose, onError]);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;

    setReconnectAttempts(prev => prev + 1);
    
    reconnectTimer.current = setTimeout(() => {
      if (mountedRef.current && reconnectAttempts < maxReconnectAttempts) {
        console.log(`🔄 Tentativa de reconexão ${reconnectAttempts + 1}/${maxReconnectAttempts} para ${url}`);
        connect();
      }
    }, reconnectInterval);
  }, [connect, reconnectInterval, reconnectAttempts, maxReconnectAttempts, url]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
    }
    
    if (ws.current) {
      ws.current.close(1000, 'Manual disconnect');
    }
    
    setIsConnected(false);
    setConnectionState('DISCONNECTED');
    setData(null);
    setError(null);
    setReconnectAttempts(0);
  }, []);

  const sendMessage = useCallback((message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        const dataToSend = typeof message === 'string' ? message : JSON.stringify(message);
        ws.current.send(dataToSend);
        return true;
      } catch (err) {
        console.error('Erro ao enviar mensagem WebSocket:', err);
        setError(`Erro ao enviar mensagem: ${err.message}`);
        return false;
      }
    } else {
      console.warn('WebSocket não está conectado. Estado:', ws.current?.readyState);
      setError('WebSocket não está conectado');
      return false;
    }
  }, []);

  const forceReconnect = useCallback(() => {
    setReconnectAttempts(0);
    disconnect();
    setTimeout(connect, 1000);
  }, [connect, disconnect]);

  // Conectar automaticamente quando o hook é inicializado
  useEffect(() => {
    if (url) {
      connect();
    }

    return () => {
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url, connect]); // Reconectar quando a URL muda

  // Monitorar mudanças na conectividade da rede
  useEffect(() => {
    const handleOnline = () => {
      if (!isConnected && shouldReconnect) {
        console.log('🌐 Rede reconectada, tentando reconectar WebSocket...');
        forceReconnect();
      }
    };

    const handleOffline = () => {
      console.log('🌐 Rede desconectada');
      setError('Conexão de rede perdida');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [isConnected, shouldReconnect, forceReconnect]);

  // Ping/Pong para manter conexão viva (opcional)
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        sendMessage({ type: 'ping', timestamp: Date.now() });
      }
    }, 30000); // Ping a cada 30 segundos

    return () => clearInterval(pingInterval);
  }, [isConnected, sendMessage]);

  // Verificação de saúde da conexão
  const isHealthy = useCallback(() => {
    return ws.current && 
           ws.current.readyState === WebSocket.OPEN && 
           isConnected && 
           connectionState === 'CONNECTED';
  }, [isConnected, connectionState]);

  // Estatísticas da conexão
  const getConnectionStats = useCallback(() => {
    return {
      url: ws.current?.url || url,
      readyState: ws.current?.readyState,
      readyStateText: ws.current?.readyState === WebSocket.CONNECTING ? 'CONNECTING' :
                     ws.current?.readyState === WebSocket.OPEN ? 'OPEN' :
                     ws.current?.readyState === WebSocket.CLOSING ? 'CLOSING' :
                     ws.current?.readyState === WebSocket.CLOSED ? 'CLOSED' : 'UNKNOWN',
      isConnected,
      connectionState,
      reconnectAttempts,
      maxReconnectAttempts,
      canReconnect: shouldReconnect && reconnectAttempts < maxReconnectAttempts,
      lastError: error
    };
  }, [url, isConnected, connectionState, reconnectAttempts, maxReconnectAttempts, shouldReconnect, error]);

  return {
    // Estado principal
    data,
    isConnected,
    error,
    connectionState,
    reconnectAttempts,
    
    // Métodos de controle
    sendMessage,
    disconnect,
    forceReconnect,
    connect,
    
    // Informações de debug
    readyState: ws.current?.readyState,
    url: ws.current?.url,
    
    // Estados úteis derivados
    isConnecting: connectionState === 'CONNECTING',
    isDisconnected: connectionState === 'DISCONNECTED',
    hasError: connectionState === 'ERROR',
    canReconnect: shouldReconnect && reconnectAttempts < maxReconnectAttempts,
    
    // Métodos utilitários
    isHealthy,
    getConnectionStats
  };
};

export default useWebSocket;
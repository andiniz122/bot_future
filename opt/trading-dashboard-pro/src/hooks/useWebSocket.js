// src/hooks/useWebSocket.js
import { useEffect, useRef, useState, useCallback } from 'react';

const useWebSocket = (url) => {
    const [data, setData] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    const ws = useRef(null);

    const connect = useCallback(() => {
        if (ws.current) {
            ws.current.close(); // Fechar conexão anterior se existir
        }
        ws.current = new WebSocket(url);

        ws.current.onopen = () => {
            setIsConnected(true);
            setError(null);
            console.log(`WebSocket connected to ${url}`);
        };

        ws.current.onmessage = (event) => {
            try {
                const parsedData = JSON.parse(event.data);
                setData(parsedData);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e, event.data);
                setError('Failed to parse WebSocket message.');
            }
        };

        ws.current.onerror = (event) => {
            console.error(`WebSocket error from ${url}:`, event);
            setError('WebSocket connection error.');
            setIsConnected(false);
        };

        ws.current.onclose = (event) => {
            setIsConnected(false);
            console.warn(`WebSocket disconnected from ${url}:`, event.code, event.reason);
            // Implementação básica de reconexão
            setTimeout(() => {
                console.log(`Attempting to reconnect to ${url}...`);
                connect();
            }, 5000); // Tentar reconectar após 5 segundos
        };
    }, [url]);

    useEffect(() => {
        connect();

        return () => {
            if (ws.current) {
                ws.current.close(); // Fechar WebSocket ao desmontar o componente
            }
        };
    }, [url, connect]);

    const sendMessage = (message) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket is not open. Cannot send message.');
        }
    };

    return { data, isConnected, error, sendMessage };
};

export default useWebSocket;
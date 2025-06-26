// src/hooks/useWebSocket.js - Versão Melhorada
import { useEffect, useRef, useState, useCallback } from 'react';

const useWebSocket = (url) => {
    const [data, setData] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    const ws = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = 10;

    const connect = useCallback(() => {
        // Limpar timeout de reconexão anterior
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (ws.current) {
            ws.current.close();
        }

        console.log(`🔄 Tentando conectar WebSocket: ${url} (Tentativa ${reconnectAttempts.current + 1})`);
        
        try {
            ws.current = new WebSocket(url);

            ws.current.onopen = () => {
                setIsConnected(true);
                setError(null);
                reconnectAttempts.current = 0; // Reset contador de tentativas
                console.log(`✅ WebSocket conectado: ${url}`);
            };

            ws.current.onmessage = (event) => {
                try {
                    const parsedData = JSON.parse(event.data);
                    setData(parsedData);
                } catch (e) {
                    console.error('❌ Falha ao parsear mensagem WebSocket:', e, event.data);
                    setError('Falha ao parsear mensagem WebSocket.');
                }
            };

            ws.current.onerror = (event) => {
                console.error(`❌ Erro WebSocket: ${url}`, event);
                setError('Erro de conexão WebSocket.');
                setIsConnected(false);
            };

            ws.current.onclose = (event) => {
                setIsConnected(false);
                console.warn(`🔌 WebSocket fechado: ${url} ${event.code}`);
                
                // Códigos que não devem tentar reconectar
                const dontReconnectCodes = [1000, 1001]; // Normal closure, going away
                
                if (!dontReconnectCodes.includes(event.code) && 
                    reconnectAttempts.current < maxReconnectAttempts) {
                    
                    reconnectAttempts.current++;
                    
                    // Backoff exponencial: 2^tentativa * 1000ms (máximo 30s)
                    const delay = Math.min(Math.pow(2, reconnectAttempts.current) * 1000, 30000);
                    
                    console.log(`⏳ Tentando reconectar em ${delay/1000}s... (${reconnectAttempts.current}/${maxReconnectAttempts})`);
                    
                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, delay);
                } else if (reconnectAttempts.current >= maxReconnectAttempts) {
                    console.error(`❌ Máximo de tentativas de reconexão atingido para ${url}`);
                    setError(`Falha na conexão após ${maxReconnectAttempts} tentativas`);
                }
            };

        } catch (error) {
            console.error(`❌ Erro ao criar WebSocket ${url}:`, error);
            setError('Erro ao criar conexão WebSocket');
            setIsConnected(false);
        }
    }, [url]);

    useEffect(() => {
        connect();

        return () => {
            // Cleanup
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (ws.current) {
                ws.current.close(1000); // Normal closure
            }
        };
    }, [connect]);

    const sendMessage = (message) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
            return true;
        } else {
            console.warn('⚠️ WebSocket não está aberto. Não é possível enviar mensagem.');
            return false;
        }
    };

    // Função manual de reconexão
    const reconnect = useCallback(() => {
        reconnectAttempts.current = 0;
        connect();
    }, [connect]);

    return { 
        data, 
        isConnected, 
        error, 
        sendMessage, 
        reconnect,
        reconnectAttempts: reconnectAttempts.current 
    };
};

export default useWebSocket;
# notification_manager.py
import asyncio
import logging
import os
from typing import Optional
import httpx 

logger = logging.getLogger('telegram_notifier')

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = False

        if self.bot_token and self.chat_id:
            self.enabled = True
            logger.info("✅ Telegram Notifier inicializado. As notificações proativas estão HABILITADAS.")
        else:
            logger.warning("⚠️ Variáveis de ambiente TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não encontradas. As notificações proativas do Telegram estão DESABILADAS.")

    async def send_message(self, message: str, chat_id: Optional[str] = None, parse_mode: Optional[str] = 'MarkdownV2') -> bool:
        """
        Envia uma mensagem para o Telegram.
        :param message: O texto da mensagem a ser enviado.
        :param chat_id: Opcional. Se não fornecido, usa o self.chat_id.
        :param parse_mode: Modo de parse (MarkdownV2, HTML ou None). Padrão é MarkdownV2.
        :return: True se a mensagem foi enviada com sucesso, False caso contrário.
        """
        target_chat_id = chat_id if chat_id else self.chat_id

        if not self.enabled or not target_chat_id or not self.bot_token:
            logger.debug(f"Notificações do Telegram desabilitadas ou CHAT_ID/TOKEN faltando. Mensagem não enviada: {message[:50]}...")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': target_chat_id,
            'text': message,
            'parse_mode': parse_mode
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10)
                response.raise_for_status() 
                logger.info(f"✅ Notificação do Telegram enviada com sucesso para {target_chat_id}.")
                return True
        except httpx.HTTPStatusError as e:
            # Capturar o erro específico de parse e logar de forma útil
            if "can't parse entities: Character" in str(e):
                logger.error(f"❌ Erro de formatação de mensagem do Telegram: {e.response.text}. Verifique se caracteres como '.', '-', '+' etc. estão escapados em MarkdownV2 se não estiverem em blocos de código/negrito.")
            else:
                logger.error(f"❌ Erro HTTP ao enviar notificação do Telegram (status: {e.response.status_code}): {e.response.text}", exc_info=True)
            return False
        except httpx.RequestError as e:
            logger.error(f"❌ Erro de requisição ao enviar notificação do Telegram: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao enviar notificação do Telegram: {e}", exc_info=True)
            return False

    async def notify_trade_event(self, trade_type: str, symbol: str, price: float, size: float,
                                 action: str, pnl_pct: Optional[float] = None) -> None:
        """
        Formata e envia uma notificação de evento de trade (compra/venda/fechamento).
        :param trade_type: Tipo de evento ('COMPRA', 'VENDA', 'FECHAMENTO_SL', 'FECHAMENTO_TP', 'FECHAMENTO_SINAL').
        :param symbol: Símbolo do ativo.
        :param price: Preço da execução.
        :param size: Tamanho da posição.
        :param action: Ação ('BUY' ou 'SELL').
        :param pnl_pct: PnL em porcentagem (apenas para fechamento).
        """
        if not self.enabled:
            return

        # Função auxiliar para escapar caracteres especiais para MarkdownV2
        def escape_markdown_v2(text: str) -> str:
            # Caracteres que precisam de escape no MarkdownV2 (os que não são números, letras ou underscores)
            # Lista completa para segurança
            special_chars = r'_*[]()~`>#+-=|{}.!'
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text

        # Escapando todos os valores que podem conter caracteres especiais
        symbol_esc = escape_markdown_v2(symbol)
        price_esc = escape_markdown_v2(f"{price:.6f}") # Formata e escapa o float
        size_esc = escape_markdown_v2(f"{size:.4f}")
        action_text_esc = escape_markdown_v2(action) # Ação também pode ter chars especiais (ex: BUY/SELL)

        formatted_message = ""

        if trade_type == 'COMPRA':
            formatted_message = (
                f"📈 \\*NOVA COMPRA\\* 📈\n" 
                f"Ativo: `{symbol_esc}`\n"
                f"Preço: `{price_esc} USDT`\n"
                f"Quantidade: `{size_esc}`\n"
                f"Tipo: \\*LONG\\* \\(Abertura\\)\n" 
                f"\\-\\-\\-\\- " 
            )
        elif trade_type == 'VENDA': 
            formatted_message = (
                f"📉 \\*NOVA VENDA\\* 📉\n"
                f"Ativo: `{symbol_esc}`\n"
                f"Preço: `{price_esc} USDT`\n"
                f"Quantidade: `{size_esc}`\n"
                f"Tipo: \\*SHORT\\* \\(Abertura\\)\n"
                f"\\-\\-\\-\\- "
            )
        elif trade_type.startswith('FECHAMENTO'):
            pnl_text = ""
            if pnl_pct is not None:
                pnl_sign = "+" if pnl_pct >= 0 else ""
                pnl_pct_esc = escape_markdown_v2(f"{pnl_sign}{pnl_pct:.2f}%")
                pnl_text = f" PnL: `{pnl_pct_esc}`"
            
            title = ""
            if trade_type == 'FECHAMENTO_SL':
                title = "💔 \\*STOP LOSS ATIVADO\\* 💔"
            elif trade_type == 'FECHAMENTO_TP':
                title = "🎉 \\*TAKE PROFIT ATIVADO\\* 🎉"
            elif trade_type == 'FECHAMENTO_SINAL':
                title = "🔄 \\*FECHAMENTO POR SINAL\\* 🔄"
            else:
                title = "✅ \\*POSIÇÃO FECHADA\\* ✅" 

            formatted_message = (
                f"{title}\n"
                f"Ativo: `{symbol_esc}`\n"
                f"Preço: `{price_esc} USDT`\n"
                f"Quantidade Fechada: `{size_esc}`\n"
                f"Ação: `{action_text_esc}`\n" # Usar a versão escapada
                f"Resultado:{pnl_text}\n"
                f"\\-\\-\\-\\- "
            )
        else:
            logger.warning(f"Tipo de evento de trade desconhecido: {trade_type}")
            return

        await self.send_message(formatted_message, parse_mode='MarkdownV2')
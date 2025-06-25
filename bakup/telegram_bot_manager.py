# telegram_bot_manager.py
import asyncio
import logging
import os
import time 
from datetime import datetime, timedelta
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from typing import TYPE_CHECKING, Optional

# Para evitar circular import, usamos TYPE_CHECKING para anotações de tipo
if TYPE_CHECKING:
    from main import IntelligentBotGate # Importa para anotação de tipo sem import real

# Configuração de log
logger = logging.getLogger('telegram_bot')
logger.setLevel(logging.INFO) 

class TelegramBotManager:
    def __init__(self, bot_instance: Optional['IntelligentBotGate'] = None):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID") 
        self.application: Optional[Application] = None
        self.bot_instance = bot_instance 
        self.running = False

        if not self.bot_token:
            logger.error("❌ TELEGRAM_BOT_TOKEN não encontrado no .env. O bot de comandos não funcionará.")
            return

        self.application = Application.builder().token(self.bot_token).build()

        # Adicionar Handlers (comandos)
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("saldo", self.saldo_command))
        self.application.add_handler(CommandHandler("posicoes", self.posicoes_command))
        self.application.add_handler(CommandHandler("perfil", self.perfil_command))
        self.application.add_handler(CommandHandler("modo", self.perfil_command)) 
        self.application.add_handler(CommandHandler("mudar_modo", self.mudar_modo_command))
        self.application.add_handler(CommandHandler("historico_trades", self.historico_trades_command))
        self.application.add_handler(CommandHandler("relatorio_diario", self.send_daily_report_command)) 
        self.application.add_handler(CommandHandler("status_bot", self.status_bot_command)) 

        # Adicionar um handler para mensagens não reconhecidas (opcional)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.unknown_command))

        logger.info("✅ Telegram Bot Manager inicializado. Pronto para receber comandos.")

    async def start_polling(self):
        """Inicia o polling para receber atualizações do Telegram."""
        if not self.application:
            logger.warning("Telegram Bot Application não inicializado. Não foi possível iniciar o polling.")
            return

        self.running = True
        logger.info("🚀 Iniciando polling do Telegram Bot (compatível com loop existente)...")
        try:
            # CORREÇÃO: Substituir run_until_disconnected() por run_polling()
            await self.application.run_polling() 
            
        except asyncio.CancelledError:
            logger.info("Polling do Telegram Bot cancelado.")
        except Exception as e:
            logger.error(f"Erro no polling do Telegram Bot: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Polling do Telegram Bot finalizado.")

    async def stop_polling(self):
        """Para o polling do Telegram Bot."""
        if self.application and self.running:
            self.running = False
            logger.info("🛑 Solicitando parada do polling do Telegram Bot via shutdown()...")
            await self.application.shutdown() 
            logger.info("Telegram Bot Polling parado.")

    async def send_message(self, message: str, chat_id: Optional[str] = None, parse_mode: Optional[str] = 'HTML') -> bool:
        """
        Envia uma mensagem proativa para um chat específico.
        Útil para notificações iniciadas pelo bot (não em resposta a um comando).
        """
        target_chat_id = chat_id if chat_id else self.chat_id
        if not target_chat_id or not self.bot_token:
            logger.warning("Não foi possível enviar mensagem: CHAT_ID ou BOT_TOKEN não configurado.")
            return False

        try:
            bot = Bot(self.bot_token)
            await bot.send_message(chat_id=target_chat_id, text=message, parse_mode=parse_mode)
            logger.info(f"✅ Mensagem proativa enviada para {target_chat_id}.")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao enviar mensagem proativa para o Telegram: {e}", exc_info=True)
            return False

    # --- Comandos do Bot ---
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Envia uma mensagem quando o comando /start é emitido."""
        user = update.effective_user
        await update.message.reply_html(
            f"Olá, {user.mention_html()}! Eu sou seu bot de trading Gate.io. "
            f"Use /help para ver os comandos disponíveis."
        )
        logger.info(f"Comando /start recebido de {user.id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Envia uma mensagem quando o comando /help é emitido."""
        help_text = (
            "Comandos disponíveis:\n\n"
            "• /saldo - Mostra seu saldo de futuros USDT.\n"
            "• /posicoes - Lista suas posições abertas.\n"
            "• /perfil ou /modo - Mostra o modo de operação atual do bot.\n"
            "• /mudar_modo &lt;modo&gt; - Muda o perfil do bot (ex: /mudar_modo AGGRESSIVE). Modos: DISCOVERY, CONSERVATIVE, MODERATE, AGGRESSIVE, EMERGENCY.\n"
            "• /historico_trades - Mostra um resumo dos seus últimos trades.\n"
            "• /relatorio_diario - Envia o relatório de desempenho do dia (manual).\n"
            "• /status_bot - Mostra um resumo do status atual do bot (ciclos, etc.).\n"
            "• /help - Exibe esta mensagem de ajuda."
        )
        await update.message.reply_html(help_text)
        logger.info(f"Comando /help recebido de {update.effective_user.id}")

    async def saldo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mostra o saldo de futuros USDT."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para consultar o saldo.")
            return

        try:
            balance = await self.bot_instance.gate_api.get_futures_balance()
            if balance:
                message = (
                    f"💰 <b>SALDO FUTUROS GATE.IO</b> 💰\n"
                    f"<b>Disponível:</b> {balance.get('free', 0.0):.2f} USDT\n"
                    f"<b>Total:</b> {balance.get('equity', 0.0):.2f} USDT\n"
                    f"<b>PNL Não Realizado:</b> {balance.get('unrealPnl', 0.0):.2f} USDT"
                )
                await update.message.reply_html(message)
            else:
                await update.message.reply_text("Não foi possível obter o saldo. Verifique a conexão com a API.")
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}", exc_info=True)
            await update.message.reply_text("Ocorreu um erro ao consultar o saldo.")
        logger.info(f"Comando /saldo recebido de {update.effective_user.id}")

    async def posicoes_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Lista as posições abertas."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para consultar posições.")
            return

        try:
            positions = await self.bot_instance.portfolio_manager.get_open_positions_ws()
            
            if not positions:
                await update.message.reply_text("Você não tem posições abertas no momento.")
                return

            message_parts = ["📊 <b>SUAS POSIÇÕES ABERTAS</b> 📊"]
            for pos in positions:
                symbol = pos.get('contract', 'N/A')
                size = float(pos.get('size', 0))
                entry_price = float(pos.get('entry_price', 0))
                mark_price = float(pos.get('mark_price', 0))
                unrealized_pnl = float(pos.get('unrealized_pnl', 0))

                side = "LONG" if size > 0 else "SHORT"
                
                pnl_pct = 0.0
                if entry_price > 0:
                    if side == 'LONG':
                        pnl_pct = ((mark_price - entry_price) / entry_price) * 100
                    else: 
                        pnl_pct = ((entry_price - mark_price) / entry_price) * 100

                message_parts.append(
                    f"\n• <b>{symbol}</b> ({side})\n"
                    f"  Quantidade: {abs(size):.4f}\n"
                    f"  Entrada: {entry_price:.6f}\n"
                    f"  Preço Atual: {mark_price:.6f}\n"
                    f"  PNL: {unrealized_pnl:+.2f} USDT ({pnl_pct:+.2f}%)"
                )
            
            final_message = "\n".join(message_parts)
            await update.message.reply_html(final_message)

        except Exception as e:
            logger.error(f"Erro ao obter posições: {e}", exc_info=True)
            await update.message.reply_text("Ocorreu um erro ao consultar as posições.")
        logger.info(f"Comando /posicoes recebido de {update.effective_user.id}")

    async def perfil_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mostra o modo de operação atual do bot."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para consultar o perfil.")
            return

        current_mode = self.bot_instance.current_mode
        message = f"⚙️ <b>MODO DE OPERAÇÃO ATUAL:</b> <b>{current_mode.upper()}</b>"
        await update.message.reply_html(message)
        logger.info(f"Comando /perfil recebido de {update.effective_user.id}")

    async def mudar_modo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Muda o perfil do bot manualmente."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para mudar o modo.")
            return

        if not context.args:
            await update.message.reply_html(
                "Por favor, especifique o novo modo. Ex: <code>/mudar_modo AGGRESSIVE</code>\n"
                "Modos disponíveis: DISCOVERY, CONSERVATIVE, MODERATE, AGGRESSIVE, EMERGENCY."
            )
            return

        new_mode_str = context.args[0].upper()
        
        from main import AdaptiveMarketMode # Importa AdaptiveMarketMode aqui
        
        valid_modes_map = {
            "DISCOVERY": AdaptiveMarketMode.DISCOVERY,
            "CONSERVATIVE": AdaptiveMarketMode.CONSERVATIVE,
            "MODERATE": AdaptiveMarketMode.MODERATE,
            "AGGRESSIVE": AdaptiveMarketMode.AGGRESSIVE,
            "EMERGENCY": AdaptiveMarketMode.EMERGENCY,
        }

        if new_mode_str not in valid_modes_map:
            await update.message.reply_text(
                f"Modo '{new_mode_str}' inválido. Modos disponíveis: DISCOVERY, CONSERVATIVE, MODERATE, AGGRESSIVE, EMERGENCY."
            )
            return

        new_mode = valid_modes_map[new_mode_str]
        
        try:
            old_mode = self.bot_instance.current_mode
            self.bot_instance.current_mode = new_mode
            self.bot_instance.signal_engine = self.bot_instance.signal_engine.__class__(
                self.bot_instance._get_adaptive_strategy_config()
            )
            self.bot_instance.last_mode_change = time.time() 

            message = (
                f"⚙️ <b>MUDANÇA DE MODO SOLICITADA</b> ⚙️\n"
                f"De: <b>{old_mode.upper()}</b>\n"
                f"Para: <b>{new_mode.upper()}</b>\n"
                f"Ajustes de estratégia aplicados."
            )
            await update.message.reply_html(message)
            logger.info(f"Comando /mudar_modo para {new_mode_str} recebido de {update.effective_user.id}")

        except Exception as e:
            logger.error(f"Erro ao mudar o modo do bot: {e}", exc_info=True)
            await update.message.reply_text("Ocorreu um erro ao tentar mudar o modo do bot.")

    async def historico_trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mostra um resumo dos últimos trades (implementação simplificada)."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para obter histórico de trades.")
            return

        try:
            # Esta é uma implementação simplificada.
            # O Gate.io Futures API tem um endpoint para 'my_trades'.
            # Você precisaria ajustar GateAPI para ter um método como `get_my_futures_trades()`
            # E chamar aqui, filtrando por data ou quantidade.
            
            # Exemplo (assumindo que GateAPI tem um método `get_my_futures_trades`):
            # trades = await self.bot_instance.gate_api.get_my_futures_trades(limit=5)
            # if trades:
            #     message_parts = ["📋 <b>ÚLTIMOS TRADES</b> 📋"]
            #     for trade in trades:
            #         message_parts.append(
            #             f"\n• <b>{trade.get('contract', 'N/A')}</b> {trade.get('side', 'N/A').upper()}\n"
            #             f"  Preço: {float(trade.get('price', 0)):.6f}\n"
            #             f"  Qtd: {float(trade.get('size', 0)):.4f}\n"
            #             f"  Data: {datetime.fromtimestamp(float(trade.get('create_time_ms', 0))/1000).strftime('%Y-%m-%d %H:%M:%S')}"
            #         )
            #     await update.message.reply_html("\n".join(message_parts))
            # else:
            #     await update.message.reply_text("Nenhum trade recente encontrado ou erro ao obter dados.")
            
            await update.message.reply_text("A funcionalidade de histórico de trades ainda está sendo implementada ou não está disponível via API para este bot.")

        except Exception as e:
            logger.error(f"Erro ao obter histórico de trades: {e}", exc_info=True)
            await update.message.reply_text("Ocorreu um erro ao consultar o histórico de trades.")
        logger.info(f"Comando /historico_trades recebido de {update.effective_user.id}")
            
    async def status_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mostra um resumo do status atual do bot."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para obter o status.")
            return

        try:
            stats = self.bot_instance.stats
            current_mode = self.bot_instance.current_mode.upper()
            cycle_interval = self.bot_instance._get_adaptive_cycle_interval()
            
            balance = await self.bot_instance.gate_api.get_futures_balance()
            positions = await self.bot_instance.portfolio_manager.get_open_positions_ws()
            
            total_equity = balance.get('equity', 0.0) if balance else 0.0
            num_positions = len(positions)
            
            uptime = "N/A"
            if stats['start_time']:
                duration = datetime.now() - stats['start_time']
                days = duration.days
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                uptime = f"{days}d {hours}h {minutes}m"

            message = (
                f"🤖 <b>STATUS ATUAL DO BOT</b> 🤖\n"
                f"Modo: <b>{current_mode}</b>\n"
                f"Uptime: {uptime}\n"
                f"Ciclos Executados: {stats['total_cycles']}\n"
                f"Ciclos Bem-Sucedidos: {stats['successful_cycles']}\n"
                f"Falhas Consecutivas: {stats['consecutive_failures']}\n"
                f"Saldo Total: {total_equity:.2f} USDT\n"
                f"Posições Abertas: {num_positions}\n"
                f"Oportunidades Encontradas: {stats['opportunities_found']}\n"
                f"Ordens Executadas: {stats['orders_executed']}\n"
                f"Próximo Ciclo em: {cycle_interval} segundos."
            )
            await update.message.reply_html(message)
            logger.info(f"Comando /status_bot recebido de {update.effective_user.id}")

        except Exception as e:
            logger.error(f"Erro ao obter status do bot: {e}", exc_info=True)
            await update.message.reply_text("Ocorreu um erro ao consultar o status do bot.")

    async def send_daily_report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Aciona o envio do relatório diário manualmente."""
        if not self.bot_instance:
            await update.message.reply_text("Bot principal não está conectado para gerar o relatório.")
            return

        await update.message.reply_text("Gerando e enviando relatório diário... Isso pode levar alguns segundos.")
        # Chamar a função do bot principal para gerar e enviar o relatório
        success = await self.bot_instance.send_daily_report()
        if success:
            await update.message.reply_text("Relatório diário enviado com sucesso!")
        else:
            await update.message.reply_text("Falha ao gerar ou enviar o relatório diário.")
        logger.info(f"Comando /relatorio_diario recebido de {update.effective_user.id}")

    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Responde a comandos desconhecidos."""
        await update.message.reply_text(f"Comando '{update.message.text}' não reconhecido. Use /help para ver os comandos.")
        logger.info(f"Comando desconhecido '{update.message.text}' recebido de {update.effective_user.id}")
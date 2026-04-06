import logging
import asyncio
from data_handler import DataHandler

logger = logging.getLogger(__name__)

class Executor:
    """
    Handles order placement for Binance Futures Testnet via ccxt.
    Manages opening positions with attached Stop-Loss and Take-Profit orders.
    """
    def __init__(self, data_handler: DataHandler):
        self.dh = data_handler
        self.exchange = data_handler.exchange
        self.symbol = data_handler.symbol
        self.market_info = None

    async def _ensure_market_loaded(self):
        if self.market_info is None:
            await self.exchange.load_markets()
            self.market_info = self.exchange.market(self.symbol)

    def _format_amount(self, amount: float) -> str:
        """Rounds amount to the correct precision and checks against minimum lot size."""
        if not self.market_info:
            # Fallback if no market info (e.g., 3 decimals for BTC)
            return f"{amount:.3f}"
            
        amount_precision = self.market_info.get('precision', {}).get('amount')
        amount_min = self.market_info.get('limits', {}).get('amount', {}).get('min', 0.001)
        
        # Enforce Minimum
        if amount < amount_min:
            logger.warning(f"Calculated amount {amount} is below market minimum {amount_min}. Clamping to min.")
            amount = amount_min
            
        # Format string
        if amount_precision is not None:
            # Construct a format string like '{:.3f}' if precision is 3
            # In ccxt some precisions are integers dictating decimal places
            # Some are float step sizes (e.g., 0.001 for BTC)
            if isinstance(amount_precision, int):
                return f"{amount:.{amount_precision}f}"
            elif isinstance(amount_precision, float):
                # Count decimal places of the step size
                decimals = len(str(amount_precision).rstrip('0').split('.')[1]) if '.' in str(amount_precision) else 0
                return f"{amount:.{decimals}f}"
                
        # Default fallback
        return f"{amount:.3f}"

    async def execute_trade(self, action: str, amount: float, sl_price: float, tp_price: float):
        """
        Executes a Market buy/sell and immediately places SL and TP limit/stop market orders.
        action: "BUY" or "SELL"
        """
        try:
            await self._ensure_market_loaded()
            formatted_amount_str = self._format_amount(amount)
            formatted_amount = float(formatted_amount_str)

            # 1. Open the main market position
            logger.info(f"Placing Market {action} order for {formatted_amount} {self.symbol}...")
            order = await self.exchange.create_market_order(self.symbol, action, formatted_amount)
            logger.info(f"Market order filled: {order['id']} at ~{order.get('average', order.get('price'))}")
            
            # 2. Place Stop-Loss and Take-Profit orders
            # Binance Futures requires specific params to set these as reduce-only closing orders.
            
            close_action = "SELL" if action == "BUY" else "BUY"
            
            # STOP_MARKET for Stop Loss
            sl_params = {
                'stopPrice': sl_price,
                'reduceOnly': True
            }
            logger.info(f"Placing Stop-Loss {close_action} at {sl_price:.2f}")
            await self.exchange.create_order(
                self.symbol, 'STOP_MARKET', close_action, formatted_amount, None, sl_params
            )
            
            # TAKE_PROFIT_MARKET for Take Profit
            tp_params = {
                'stopPrice': tp_price,
                'reduceOnly': True
            }
            logger.info(f"Placing Take-Profit {close_action} at {tp_price:.2f}")
            await self.exchange.create_order(
                self.symbol, 'TAKE_PROFIT_MARKET', close_action, formatted_amount, None, tp_params
            )
            
            logger.info("Trade execution sequence completed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error during execution sequence: {e}")
            return False

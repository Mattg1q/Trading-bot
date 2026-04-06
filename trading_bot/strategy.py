import numpy as np
import logging

logger = logging.getLogger(__name__)

class Strategy:
    """
    Strategy Logic:
    - Gating Mechanism: Execution only occurs if sign(SNN_pred) == sign(FinBERT_score).
    - Risk: 1% fixed fractional position sizing, 0.5% Stop-Loss, 1.0% Take-Profit.
    """
    def __init__(self, risk_per_trade=0.01, stop_loss=0.005, take_profit=0.01):
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signal(self, lob_pred, sentiment_score):
        """
        Gating Mechanism: Execution only occurs if sign(SNN_pred) == sign(FinBERT_score).
        Returns +1 for LONG, -1 for SHORT, 0 for HOLD.
        """
        lob_sign = np.sign(lob_pred)
        sentiment_sign = np.sign(sentiment_score)
        
        # Add a small threshold to avoid spurious entries due to model noise
        if abs(lob_pred) < 0.05 or abs(sentiment_score) < 0.05:
            return 0
            
        if lob_sign == sentiment_sign and lob_sign != 0:
            return int(lob_sign)
        
        return 0

    def calculate_position_size(self, current_price, current_capital):
        """
        1% fixed fractional position sizing based on real-time capital.
        Dollar Risk = Capital * Risk Per Trade (1%)
        Risk Distance per Unit = Entry Price * Stop Loss %
        Units = Dollar Risk / Risk Distance
        """
        if not current_capital or current_capital <= 0:
            logger.warning("Balance returned 0 or failed. Using fallback testing position size: 0.001")
            return 0.001
            
        amount = (current_capital * self.risk_per_trade) / current_price
        
        # Clamp to minimum notional value (110 USDT for Binance Futures)
        amount = max(amount, 110 / current_price)
        
        # Exact debug requested by user
        risk_dollar = current_capital * self.risk_per_trade
        logger.info(f"DEBUG: Risking ${risk_dollar:.0f} ({(self.risk_per_trade * 100):.1f}% of {current_capital:.0f}) -> Target Size: {amount:.3f} BTC.")
            
        # Margin Check (Assuming 1x Leverage)
        required_margin = amount * current_price
        logger.info(f"Available Margin: {current_capital:.2f} USDT | Required Margin: {required_margin:.2f} USDT")
        
        if current_capital < required_margin:
            logger.warning("Insufficient Margin for calculated sizing. Scaling down to minimum allowed (0.001).")
            amount = 0.001
        
        # Maximum allowed physical size based on available account value (1x leverage max)
        max_size = current_capital / current_price
        amount = min(amount, max_size)
        
        # Round to 3 decimal places to meet exchange precision requirements
        amount = round(amount, 3)
        
        return amount

    def calculate_sl_tp(self, entry_price, signal):
        """
        Returns Stop-Loss and Take-Profit prices based on entry price and signal (+1 Long, -1 Short).
        """
        if signal == 1:
            sl_price = entry_price * (1 - self.stop_loss)
            tp_price = entry_price * (1 + self.take_profit)
        elif signal == -1:
            sl_price = entry_price * (1 + self.stop_loss)
            tp_price = entry_price * (1 - self.take_profit)
        else:
            return None, None
            
        return sl_price, tp_price

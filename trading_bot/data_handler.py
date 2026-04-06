import ccxt.async_support as ccxt
import finnhub
import torch
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, binance_symbol=config.TICKER, finnhub_api_key=config.FINNHUB_API_KEY):
        self.symbol = binance_symbol
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        # Enable Binance Demo Trading (works transparently for Future configs in modern ccxt)
        self.exchange.enable_demo_trading(True)
        
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        self._leverage_set = False
        
    async def initialize_futures(self):
        """Pre-flight checks and settings like leverage."""
        if not self._leverage_set:
            try:
                await self.exchange.load_markets()
                # Determine binance specific symbol format (usually BTC/USDT)
                market = self.exchange.market(self.symbol)
                await self.exchange.set_leverage(config.LEVERAGE, market['id'])
                logger.info(f"Successfully set {config.LEVERAGE}x leverage for {self.symbol} on Futures Testnet.")
                
                # Ensure One-way Mode explicitly
                try:
                    await self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'false'})
                    logger.info("Successfully set One-way Mode (Hedge Mode disabled).")
                except Exception as mode_e:
                    # Binance returns error -4059 if it's already in the target state
                    if "-4059" not in str(mode_e):
                        logger.warning(f"Could not change Position Mode: {mode_e}")
                    else:
                        logger.info("One-way Mode is already active.")
                        
                self._leverage_set = True
            except Exception as e:
                logger.error(f"Failed to set leverage: {e}")

    async def get_futures_balance(self):
        """Fetch the available USDT balance from the Futures wallet."""
        try:
            balance = await self.exchange.fetch_balance()
            if 'USDT' in balance and 'free' in balance['USDT']:
                return balance['USDT']['free']
            return config.DEFAULT_CAPITAL
        except ccxt.PermissionDenied as e:
            logger.error(f"API Key lacks permissions for fetching balance. Falling back to {config.DEFAULT_CAPITAL}. Ensure API keys have Futures enabled on demo.binance.com")
            return config.DEFAULT_CAPITAL
        except ccxt.AuthenticationError as e:
            logger.error(f"Invalid API Keys supplied for fetching balance. Falling back to {config.DEFAULT_CAPITAL}. Details: {e}")
            return config.DEFAULT_CAPITAL
        except Exception as e:
            logger.error(f"Error fetching futures balance: {e}")
            return config.DEFAULT_CAPITAL
        
    async def get_lob_data(self):
        """Fetch 20 levels of LOB depth from Binance and normalize into a 20x4 tensor relative to mid-price."""
        try:
            orderbook = await self.exchange.fetch_order_book(self.symbol, limit=20)
            
            bids = orderbook['bids'][:20] # List of [price, amount]
            asks = orderbook['asks'][:20]
            
            if not bids or not asks:
                return None, None
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2.0
            
            # Normalize into 20x4 tensor: [bid_price_rel, bid_qty, ask_price_rel, ask_qty]
            tensor_data = []
            for i in range(20):
                bid_p = (bids[i][0] / mid_price) - 1.0 if i < len(bids) else 0.0
                bid_q = bids[i][1] if i < len(bids) else 0.0
                
                ask_p = (asks[i][0] / mid_price) - 1.0 if i < len(asks) else 0.0
                ask_q = asks[i][1] if i < len(asks) else 0.0
                
                tensor_data.append([bid_p, bid_q, ask_p, ask_q])
                
            tensor = torch.tensor(tensor_data, dtype=torch.float32)
            return tensor, mid_price
        except Exception as e:
            logger.error(f"Error fetching LOB data: {e}")
            return None, None

    async def get_finnhub_sentiment_news(self):
        """Fetch real-time crypto news via Finnhub API."""
        try:
            # Finnhub requires 'category' for general news. We can use 'crypto'
            news = self.finnhub_client.general_news('crypto', min_id=0)
            if not news:
                return []
            
            # Extract headlines from the latest news (e.g., top 5)
            headlines = [article.get('headline', '') for article in news[:5] if 'headline' in article]
            return headlines
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return []
            
    async def close(self):
        await self.exchange.close()

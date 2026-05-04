import ccxt.async_support as ccxt
import finnhub
import torch
import logging
import asyncio
import contextlib
import json
import aiohttp
import time

import config

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, binance_symbol=config.TICKER, finnhub_api_key=config.FINNHUB_API_KEY):
        self.symbol = binance_symbol
        exchange_config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        }
        if config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY:
            exchange_config['apiKey'] = config.BINANCE_API_KEY
            exchange_config['secret'] = config.BINANCE_SECRET_KEY

        self.exchange = ccxt.binance(exchange_config)
        # Enable Binance Demo Trading (works transparently for Future configs in modern ccxt)
        self.exchange.enable_demo_trading(True)
        
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        self._leverage_set = False
        self._lob_tensor = None
        self._lob_mid_price = None
        self._lob_event = asyncio.Event()
        self._lob_task = None
        self._http_session = None
        self._news_cache = []
        self._news_cache_at = 0.0
        
    async def initialize_futures(self):
        """Pre-flight checks and settings like leverage."""
        if config.TRADING_MODE != "Demo Futures":
            logger.info("Paper trading mode active; skipping Futures account setup.")
            return

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
        if config.TRADING_MODE != "Demo Futures":
            return config.DEFAULT_CAPITAL

        if not config.BINANCE_API_KEY or not config.BINANCE_SECRET_KEY:
            logger.error("Binance API credentials are missing; cannot fetch Futures balance.")
            return None

        try:
            balance = await self.exchange.fetch_balance()
            if 'USDT' in balance and 'free' in balance['USDT']:
                return balance['USDT']['free']
            logger.error("USDT Futures balance was not present in the exchange response.")
            return None
        except ccxt.PermissionDenied as e:
            logger.error(f"API Key lacks permissions for fetching balance. Ensure API keys have Futures enabled on demo.binance.com. Details: {e}")
            return None
        except ccxt.AuthenticationError as e:
            logger.error(f"Invalid API Keys supplied for fetching balance. Details: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching futures balance: {e}")
            return None

    def _build_lob_tensor(self, bids, asks):
        if not bids or not asks:
            return None, None

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0

        tensor_data = []
        for i in range(config.LOB_LEVELS):
            bid_p = (float(bids[i][0]) / mid_price) - 1.0 if i < len(bids) else 0.0
            bid_q = float(bids[i][1]) if i < len(bids) else 0.0
            ask_p = (float(asks[i][0]) / mid_price) - 1.0 if i < len(asks) else 0.0
            ask_q = float(asks[i][1]) if i < len(asks) else 0.0
            tensor_data.append([bid_p, bid_q, ask_p, ask_q])

        return torch.tensor(tensor_data, dtype=torch.float32), mid_price

    async def start_lob_stream(self):
        """Start the continuous Binance Futures partial-depth WebSocket cache."""
        if self._lob_task and not self._lob_task.done():
            return

        self._http_session = aiohttp.ClientSession()
        self._lob_task = asyncio.create_task(self._lob_stream_loop())
        try:
            await asyncio.wait_for(self._lob_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("LOB WebSocket did not deliver an initial snapshot within 10 seconds.")

    async def _lob_stream_loop(self):
        stream = f"{config.BINANCE_STREAM_SYMBOL}@depth{config.LOB_LEVELS}@{config.LOB_STREAM_SPEED}"
        url = f"wss://fstream.binance.com/public/ws/{stream}"
        backoff = 1

        while True:
            try:
                logger.info(f"Connecting Binance Futures LOB stream: {stream}")
                async with self._http_session.ws_connect(url, heartbeat=180) as ws:
                    backoff = 1
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(msg.data)
                            data = payload.get("data", payload)
                            tensor, mid_price = self._build_lob_tensor(data.get("b", []), data.get("a", []))
                            if tensor is not None:
                                self._lob_tensor = tensor
                                self._lob_mid_price = mid_price
                                self._lob_event.set()
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"LOB WebSocket error: {e}. Reconnecting in {backoff}s.")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
        
    async def get_lob_data(self):
        """Return the latest 20-level LOB tensor from the Binance Futures WebSocket cache."""
        if not self._lob_task or self._lob_task.done():
            await self.start_lob_stream()

        if self._lob_tensor is not None and self._lob_mid_price is not None:
            return self._lob_tensor.clone(), self._lob_mid_price

        logger.warning("LOB WebSocket cache is empty; falling back to a one-off REST snapshot.")
        try:
            orderbook = await self.exchange.fetch_order_book(self.symbol, limit=config.LOB_LEVELS)
            return self._build_lob_tensor(orderbook['bids'][:config.LOB_LEVELS], orderbook['asks'][:config.LOB_LEVELS])
        except Exception as e:
            logger.error(f"Error fetching LOB data: {e}")
            return None, None

    async def get_lob_snapshot_rows(self):
        """Fetch a raw 20-level LOB snapshot for CSV/JSON export."""
        orderbook = await self.exchange.fetch_order_book(self.symbol, limit=config.LOB_LEVELS)
        bids = orderbook['bids'][:config.LOB_LEVELS]
        asks = orderbook['asks'][:config.LOB_LEVELS]

        if not bids or not asks:
            return [], None

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0
        rows = []

        for level in range(config.LOB_LEVELS):
            bid = bids[level] if level < len(bids) else [None, None]
            ask = asks[level] if level < len(asks) else [None, None]
            rows.append({
                "level": level + 1,
                "bid_price": bid[0],
                "bid_quantity": bid[1],
                "ask_price": ask[0],
                "ask_quantity": ask[1],
                "mid_price": mid_price,
            })

        return rows, mid_price

    async def get_finnhub_sentiment_news(self):
        """Fetch crypto news via Finnhub without blocking the event loop."""
        if not config.FINNHUB_API_KEY:
            logger.error("FINNHUB_API_KEY is missing; cannot fetch Finnhub news.")
            return []

        now = time.monotonic()
        if self._news_cache and now - self._news_cache_at < config.FINNHUB_NEWS_TTL_SECONDS:
            return self._news_cache

        try:
            # Finnhub requires 'category' for general news. We can use 'crypto'
            news = await asyncio.to_thread(self.finnhub_client.general_news, 'crypto', min_id=0)
            if not news:
                return self._news_cache
            
            headlines = [
                article.get('headline', '')
                for article in news[:config.FINNHUB_MAX_HEADLINES]
                if article.get('headline')
            ]
            self._news_cache = headlines
            self._news_cache_at = now
            return headlines
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return self._news_cache
            
    async def close(self):
        if self._lob_task:
            self._lob_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lob_task
        if self._http_session:
            await self._http_session.close()
        await self.exchange.close()

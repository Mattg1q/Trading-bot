import asyncio
import ccxt.async_support as ccxt
import sys
import logging
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

async def test_demo_connection():
    logger.info("Initializing ccxt for Binance Demo Validation...")
    
    # By default ccxt points binance to production.
    # The actual demo/mock platform for Binance Futures usually resides at testnet.binancefuture.com
    # However, there's also a 'demo.binance.com' used for some mock accounts.
    # Let's try the standard testnet URLs which represents Mock Trading on Futures.
    
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'warnOnFetchOpenOrdersWithoutSymbol': False,
        }
    })
    
    # The standard way to set sandbox (Mock Trading) for Binance Futures
    exchange.set_sandbox_mode(True)
    
    try:
        # Step 1: Ping Server
        logger.info(f"Pinging Server URL: {exchange.urls['api']['fapiPublic']}")
        time = await exchange.fetch_time()
        logger.info(f"[SUCCESS] Server Time Fetched: {time}")
        
    except Exception as e:
        logger.error(f"[FAILED] Could not ping server. Is the URL correct? Error: {e}")
        await exchange.close()
        return

    try:
        # Step 2: Validate API Keys
        logger.info("Testing Balance fetch to validate keys and permissions...")
        balance = await exchange.fetch_balance()
        logger.info(f"[SUCCESS] Keys authenticated! Balance found: {list(balance.get('free', {}).keys())[:3]}...")
    except ccxt.AuthenticationError as e:
        logger.error(f"[ERROR -2008] Invalid API Key or Secret. Ensure you generated keys from the proper Demo environment. Details: {e}")
    except ccxt.PermissionDenied as e:
        logger.error(f"[ERROR] API Key lacks Futures permissions or IP restrictions are blocking access. Details: {e}")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected validation error: {e}")
        
    finally:
        await exchange.close()
        logger.info("Connection closed.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_demo_connection())

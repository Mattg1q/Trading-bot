import asyncio
import logging
import torch
import time
import os
import sys

from data_handler import DataHandler
from models import SpatialLOBModel, SentimentModel
from strategy import Strategy
from executor import Executor
import config

# Use Config values
TICKER = config.TICKER
POLL_INTERVAL = config.POLL_INTERVAL

bot_state = {
    "ticker": TICKER,
    "paper_trading": config.TRADING_MODE,
    "mid_price": 0.0,
    "snn_prediction": 0.0,
    "sentiment_score": 0.0,
    "current_signal": 0,
    "capital": config.DEFAULT_CAPITAL,
    "position_size": 0.0,
    "risk_per_trade": 0.01,
    "poll_interval": POLL_INTERVAL,
    "logs": [],
    "ready": False,
    "runtime_error": ""
}

class ListHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        bot_state["logs"].append(msg)
        if len(bot_state["logs"]) > 100:
            bot_state["logs"].pop(0)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), ListHandler()])
logger = logging.getLogger(__name__)

def _select_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if config.REQUIRE_GPU:
        raise RuntimeError(
            "GPU acceleration is required but PyTorch cannot see CUDA. "
            "Install a CUDA-enabled torch build or set REQUIRE_GPU=0 for local dry-runs."
        )
    return torch.device("cpu")

def _load_lob_model(device):
    weights_path = config.SNN_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        if config.ALLOW_UNTRAINED_SNN and config.TRADING_MODE != "Demo Futures":
            logger.warning(
                "Using untrained SNN fallback for paper trading only. "
                "Provide trained weights before enabling Demo Futures."
            )
            model = SpatialLOBModel().to(device)
            model.eval()
            return model
        raise RuntimeError(
            f"Missing trained SNN weights at {weights_path}. "
            "Set SNN_WEIGHTS_PATH to a valid .pth/.pt checkpoint before trading."
        )

    model = SpatialLOBModel().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

async def main_loop():
    try:
        config.validate_python_runtime()
        config.validate_required_secrets()
        device = _select_torch_device()
    except Exception as e:
        bot_state["runtime_error"] = str(e)
        bot_state["ready"] = False
        logger.error(f"Startup validation failed: {e}")
        return

    logger.info("Initializing multi-modal trading bot components...")
    logger.info(f"Using torch device: {device}")
    
    # Initialize components
    data_handler = DataHandler()
    executor = Executor(data_handler)

    try:
        lob_model = _load_lob_model(device)
        sentiment_model = SentimentModel(pipeline_device=0 if device.type == "cuda" else -1)
        strategy = Strategy()
    except Exception as e:
        bot_state["runtime_error"] = str(e)
        bot_state["ready"] = False
        logger.error(f"Model initialization failed: {e}")
        await data_handler.close()
        return
    
    # Pre-flight checks and Leverage setup for Futures
    await data_handler.initialize_futures()
    await data_handler.start_lob_stream()
    
    logger.info(f"Bot initialized. Mode: {config.TRADING_MODE}")
    logger.info(f"Configuration -> Ticker: {TICKER} | Interval: {POLL_INTERVAL}s | Leverage: {config.LEVERAGE}x")
    bot_state["ready"] = True
    bot_state["runtime_error"] = ""
    
    try:
        while True:
            start_time = time.time()
            logger.info("-" * 40)
            logger.info(f"Fetching Data for {TICKER} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Grab dynamic variables controlled by UI layout
            current_poll = bot_state["poll_interval"]
            strategy.risk_per_trade = bot_state["risk_per_trade"]
            
            # Fetch data concurrently
            lob_task = data_handler.get_lob_data()
            news_task = data_handler.get_finnhub_sentiment_news()
            balance_task = data_handler.get_futures_balance()
            
            # Gather responses
            (lob_tensor, mid_price), headlines, current_capital = await asyncio.gather(lob_task, news_task, balance_task)
            
            if lob_tensor is None or mid_price is None:
                logger.warning("Failed to fetch LOB data. Skipping iteration.")
                # We skip to the next iteration safely
                await asyncio.sleep(current_poll)
                continue
            if current_capital is None:
                logger.warning("Failed to fetch account balance. Skipping iteration to avoid unsafe sizing.")
                await asyncio.sleep(current_poll)
                continue
                
            bot_state["capital"] = current_capital
                
            # Process LOB prediction using the CNN
            with torch.no_grad():
                # Add batch dimension: [1, 20, 4]
                lob_input = lob_tensor.unsqueeze(0).to(device)
                lob_pred_tensor = lob_model(lob_input)
                lob_pred = lob_pred_tensor.item()
                
            # Process Sentiment score using NLP transformers
            sentiment_score = sentiment_model.score_headlines(headlines)
            
            # Update State
            bot_state["mid_price"] = mid_price
            bot_state["snn_prediction"] = lob_pred
            bot_state["sentiment_score"] = sentiment_score
            
            # Output signals
            logger.info(f"Mid Price          : {mid_price:.2f} USDT")
            logger.info(f"SNN LOB Prediction : {lob_pred:.4f}")
            logger.info(f"FinBERT Sentiment  : {sentiment_score:.4f} (based on {len(headlines)} headlines)")
            
            # Generate actionable signal
            signal = strategy.generate_signal(lob_pred, sentiment_score)
            
            if signal == 1:
                logger.info("SIGNAL             : LONG (+1)")
            elif signal == -1:
                logger.info("SIGNAL             : SHORT (-1)")
            else:
                logger.info("SIGNAL             : HOLD (0)")
                
            bot_state["current_signal"] = signal
            
            # Execute execution logic
            if signal != 0:
                position_size = strategy.calculate_position_size(mid_price, current_capital)
                bot_state["position_size"] = position_size
                if position_size <= 0:
                    logger.info("Signal skipped because exact risk sizing is not executable with current balance/settings.")
                    elapsed = time.time() - start_time
                    sleep_time = max(0, current_poll - elapsed)
                    await asyncio.sleep(sleep_time)
                    continue
                sl, tp = strategy.calculate_sl_tp(mid_price, signal)
                action = "BUY" if signal == 1 else "SELL"
                
                logger.info(f"[{config.TRADING_MODE}] Executing {action} of {position_size:.4f} {TICKER} at {mid_price:.2f}")
                logger.info(f"[{config.TRADING_MODE}] Placed Target SL: {sl:.2f}, Target TP: {tp:.2f}")
                
                if config.TRADING_MODE == "Demo Futures":
                    await executor.execute_trade(action, position_size, sl, tp)
                
            # Dynamic sleep to align with actively controlled poll interval
            elapsed = time.time() - start_time
            sleep_time = max(0, current_poll - elapsed)
            logger.info(f"Sleeping for {sleep_time:.2f} seconds until next iteration...")
            await asyncio.sleep(sleep_time)
            
    except asyncio.CancelledError:
        logger.info("Bot execution cancelled.")
    except KeyboardInterrupt:
        logger.info("Bot shutting down (KeyboardInterrupt)...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        await data_handler.close()
        logger.info("Exchange connection closed.")

import logging
import asyncio
from typing import Dict, Any, Optional
from src.interfaces.broker_interface import BrokerInterface, Order, Fill
from src.brokers.paper_broker import PaperBroker
# from src.brokers.ibkr_broker import IBKRBroker # Future
# from src.brokers.alpaca_broker import AlpacaBroker # Future

logger = logging.getLogger(__name__)

class ExecutionGateway(BrokerInterface):
    """
    Central gateway for order execution.
    Abstracts the underlying broker implementation and handles:
    - Broker selection (Paper vs Live)
    - Retry logic (Async)
    - Error handling
    - Centralized logging
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "paper")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        self.broker = self._initialize_broker()
        logger.info(f"ExecutionGateway initialized in {self.mode} mode")

    def _initialize_broker(self) -> BrokerInterface:
        if self.mode == "paper":
            return PaperBroker()
        elif self.mode == "ibkr":
            # return IBKRBroker(self.config)
            raise NotImplementedError("IBKR broker not yet implemented")
        elif self.mode == "alpaca":
            # return AlpacaBroker(self.config)
            raise NotImplementedError("Alpaca broker not yet implemented")
        else:
            raise ValueError(f"Unknown execution mode: {self.mode}")

    def get_cash(self) -> float:
        try:
            return self.broker.get_cash()
        except Exception as e:
            logger.error(f"Failed to get cash: {e}")
            raise e

    def get_positions(self) -> Dict[str, float]:
        try:
            return self.broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise e

    def get_portfolio_value(self) -> float:
        try:
            return self.broker.get_portfolio_value()
        except Exception as e:
            logger.error(f"Failed to get portfolio value: {e}")
            raise e

    async def place_order(self, order: Order) -> Fill:
        """
        Place an order with retry logic (Async).
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                logger.info(f"Placing order: {order} (Attempt {attempt + 1})")
                # Assuming broker.place_order is synchronous for now, 
                # but we wrap the retry delay asynchronously.
                # Ideally, broker.place_order should also be async.
                fill = self.broker.place_order(order)
                
                if fill.status == "FILLED":
                    logger.info(f"Order filled: {fill}")
                    return fill
                elif fill.status == "REJECTED_RISK":
                    logger.warning(f"Order rejected by risk checks: {fill}")
                    return fill # Do not retry risk rejections
                else:
                    logger.warning(f"Order rejected: {fill}")
                    return fill # Do not retry logic rejections (e.g. insufficient funds)
                    
            except Exception as e:
                logger.error(f"Order execution failed (Attempt {attempt + 1}): {e}")
                last_error = e
                attempt += 1
                await asyncio.sleep(self.retry_delay)
        
        # If we exhausted retries
        logger.critical(f"Order failed after {self.max_retries} attempts: {last_error}")
        # Return a failed fill object instead of crashing? 
        # Or raise exception? The interface expects a Fill.
        from datetime import datetime
        import uuid
        return Fill(
            order_id=str(uuid.uuid4()),
            symbol=order.symbol,
            action=order.action,
            quantity=order.quantity,
            price=0,
            timestamp=datetime.now(),
            status="FAILED_ERROR"
        )

from typing import Dict, Any, List, Optional
import logging
from src.core.state import PipelineGraphState as GraphState

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """
    Supervisor agent that coordinates the workflow and manages specialized sub-agents.
    Acts as the central brain for dynamic routing and high-level decision making.
    """
    def __init__(self, llm=None, config: Dict = None):
        self.llm = llm
        self.config = config or {}
        self.sub_agents = {}
        logger.info("SupervisorAgent initialized")

    def register_agent(self, name: str, agent: Any):
        """Register a specialized sub-agent."""
        self.sub_agents[name] = agent
        logger.info(f"Registered sub-agent: {name}")

    def run(self, ctx: Any, symbols: List[str]):
        """
        Execute the mission based on the run context.
        This is the main entry point for the agentic workflow.
        """
        logger.info(f"Supervisor accepting mission: {ctx.run_type.value} (ID: {ctx.run_id})")
        
        # Future: Here we can add dynamic logic to check market status, 
        # resource availability, or override the run_type based on urgency.
        
        # For now, we delegate to the standard pipeline execution, 
        # but the Supervisor is now the explicit controller.
        from src.pipeline_orchestrator import run_pipeline
        
        try:
            run_pipeline(ctx, symbols, self.config)
            logger.info("Mission accomplished.")
        except Exception as e:
            logger.error(f"Mission failed: {e}")
            raise e

    def coordinate_workflow(self, state: GraphState) -> str:
        """
        Determine the next step in the workflow based on the current state.
        Can use LLM for dynamic routing if available.
        """
        # Check feature flag for dynamic routing
        dynamic_routing = self.config.get('orchestrator', {}).get('dynamic_routing_enabled', True) # Default to True for now
        
        # 0. Check run status
        if state.get('run_status') in ['COMPLETED', 'FAILED']:
            return 'end'

        # 1. Check for critical errors or blockers
        if state.get('errors'):
            logger.warning(f"Errors detected: {state['errors']}")
            # Could route to an error handling agent
            
        # 2. Check for HPO triggers (priority)
        # Only trigger if we haven't produced results yet to avoid loops
        if state.get('hpo_triggered') and not state.get('hpo_results'):
            return "hpo"
            
        # 3. Check for Drift triggers
        # Only trigger if we haven't retrained yet
        if state.get('drift_detected') and not state.get('retrained_models'):
            return "retrain"

        # 4. Dynamic Routing based on Forecast Confidence
        if dynamic_routing:
            horizon_forecasts = state.get('horizon_forecasts', {})
            low_confidence_detected = False
            
            for symbol, forecasts in horizon_forecasts.items():
                # forecasts is a list of HorizonForecast objects (or dicts if serialized)
                for forecast in forecasts:
                    # Handle both object and dict access
                    conf = getattr(forecast, 'confidence', None)
                    if conf is None and isinstance(forecast, dict):
                        conf = forecast.get('confidence')
                    
                    if conf == "Low":
                        logger.info(f"Low confidence detected for {symbol}. Routing to Deep Research.")
                        low_confidence_detected = True
                        break
                if low_confidence_detected:
                    break
            
            if low_confidence_detected:
                # Check if we already did deep research to avoid loops
                if not state.get('deep_research_conducted', False):
                    return "deep_research"

        # 5. Definition of Done
        # If we have interpreted forecasts, we proceed to decision making (continue).
        # The graph will end naturally after reporting.
        # if state.get('interpreted_forecasts'):
        #    return "continue"

        # 6. Default workflow progression
        # If we are at the end of a standard flow, check if we need to loop or end
        if state.get('next_step'):
            return state.next_step
            
        return "continue"

    def review_plan(self, state: GraphState) -> Dict[str, Any]:
        """
        Review the current execution plan and suggest adjustments.
        """
        # Placeholder for LLM-based plan review
        return {"status": "approved", "modifications": []}

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system and all registered agents.
        """
        status = {
            "agent_type": "SupervisorAgent",
            "status": "active",
            "sub_agents": list(self.sub_agents.keys())
        }
        return status

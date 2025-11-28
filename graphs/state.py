from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class GraphState:
    """
    State object for the graph workflow.
    """
    messages: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    next_step: Optional[str] = None
    use_spectral: bool = False
    run_hpo: bool = False

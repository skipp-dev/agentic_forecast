import re
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def extract_json_from_llm_output(raw: str) -> str:
    """
    Extract JSON string from LLM output.
    Handles markdown code blocks and raw JSON strings.
    """
    # Try to find JSON inside markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON inside generic code blocks
    code_match = re.search(r'```\s*(.*?)\s*```', raw, re.DOTALL)
    if code_match:
        return code_match.group(1)

    # Fallback: find first { and last }
    start = raw.find('{')
    end = raw.rfind('}')
    
    if start != -1 and end != -1:
        return raw[start:end+1]
        
    return raw

def parse_llm_json(raw: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM output with error handling.
    """
    if default is None:
        default = {}
        
    try:
        json_str = extract_json_from_llm_output(raw)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}. Raw: {raw[:200]}...")
        return default
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM JSON: {e}")
        return default

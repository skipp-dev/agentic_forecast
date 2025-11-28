import os
import yaml
import llama_cpp
import atexit
from typing import Optional, Any
try:
    from src.llm.local_client import LocalLlamaClient
except ImportError:
    try:
        from .llm.local_client import LocalLlamaClient
    except ImportError:
        # Fallback for direct execution
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'llm'))
        from local_client import LocalLlamaClient

from src.config.config_loader import config_loader

# Path to the configuration file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

class LLMFactory:
    _instances = {}
    _config = None

    @classmethod
    def _load_config(cls):
        if cls._config is None:
            cls._config = config_loader.load_master_config()
        return cls._config

    @classmethod
    def get_llm(cls, agent_name: str) -> Any:
        """
        Get an LLM instance for a specific agent role.
        
        Args:
            agent_name: One of 'research_openai', 'planner', 'writer', 'tool_user'
        """
        if agent_name in cls._instances:
            return cls._instances[agent_name]

        config = cls._load_config()
        if "llm" not in config or agent_name not in config["llm"]:
            raise ValueError(f"No configuration found for agent: {agent_name}")

        llm_config = config["llm"][agent_name]
        backend = llm_config.get("backend")

        if backend == "openai":
            # Placeholder for OpenAI integration
            # Assuming langchain or openai package is used
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv(llm_config["api_key_env"])
                instance = ChatOpenAI(
                    model=llm_config["model"],
                    api_key=api_key
                )
            except ImportError:
                # Fallback if langchain is not installed, or return a mock/wrapper
                print("Warning: langchain_openai not found. Returning config dict.")
                instance = llm_config
            
        elif backend == "local":
            # Unload other local models to prevent VRAM overflow
            # We iterate over a copy of keys to avoid runtime error
            for name, inst in list(cls._instances.items()):
                if isinstance(inst, LocalLlamaClient):
                    print(f"Unloading {name} to free VRAM...")
                    inst.close()
                    del cls._instances[name]
            
            # Force garbage collection
            import gc
            gc.collect()

            model_path = llm_config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading local model for {agent_name} from {model_path}...")
            instance = LocalLlamaClient(
                model_path=model_path,
                n_ctx=llm_config.get("n_ctx", 2048),
                n_gpu_layers=llm_config.get("n_gpu_layers", -1),
                verbose=llm_config.get("verbose", False)
            )
            print(f"Model {agent_name} loaded.")
            
        else:
            raise ValueError(f"Unknown backend: {backend}")

        cls._instances[agent_name] = instance
        return instance

    @classmethod
    def unload_all(cls):
        """Unload all loaded models to free VRAM."""
        for name, instance in list(cls._instances.items()):
            if hasattr(instance, "close"):
                print(f"Unloading {name} on exit...")
                instance.close()
        cls._instances.clear()
        import gc
        gc.collect()

# Register cleanup on exit
atexit.register(LLMFactory.unload_all)

if __name__ == "__main__":
    # Test the factory
    print("Testing LLM Factory...")
    
    # Test Tool User (Phi-3 - Fast)
    try:
        tool_llm = LLMFactory.get_llm("tool_user")
        output = tool_llm("Q: Say hello. A: ", max_tokens=10, stop=["\n"])
        print("Tool User Output:", output['choices'][0]['text'])
    except Exception as e:
        print(f"Tool User failed: {e}")

    # Test Planner (Gemma 3 - Reasoning)
    # Note: This might trigger model loading/unloading if VRAM is tight
    # For this test, we just check if it loads.
    # try:
    #     planner_llm = LLMFactory.get_llm("planner")
    #     print("Planner loaded.")
    # except Exception as e:
    #     print(f"Planner failed: {e}")

"""
Documentation Generator

Automated generation of API documentation, deployment guides, and user manuals
for the IB Forecast system.
"""

import os
import sys
import inspect
import json
from typing import Dict, List, Any, Optional, get_type_hints
import logging
from pathlib import Path
from datetime import datetime, timedelta
import markdown
import jinja2

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class DocumentationGenerator:
    """
    Automated documentation generator for the IB Forecast system.

    Generates:
    - API documentation
    - Deployment guides
    - User manuals
    - Architecture diagrams
    - Performance reports
    """

    def __init__(self, docs_path: str = '/tmp/AGENTIC_FORECAST_docs'):
        """
        Initialize documentation generator.

        Args:
            docs_path: Path to store generated documentation
        """
        self.docs_path = Path(docs_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)

        # Template environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('/tmp'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        # Documentation structure
        self.api_docs = {}
        self.deployment_guides = {}
        self.user_manuals = {}

        logger.info(f"Documentation Generator initialized at {docs_path}")

    def generate_api_documentation(self):
        """Generate comprehensive API documentation."""
        logger.info("Generating API documentation...")

        # Import all modules to document
        modules_to_document = [
            'agents.hyperparameter_search_agent',
            'agents.drift_monitor_agent',
            'agents.feature_engineer_agent',
            'agents.forecast_agent',
            'services.gpu_training_service',
            'services.model_registry_service',
            'services.inference_service',
            'data.feature_store',
            'data.metrics_database',
            'src.gpu_services'
        ]

        api_docs = {}

        for module_name in modules_to_document:
            try:
                module = __import__(module_name, fromlist=[''])
                api_docs[module_name] = self._document_module(module)
                logger.info(f"Documented module: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to document {module_name}: {e}")

        # Generate API docs
        self._generate_api_html(api_docs)
        self._generate_api_markdown(api_docs)
        self._generate_openapi_spec(api_docs)

        logger.info("API documentation generated")

    def generate_deployment_guide(self):
        """Generate deployment and installation guides."""
        logger.info("Generating deployment guide...")

        deployment_info = {
            'system_requirements': self._get_system_requirements(),
            'installation_steps': self._get_installation_steps(),
            'configuration_options': self._get_configuration_options(),
            'deployment_scenarios': self._get_deployment_scenarios(),
            'troubleshooting': self._get_troubleshooting_guide()
        }

        # Generate deployment guide
        self._generate_deployment_html(deployment_info)
        self._generate_deployment_markdown(deployment_info)
        self._generate_docker_compose(deployment_info)

        logger.info("Deployment guide generated")

    def generate_user_manual(self):
        """Generate user manuals and tutorials."""
        logger.info("Generating user manual...")

        manual_content = {
            'getting_started': self._get_getting_started(),
            'tutorials': self._get_tutorials(),
            'api_examples': self._get_api_examples(),
            'best_practices': self._get_best_practices(),
            'faq': self._get_faq()
        }

        # Generate user manual
        self._generate_manual_html(manual_content)
        self._generate_manual_markdown(manual_content)

        logger.info("User manual generated")

    def generate_architecture_diagram(self):
        """Generate architecture diagram documentation."""
        logger.info("Generating architecture diagram...")

        # Create PlantUML diagram
        diagram_content = self._create_architecture_diagram()

        # Save diagram
        diagram_path = self.docs_path / 'architecture.puml'
        with open(diagram_path, 'w') as f:
            f.write(diagram_content)

        logger.info(f"Architecture diagram saved to {diagram_path}")

    def generate_performance_report(self, metrics_db):
        """Generate performance report from metrics database."""
        logger.info("Generating performance report...")

        try:
            from data.metrics_database import MetricQuery

            # Query performance metrics
            query = MetricQuery(
                metric_names=['gpu.memory.feature_engineering', 'gpu.memory.training',
                             'gpu.utilization.avg', 'gpu.throughput.samples_per_sec',
                             'gpu.inference.latency.avg', 'pipeline.feature_engineering_time'],
                start_time=datetime.now() - timedelta(days=7),
                end_time=datetime.now()
            )

            metrics = metrics_db.query_metrics(query)

            if not metrics.empty:
                performance_data = self._analyze_performance_metrics(metrics)
                self._generate_performance_html(performance_data)
                self._generate_performance_markdown(performance_data)

                logger.info("Performance report generated")
            else:
                logger.warning("No performance metrics available for report")

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")

    def _document_module(self, module) -> Dict[str, Any]:
        """Document a Python module."""
        module_doc = {
            'name': module.__name__,
            'docstring': module.__doc__ or '',
            'classes': {},
            'functions': {}
        }

        # Document classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                module_doc['classes'][name] = self._document_class(obj)

        # Document functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                module_doc['functions'][name] = self._document_function(obj)

        return module_doc

    def _document_class(self, cls) -> Dict[str, Any]:
        """Document a Python class."""
        class_doc = {
            'name': cls.__name__,
            'docstring': cls.__doc__ or '',
            'methods': {},
            'attributes': {}
        }

        # Document methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_'):
                class_doc['methods'][name] = self._document_function(method)

        # Document attributes
        for name, value in inspect.getmembers(cls):
            if not name.startswith('_') and not callable(value):
                class_doc['attributes'][name] = {
                    'type': str(type(value).__name__),
                    'value': str(value)[:100]  # Truncate long values
                }

        return class_doc

    def _document_function(self, func) -> Dict[str, Any]:
        """Document a Python function."""
        try:
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            func_doc = {
                'name': func.__name__,
                'docstring': func.__doc__ or '',
                'signature': str(sig),
                'parameters': {},
                'return_type': str(type_hints.get('return', 'Any'))
            }

            # Document parameters
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'type': str(type_hints.get(param_name, 'Any')),
                    'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                    'kind': str(param.kind)
                }
                func_doc['parameters'][param_name] = param_info

            return func_doc

        except Exception as e:
            return {
                'name': func.__name__,
                'docstring': func.__doc__ or '',
                'error': str(e)
            }

    def _generate_api_html(self, api_docs: Dict[str, Any]):
        """Generate HTML API documentation."""
        html_content = self._create_api_html_template(api_docs)

        api_html_path = self.docs_path / 'api_documentation.html'
        with open(api_html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"API HTML documentation saved to {api_html_path}")

    def _generate_api_markdown(self, api_docs: Dict[str, Any]):
        """Generate Markdown API documentation."""
        md_content = self._create_api_markdown_template(api_docs)

        api_md_path = self.docs_path / 'API_DOCUMENTATION.md'
        with open(api_md_path, 'w') as f:
            f.write(md_content)

        logger.info(f"API Markdown documentation saved to {api_md_path}")

    def _generate_openapi_spec(self, api_docs: Dict[str, Any]):
        """Generate OpenAPI specification."""
        openapi_spec = {
            'openapi': '3.0.0',
            'info': {
                'title': 'IB Forecast API',
                'version': '1.0.0',
                'description': 'API for the IB Forecast system'
            },
            'paths': {},
            'components': {
                'schemas': {}
            }
        }

        # Convert API docs to OpenAPI format
        for module_name, module_doc in api_docs.items():
            for class_name, class_doc in module_doc.get('classes', {}).items():
                for method_name, method_doc in class_doc.get('methods', {}).items():
                    if method_name in ['predict', 'train_model', 'store_features']:  # API methods
                        path = f'/{module_name.split(".")[-1]}/{method_name}'
                        openapi_spec['paths'][path] = {
                            'post': {
                                'summary': method_doc.get('docstring', '').split('\n')[0],
                                'parameters': [],
                                'responses': {
                                    '200': {'description': 'Success'}
                                }
                            }
                        }

        openapi_path = self.docs_path / 'openapi.json'
        with open(openapi_path, 'w') as f:
            json.dump(openapi_spec, f, indent=2)

        logger.info(f"OpenAPI specification saved to {openapi_path}")

    def _get_system_requirements(self) -> Dict[str, Any]:
        """Get system requirements for deployment."""
        return {
            'hardware': {
                'cpu': 'Intel i5 or equivalent (8+ cores recommended)',
                'ram': '16GB minimum, 32GB recommended',
                'gpu': 'NVIDIA GPU with CUDA support (8GB+ VRAM)',
                'storage': '100GB SSD for data and models'
            },
            'software': {
                'os': 'Ubuntu 20.04+, CentOS 7+, or Windows 10+ with WSL2',
                'python': 'Python 3.8+',
                'cuda': 'CUDA 11.0+',
                'docker': 'Docker 20.0+ (optional but recommended)'
            },
            'dependencies': [
                'torch==2.0.0',
                'tensorflow==2.13.0',
                'numpy==1.24.0',
                'pandas==2.0.0',
                'scikit-learn==1.3.0'
            ]
        }

    def _get_installation_steps(self) -> List[str]:
        """Get installation steps."""
        return [
            'Clone the repository: git clone https://github.com/skipp-dev/AGENTIC_FORECAST.git',
            'Create virtual environment: python -m venv AGENTIC_FORECAST_env',
            'Activate environment: source AGENTIC_FORECAST_env/bin/activate (Linux/Mac) or AGENTIC_FORECAST_env\\Scripts\\activate (Windows)',
            'Install dependencies: pip install -r requirements-gpu.txt',
            'Install CUDA toolkit if not present',
            'Run setup script: python setup.py install',
            'Verify installation: python -c "import torch; print(torch.cuda.is_available())"'
        ]

    def _get_configuration_options(self) -> Dict[str, Any]:
        """Get configuration options."""
        return {
            'gpu_settings': {
                'cuda_visible_devices': 'GPU device IDs to use (comma-separated)',
                'memory_fraction': 'GPU memory fraction to allocate (0.0-1.0)',
                'allow_growth': 'Allow GPU memory to grow dynamically'
            },
            'model_settings': {
                'batch_size': 'Training batch size (16-512)',
                'learning_rate': 'Initial learning rate (0.0001-0.01)',
                'epochs': 'Maximum training epochs (10-1000)'
            },
            'data_settings': {
                'cache_size': 'Feature cache size in MB',
                'retention_days': 'Data retention period in days',
                'parallel_workers': 'Number of parallel data loading workers'
            }
        }

    def _get_deployment_scenarios(self) -> Dict[str, Any]:
        """Get deployment scenarios."""
        return {
            'development': {
                'description': 'Single-machine development setup',
                'components': ['All services on localhost'],
                'scaling': 'No scaling, single GPU'
            },
            'production_single': {
                'description': 'Single-server production deployment',
                'components': ['All services containerized with Docker'],
                'scaling': 'Single GPU, horizontal scaling possible'
            },
            'production_distributed': {
                'description': 'Distributed production deployment',
                'components': ['Kubernetes orchestration', 'Multiple GPU nodes'],
                'scaling': 'Auto-scaling, load balancing'
            },
            'cloud': {
                'description': 'Cloud-native deployment',
                'components': ['AWS ECS/EKS, GCP Cloud Run/GKE, Azure ACI/AKS'],
                'scaling': 'Serverless scaling, multi-region'
            }
        }

    def _get_troubleshooting_guide(self) -> Dict[str, Any]:
        """Get troubleshooting guide."""
        return {
            'gpu_issues': {
                'cuda_not_found': 'Install CUDA toolkit and ensure PATH is set correctly',
                'memory_errors': 'Reduce batch size or use gradient checkpointing',
                'driver_issues': 'Update GPU drivers to latest version'
            },
            'data_issues': {
                'api_limits': 'Implement rate limiting and caching',
                'data_quality': 'Add data validation and cleaning steps',
                'storage_full': 'Implement data retention policies'
            },
            'performance_issues': {
                'slow_training': 'Use mixed precision training, optimize batch size',
                'high_latency': 'Implement model caching and batching',
                'memory_leaks': 'Use PyTorch\'s memory profiler and fix leaks'
            }
        }

    def _create_architecture_diagram(self) -> str:
        """Create PlantUML architecture diagram."""
        return """
@startuml IB Forecast Architecture

title IB Forecast System Architecture

package "User Interface" as UI {
    [Web Dashboard]
    [API Gateway]
    [CLI Tools]
}

package "Core Services" as Services {
    [Inference Service] as IS
    [GPU Training Service] as GTS
    [Model Registry Service] as MRS
}

package "Intelligent Agents" as Agents {
    [Hyperparameter Search Agent] as HSA
    [Drift Monitor Agent] as DMA
    [Feature Engineer Agent] as FEA
    [Forecast Agent] as FA
}

package "Data Layer" as Data {
    [Time-Series Feature Store] as FS
    [Metrics Database] as MD
}

package "Infrastructure" as Infra {
    [GPU Services] as GS
    [Data Pipeline] as DP
    [External APIs] as API
}

UI --> IS
IS --> MRS
IS --> FEA
GTS --> GS
GTS --> MRS
HSA --> GTS
DMA --> MD
FEA --> FS
FA --> IS
FS --> DP
MD --> DP
DP --> API

note right of GS
  GPU acceleration,
  CUDA optimization,
  Memory management
end note

note right of FS
  Time-series storage,
  Feature versioning,
  Partitioning
end note

@enduml
"""

    def _create_api_html_template(self, api_docs: Dict[str, Any]) -> str:
        """Create HTML template for API documentation."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IB Forecast API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .module {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; }}
        .class {{ margin-left: 20px; margin-bottom: 20px; }}
        .function {{ margin-left: 40px; margin-bottom: 10px; }}
        .signature {{ font-family: monospace; background: #f5f5f5; padding: 5px; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>IB Forecast API Documentation</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        for module_name, module_doc in api_docs.items():
            html += f"""
    <div class="module">
        <h2>Module: {module_name}</h2>
        <p>{module_doc.get('docstring', 'No description available')}</p>
"""

            for class_name, class_doc in module_doc.get('classes', {}).items():
                html += f"""
        <div class="class">
            <h3>Class: {class_name}</h3>
            <p>{class_doc.get('docstring', 'No description available')}</p>
"""

                for method_name, method_doc in class_doc.get('methods', {}).items():
                    html += f"""
            <div class="function">
                <h4>Method: {method_name}</h4>
                <div class="signature">{method_doc.get('signature', 'N/A')}</div>
                <p>{method_doc.get('docstring', 'No description available')}</p>
            </div>
"""

                html += "        </div>"

            html += "    </div>"

        html += """
</body>
</html>
"""

        return html

    def _create_api_markdown_template(self, api_docs: Dict[str, Any]) -> str:
        """Create Markdown template for API documentation."""
        md = f"""# IB Forecast API Documentation

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        for module_name, module_doc in api_docs.items():
            md += f"""## Module: {module_name}

{module_doc.get('docstring', 'No description available')}

"""

            for class_name, class_doc in module_doc.get('classes', {}).items():
                md += f"""### Class: {class_name}

{class_doc.get('docstring', 'No description available')}

"""

                for method_name, method_doc in class_doc.get('methods', {}).items():
                    md += f"""#### Method: {method_name}

**Signature:** `{method_doc.get('signature', 'N/A')}`

{method_doc.get('docstring', 'No description available')}

"""

        return md

    def _generate_deployment_html(self, deployment_info: Dict[str, Any]):
        """Generate HTML deployment guide."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IB Forecast Deployment Guide</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin-bottom: 30px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f5f5f5; padding: 2px 5px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>IB Forecast Deployment Guide</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="section">
        <h2>System Requirements</h2>
        <h3>Hardware</h3>
        <ul>
"""

        for req, value in deployment_info['system_requirements']['hardware'].items():
            html += f"            <li><strong>{req.upper()}:</strong> {value}</li>\n"

        html += """
        </ul>
        <h3>Software</h3>
        <ul>
"""

        for req, value in deployment_info['system_requirements']['software'].items():
            html += f"            <li><strong>{req.upper()}:</strong> {value}</li>\n"

        html += """
        </ul>
    </div>

    <div class="section">
        <h2>Installation Steps</h2>
        <ol>
"""

        for step in deployment_info['installation_steps']:
            html += f"            <li>{step}</li>\n"

        html += """
        </ol>
    </div>

    <div class="section">
        <h2>Configuration Options</h2>
"""

        for category, options in deployment_info['configuration_options'].items():
            html += f"        <h3>{category.replace('_', ' ').title()}</h3>\n        <ul>\n"

            for option, description in options.items():
                html += f"            <li><code>{option}</code>: {description}</li>\n"

            html += "        </ul>\n"

        html += """
    </div>
</body>
</html>
"""

        deployment_html_path = self.docs_path / 'deployment_guide.html'
        with open(deployment_html_path, 'w') as f:
            f.write(html)

        logger.info(f"Deployment HTML guide saved to {deployment_html_path}")

    def _generate_deployment_markdown(self, deployment_info: Dict[str, Any]):
        """Generate Markdown deployment guide."""
        md = f"""# IB Forecast Deployment Guide

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Requirements

### Hardware
"""

        for req, value in deployment_info['system_requirements']['hardware'].items():
            md += f"- **{req.upper()}**: {value}\n"

        md += "\n### Software\n"

        for req, value in deployment_info['system_requirements']['software'].items():
            md += f"- **{req.upper()}**: {value}\n"

        md += "\n## Installation Steps\n\n"

        for i, step in enumerate(deployment_info['installation_steps'], 1):
            md += f"{i}. {step}\n"

        md += "\n## Configuration Options\n\n"

        for category, options in deployment_info['configuration_options'].items():
            md += f"### {category.replace('_', ' ').title()}\n\n"

            for option, description in options.items():
                md += f"- `{option}`: {description}\n"

            md += "\n"

        deployment_md_path = self.docs_path / 'DEPLOYMENT_GUIDE.md'
        with open(deployment_md_path, 'w') as f:
            f.write(md)

        logger.info(f"Deployment Markdown guide saved to {deployment_md_path}")

    def _get_getting_started(self) -> str:
        """Get getting started guide."""
        return """
# Getting Started with IB Forecast

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements-gpu.txt
   ```

2. **Basic Usage**
   ```python
   from agents.forecast_agent import ForecastAgent

   agent = ForecastAgent()
   forecast = agent.generate_forecast('AAPL', horizon=1)
   print(f"AAPL forecast: {forecast['prediction']:.4f}")
   ```

3. **Advanced Usage**
   ```python
   from services.inference_service import InferenceService

   service = InferenceService()
   result = await service.predict_async({'symbol': 'AAPL'})
   print(f"Prediction: {result.prediction}")
   ```
"""

    def _get_tutorials(self) -> List[Dict[str, str]]:
        """Get tutorial content."""
        return [
            {
                'title': 'Training Your First Model',
                'content': '''
## Training Your First Model

1. **Prepare Data**
   ```python
   from agents.feature_engineer_agent import FeatureEngineerAgent

   engineer = FeatureEngineerAgent()
   features = engineer.engineer_features('AAPL')
   ```

2. **Train Model**
   ```python
   from services.gpu_training_service import GPUTrainingService

   trainer = GPUTrainingService()
   results = trainer.train_model('AAPL', {'type': 'lstm'})
   ```

3. **Monitor Training**
   ```python
   from data.metrics_database import MetricsDatabase

   metrics = MetricsDatabase()
   stats = metrics.get_metric_stats('gpu.memory.training')
   ```
'''
            },
            {
                'title': 'Real-time Forecasting',
                'content': '''
## Real-time Forecasting

1. **Set Up Inference Service**
   ```python
   from services.inference_service import InferenceService

   service = InferenceService()
   ```

2. **Make Predictions**
   ```python
   result = await service.predict_async({'symbol': 'AAPL'})
   print(f"Price prediction: ${result.prediction:.2f}")
   ```

3. **Batch Predictions**
   ```python
   results = service.predict_batch([
       {'symbol': 'AAPL'},
       {'symbol': 'GOOGL'},
       {'symbol': 'MSFT'}
   ])
   ```
'''
            }
        ]

    def _get_api_examples(self) -> List[Dict[str, str]]:
        """Get API usage examples."""
        return [
            {
                'title': 'Feature Engineering',
                'code': '''
from agents.feature_engineer_agent import FeatureEngineerAgent

# Initialize agent
engineer = FeatureEngineerAgent()

# Engineer features for a symbol
features = engineer.engineer_features('AAPL', feature_sets=['basic', 'spectral'])

# Select best features
selected = engineer.select_features(features.drop('target', axis=1), features['target'])
'''
            },
            {
                'title': 'Model Training',
                'code': '''
from services.gpu_training_service import GPUTrainingService

# Initialize training service
trainer = GPUTrainingService()

# Configure model
model_config = {
    'type': 'lstm',
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2
}

# Train model
results = trainer.train_model('AAPL', model_config)
print(f"Training completed with MAE: {results['final_metrics']['mae']:.4f}")
'''
            }
        ]

    def _get_best_practices(self) -> List[str]:
        """Get best practices."""
        return [
            "Use GPU acceleration for all training and inference operations",
            "Implement proper data validation and error handling",
            "Monitor system performance and resource usage",
            "Use feature versioning and model lineage tracking",
            "Implement proper logging and monitoring",
            "Regularly update and retrain models",
            "Use appropriate batch sizes for optimal GPU utilization",
            "Implement data retention policies to manage storage",
            "Use async operations for better performance",
            "Validate model performance before deployment"
        ]

    def _get_faq(self) -> List[Dict[str, str]]:
        """Get frequently asked questions."""
        return [
            {
                'question': 'How do I optimize GPU memory usage?',
                'answer': 'Use gradient checkpointing, reduce batch sizes, and implement proper memory cleanup. Monitor memory usage with torch.cuda.memory_summary().'
            },
            {
                'question': 'What should I do if training is slow?',
                'answer': 'Check GPU utilization, optimize batch size, use mixed precision training, and ensure data loading is not a bottleneck.'
            },
            {
                'question': 'How do I handle model drift?',
                'answer': 'Use the Drift Monitor Agent to detect performance and data drift. Retrain models when drift exceeds thresholds.'
            },
            {
                'question': 'Can I deploy on multiple GPUs?',
                'answer': 'Yes, the system supports distributed training. Configure CUDA_VISIBLE_DEVICES and use the distributed training options.'
            }
        ]

    def _generate_manual_html(self, manual_content: Dict[str, Any]):
        """Generate HTML user manual."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IB Forecast User Manual</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin-bottom: 30px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f5f5f5; padding: 2px 5px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
        .tutorial {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>IB Forecast User Manual</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="section">
        <h2>Getting Started</h2>
        <pre>{manual_content['getting_started']}</pre>
    </div>

    <div class="section">
        <h2>Tutorials</h2>
"""

        for tutorial in manual_content['tutorials']:
            html += f"""
        <div class="tutorial">
            <h3>{tutorial['title']}</h3>
            <pre>{tutorial['content']}</pre>
        </div>
"""

        html += """
    </div>

    <div class="section">
        <h2>API Examples</h2>
"""

        for example in manual_content['api_examples']:
            html += f"""
        <div class="tutorial">
            <h3>{example['title']}</h3>
            <pre>{example['code']}</pre>
        </div>
"""

        html += """
    </div>

    <div class="section">
        <h2>Best Practices</h2>
        <ul>
"""

        for practice in manual_content['best_practices']:
            html += f"            <li>{practice}</li>\n"

        html += """
        </ul>
    </div>

    <div class="section">
        <h2>Frequently Asked Questions</h2>
"""

        for faq in manual_content['faq']:
            html += f"""
        <div class="tutorial">
            <h3>{faq['question']}</h3>
            <p>{faq['answer']}</p>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        manual_html_path = self.docs_path / 'user_manual.html'
        with open(manual_html_path, 'w') as f:
            f.write(html)

        logger.info(f"User manual HTML saved to {manual_html_path}")

    def _generate_manual_markdown(self, manual_content: Dict[str, Any]):
        """Generate Markdown user manual."""
        md = f"""# IB Forecast User Manual

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{manual_content['getting_started']}

## Tutorials

"""

        for tutorial in manual_content['tutorials']:
            md += f"""### {tutorial['title']}

{tutorial['content']}

"""

        md += "## API Examples\n\n"

        for example in manual_content['api_examples']:
            md += f"""### {example['title']}

```python
{example['code']}
```

"""

        md += "## Best Practices\n\n"

        for practice in manual_content['best_practices']:
            md += f"- {practice}\n"

        md += "\n## Frequently Asked Questions\n\n"

        for faq in manual_content['faq']:
            md += f"""### {faq['question']}

{faq['answer']}

"""

        manual_md_path = self.docs_path / 'USER_MANUAL.md'
        with open(manual_md_path, 'w') as f:
            f.write(md)

        logger.info(f"User manual Markdown saved to {manual_md_path}")

    def _analyze_performance_metrics(self, metrics_df) -> Dict[str, Any]:
        """Analyze performance metrics for reporting."""
        analysis = {}

        if not metrics_df.empty:
            # Group by metric type
            for metric_name in metrics_df['metric_name'].unique():
                metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
                analysis[metric_name] = {
                    'mean': metric_data['value'].mean(),
                    'std': metric_data['value'].std(),
                    'min': metric_data['value'].min(),
                    'max': metric_data['value'].max(),
                    'count': len(metric_data)
                }

        return analysis

    def _generate_performance_html(self, performance_data: Dict[str, Any]):
        """Generate HTML performance report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IB Forecast Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>IB Forecast Performance Report</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Performance Metrics Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Std Dev</th>
            <th>Min</th>
            <th>Max</th>
            <th>Count</th>
        </tr>
"""

        for metric_name, stats in performance_data.items():
            html += f"""
        <tr>
            <td>{metric_name}</td>
            <td>{stats['mean']:.4f}</td>
            <td>{stats['std']:.4f}</td>
            <td>{stats['min']:.4f}</td>
            <td>{stats['max']:.4f}</td>
            <td>{stats['count']}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        perf_html_path = self.docs_path / 'performance_report.html'
        with open(perf_html_path, 'w') as f:
            f.write(html)

        logger.info(f"Performance HTML report saved to {perf_html_path}")

    def _generate_performance_markdown(self, performance_data: Dict[str, Any]):
        """Generate Markdown performance report."""
        md = f"""# IB Forecast Performance Report

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics Summary

| Metric | Mean | Std Dev | Min | Max | Count |
|--------|------|---------|-----|-----|-------|
"""

        for metric_name, stats in performance_data.items():
            md += f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"

        md += "\n## Recommendations\n\n"

        # Add performance recommendations based on metrics
        if performance_data:
            memory_metrics = [k for k in performance_data.keys() if 'memory' in k]
            if memory_metrics:
                avg_memory = sum(performance_data[m]['mean'] for m in memory_metrics) / len(memory_metrics)
                if avg_memory > 4.0:
                    md += "- Consider reducing batch sizes or implementing gradient checkpointing to reduce memory usage\n"

            latency_metrics = [k for k in performance_data.keys() if 'latency' in k]
            if latency_metrics:
                avg_latency = sum(performance_data[m]['mean'] for m in latency_metrics) / len(latency_metrics)
                if avg_latency > 1.0:
                    md += "- Inference latency is high; consider model optimization or caching strategies\n"

        perf_md_path = self.docs_path / 'PERFORMANCE_REPORT.md'
        with open(perf_md_path, 'w') as f:
            f.write(md)

        logger.info(f"Performance Markdown report saved to {perf_md_path}")

    def _generate_docker_compose(self, deployment_info: Dict[str, Any]):
        """Generate Docker Compose configuration."""
        compose_config = {
            'version': '3.8',
            'services': {
                'agentic-forecast': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.optimized'
                    },
                    'ports': ['8000:8000'],
                    'environment': [
                        'CUDA_VISIBLE_DEVICES=0',
                        'PYTHONPATH=/workspaces/AGENTIC_FORECAST'
                    ],
                    'volumes': [
                        './data:/app/data',
                        './models:/app/models'
                    ],
                    'deploy': {
                        'resources': {
                            'reservations': {
                                'devices': [{
                                    'driver': 'nvidia',
                                    'count': 1,
                                    'capabilities': [['gpu']]
                                }]
                            }
                        }
                    }
                }
            }
        }

        import yaml
        compose_path = self.docs_path / 'docker-compose.yml'
        with open(compose_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)

        logger.info(f"Docker Compose configuration saved to {compose_path}")

# Convenience functions
def generate_full_documentation(docs_path: str = '/tmp/AGENTIC_FORECAST_docs'):
    """Generate complete documentation suite."""
    generator = DocumentationGenerator(docs_path)

    # Generate all documentation
    generator.generate_api_documentation()
    generator.generate_deployment_guide()
    generator.generate_user_manual()
    generator.generate_architecture_diagram()

    # Generate performance report if metrics available
    try:
        from data.metrics_database import MetricsDatabase
        metrics_db = MetricsDatabase()
        generator.generate_performance_report(metrics_db)
    except Exception as e:
        logger.warning(f"Could not generate performance report: {e}")

    logger.info(f"Complete documentation generated in {docs_path}")

if __name__ == '__main__':
    # Generate documentation
    generate_full_documentation()

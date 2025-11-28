"""
Enterprise Integration Service

Handles enterprise integrations, API connections, and data synchronization.
Provides adapters for various enterprise systems and protocols.
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis
import kafka
from kafka import KafkaProducer, KafkaConsumer
import pika
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage
import paramiko
import ftplib

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class IntegrationAdapter(ABC):
    """Base class for integration adapters."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the external system."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection to the external system."""
        pass

    @abstractmethod
    async def send_data(self, data: Any, endpoint: str = None) -> Dict[str, Any]:
        """Send data to the external system."""
        pass

    @abstractmethod
    async def receive_data(self, endpoint: str = None) -> Any:
        """Receive data from the external system."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get connection status and health information."""
        pass

class RESTAPIAdapter(IntegrationAdapter):
    """REST API integration adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.base_url = config['base_url']
        self.auth = self._setup_auth()
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)

    def _setup_auth(self):
        """Setup authentication."""
        auth_type = self.config.get('auth_type', 'none')

        if auth_type == 'basic':
            return aiohttp.BasicAuth(
                self.config['username'],
                self.config['password']
            )
        elif auth_type == 'bearer':
            self.headers['Authorization'] = f"Bearer {self.config['token']}"
            return None
        elif auth_type == 'api_key':
            self.headers[self.config['api_key_header']] = self.config['api_key']
            return None

        return None

    async def connect(self) -> bool:
        """Establish HTTP connection."""
        try:
            self.session = aiohttp.ClientSession(
                auth=self.auth,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to REST API: {e}")
            return False

    async def disconnect(self):
        """Close HTTP connection."""
        if self.session:
            await self.session.close()
            self.session = None

    async def send_data(self, data: Any, endpoint: str = None) -> Dict[str, Any]:
        """Send data via REST API."""
        if not self.session:
            return {'success': False, 'error': 'Not connected'}

        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url

        try:
            if isinstance(data, dict):
                data = json.dumps(data)
                headers = {'Content-Type': 'application/json'}
            else:
                headers = {'Content-Type': 'application/octet-stream'}

            async with self.session.post(url, data=data, headers=headers) as response:
                result = await response.json() if response.content_type == 'application/json' else await response.text()
                return {
                    'success': response.status < 400,
                    'status_code': response.status,
                    'data': result
                }

        except Exception as e:
            logger.error(f"Failed to send data to REST API: {e}")
            return {'success': False, 'error': str(e)}

    async def receive_data(self, endpoint: str = None) -> Any:
        """Receive data via REST API."""
        if not self.session:
            return {'success': False, 'error': 'Not connected'}

        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url

        try:
            async with self.session.get(url) as response:
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    return await response.text()

        except Exception as e:
            logger.error(f"Failed to receive data from REST API: {e}")
            return {'success': False, 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get REST API connection status."""
        return {
            'connected': self.session is not None and not self.session.closed,
            'base_url': self.base_url,
            'auth_type': self.config.get('auth_type', 'none')
        }

class SOAPAdapter(IntegrationAdapter):
    """SOAP API integration adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.wsdl_url = config['wsdl_url']
        self.endpoint = config.get('endpoint')
        self.auth = self._setup_auth()

    def _setup_auth(self):
        """Setup SOAP authentication."""
        # Implement SOAP authentication (WS-Security, etc.)
        return None

    async def connect(self) -> bool:
        """Establish SOAP connection."""
        try:
            # In a real implementation, you'd use zeep or similar SOAP library
            self.session = aiohttp.ClientSession(auth=self.auth)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SOAP service: {e}")
            return False

    async def disconnect(self):
        """Close SOAP connection."""
        if self.session:
            await self.session.close()
            self.session = None

    async def send_data(self, data: Any, endpoint: str = None) -> Dict[str, Any]:
        """Send SOAP request."""
        # Implement SOAP request sending
        return {'success': False, 'error': 'SOAP implementation pending'}

    async def receive_data(self, endpoint: str = None) -> Any:
        """Receive SOAP response."""
        # Implement SOAP response receiving
        return {'success': False, 'error': 'SOAP implementation pending'}

    def get_status(self) -> Dict[str, Any]:
        """Get SOAP connection status."""
        return {
            'connected': self.session is not None and not self.session.closed,
            'wsdl_url': self.wsdl_url
        }

class KafkaAdapter(IntegrationAdapter):
    """Apache Kafka integration adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = None
        self.consumer = None
        self.bootstrap_servers = config['bootstrap_servers']
        self.group_id = config.get('group_id', 'AGENTIC_FORECAST')

    async def connect(self) -> bool:
        """Establish Kafka connection."""
        try:
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )

            # Initialize consumer if needed
            if self.config.get('consume_topics'):
                self.consumer = KafkaConsumer(
                    *self.config['consume_topics'],
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )

            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    async def disconnect(self):
        """Close Kafka connection."""
        if self.producer:
            self.producer.close()
            self.producer = None

        if self.consumer:
            self.consumer.close()
            self.consumer = None

    async def send_data(self, data: Any, endpoint: str = None) -> Dict[str, Any]:
        """Send message to Kafka topic."""
        if not self.producer:
            return {'success': False, 'error': 'Producer not connected'}

        topic = endpoint or self.config.get('default_topic', 'AGENTIC_FORECAST')

        try:
            future = self.producer.send(topic, data)
            record_metadata = future.get(timeout=10)

            return {
                'success': True,
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset
            }

        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return {'success': False, 'error': str(e)}

    async def receive_data(self, endpoint: str = None) -> Any:
        """Receive message from Kafka topic."""
        if not self.consumer:
            return {'success': False, 'error': 'Consumer not connected'}

        try:
            # Poll for messages
            message = self.consumer.poll(timeout_ms=1000)

            if message:
                # Get the first message from any partition
                for topic_partition, messages in message.items():
                    for msg in messages:
                        return msg.value

            return None

        except Exception as e:
            logger.error(f"Failed to receive message from Kafka: {e}")
            return {'success': False, 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get Kafka connection status."""
        return {
            'producer_connected': self.producer is not None,
            'consumer_connected': self.consumer is not None,
            'bootstrap_servers': self.bootstrap_servers
        }

class DatabaseAdapter(IntegrationAdapter):
    """Database integration adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.db_type = config['type']  # 'postgresql', 'mysql', 'oracle', etc.
        self.connection_string = config['connection_string']

    async def connect(self) -> bool:
        """Establish database connection."""
        try:
            if self.db_type == 'postgresql':
                import asyncpg
                self.connection = await asyncpg.connect(self.connection_string)
            elif self.db_type == 'mysql':
                import aiomysql
                # Parse connection string and connect
                self.connection = await aiomysql.connect(
                    host=self.config['host'],
                    port=self.config.get('port', 3306),
                    user=self.config['username'],
                    password=self.config['password'],
                    db=self.config['database']
                )
            else:
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    async def disconnect(self):
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None

    async def send_data(self, data: Any, endpoint: str = None) -> Dict[str, Any]:
        """Execute database query/insert."""
        if not self.connection:
            return {'success': False, 'error': 'Not connected'}

        try:
            if isinstance(data, str):
                # Execute raw query
                if self.db_type == 'postgresql':
                    result = await self.connection.execute(data)
                else:
                    async with self.connection.cursor() as cursor:
                        await cursor.execute(data)
                        result = await cursor.fetchall()
            else:
                # Assume data is dict for insert
                table = endpoint or self.config.get('default_table')
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['$' + str(i+1) for i in range(len(data))])
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

                if self.db_type == 'postgresql':
                    await self.connection.execute(query, *data.values())
                else:
                    async with self.connection.cursor() as cursor:
                        await cursor.execute(query, list(data.values()))

                result = "Insert successful"

            return {'success': True, 'result': result}

        except Exception as e:
            logger.error(f"Failed to execute database operation: {e}")
            return {'success': False, 'error': str(e)}

    async def receive_data(self, endpoint: str = None) -> Any:
        """Execute database query."""
        if not self.connection:
            return {'success': False, 'error': 'Not connected'}

        try:
            query = endpoint  # Assume endpoint is the query

            if self.db_type == 'postgresql':
                result = await self.connection.fetch(query)
                return [dict(row) for row in result]
            else:
                async with self.connection.cursor() as cursor:
                    await cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = await cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to execute database query: {e}")
            return {'success': False, 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get database connection status."""
        return {
            'connected': self.connection is not None,
            'db_type': self.db_type
        }

class CloudStorageAdapter(IntegrationAdapter):
    """Cloud storage integration adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config['provider']  # 'aws', 'azure', 'gcp'
        self.client = None
        self.bucket = config['bucket']

    async def connect(self) -> bool:
        """Establish cloud storage connection."""
        try:
            if self.provider == 'aws':
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.config['access_key'],
                    aws_secret_access_key=self.config['secret_key'],
                    region_name=self.config.get('region', 'us-east-1')
                )
            elif self.provider == 'azure':
                account_url = f"https://{self.config['account_name']}.blob.core.windows.net"
                self.client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.config['account_key']
                )
            elif self.provider == 'gcp':
                self.client = storage.Client.from_service_account_json(
                    self.config['service_account_file']
                )

            return True
        except Exception as e:
            logger.error(f"Failed to connect to cloud storage: {e}")
            return False

    async def disconnect(self):
        """Close cloud storage connection."""
        # Cloud clients don't need explicit disconnection
        pass

    async def send_data(self, data: Any, endpoint: str = None) -> Dict[str, Any]:
        """Upload data to cloud storage."""
        if not self.client:
            return {'success': False, 'error': 'Not connected'}

        key = endpoint or f"data_{datetime.now().isoformat()}.json"

        try:
            if isinstance(data, dict):
                data = json.dumps(data)

            if self.provider == 'aws':
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=data.encode('utf-8')
                )
            elif self.provider == 'azure':
                blob_client = self.client.get_blob_client(
                    container=self.bucket, blob=key
                )
                blob_client.upload_blob(data, overwrite=True)
            elif self.provider == 'gcp':
                bucket = self.client.bucket(self.bucket)
                blob = bucket.blob(key)
                blob.upload_from_string(data)

            return {'success': True, 'key': key}

        except Exception as e:
            logger.error(f"Failed to upload to cloud storage: {e}")
            return {'success': False, 'error': str(e)}

    async def receive_data(self, endpoint: str = None) -> Any:
        """Download data from cloud storage."""
        if not self.client or not endpoint:
            return {'success': False, 'error': 'Not connected or no key specified'}

        try:
            if self.provider == 'aws':
                response = self.client.get_object(Bucket=self.bucket, Key=endpoint)
                data = response['Body'].read().decode('utf-8')
            elif self.provider == 'azure':
                blob_client = self.client.get_blob_client(
                    container=self.bucket, blob=endpoint
                )
                data = blob_client.download_blob().readall().decode('utf-8')
            elif self.provider == 'gcp':
                bucket = self.client.bucket(self.bucket)
                blob = bucket.blob(endpoint)
                data = blob.download_as_text()

            # Try to parse as JSON
            try:
                return json.loads(data)
            except:
                return data

        except Exception as e:
            logger.error(f"Failed to download from cloud storage: {e}")
            return {'success': False, 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get cloud storage connection status."""
        return {
            'connected': self.client is not None,
            'provider': self.provider,
            'bucket': self.bucket
        }

class EnterpriseIntegrationService:
    """
    Enterprise integration service.

    Provides:
    - Multiple integration adapters
    - Data synchronization
    - Enterprise system connectivity
    - Message queuing
    - File transfer protocols
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enterprise integration service.

        Args:
            config: Service configuration
        """
        self.config = config or {
            'redis_url': 'redis://localhost:6379',
            'sync_interval': 300,  # 5 minutes
            'max_retries': 3,
            'retry_delay': 60,  # 1 minute
            'batch_size': 1000
        }

        # Initialize components
        self.redis_client = redis.Redis.from_url(self.config['redis_url'])
        self.adapters = {}
        self.sync_tasks = {}

        # Setup adapters
        self._setup_adapters()

        logger.info("Enterprise Integration Service initialized")

    def _setup_adapters(self):
        """Setup integration adapters."""
        # This would be configured from external config file
        # For demo, we'll initialize empty adapters dict
        pass

    def register_adapter(self, name: str, adapter_type: str, config: Dict[str, Any]):
        """
        Register an integration adapter.

        Args:
            name: Adapter name
            adapter_type: Type of adapter ('rest', 'soap', 'kafka', 'database', 'cloud')
            config: Adapter configuration
        """
        if adapter_type == 'rest':
            adapter = RESTAPIAdapter(config)
        elif adapter_type == 'soap':
            adapter = SOAPAdapter(config)
        elif adapter_type == 'kafka':
            adapter = KafkaAdapter(config)
        elif adapter_type == 'database':
            adapter = DatabaseAdapter(config)
        elif adapter_type == 'cloud':
            adapter = CloudStorageAdapter(config)
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        self.adapters[name] = adapter
        logger.info(f"Registered {adapter_type} adapter: {name}")

    async def connect_adapter(self, name: str) -> bool:
        """
        Connect an adapter.

        Args:
            name: Adapter name

        Returns:
            Connection success
        """
        if name not in self.adapters:
            logger.error(f"Adapter {name} not found")
            return False

        return await self.adapters[name].connect()

    async def disconnect_adapter(self, name: str):
        """
        Disconnect an adapter.

        Args:
            name: Adapter name
        """
        if name in self.adapters:
            await self.adapters[name].disconnect()

    async def send_to_adapter(self, adapter_name: str, data: Any,
                            endpoint: str = None) -> Dict[str, Any]:
        """
        Send data through an adapter.

        Args:
            adapter_name: Name of the adapter
            data: Data to send
            endpoint: Target endpoint

        Returns:
            Send result
        """
        if adapter_name not in self.adapters:
            return {'success': False, 'error': 'Adapter not found'}

        adapter = self.adapters[adapter_name]

        # Retry logic
        for attempt in range(self.config['max_retries']):
            try:
                result = await adapter.send_data(data, endpoint)
                if result.get('success'):
                    return result

                if attempt < self.config['max_retries'] - 1:
                    await asyncio.sleep(self.config['retry_delay'])

            except Exception as e:
                logger.error(f"Send attempt {attempt + 1} failed: {e}")
                if attempt < self.config['max_retries'] - 1:
                    await asyncio.sleep(self.config['retry_delay'])

        return {'success': False, 'error': 'All send attempts failed'}

    async def receive_from_adapter(self, adapter_name: str,
                                 endpoint: str = None) -> Any:
        """
        Receive data from an adapter.

        Args:
            adapter_name: Name of the adapter
            endpoint: Source endpoint

        Returns:
            Received data
        """
        if adapter_name not in self.adapters:
            return {'success': False, 'error': 'Adapter not found'}

        adapter = self.adapters[adapter_name]

        for attempt in range(self.config['max_retries']):
            try:
                result = await adapter.receive_data(endpoint)
                if result is not None:
                    return result

                if attempt < self.config['max_retries'] - 1:
                    await asyncio.sleep(self.config['retry_delay'])

            except Exception as e:
                logger.error(f"Receive attempt {attempt + 1} failed: {e}")
                if attempt < self.config['max_retries'] - 1:
                    await asyncio.sleep(self.config['retry_delay'])

        return {'success': False, 'error': 'All receive attempts failed'}

    def get_adapter_status(self, name: str = None) -> Dict[str, Any]:
        """
        Get adapter status.

        Args:
            name: Specific adapter name (optional)

        Returns:
            Status information
        """
        if name:
            if name in self.adapters:
                return self.adapters[name].get_status()
            else:
                return {'error': 'Adapter not found'}
        else:
            return {name: adapter.get_status() for name, adapter in self.adapters.items()}

    async def sync_data(self, source_adapter: str, target_adapter: str,
                       source_endpoint: str = None, target_endpoint: str = None,
                       transform_func: callable = None) -> Dict[str, Any]:
        """
        Synchronize data between adapters.

        Args:
            source_adapter: Source adapter name
            target_adapter: Target adapter name
            source_endpoint: Source endpoint
            target_endpoint: Target endpoint
            transform_func: Data transformation function

        Returns:
            Sync result
        """
        # Receive data from source
        source_data = await self.receive_from_adapter(source_adapter, source_endpoint)
        if not source_data or isinstance(source_data, dict) and not source_data.get('success'):
            return {'success': False, 'error': 'Failed to receive data from source'}

        # Apply transformation if provided
        if transform_func:
            try:
                source_data = transform_func(source_data)
            except Exception as e:
                return {'success': False, 'error': f'Transformation failed: {e}'}

        # Send data to target
        result = await self.send_to_adapter(target_adapter, source_data, target_endpoint)

        return result

    def schedule_sync_task(self, name: str, source_adapter: str, target_adapter: str,
                          interval: int = None, **kwargs):
        """
        Schedule a data synchronization task.

        Args:
            name: Task name
            source_adapter: Source adapter name
            target_adapter: Target adapter name
            interval: Sync interval in seconds
            **kwargs: Additional sync parameters
        """
        if interval is None:
            interval = self.config['sync_interval']

        async def sync_task():
            while True:
                try:
                    result = await self.sync_data(source_adapter, target_adapter, **kwargs)
                    if result.get('success'):
                        logger.info(f"Sync task {name} completed successfully")
                    else:
                        logger.error(f"Sync task {name} failed: {result.get('error')}")

                except Exception as e:
                    logger.error(f"Sync task {name} error: {e}")

                await asyncio.sleep(interval)

        self.sync_tasks[name] = asyncio.create_task(sync_task())
        logger.info(f"Scheduled sync task: {name}")

    def cancel_sync_task(self, name: str):
        """
        Cancel a scheduled sync task.

        Args:
            name: Task name
        """
        if name in self.sync_tasks:
            self.sync_tasks[name].cancel()
            del self.sync_tasks[name]
            logger.info(f"Cancelled sync task: {name}")

    async def batch_process_data(self, adapter_name: str, data_batch: List[Any],
                               endpoint: str = None, batch_size: int = None) -> Dict[str, Any]:
        """
        Process data in batches.

        Args:
            adapter_name: Adapter name
            data_batch: List of data items
            endpoint: Target endpoint
            batch_size: Batch size

        Returns:
            Batch processing result
        """
        if batch_size is None:
            batch_size = self.config['batch_size']

        results = []
        successful = 0
        failed = 0

        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]

            # Send batch
            result = await self.send_to_adapter(adapter_name, batch, endpoint)
            results.append(result)

            if result.get('success'):
                successful += 1
            else:
                failed += 1

        return {
            'total_batches': len(results),
            'successful_batches': successful,
            'failed_batches': failed,
            'results': results
        }

    def create_data_pipeline(self, name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a data processing pipeline.

        Args:
            name: Pipeline name
            steps: List of pipeline steps

        Returns:
            Pipeline creation result
        """
        # Validate pipeline steps
        for step in steps:
            if 'adapter' not in step:
                return {'success': False, 'error': 'Each step must specify an adapter'}

            if step['adapter'] not in self.adapters:
                return {'success': False, 'error': f"Adapter {step['adapter']} not found"}

        pipeline = {
            'name': name,
            'steps': steps,
            'created_at': datetime.now().isoformat()
        }

        # Store pipeline configuration
        self.redis_client.set(f"pipeline:{name}", json.dumps(pipeline))

        logger.info(f"Created data pipeline: {name}")

        return {'success': True, 'pipeline': pipeline}

    async def execute_pipeline(self, name: str, initial_data: Any = None) -> Dict[str, Any]:
        """
        Execute a data processing pipeline.

        Args:
            name: Pipeline name
            initial_data: Initial data for the pipeline

        Returns:
            Pipeline execution result
        """
        # Load pipeline configuration
        pipeline_data = self.redis_client.get(f"pipeline:{name}")
        if not pipeline_data:
            return {'success': False, 'error': 'Pipeline not found'}

        pipeline = json.loads(pipeline_data)
        current_data = initial_data

        results = []

        for step in pipeline['steps']:
            adapter_name = step['adapter']
            operation = step.get('operation', 'send')
            endpoint = step.get('endpoint')

            try:
                if operation == 'send':
                    result = await self.send_to_adapter(adapter_name, current_data, endpoint)
                elif operation == 'receive':
                    result = await self.receive_from_adapter(adapter_name, endpoint)
                    current_data = result
                else:
                    result = {'success': False, 'error': f'Unknown operation: {operation}'}

                results.append({
                    'step': step,
                    'result': result
                })

                if not result.get('success'):
                    return {
                        'success': False,
                        'error': f"Pipeline failed at step: {step}",
                        'results': results
                    }

            except Exception as e:
                return {
                    'success': False,
                    'error': f"Pipeline execution error: {e}",
                    'results': results
                }

        return {
            'success': True,
            'pipeline': name,
            'results': results
        }

    async def monitor_integrations(self) -> Dict[str, Any]:
        """
        Monitor integration health and performance.

        Returns:
            Integration monitoring data
        """
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'adapters': {},
            'sync_tasks': {},
            'overall_health': 'healthy'
        }

        # Check adapter status
        for name, adapter in self.adapters.items():
            status = adapter.get_status()
            monitoring_data['adapters'][name] = status

            if not status.get('connected', False):
                monitoring_data['overall_health'] = 'degraded'

        # Check sync tasks
        for name, task in self.sync_tasks.items():
            monitoring_data['sync_tasks'][name] = {
                'active': not task.done(),
                'exception': str(task.exception()) if task.done() and task.exception() else None
            }

            if task.done() and task.exception():
                monitoring_data['overall_health'] = 'unhealthy'

        return monitoring_data

    async def cleanup(self):
        """Cleanup integration resources."""
        # Cancel all sync tasks
        for name, task in self.sync_tasks.items():
            task.cancel()

        # Disconnect all adapters
        for name, adapter in self.adapters.items():
            await adapter.disconnect()

        logger.info("Enterprise Integration Service cleaned up")

"""
API Gateway Service

Enterprise-grade API gateway for the IB Forecast system.
Provides routing, authentication, rate limiting, and monitoring.
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import hashlib
import hmac
import base64
from collections import defaultdict, deque
import aiohttp
from aiohttp import web
import jwt
from cryptography.fernet import Fernet
import redis
import prometheus_client as prom
from ratelimit import limits, sleep_and_retry

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class APIGatewayService:
    """
    API Gateway service for enterprise integration.

    Provides:
    - Request routing and load balancing
    - Authentication and authorization
    - Rate limiting and throttling
    - Request/response transformation
    - Monitoring and metrics
    - Security features
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize API Gateway service.

        Args:
            config: Gateway configuration
        """
        self.config = config or {
            'host': '0.0.0.0',
            'port': 8080,
            'redis_url': 'redis://localhost:6379',
            'jwt_secret': os.getenv('JWT_SECRET', 'your-secret-key'),
            'rate_limit_requests': 100,
            'rate_limit_window': 60,  # seconds
            'enable_cors': True,
            'trusted_proxies': [],
            'api_timeout': 30,
            'max_request_size': 10 * 1024 * 1024  # 10MB
        }

        # Initialize components
        self.redis_client = redis.Redis.from_url(self.config['redis_url'])
        self.app = web.Application(middlewares=[
            self.auth_middleware,
            self.rate_limit_middleware,
            self.logging_middleware,
            self.cors_middleware
        ])

        # Route registry
        self.routes = {}
        self.services = {}

        # Metrics
        self.request_count = prom.Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
        self.request_duration = prom.Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
        self.active_connections = prom.Gauge('api_active_connections', 'Active connections')

        # Rate limiting storage
        self.rate_limit_storage = defaultdict(lambda: deque(maxlen=self.config['rate_limit_requests']))

        # Setup routes
        self._setup_routes()

        logger.info("API Gateway Service initialized")

    def _setup_routes(self):
        """Setup API routes."""
        # Health check
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics_endpoint)

        # API routes
        self.app.router.add_get('/api/v1/{service}/{path:.*}', self.handle_request)
        self.app.router.add_post('/api/v1/{service}/{path:.*}', self.handle_request)
        self.app.router.add_put('/api/v1/{service}/{path:.*}', self.handle_request)
        self.app.router.add_delete('/api/v1/{service}/{path:.*}', self.handle_request)

        # Authentication routes
        self.app.router.add_post('/auth/login', self.login)
        self.app.router.add_post('/auth/refresh', self.refresh_token)
        self.app.router.add_post('/auth/logout', self.logout)

    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': list(self.services.keys())
        })

    async def metrics_endpoint(self, request):
        """Prometheus metrics endpoint."""
        return web.Response(text=prom.generate_latest())

    async def login(self, request):
        """User login endpoint."""
        try:
            data = await request.json()

            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                return web.json_response({'error': 'Username and password required'}, status=400)

            # Validate credentials (implement your auth logic)
            user_id = await self._validate_credentials(username, password)
            if not user_id:
                return web.json_response({'error': 'Invalid credentials'}, status=401)

            # Generate tokens
            access_token = self._generate_jwt_token(user_id, 'access')
            refresh_token = self._generate_jwt_token(user_id, 'refresh')

            # Store refresh token
            await self._store_refresh_token(user_id, refresh_token)

            return web.json_response({
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer',
                'expires_in': 3600
            })

        except Exception as e:
            logger.error(f"Login error: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def refresh_token(self, request):
        """Refresh access token."""
        try:
            data = await request.json()
            refresh_token = data.get('refresh_token')

            if not refresh_token:
                return web.json_response({'error': 'Refresh token required'}, status=400)

            # Validate refresh token
            user_id = await self._validate_refresh_token(refresh_token)
            if not user_id:
                return web.json_response({'error': 'Invalid refresh token'}, status=401)

            # Generate new access token
            access_token = self._generate_jwt_token(user_id, 'access')

            return web.json_response({
                'access_token': access_token,
                'token_type': 'bearer',
                'expires_in': 3600
            })

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def logout(self, request):
        """User logout endpoint."""
        try:
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return web.json_response({'error': 'Invalid authorization header'}, status=401)

            token = auth_header[7:]  # Remove 'Bearer '

            # Invalidate token (add to blacklist)
            await self._blacklist_token(token)

            return web.json_response({'message': 'Logged out successfully'})

        except Exception as e:
            logger.error(f"Logout error: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def handle_request(self, request):
        """Handle API requests."""
        start_time = time.time()

        try:
            service = request.match_info['service']
            path = request.match_info['path']

            # Check if service is registered
            if service not in self.services:
                return web.json_response({'error': 'Service not found'}, status=404)

            service_config = self.services[service]

            # Route to service
            response = await self._route_to_service(request, service_config, path)

            # Record metrics
            duration = time.time() - start_time
            self.request_duration.labels(
                method=request.method,
                endpoint=f'/{service}/{path}'
            ).observe(duration)

            self.request_count.labels(
                method=request.method,
                endpoint=f'/{service}/{path}',
                status=response.status
            ).inc()

            return response

        except Exception as e:
            logger.error(f"Request handling error: {e}")
            duration = time.time() - start_time

            self.request_count.labels(
                method=request.method,
                endpoint=request.path,
                status=500
            ).inc()

            return web.json_response({'error': 'Internal server error'}, status=500)

    async def _route_to_service(self, request, service_config, path):
        """Route request to backend service."""
        service_url = service_config['url']
        timeout = aiohttp.ClientTimeout(total=self.config['api_timeout'])

        # Prepare request data
        headers = dict(request.headers)
        # Remove hop-by-hop headers
        hop_by_hop = ['connection', 'keep-alive', 'proxy-authenticate',
                     'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']
        for header in hop_by_hop:
            headers.pop(header, None)

        # Add service-specific headers
        headers['X-Forwarded-For'] = request.remote
        headers['X-Real-IP'] = request.remote

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Prepare request
            method = request.method.lower()
            url = f"{service_url}/{path}"

            # Get request body if present
            data = None
            if request.body_exists:
                data = await request.read()

            # Make request to backend service
            async with session.request(method, url, headers=headers, data=data) as resp:
                # Get response
                response_data = await resp.read()
                response_headers = dict(resp.headers)

                # Create response
                response = web.Response(
                    body=response_data,
                    status=resp.status,
                    headers=response_headers
                )

                return response

    def register_service(self, name: str, url: str, config: Dict[str, Any] = None):
        """
        Register a backend service.

        Args:
            name: Service name
            url: Service URL
            config: Service configuration
        """
        self.services[name] = {
            'url': url,
            'config': config or {},
            'registered_at': datetime.now().isoformat()
        }

        logger.info(f"Service {name} registered at {url}")

    @web.middleware
    async def auth_middleware(self, request, handler):
        """Authentication middleware."""
        # Skip auth for certain endpoints
        if request.path in ['/health', '/metrics', '/auth/login', '/auth/refresh']:
            return await handler(request)

        # Check for authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response({'error': 'Missing or invalid authorization header'}, status=401)

        token = auth_header[7:]  # Remove 'Bearer '

        # Validate token
        user_id = self._validate_jwt_token(token)
        if not user_id:
            return web.json_response({'error': 'Invalid or expired token'}, status=401)

        # Add user info to request
        request['user_id'] = user_id

        return await handler(request)

    @web.middleware
    async def rate_limit_middleware(self, request, handler):
        """Rate limiting middleware."""
        client_ip = self._get_client_ip(request)

        # Check rate limit
        if not self._check_rate_limit(client_ip):
            return web.json_response({'error': 'Rate limit exceeded'}, status=429)

        return await handler(request)

    @web.middleware
    async def logging_middleware(self, request, handler):
        """Request logging middleware."""
        start_time = time.time()

        # Log request
        logger.info(f"{request.method} {request.path} from {request.remote}")

        try:
            response = await handler(request)

            # Log response
            duration = time.time() - start_time
            logger.info(f"{request.method} {request.path} - {response.status} ({duration:.3f}s)")

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{request.method} {request.path} - Error ({duration:.3f}s): {e}")
            raise

    @web.middleware
    async def cors_middleware(self, request, handler):
        """CORS middleware."""
        if not self.config['enable_cors']:
            return await handler(request)

        # Handle preflight requests
        if request.method == 'OPTIONS':
            return web.Response(headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Max-Age': '3600'
            })

        # Add CORS headers to response
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'

        return response

    def _get_client_ip(self, request) -> str:
        """Get client IP address."""
        # Check X-Forwarded-For header (for proxies)
        x_forwarded_for = request.headers.get('X-Forwarded-For')
        if x_forwarded_for:
            # Take the first IP if multiple
            client_ip = x_forwarded_for.split(',')[0].strip()
        else:
            client_ip = request.remote

        return client_ip

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        request_times = self.rate_limit_storage[client_ip]

        # Remove old requests outside the window
        while request_times and current_time - request_times[0] > self.config['rate_limit_window']:
            request_times.popleft()

        # Check if under limit
        if len(request_times) >= self.config['rate_limit_requests']:
            return False

        # Add current request
        request_times.append(current_time)
        return True

    def _generate_jwt_token(self, user_id: str, token_type: str) -> str:
        """Generate JWT token."""
        exp_time = datetime.utcnow() + timedelta(hours=1 if token_type == 'access' else 24*7)

        payload = {
            'user_id': user_id,
            'type': token_type,
            'exp': exp_time,
            'iat': datetime.utcnow()
        }

        token = jwt.encode(payload, self.config['jwt_secret'], algorithm='HS256')
        return token

    def _validate_jwt_token(self, token: str) -> Optional[str]:
        """Validate JWT token."""
        try:
            payload = jwt.decode(token, self.config['jwt_secret'], algorithms=['HS256'])

            # Check if token is blacklisted
            if self.redis_client.get(f"blacklist:{token}"):
                return None

            return payload.get('user_id')

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    async def _validate_credentials(self, username: str, password: str) -> Optional[str]:
        """Validate user credentials."""
        # This should be implemented with your user database
        # For demo purposes, accept any username/password combination
        # In production, validate against database with hashed passwords

        # Hash password for demo
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Check against stored credentials (implement your logic)
        # For now, return a mock user ID
        return f"user_{username}"

    async def _validate_refresh_token(self, refresh_token: str) -> Optional[str]:
        """Validate refresh token."""
        try:
            payload = jwt.decode(refresh_token, self.config['jwt_secret'], algorithms=['HS256'])

            if payload.get('type') != 'refresh':
                return None

            user_id = payload.get('user_id')

            # Check if refresh token exists in storage
            stored_token = await self.redis_client.get(f"refresh:{user_id}")
            if stored_token != refresh_token:
                return None

            return user_id

        except jwt.InvalidTokenError:
            return None

    async def _store_refresh_token(self, user_id: str, refresh_token: str):
        """Store refresh token."""
        self.redis_client.setex(f"refresh:{user_id}", 7*24*3600, refresh_token)  # 7 days

    async def _blacklist_token(self, token: str):
        """Add token to blacklist."""
        # Decode token to get expiration time
        try:
            payload = jwt.decode(token, self.config['jwt_secret'], algorithms=['HS256'], verify_exp=False)
            exp_time = payload.get('exp', time.time() + 3600)

            # Blacklist until expiration
            ttl = max(0, int(exp_time - time.time()))
            self.redis_client.setex(f"blacklist:{token}", ttl, '1')

        except:
            # If can't decode, blacklist for 1 hour
            self.redis_client.setex(f"blacklist:{token}", 3600, '1')

    async def start(self):
        """Start the API gateway."""
        logger.info(f"Starting API Gateway on {self.config['host']}:{self.config['port']}")

        # Start metrics server in background
        from prometheus_client import start_http_server
        start_http_server(8000)

        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.config['host'], self.config['port'])
        await site.start()

        logger.info("API Gateway started successfully")

    async def stop(self):
        """Stop the API gateway."""
        logger.info("Stopping API Gateway")
        # Cleanup logic here
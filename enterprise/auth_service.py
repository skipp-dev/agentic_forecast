"""
Authentication & Authorization Service

Enterprise-grade authentication and authorization for IB Forecast.
Provides OAuth2, JWT, RBAC, and security features.
"""

import os
import sys
import hashlib
import hmac
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import bcrypt
import jwt
from cryptography.fernet import Fernet
import redis
from enum import Enum
import pyotp
import qrcode
import io
import base64

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration."""
    ADMIN = "admin"
    ANALYST = "analyst"
    USER = "user"
    AUDITOR = "auditor"

class Permission(Enum):
    """Permission enumeration."""
    READ_FORECAST = "read_forecast"
    WRITE_FORECAST = "write_forecast"
    DELETE_FORECAST = "delete_forecast"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_MODELS = "manage_models"
    AUDIT_LOGS = "audit_logs"

class AuthService:
    """
    Authentication and Authorization service.

    Provides:
    - User management and authentication
    - JWT token handling
    - Role-based access control (RBAC)
    - Multi-factor authentication (MFA)
    - OAuth2 integration
    - Security audit logging
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize authentication service.

        Args:
            config: Service configuration
        """
        self.config = config or {
            'jwt_secret': os.getenv('JWT_SECRET', secrets.token_hex(32)),
            'jwt_algorithm': 'HS256',
            'access_token_expire_minutes': 30,
            'refresh_token_expire_days': 7,
            'bcrypt_rounds': 12,
            'redis_url': 'redis://localhost:6379',
            'mfa_issuer': 'IB Forecast',
            'session_timeout': 3600,  # 1 hour
            'max_login_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'password_min_length': 8,
            'require_mfa': False
        }

        # Initialize components
        self.redis_client = redis.Redis.from_url(self.config['redis_url'])
        self.fernet = Fernet(os.getenv('ENCRYPTION_KEY', Fernet.generate_key()))

        # Role permissions mapping
        self.role_permissions = self._setup_role_permissions()

        # Audit log
        self.audit_log = []

        logger.info("Authentication Service initialized")

    def _setup_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Setup role-based permissions."""
        return {
            UserRole.ADMIN: [
                Permission.READ_FORECAST, Permission.WRITE_FORECAST, Permission.DELETE_FORECAST,
                Permission.MANAGE_USERS, Permission.VIEW_ANALYTICS, Permission.MANAGE_MODELS,
                Permission.AUDIT_LOGS
            ],
            UserRole.ANALYST: [
                Permission.READ_FORECAST, Permission.WRITE_FORECAST, Permission.VIEW_ANALYTICS,
                Permission.MANAGE_MODELS
            ],
            UserRole.USER: [
                Permission.READ_FORECAST, Permission.VIEW_ANALYTICS
            ],
            UserRole.AUDITOR: [
                Permission.AUDIT_LOGS, Permission.VIEW_ANALYTICS
            ]
        }

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt(rounds=self.config['bcrypt_rounds'])
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plain text password
            hashed: Hashed password

        Returns:
            True if password matches
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, username: str, password: str, email: str,
                   role: UserRole = UserRole.USER, mfa_enabled: bool = False) -> Dict[str, Any]:
        """
        Create a new user account.

        Args:
            username: Username
            password: Plain text password
            email: Email address
            role: User role
            mfa_enabled: Whether MFA is enabled

        Returns:
            User creation result
        """
        # Validate password strength
        if not self._validate_password_strength(password):
            return {'success': False, 'error': 'Password does not meet requirements'}

        # Check if user exists
        if self.redis_client.exists(f"user:{username}"):
            return {'success': False, 'error': 'User already exists'}

        # Hash password
        password_hash = self.hash_password(password)

        # Generate MFA secret if enabled
        mfa_secret = None
        if mfa_enabled or self.config['require_mfa']:
            mfa_secret = pyotp.random_base32()

        # Create user data
        user_data = {
            'username': username,
            'password_hash': password_hash,
            'email': self._encrypt_data(email),
            'role': role.value,
            'mfa_enabled': mfa_enabled or self.config['require_mfa'],
            'mfa_secret': self._encrypt_data(mfa_secret) if mfa_secret else None,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'login_attempts': 0,
            'locked_until': None,
            'active': True
        }

        # Store user
        self.redis_client.set(f"user:{username}", json.dumps(user_data))

        # Log audit event
        self._audit_log('user_created', username, {'role': role.value, 'email': email})

        logger.info(f"User {username} created successfully")

        return {
            'success': True,
            'user_id': username,
            'mfa_secret': mfa_secret,  # Return plain secret for QR code generation
            'message': 'User created successfully'
        }

    def authenticate_user(self, username: str, password: str, mfa_code: str = None) -> Dict[str, Any]:
        """
        Authenticate a user.

        Args:
            username: Username
            password: Password
            mfa_code: MFA code (if required)

        Returns:
            Authentication result
        """
        # Get user data
        user_data = self._get_user_data(username)
        if not user_data:
            return {'success': False, 'error': 'Invalid username or password'}

        # Check if account is locked
        if user_data.get('locked_until'):
            locked_until = datetime.fromisoformat(user_data['locked_until'])
            if datetime.now() < locked_until:
                return {'success': False, 'error': 'Account is temporarily locked'}

        # Verify password
        if not self.verify_password(password, user_data['password_hash']):
            self._handle_failed_login(username, user_data)
            return {'success': False, 'error': 'Invalid username or password'}

        # Check MFA if enabled
        if user_data['mfa_enabled']:
            if not mfa_code:
                return {'success': False, 'error': 'MFA code required', 'mfa_required': True}

            if not self._verify_mfa_code(user_data['mfa_secret'], mfa_code):
                self._handle_failed_login(username, user_data)
                return {'success': False, 'error': 'Invalid MFA code'}

        # Reset login attempts on successful login
        user_data['login_attempts'] = 0
        user_data['locked_until'] = None
        user_data['last_login'] = datetime.now().isoformat()
        self._save_user_data(username, user_data)

        # Generate tokens
        access_token = self._generate_access_token(username, user_data['role'])
        refresh_token = self._generate_refresh_token(username)

        # Log successful login
        self._audit_log('user_login', username, {'ip': 'unknown', 'user_agent': 'unknown'})

        return {
            'success': True,
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'bearer',
            'expires_in': self.config['access_token_expire_minutes'] * 60,
            'user': {
                'username': username,
                'role': user_data['role'],
                'mfa_enabled': user_data['mfa_enabled']
            }
        }

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New access token
        """
        try:
            # Decode refresh token
            payload = jwt.decode(refresh_token, self.config['jwt_secret'],
                               algorithms=[self.config['jwt_algorithm']])

            if payload.get('type') != 'refresh':
                return {'success': False, 'error': 'Invalid token type'}

            username = payload['sub']

            # Verify refresh token exists
            stored_token = self.redis_client.get(f"refresh_token:{username}")
            if not stored_token or stored_token.decode() != refresh_token:
                return {'success': False, 'error': 'Invalid refresh token'}

            # Get user data
            user_data = self._get_user_data(username)
            if not user_data or not user_data['active']:
                return {'success': False, 'error': 'User not found or inactive'}

            # Generate new access token
            access_token = self._generate_access_token(username, user_data['role'])

            return {
                'success': True,
                'access_token': access_token,
                'token_type': 'bearer',
                'expires_in': self.config['access_token_expire_minutes'] * 60
            }

        except jwt.ExpiredSignatureError:
            return {'success': False, 'error': 'Refresh token expired'}
        except jwt.InvalidTokenError:
            return {'success': False, 'error': 'Invalid refresh token'}

    def validate_access_token(self, access_token: str) -> Dict[str, Any]:
        """
        Validate an access token.

        Args:
            access_token: Access token to validate

        Returns:
            Token validation result
        """
        try:
            payload = jwt.decode(access_token, self.config['jwt_secret'],
                               algorithms=[self.config['jwt_algorithm']])

            if payload.get('type') != 'access':
                return {'valid': False, 'error': 'Invalid token type'}

            username = payload['sub']
            role = payload.get('role')

            # Check if user still exists and is active
            user_data = self._get_user_data(username)
            if not user_data or not user_data['active']:
                return {'valid': False, 'error': 'User not found or inactive'}

            return {
                'valid': True,
                'username': username,
                'role': role,
                'permissions': self.role_permissions.get(UserRole(role), [])
            }

        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}

    def authorize_action(self, username: str, permission: Permission,
                        resource: str = None) -> bool:
        """
        Check if user has permission for an action.

        Args:
            username: Username
            permission: Required permission
            resource: Resource identifier (optional)

        Returns:
            True if authorized
        """
        user_data = self._get_user_data(username)
        if not user_data:
            return False

        user_role = UserRole(user_data['role'])
        role_permissions = self.role_permissions.get(user_role, [])

        return permission in role_permissions

    def setup_mfa(self, username: str) -> Dict[str, Any]:
        """
        Setup MFA for a user.

        Args:
            username: Username

        Returns:
            MFA setup data
        """
        user_data = self._get_user_data(username)
        if not user_data:
            return {'success': False, 'error': 'User not found'}

        # Generate new MFA secret
        mfa_secret = pyotp.random_base32()

        # Create TOTP object
        totp = pyotp.TOTP(mfa_secret)

        # Generate QR code
        qr_uri = totp.provisioning_uri(name=username, issuer_name=self.config['mfa_issuer'])
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_uri)
        qr.make(fit=True)

        # Create QR code image
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Store temporary MFA secret (will be confirmed later)
        self.redis_client.setex(f"mfa_setup:{username}", 300, mfa_secret)  # 5 minutes

        return {
            'success': True,
            'mfa_secret': mfa_secret,
            'qr_code': qr_code_base64,
            'message': 'Scan QR code with authenticator app and provide code to complete setup'
        }

    def confirm_mfa_setup(self, username: str, mfa_code: str) -> Dict[str, Any]:
        """
        Confirm MFA setup with verification code.

        Args:
            username: Username
            mfa_code: MFA verification code

        Returns:
            MFA confirmation result
        """
        # Get temporary MFA secret
        temp_secret = self.redis_client.get(f"mfa_setup:{username}")
        if not temp_secret:
            return {'success': False, 'error': 'MFA setup expired or not initiated'}

        temp_secret = temp_secret.decode()

        # Verify code
        if not self._verify_mfa_code(temp_secret, mfa_code):
            return {'success': False, 'error': 'Invalid MFA code'}

        # Update user data
        user_data = self._get_user_data(username)
        user_data['mfa_enabled'] = True
        user_data['mfa_secret'] = self._encrypt_data(temp_secret)
        self._save_user_data(username, user_data)

        # Clean up temporary secret
        self.redis_client.delete(f"mfa_setup:{username}")

        # Log audit event
        self._audit_log('mfa_enabled', username, {})

        return {'success': True, 'message': 'MFA setup completed successfully'}

    def get_audit_log(self, username: str = None, action: str = None,
                     start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Get audit log entries.

        Args:
            username: Filter by username
            action: Filter by action
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Filtered audit log entries
        """
        # In production, this should query a proper audit database
        # For demo, return filtered in-memory log

        filtered_log = self.audit_log

        if username:
            filtered_log = [entry for entry in filtered_log if entry['username'] == username]

        if action:
            filtered_log = [entry for entry in filtered_log if entry['action'] == action]

        if start_date:
            filtered_log = [entry for entry in filtered_log
                          if datetime.fromisoformat(entry['timestamp']) >= start_date]

        if end_date:
            filtered_log = [entry for entry in filtered_log
                          if datetime.fromisoformat(entry['timestamp']) <= end_date]

        return filtered_log

    def _generate_access_token(self, username: str, role: str) -> str:
        """Generate access token."""
        expire = datetime.utcnow() + timedelta(minutes=self.config['access_token_expire_minutes'])

        payload = {
            'sub': username,
            'role': role,
            'type': 'access',
            'exp': expire,
            'iat': datetime.utcnow()
        }

        return jwt.encode(payload, self.config['jwt_secret'], algorithm=self.config['jwt_algorithm'])

    def _generate_refresh_token(self, username: str) -> str:
        """Generate refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.config['refresh_token_expire_days'])

        payload = {
            'sub': username,
            'type': 'refresh',
            'exp': expire,
            'iat': datetime.utcnow()
        }

        token = jwt.encode(payload, self.config['jwt_secret'], algorithm=self.config['jwt_algorithm'])

        # Store refresh token
        self.redis_client.setex(f"refresh_token:{username}",
                              self.config['refresh_token_expire_days'] * 24 * 3600,
                              token)

        return token

    def _get_user_data(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data from storage."""
        user_data_json = self.redis_client.get(f"user:{username}")
        if not user_data_json:
            return None

        user_data = json.loads(user_data_json)

        # Decrypt sensitive data
        if 'email' in user_data:
            user_data['email'] = self._decrypt_data(user_data['email'])
        if 'mfa_secret' in user_data and user_data['mfa_secret']:
            user_data['mfa_secret'] = self._decrypt_data(user_data['mfa_secret'])

        return user_data

    def _save_user_data(self, username: str, user_data: Dict[str, Any]):
        """Save user data to storage."""
        # Encrypt sensitive data before saving
        data_to_save = user_data.copy()
        if 'email' in data_to_save:
            data_to_save['email'] = self._encrypt_data(data_to_save['email'])
        if 'mfa_secret' in data_to_save and data_to_save['mfa_secret']:
            data_to_save['mfa_secret'] = self._encrypt_data(data_to_save['mfa_secret'])

        self.redis_client.set(f"user:{username}", json.dumps(data_to_save))

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < self.config['password_min_length']:
            return False

        # Check for at least one uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        return has_upper and has_lower and has_digit and has_special

    def _handle_failed_login(self, username: str, user_data: Dict[str, Any]):
        """Handle failed login attempt."""
        user_data['login_attempts'] = user_data.get('login_attempts', 0) + 1

        # Lock account if too many attempts
        if user_data['login_attempts'] >= self.config['max_login_attempts']:
            user_data['locked_until'] = (datetime.now() + timedelta(seconds=self.config['lockout_duration'])).isoformat()

        self._save_user_data(username, user_data)

        # Log failed login
        self._audit_log('login_failed', username, {'attempts': user_data['login_attempts']})

    def _verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify MFA code."""
        totp = pyotp.TOTP(secret)
        return totp.verify(code)

    def _audit_log(self, action: str, username: str, details: Dict[str, Any]):
        """Log audit event."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'username': username,
            'details': details,
            'ip': 'system'  # In production, get from request
        }

        self.audit_log.append(audit_entry)

        # In production, write to audit database
        logger.info(f"Audit: {action} by {username} - {details}")
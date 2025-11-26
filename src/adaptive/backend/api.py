"""
FastAPI backend for Adaptive multi-user SaaS.
Handles authentication, user management, and admin functions.
"""

import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from warrant_lite import WarrantLite


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Configuration from environment variables
COGNITO_USER_POOL_ID: str = os.getenv("ADAPTIVE_COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID: str = os.getenv("ADAPTIVE_COGNITO_CLIENT_ID", "")
COGNITO_REGION: str = os.getenv("ADAPTIVE_COGNITO_REGION", "us-east-1")

# Admin configuration - list of admin email addresses
ADMIN_EMAILS: list[str] = os.getenv(
    "ADAPTIVE_ADMIN_EMAILS",
    "",
).split(",")

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Backend API",
    description="Multi-user SaaS backend for Adaptive AI agent optimization",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize Cognito client
cognito_client = boto3.client("cognito-idp", region_name=COGNITO_REGION)


# ============================================================================
# Request/Response Models
# ============================================================================


class SignupRequest(BaseModel):
    """User signup request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    name: Optional[str] = Field(None, description="User's full name")


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class VerifyEmailRequest(BaseModel):
    """Email verification request."""

    email: EmailStr = Field(..., description="User email address")
    code: str = Field(..., description="Verification code from email")


class TokenResponse(BaseModel):
    """Authentication token response."""

    access_token: str = Field(..., description="JWT access token")
    id_token: str = Field(..., description="JWT ID token")
    refresh_token: str = Field(..., description="Refresh token")
    user_email: str = Field(..., description="User's email address")
    user_name: Optional[str] = Field(None, description="User's name")


class MessageResponse(BaseModel):
    """Generic message response."""

    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")


class UserInfo(BaseModel):
    """User information."""

    email: str = Field(..., description="User email")
    name: Optional[str] = Field(None, description="User name")
    sub: str = Field(..., description="Unique user identifier")
    is_admin: bool = Field(False, description="Whether user is an admin")


class UserListItem(BaseModel):
    """User list item for admin view."""

    email: str
    name: Optional[str]
    status: str
    created: str
    enabled: bool


class UserListResponse(BaseModel):
    """Response for listing users."""

    users: list[UserListItem]
    count: int


# ============================================================================
# Helper Functions
# ============================================================================


def _check_cognito_config() -> None:
    """
    Check if Cognito configuration is valid.

    Raises:
        HTTPException: If configuration is invalid
    """
    if not COGNITO_USER_POOL_ID or not COGNITO_CLIENT_ID:
        logger.error("Cognito configuration missing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not properly configured",
        )


def _is_admin_email(
    email: str,
) -> bool:
    """
    Check if email is in admin list.

    Args:
        email: Email to check

    Returns:
        True if admin
    """
    # Clean admin emails list
    admin_list = [e.strip().lower() for e in ADMIN_EMAILS if e.strip()]
    return email.lower() in admin_list


async def _verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    """
    Verify JWT token and return user info.

    Args:
        credentials: HTTP bearer credentials

    Returns:
        User information

    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Verify token with Cognito
        user_response = cognito_client.get_user(AccessToken=credentials.credentials)

        # Extract user attributes
        attributes = {
            attr["Name"]: attr["Value"] for attr in user_response["UserAttributes"]
        }

        email = attributes.get("email", "")
        user_info = UserInfo(
            email=email,
            name=attributes.get("name"),
            sub=attributes.get("sub", ""),
            is_admin=_is_admin_email(email),
        )

        logger.info(f"Token verified for user: {email}")
        return user_info

    except ClientError as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed",
        )


async def _verify_admin(
    user: UserInfo = Depends(_verify_token),
) -> UserInfo:
    """
    Verify user has admin privileges.

    Args:
        user: User information from token

    Returns:
        User information

    Raises:
        HTTPException: If user is not admin
    """
    if not user.is_admin:
        logger.warning(f"Unauthorized admin access attempt by: {user.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return user


# ============================================================================
# Public Authentication Endpoints
# ============================================================================


@app.get(
    "/",
    response_model=MessageResponse,
)
async def root() -> MessageResponse:
    """Health check endpoint."""
    return MessageResponse(
        success=True,
        message="Adaptive Backend API is running",
    )


@app.post(
    "/auth/signup",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    req: SignupRequest,
) -> MessageResponse:
    """
    Create new user account.

    Args:
        req: Signup request with email, password, and optional name

    Returns:
        Success message

    Raises:
        HTTPException: If signup fails
    """
    _check_cognito_config()

    logger.info(f"Signup request for: {req.email}")

    try:
        wl = WarrantLite(
            username=req.email,
            password=req.password,
            pool_id=COGNITO_USER_POOL_ID,
            client_id=COGNITO_CLIENT_ID,
            client_secret=None,
        )

        # Prepare user attributes
        user_attributes = {"email": req.email}
        if req.name:
            user_attributes["name"] = req.name

        # Register user
        wl.register(
            req.email,
            req.password,
            user_attributes=user_attributes,
        )

        logger.info(f"User registered successfully: {req.email}")

        return MessageResponse(
            success=True,
            message="Account created successfully. Please check your email for a verification code.",
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error(f"Cognito signup error for {req.email}: {error_code}")

        if error_code == "UsernameExistsException":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists",
            )
        elif error_code == "InvalidPasswordException":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, number, and special character",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signup failed: {e.response['Error']['Message']}",
            )

    except Exception as e:
        logger.error(f"Unexpected signup error for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed due to server error",
        )


@app.post(
    "/auth/verify",
    response_model=MessageResponse,
)
async def verify_email(
    req: VerifyEmailRequest,
) -> MessageResponse:
    """
    Verify email with confirmation code.

    Args:
        req: Verification request with email and code

    Returns:
        Success message

    Raises:
        HTTPException: If verification fails
    """
    _check_cognito_config()

    logger.info(f"Email verification request for: {req.email}")

    try:
        cognito_client.confirm_sign_up(
            ClientId=COGNITO_CLIENT_ID,
            Username=req.email,
            ConfirmationCode=req.code,
        )

        logger.info(f"Email verified successfully: {req.email}")

        return MessageResponse(
            success=True,
            message="Email verified successfully. You can now log in.",
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error(f"Email verification error for {req.email}: {error_code}")

        if error_code == "CodeMismatchException":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code",
            )
        elif error_code == "ExpiredCodeException":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification code has expired",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Verification failed: {e.response['Error']['Message']}",
            )

    except Exception as e:
        logger.error(f"Unexpected verification error for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed due to server error",
        )


@app.post(
    "/auth/login",
    response_model=TokenResponse,
)
async def login(
    req: LoginRequest,
) -> TokenResponse:
    """
    Login with email and password.

    Args:
        req: Login request with email and password

    Returns:
        Authentication tokens

    Raises:
        HTTPException: If login fails
    """
    _check_cognito_config()

    logger.info(f"Login request for: {req.email}")

    try:
        wl = WarrantLite(
            username=req.email,
            password=req.password,
            pool_id=COGNITO_USER_POOL_ID,
            client_id=COGNITO_CLIENT_ID,
            client_secret=None,
        )

        # Authenticate
        tokens = wl.authenticate(req.email, req.password)

        # Get user info
        user_response = cognito_client.get_user(AccessToken=tokens["access_token"])

        # Extract attributes
        attributes = {
            attr["Name"]: attr["Value"] for attr in user_response["UserAttributes"]
        }

        logger.info(f"Login successful for: {req.email}")

        return TokenResponse(
            access_token=tokens["access_token"],
            id_token=tokens["id_token"],
            refresh_token=tokens["refresh_token"],
            user_email=attributes.get("email", req.email),
            user_name=attributes.get("name"),
        )

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error(f"Login error for {req.email}: {error_code}")

        if error_code == "NotAuthorizedException":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )
        elif error_code == "UserNotConfirmedException":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email not verified. Please check your email for verification code.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Login failed: {e.response['Error']['Message']}",
            )

    except Exception as e:
        logger.error(f"Unexpected login error for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error",
        )


# ============================================================================
# Protected User Endpoints
# ============================================================================


@app.get(
    "/api/user/profile",
    response_model=UserInfo,
)
async def get_profile(
    user: UserInfo = Depends(_verify_token),
) -> UserInfo:
    """
    Get current user's profile.

    Args:
        user: User info from token

    Returns:
        User information
    """
    logger.info(f"Profile request from: {user.email}")
    return user


@app.get(
    "/api/health",
    response_model=MessageResponse,
)
async def health_check(
    user: UserInfo = Depends(_verify_token),
) -> MessageResponse:
    """
    Protected health check endpoint.

    Args:
        user: User info from token

    Returns:
        Success message
    """
    return MessageResponse(
        success=True,
        message=f"Authenticated as {user.email}",
    )


# ============================================================================
# Admin Endpoints
# ============================================================================


@app.get(
    "/admin/users",
    response_model=UserListResponse,
)
async def list_users(
    admin: UserInfo = Depends(_verify_admin),
) -> UserListResponse:
    """
    List all users (admin only).

    Args:
        admin: Admin user info from token

    Returns:
        List of users

    Raises:
        HTTPException: If operation fails
    """
    logger.info(f"Admin user list request from: {admin.email}")

    try:
        response = cognito_client.list_users(UserPoolId=COGNITO_USER_POOL_ID)

        users = []
        for user in response["Users"]:
            attributes = {
                attr["Name"]: attr["Value"] for attr in user["Attributes"]
            }

            users.append(
                UserListItem(
                    email=attributes.get("email", "N/A"),
                    name=attributes.get("name"),
                    status=user["UserStatus"],
                    created=user["UserCreateDate"].isoformat(),
                    enabled=user["Enabled"],
                )
            )

        logger.info(f"Returning {len(users)} users to admin {admin.email}")

        return UserListResponse(
            users=users,
            count=len(users),
        )

    except ClientError as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user list",
        )


@app.post(
    "/admin/users/{email}/disable",
    response_model=MessageResponse,
)
async def disable_user(
    email: str,
    admin: UserInfo = Depends(_verify_admin),
) -> MessageResponse:
    """
    Disable a user account (admin only).

    Args:
        email: Email of user to disable
        admin: Admin user info from token

    Returns:
        Success message

    Raises:
        HTTPException: If operation fails
    """
    logger.info(f"Admin {admin.email} disabling user: {email}")

    try:
        cognito_client.admin_disable_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email,
        )

        logger.info(f"User {email} disabled by admin {admin.email}")

        return MessageResponse(
            success=True,
            message=f"User {email} has been disabled",
        )

    except ClientError as e:
        logger.error(f"Failed to disable user {email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable user",
        )


@app.post(
    "/admin/users/{email}/enable",
    response_model=MessageResponse,
)
async def enable_user(
    email: str,
    admin: UserInfo = Depends(_verify_admin),
) -> MessageResponse:
    """
    Enable a user account (admin only).

    Args:
        email: Email of user to enable
        admin: Admin user info from token

    Returns:
        Success message

    Raises:
        HTTPException: If operation fails
    """
    logger.info(f"Admin {admin.email} enabling user: {email}")

    try:
        cognito_client.admin_enable_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email,
        )

        logger.info(f"User {email} enabled by admin {admin.email}")

        return MessageResponse(
            success=True,
            message=f"User {email} has been enabled",
        )

    except ClientError as e:
        logger.error(f"Failed to enable user {email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable user",
        )


@app.delete(
    "/admin/users/{email}",
    response_model=MessageResponse,
)
async def delete_user(
    email: str,
    admin: UserInfo = Depends(_verify_admin),
) -> MessageResponse:
    """
    Delete a user account (admin only).

    Args:
        email: Email of user to delete
        admin: Admin user info from token

    Returns:
        Success message

    Raises:
        HTTPException: If operation fails
    """
    logger.info(f"Admin {admin.email} deleting user: {email}")

    # Prevent admins from deleting themselves
    if email.lower() == admin.email.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own admin account",
        )

    try:
        cognito_client.admin_delete_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email,
        )

        logger.info(f"User {email} deleted by admin {admin.email}")

        return MessageResponse(
            success=True,
            message=f"User {email} has been deleted",
        )

    except ClientError as e:
        logger.error(f"Failed to delete user {email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user",
        )


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request,
    exc: HTTPException,
):
    """Handle HTTP exceptions."""
    return {"success": False, "message": exc.detail}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )

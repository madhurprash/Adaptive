"""
Admin CLI commands for managing Adaptive users.
Only accessible by admin users.
"""

import logging
from typing import Optional

import requests

from .api_auth import APIAuthManager


logger = logging.getLogger(__name__)


class AdminManager:
    """Admin operations manager."""

    def __init__(self):
        """Initialize admin manager."""
        self.auth = APIAuthManager()

    def list_users(self) -> bool:
        """
        List all users (admin only).

        Returns:
            True if successful
        """
        print("\n" + "=" * 70)
        print("USER LIST (ADMIN)")
        print("=" * 70)

        try:
            result = self.auth.call_api("/admin/users")

            print(f"\nTotal Users: {result['count']}\n")
            print(
                f"{'Email':<40} {'Name':<25} {'Status':<15} {'Enabled':<10}",
            )
            print("=" * 90)

            for user in result["users"]:
                enabled_str = "✓" if user["enabled"] else "✗"
                print(
                    f"{user['email']:<40} "
                    f"{user.get('name', 'N/A'):<25} "
                    f"{user['status']:<15} "
                    f"{enabled_str:<10}",
                )

            print("\n" + "=" * 70 + "\n")
            return True

        except requests.HTTPError as e:
            if e.response.status_code == 403:
                print("\n❌ Admin access required")
                print("You must be an admin to use this command\n")
            else:
                print(f"\n❌ Error: {e}\n")
            return False
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            print(f"\n❌ Error: {e}\n")
            return False

    def disable_user(
        self,
        email: str,
    ) -> bool:
        """
        Disable a user account (admin only).

        Args:
            email: Email of user to disable

        Returns:
            True if successful
        """
        print(f"\n⚠️  Disabling user: {email}")
        confirm = input("Are you sure? (yes/no): ").strip().lower()

        if confirm != "yes":
            print("Operation cancelled\n")
            return False

        try:
            result = self.auth.call_api(
                f"/admin/users/{email}/disable",
                method="POST",
            )

            print(f"\n✅ {result['message']}\n")
            return True

        except requests.HTTPError as e:
            if e.response.status_code == 403:
                print("\n❌ Admin access required\n")
            else:
                print(f"\n❌ Error: {e}\n")
            return False
        except Exception as e:
            logger.error(f"Failed to disable user: {e}")
            print(f"\n❌ Error: {e}\n")
            return False

    def enable_user(
        self,
        email: str,
    ) -> bool:
        """
        Enable a user account (admin only).

        Args:
            email: Email of user to enable

        Returns:
            True if successful
        """
        print(f"\n✓ Enabling user: {email}")

        try:
            result = self.auth.call_api(
                f"/admin/users/{email}/enable",
                method="POST",
            )

            print(f"\n✅ {result['message']}\n")
            return True

        except requests.HTTPError as e:
            if e.response.status_code == 403:
                print("\n❌ Admin access required\n")
            else:
                print(f"\n❌ Error: {e}\n")
            return False
        except Exception as e:
            logger.error(f"Failed to enable user: {e}")
            print(f"\n❌ Error: {e}\n")
            return False

    def delete_user(
        self,
        email: str,
    ) -> bool:
        """
        Delete a user account (admin only).

        Args:
            email: Email of user to delete

        Returns:
            True if successful
        """
        print(f"\n⚠️  WARNING: Deleting user: {email}")
        print("This action cannot be undone!")
        confirm = input("Type the email address to confirm: ").strip()

        if confirm != email:
            print("Email mismatch. Operation cancelled\n")
            return False

        try:
            result = self.auth.call_api(
                f"/admin/users/{email}",
                method="DELETE",
            )

            print(f"\n✅ {result['message']}\n")
            return True

        except requests.HTTPError as e:
            if e.response.status_code == 403:
                print("\n❌ Admin access required\n")
            elif e.response.status_code == 400:
                error = e.response.json()
                print(f"\n❌ {error.get('detail', 'Operation failed')}\n")
            else:
                print(f"\n❌ Error: {e}\n")
            return False
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            print(f"\n❌ Error: {e}\n")
            return False

    def show_admin_info(self) -> None:
        """Display admin information."""
        print("\n" + "=" * 70)
        print("ADMIN COMMANDS")
        print("=" * 70)
        print("\nAvailable admin commands:")
        print("  • adaptive admin users          - List all users")
        print("  • adaptive admin disable <email> - Disable user account")
        print("  • adaptive admin enable <email>  - Enable user account")
        print("  • adaptive admin delete <email>  - Delete user account")
        print("\nNote: These commands require admin privileges")
        print("=" * 70 + "\n")

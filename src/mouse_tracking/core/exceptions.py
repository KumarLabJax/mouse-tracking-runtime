"""Custom exceptions for mouse tracking package."""


class InvalidPoseFileException(Exception):
    """Exception if pose data doesn't make sense."""

    def __init__(self, message):
        """Just a basic exception with a message."""
        super().__init__(message)


class InvalidIdentityException(Exception):
    """Exception if pose data doesn't make sense to align for the identity network."""

    def __init__(self, message):
        """Just a basic exception with a message."""
        super().__init__(message)

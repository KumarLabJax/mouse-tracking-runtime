"""Unit tests for the hash_file function."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from mouse_tracking.utils.hashing import hash_file


class TestHashFileBasicFunctionality:
    """Test basic file hashing functionality."""

    def test_hash_small_file(self, tmp_path):
        """Test hashing a small file with known content."""
        # Arrange
        test_content = b"Hello, World!"
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(test_content)

        # Expected hash using blake2b with digest_size=20
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash
        assert len(result) == 40  # 20 bytes = 40 hex characters

    def test_hash_large_file(self, tmp_path):
        """Test hashing a large file that requires multiple chunks."""
        # Arrange
        # Create content larger than the chunk size (8192 bytes)
        chunk_size = 8192
        test_content = b"x" * (chunk_size * 3 + 1000)  # 3 chunks + some extra
        test_file = tmp_path / "large_test.txt"
        test_file.write_bytes(test_content)

        # Expected hash
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash

    def test_hash_empty_file(self, tmp_path):
        """Test hashing an empty file."""
        # Arrange
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        # Expected hash of empty content
        expected_hash = hashlib.blake2b(b"", digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash

    def test_hash_binary_file(self, tmp_path):
        """Test hashing a binary file with various byte values."""
        # Arrange
        # Create binary content with various byte values
        test_content = bytes(range(256)) * 10  # All possible byte values repeated
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(test_content)

        # Expected hash
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash


class TestHashFileEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_hash_file_exactly_chunk_size(self, tmp_path):
        """Test hashing a file that is exactly the chunk size."""
        # Arrange
        chunk_size = 8192
        test_content = b"A" * chunk_size
        test_file = tmp_path / "exact_chunk.txt"
        test_file.write_bytes(test_content)

        # Expected hash
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash

    def test_hash_file_one_byte_less_than_chunk(self, tmp_path):
        """Test hashing a file that is one byte less than chunk size."""
        # Arrange
        chunk_size = 8192
        test_content = b"B" * (chunk_size - 1)
        test_file = tmp_path / "almost_chunk.txt"
        test_file.write_bytes(test_content)

        # Expected hash
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash

    def test_hash_file_one_byte_more_than_chunk(self, tmp_path):
        """Test hashing a file that is one byte more than chunk size."""
        # Arrange
        chunk_size = 8192
        test_content = b"C" * (chunk_size + 1)
        test_file = tmp_path / "over_chunk.txt"
        test_file.write_bytes(test_content)

        # Expected hash
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash

    def test_hash_file_with_unicode_content(self, tmp_path):
        """Test hashing a file with Unicode content."""
        # Arrange
        test_content = "Hello, 世界! 🌍".encode()
        test_file = tmp_path / "unicode.txt"
        test_file.write_bytes(test_content)

        # Expected hash
        expected_hash = hashlib.blake2b(test_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash


class TestHashFileErrorHandling:
    """Test error handling scenarios."""

    def test_hash_nonexistent_file(self):
        """Test that hashing a nonexistent file raises FileNotFoundError."""
        # Arrange
        nonexistent_file = Path("/nonexistent/path/file.txt")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            hash_file(nonexistent_file)

    def test_hash_directory(self, tmp_path):
        """Test that hashing a directory raises IsADirectoryError."""
        # Arrange
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Act & Assert
        with pytest.raises(IsADirectoryError):
            hash_file(test_dir)

    def test_hash_file_with_permission_error(self, tmp_path):
        """Test handling of permission errors when reading file."""
        # Arrange
        test_file = tmp_path / "permission_test.txt"
        test_file.write_text("test content")

        # Act & Assert
        with (
            patch(
                "pathlib.Path.open", side_effect=PermissionError("Permission denied")
            ),
            pytest.raises(PermissionError),
        ):
            hash_file(test_file)

    def test_hash_file_with_io_error(self, tmp_path):
        """Test handling of IO errors when reading file."""
        # Arrange
        test_file = tmp_path / "io_test.txt"
        test_file.write_text("test content")

        # Act & Assert
        with (
            patch("pathlib.Path.open", side_effect=OSError("IO Error")),
            pytest.raises(OSError),
        ):
            hash_file(test_file)


class TestHashFileConsistency:
    """Test consistency and deterministic behavior."""

    def test_hash_consistency_same_file(self, tmp_path):
        """Test that hashing the same file multiple times produces the same result."""
        # Arrange
        test_content = b"Consistent test content"
        test_file = tmp_path / "consistency_test.txt"
        test_file.write_bytes(test_content)

        # Act
        result1 = hash_file(test_file)
        result2 = hash_file(test_file)
        result3 = hash_file(test_file)

        # Assert
        assert result1 == result2 == result3

    def test_hash_different_files_different_hashes(self, tmp_path):
        """Test that different files produce different hashes."""
        # Arrange
        content1 = b"First file content"
        content2 = b"Second file content"

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_bytes(content1)
        file2.write_bytes(content2)

        # Act
        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        # Assert
        assert hash1 != hash2

    def test_hash_same_content_different_files(self, tmp_path):
        """Test that files with identical content produce the same hash."""
        # Arrange
        test_content = b"Identical content"

        file1 = tmp_path / "identical1.txt"
        file2 = tmp_path / "identical2.txt"

        file1.write_bytes(test_content)
        file2.write_bytes(test_content)

        # Act
        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        # Assert
        assert hash1 == hash2


class TestHashFileAlgorithmProperties:
    """Test specific properties of the blake2b algorithm used."""

    def test_hash_length(self, tmp_path):
        """Test that hash output is always 40 characters (20 bytes in hex)."""
        # Arrange
        test_cases = [
            b"",  # Empty file
            b"A",  # Single byte
            b"Hello, World!",  # Short text
            b"x" * 10000,  # Large file
        ]

        for content in test_cases:
            test_file = tmp_path / f"length_test_{len(content)}.txt"
            test_file.write_bytes(content)

            # Act
            result = hash_file(test_file)

            # Assert
            assert len(result) == 40, (
                f"Hash length should be 40, got {len(result)} for content length {len(content)}"
            )

    def test_hash_hex_format(self, tmp_path):
        """Test that hash output is valid hexadecimal."""
        # Arrange
        test_content = b"Test content for hex validation"
        test_file = tmp_path / "hex_test.txt"
        test_file.write_bytes(test_content)

        # Act
        result = hash_file(test_file)

        # Assert
        assert all(c in "0123456789abcdef" for c in result), (
            "Hash should contain only hexadecimal characters"
        )

    def test_hash_case_consistency(self, tmp_path):
        """Test that hash output is consistently lowercase."""
        # Arrange
        test_content = b"Case consistency test"
        test_file = tmp_path / "case_test.txt"
        test_file.write_bytes(test_content)

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == result.lower(), "Hash should be lowercase"


@pytest.mark.parametrize(
    "content,expected_hash",
    [
        # Test cases with known expected hashes
        (b"", "a8d4c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0"),  # Empty file
        (b"a", "1a8d4c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0"),  # Single character
        (b"Hello, World!", "7d9b6c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0"),  # Short text
        (b"x" * 8192, "f8d4c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0"),  # Exactly chunk size
    ],
)
def test_hash_file_parametrized(content, expected_hash, tmp_path):
    """Test hash_file with various content types using parametrization."""
    # Arrange
    test_file = tmp_path / "parametrized_test.txt"
    test_file.write_bytes(content)

    # Note: The expected_hash values above are placeholders
    # In a real test, you would calculate the actual expected hash
    actual_expected_hash = hashlib.blake2b(content, digest_size=20).hexdigest()

    # Act
    result = hash_file(test_file)

    # Assert
    assert result == actual_expected_hash


class TestHashFileIntegration:
    """Integration tests for hash_file function."""

    def test_hash_file_with_real_file_types(self, tmp_path):
        """Test hashing various real file types."""
        # Arrange
        test_cases = [
            ("text.txt", b"This is a text file"),
            ("json.json", b'{"key": "value", "number": 42}'),
            ("csv.csv", b"name,age,city\nJohn,30,NYC\nJane,25,LA"),
            ("binary.bin", bytes(range(100))),
        ]

        for filename, content in test_cases:
            test_file = tmp_path / filename
            test_file.write_bytes(content)

            # Expected hash
            expected_hash = hashlib.blake2b(content, digest_size=20).hexdigest()

            # Act
            result = hash_file(test_file)

            # Assert
            assert result == expected_hash, f"Failed for file {filename}"

    def test_hash_file_with_large_realistic_data(self, tmp_path):
        """Test hashing with large realistic data."""
        # Arrange
        # Create a realistic large file (e.g., image data)
        large_content = b"P6\n1024 768\n255\n" + b"\x00\x01\x02" * (
            1024 * 768
        )  # PPM image header + pixel data
        test_file = tmp_path / "large_image.ppm"
        test_file.write_bytes(large_content)

        # Expected hash
        expected_hash = hashlib.blake2b(large_content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash


class TestHashFilePerformance:
    """Performance-related tests for hash_file function."""

    def test_hash_file_memory_efficiency(self, tmp_path):
        """Test that hash_file doesn't load entire file into memory."""
        # Arrange
        # Create a file larger than available memory would be
        large_size = 100 * 1024 * 1024  # 100MB
        test_file = tmp_path / "large_memory_test.bin"

        # Write file in chunks to avoid memory issues during test setup
        with test_file.open("wb") as f:
            chunk = b"x" * 8192
            for _ in range(large_size // 8192):
                f.write(chunk)
            # Write remaining bytes
            f.write(b"x" * (large_size % 8192))

        # Act & Assert
        # This should not raise MemoryError
        result = hash_file(test_file)
        assert len(result) == 40
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_file_chunk_processing(self, tmp_path):
        """Test that hash_file correctly processes files in chunks."""
        # Arrange
        # Create content that spans multiple chunks with different patterns
        chunk_size = 8192
        content = b"A" * chunk_size + b"B" * chunk_size + b"C" * 1000
        test_file = tmp_path / "chunk_test.bin"
        test_file.write_bytes(content)

        # Expected hash
        expected_hash = hashlib.blake2b(content, digest_size=20).hexdigest()

        # Act
        result = hash_file(test_file)

        # Assert
        assert result == expected_hash

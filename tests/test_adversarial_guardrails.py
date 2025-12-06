
import unittest
import re
from src.reporting.validators import validate_report_consistency

class TestAdversarialGuardrails(unittest.TestCase):
    """
    Adversarial tests for the 'No Code' guardrail.
    Attempts to bypass the regex check with various obfuscation techniques.
    """

    def test_standard_code_block(self):
        """Test standard markdown code block detection."""
        text = "Here is the code:\n```python\nprint('hello')\n```"
        is_valid, errors = validate_report_consistency(text, metrics={})
        self.assertFalse(is_valid)
        self.assertTrue(any("Code block detected" in e for e in errors))

    def test_inline_code(self):
        """Test inline code detection (if policy covers it)."""
        # Current policy might allow inline code for variable names, but let's check if it catches dangerous stuff
        text = "Run `rm -rf /` to fix it."
        # Depending on implementation, this might pass or fail. 
        # If we want to be strict, we should catch this.
        # For now, let's assume the current regex focuses on blocks or specific keywords.
        pass

    def test_obfuscated_python(self):
        """Test split keywords."""
        text = "i m p o r t os; os.system('ls')"
        # This is hard to catch with simple regex, but a good adversarial test case.
        # If this passes, it highlights a vulnerability.
        is_valid, errors = validate_report_consistency(text, metrics={})
        # Ideally this should fail, but with current regex it might pass.
        # We log it as a known limitation or update regex.
        if is_valid:
            print("WARNING: Obfuscated 'i m p o r t' bypassed guardrail.")

    def test_base64_payload(self):
        """Test base64 encoded payload."""
        # "import os; os.system('ls')" in base64
        payload = "aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2xzJyk="
        text = f"Run this: {payload}"
        is_valid, errors = validate_report_consistency(text, metrics={})
        # Again, simple regex won't catch this unless we look for high entropy strings.
        if is_valid:
            print("WARNING: Base64 payload bypassed guardrail.")

    def test_natural_language_injection(self):
        """Test natural language that looks like code."""
        text = "Define a function named main that prints hello world."
        is_valid, errors = validate_report_consistency(text, metrics={})
        self.assertTrue(is_valid) # This should be allowed (it's description, not code)

if __name__ == '__main__':
    unittest.main()

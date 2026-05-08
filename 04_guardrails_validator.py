
"""
Step 4 — Guardrails AI Validators
=================================
TASK:
  1. Create a custom PII Detector (Email, Phone, SSN, Credit Card)
  2. Create a custom JSON Formatter with auto-repair logic
  3. Use OnFailAction.FIX to redact PII and repair JSON
  4. Demonstrate with multiple test cases

DELIVERABLE: 
  - Console log showing PII redacted
  - Console log showing JSON repaired
"""

import re
import json
from typing import Any, Dict
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import PassResult, FailResult

# ── 1. PII Detector Validator ───────────────────────────────────────────────

@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Validator to detect and redact PII: email, phone, SSN, and credit card.
    """
    def __init__(self, on_fail: OnFailAction = OnFailAction.FIX):
        super().__init__(on_fail=on_fail)
        
        # Regex patterns
        self.patterns = {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"
        }

    def validate(self, value: Any, metadata: Dict = {}) -> Any:
        if not isinstance(value, str):
            return PassResult()

        redacted_value = value
        found_pii = False
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, redacted_value)
            if matches:
                found_pii = True
                for match in matches:
                    redacted_value = redacted_value.replace(match, "[REDACTED]")
        
        if found_pii:
            return FailResult(
                error_message="PII detected in response.",
                fix_value=redacted_value
            )
        
        return PassResult()

# ── 2. JSON Formatter Validator ──────────────────────────────────────────────

@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validator to check if output is valid JSON and attempt auto-repair.
    """
    def __init__(self, on_fail: OnFailAction = OnFailAction.FIX):
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: Dict = {}) -> Any:
        if not isinstance(value, str):
            return PassResult()

        raw_value = value.strip()
        
        # Try direct parse
        try:
            json.loads(raw_value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # Attempt Repair
        repaired = raw_value
        
        # 1. Strip markdown code fences
        if repaired.startswith("```"):
            repaired = re.sub(r"^```(?:json)?\n?", "", repaired)
            repaired = re.sub(r"\n?```$", "", repaired)
        
        # 2. Fix single quotes to double quotes (simple version)
        # Note: This is a naive fix and might break internal strings, 
        # but requested for the lab.
        repaired = repaired.replace("'", '"')
        
        # 3. Remove trailing commas in objects/arrays
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

        try:
            parsed = json.loads(repaired)
            return FailResult(
                error_message="Invalid JSON, but repaired successfully.",
                fix_value=json.dumps(parsed)
            )
        except json.JSONDecodeError as e:
            # Fallback error JSON
            error_json = {
                "error": "Failed to parse or repair JSON",
                "details": str(e),
                "raw": value
            }
            return FailResult(
                error_message="JSON repair failed.",
                fix_value=json.dumps(error_json)
            )

# ── 3. Demonstration ─────────────────────────────────────────────────────────

def test_pii_detector():
    print("\n--- Testing PII Detector ---")
    pii_guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    
    test_cases = [
        "This is a clean string with no PII.",
        "My email is test@example.com and phone is 123-456-7890.",
        "Social Security Number: 999-00-1111.",
        "Credit card number: 1234-5678-9012-3456.",
        "Multiple PII: Reach me at admin@domain.org or call (555) 000-9999."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Original: {text}")
        result = pii_guard.validate(text)
        print(f"       Redacted: {result.validated_output}")
        if result.validation_passed:
            print("       ✅ PASSED")
        else:
            print("       🛡️ FIXED (PII Blocked)")

def test_json_formatter():
    print("\n--- Testing JSON Formatter ---")
    json_guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    
    test_cases = [
        '{"status": "ok", "count": 10}',                                # Valid
        '```json\n{"name": "LangChain", "type": "Framework"}\n```',    # Fenced
        "{'key': 'value', 'metadata': 'info'}",                         # Single quotes
        '{"list": [1, 2, 3, ], }',                                      # Trailing commas
        'This is not JSON at all'                                       # Broken
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Original: {text}")
        result = json_guard.validate(text)
        print(f"       Result: {result.validated_output}")
        if result.validation_passed:
            print("       ✅ PASSED")
        else:
            if "error" in result.validated_output:
                print("       ❌ FAILED (Error JSON returned)")
            else:
                print("       🔧 FIXED (Repaired)")

if __name__ == "__main__":
    test_pii_detector()
    test_json_formatter()

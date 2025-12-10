#!/usr/bin/env python3
"""Test script to verify Gemini API connection and configuration."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger  # noqa: E402

from swipeflix.config import settings  # noqa: E402
from swipeflix.llm.gemini_client import get_gemini_client  # noqa: E402


def test_gemini_api():
    """Test Gemini API connection and configuration."""
    logger.info("=" * 60)
    logger.info("Gemini API Test")
    logger.info("=" * 60)

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
    if not api_key:
        logger.error("❌ GEMINI_API_KEY not set!")
        logger.info("Set it with: export GEMINI_API_KEY=your_key")
        return False

    logger.info(f"✅ API Key found: {api_key[:10]}...")

    # Check model name
    model_name = settings.gemini_model
    logger.info(f"📋 Model: {model_name}")

    # Check rate limits
    logger.info("📊 Rate Limits:")
    logger.info(f"   - RPM: {settings.llm_rpm_limit}")
    logger.info(f"   - TPM: {settings.llm_tpm_limit}")
    logger.info(f"   - Daily: {settings.llm_daily_limit}")

    # Initialize client
    logger.info("\n🔌 Initializing Gemini client...")
    try:
        client = get_gemini_client()

        if not client.is_available():
            logger.error("❌ Client not initialized!")
            logger.error("Check the error messages above for details.")
            return False

        logger.info("✅ Client initialized successfully!")

        # Test API call
        logger.info("\n🧪 Testing API call...")
        test_prompt = "Say 'Hello, SwipeFlix!' in one sentence."

        result = client.generate(
            prompt=test_prompt,
            temperature=0.7,
            max_output_tokens=50,
        )

        if "error" in result:
            logger.error(f"❌ API Error: {result['error']}")
            return False

        logger.info(f"✅ API Response: {result['text']}")
        logger.info(f"📊 Tokens used: {result['tokens_used']}")
        logger.info(f"⏱️  Latency: {result['latency_ms']:.0f}ms")
        logger.info(f"💾 Cached: {result.get('cached', False)}")

        # Test rate limiter
        logger.info("\n📊 Rate Limiter Status:")
        status = client.rate_limiter.get_status()
        logger.info(
            f"   - Requests this minute: {status['requests_this_minute']}/{status['rpm_limit']}"
        )
        logger.info(
            f"   - Tokens this minute: {status['tokens_this_minute']}/{status['tpm_limit']}"
        )
        logger.info(
            f"   - Requests today: {status['requests_today']}/{status['daily_limit']}"
        )

        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests passed!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gemini_api()
    sys.exit(0 if success else 1)

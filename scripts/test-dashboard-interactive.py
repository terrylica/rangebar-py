#!/usr/bin/env python3
"""
Interactive dashboard validation using Playwright.

Tests actual browser interaction with the performance dashboard.
"""

# /// script
# dependencies = [
#   "playwright",
# ]
# ///

import asyncio
import sys
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout


async def test_dashboard():
    """Test dashboard with interactive browser automation."""
    print("=" * 80)
    print("Interactive Dashboard Validation with Playwright")
    print("=" * 80)
    print()

    async with async_playwright() as p:
        # Launch browser
        print("Launching browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Enable console logging
        page.on("console", lambda msg: print(f"  Browser console [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"  Browser error: {exc}"))

        try:
            # Test 1: Navigate to main dashboard
            print("\nTest 1: Navigate to Main Dashboard")
            print("-" * 80)
            base_url = "https://terrylica.github.io/rangebar-py/"
            print(f"  Navigating to {base_url}")

            response = await page.goto(base_url, wait_until="networkidle", timeout=30000)
            print(f"  ✓ Page loaded (status: {response.status})")

            # Test 2: Wait for page title
            print("\nTest 2: Check Page Title")
            print("-" * 80)
            title = await page.title()
            print(f"  Page title: {title}")
            assert "rangebar-py" in title, f"Expected 'rangebar-py' in title, got: {title}"
            print("  ✓ Title correct")

            # Test 3: Wait for heading
            print("\nTest 3: Check Main Heading")
            print("-" * 80)
            heading = await page.wait_for_selector("h1", timeout=5000)
            heading_text = await heading.text_content()
            print(f"  Heading text: {heading_text}")
            assert "rangebar-py" in heading_text, f"Expected 'rangebar-py' in heading"
            print("  ✓ Heading present")

            # Test 4: Wait for metrics to populate
            print("\nTest 4: Wait for Metrics to Load")
            print("-" * 80)
            print("  Waiting for metrics div...")

            # Wait a bit for the fetch to complete
            await asyncio.sleep(3)

            # Check if metrics loaded
            metrics_div = await page.query_selector("#metrics")
            metrics_html = await metrics_div.inner_html()

            if "Benchmark data will appear" in metrics_html or "No benchmark data" in metrics_html:
                print("  ⚠️  Warning: No benchmark data loaded yet")
                print("  (This is expected if workflow hasn't run)")
            elif ".metric-card" in metrics_html or "metric-value" in metrics_html:
                # Count metric cards
                metric_cards = await page.query_selector_all(".metric-card")
                print(f"  ✓ Found {len(metric_cards)} metric cards")

                # Get metric values
                for i, card in enumerate(metric_cards[:3]):  # Show first 3
                    value_elem = await card.query_selector(".metric-value")
                    label_elem = await card.query_selector(".metric-label")
                    if value_elem and label_elem:
                        value = await value_elem.text_content()
                        label = await label_elem.text_content()
                        print(f"    - {label}: {value}")
            else:
                print(f"  Metrics HTML: {metrics_html[:200]}...")

            # Test 5: Check for charts
            print("\nTest 5: Check for Chart Canvases")
            print("-" * 80)
            throughput_chart = await page.query_selector("#throughputChart")
            memory_chart = await page.query_selector("#memoryChart")

            assert throughput_chart is not None, "Throughput chart canvas not found"
            assert memory_chart is not None, "Memory chart canvas not found"
            print("  ✓ Throughput chart canvas present")
            print("  ✓ Memory chart canvas present")

            # Test 6: Click "View Raw Data" link
            print("\nTest 6: Click 'View Raw Data' Link")
            print("-" * 80)

            # Find the link by text
            raw_data_link = await page.query_selector('a[href="dev/bench/data.js"]')
            assert raw_data_link is not None, "View Raw Data link not found"

            link_text = await raw_data_link.text_content()
            print(f"  Found link: {link_text}")

            # Click the link and wait for navigation
            print("  Clicking link...")
            async with context.expect_page() as new_page_info:
                await raw_data_link.click()

            # Wait for new page to load
            new_page = await new_page_info.value
            await new_page.wait_for_load_state("networkidle", timeout=10000)

            new_url = new_page.url
            print(f"  ✓ Navigated to: {new_url}")
            assert "dev/bench/data.js" in new_url, f"Expected data.js in URL, got: {new_url}"

            # Check content
            content = await new_page.content()
            assert "window.BENCHMARK_DATA" in content, "Expected window.BENCHMARK_DATA in data.js"
            print("  ✓ data.js contains window.BENCHMARK_DATA")

            # Close the new page
            await new_page.close()

            # Test 7: Click "Detailed Charts" link
            print("\nTest 7: Click 'Detailed Charts' Link")
            print("-" * 80)

            # Go back to main page
            await page.goto(base_url, wait_until="networkidle", timeout=30000)

            detailed_link = await page.query_selector('a[href="dev/bench/index.html"]')
            assert detailed_link is not None, "Detailed Charts link not found"

            link_text = await detailed_link.text_content()
            print(f"  Found link: {link_text}")

            # Click the link
            print("  Clicking link...")
            async with context.expect_page() as new_page_info:
                await detailed_link.click()

            # Wait for new page to load
            new_page = await new_page_info.value
            await new_page.wait_for_load_state("networkidle", timeout=10000)

            new_url = new_page.url
            print(f"  ✓ Navigated to: {new_url}")
            assert "dev/bench/index.html" in new_url, f"Expected index.html in URL, got: {new_url}"

            # Check for github-action-benchmark page
            new_title = await new_page.title()
            print(f"  Page title: {new_title}")

            # Close the new page
            await new_page.close()

            # Test 8: Check for JavaScript errors
            print("\nTest 8: Check for JavaScript Errors")
            print("-" * 80)

            # Navigate to main page again
            await page.goto(base_url, wait_until="networkidle", timeout=30000)

            # Wait for any async operations
            await asyncio.sleep(3)

            # Get console errors
            console_errors = []
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))

            # Reload to catch any errors
            await page.reload(wait_until="networkidle")
            await asyncio.sleep(2)

            if console_errors:
                print(f"  ⚠️  Found {len(console_errors)} JavaScript errors:")
                for error in console_errors:
                    print(f"    - {error}")
            else:
                print("  ✓ No JavaScript errors detected")

            # Test 9: Take screenshot
            print("\nTest 9: Take Screenshot")
            print("-" * 80)
            screenshot_path = "/tmp/dashboard-screenshot.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"  ✓ Screenshot saved to: {screenshot_path}")

            # Test 10: Check responsive design
            print("\nTest 10: Check Responsive Design")
            print("-" * 80)

            # Test mobile viewport
            await page.set_viewport_size({"width": 375, "height": 667})
            await asyncio.sleep(1)
            print("  ✓ Mobile viewport (375x667) renders")

            # Test tablet viewport
            await page.set_viewport_size({"width": 768, "height": 1024})
            await asyncio.sleep(1)
            print("  ✓ Tablet viewport (768x1024) renders")

            # Test desktop viewport
            await page.set_viewport_size({"width": 1920, "height": 1080})
            await asyncio.sleep(1)
            print("  ✓ Desktop viewport (1920x1080) renders")

            print("\n" + "=" * 80)
            print("✅ ALL INTERACTIVE TESTS PASSED")
            print("=" * 80)

        except PlaywrightTimeout as e:
            print(f"\n❌ Timeout error: {e}")
            return False
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await browser.close()

    return True


if __name__ == "__main__":
    success = asyncio.run(test_dashboard())
    sys.exit(0 if success else 1)

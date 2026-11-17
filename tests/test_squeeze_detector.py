import math
import unittest

from engine.squeeze_detector import MacroSqueezeDetector


class MacroSqueezeDetectorTests(unittest.TestCase):
    def test_missing_inputs_never_create_zero_threshold_signals(self):
        result = MacroSqueezeDetector.evaluate({})
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["status"], "REGIME_NORMAL")
        self.assertFalse(result["alert_needed"])
        self.assertFalse(any(result["signals"].values()))
        self.assertIn("Data unavailable", result["diagnostic_text"])

    def test_complete_confirmed_squeeze(self):
        result = MacroSqueezeDetector.evaluate({
            "spot_price": 500, "iv_change_pct": -3.5,
            "gamma_flip_level": 495, "call_flow_pct": 0.60,
            "cta_position_zscore": -1.5, "cta_cover_trigger": 498,
            "tqqq_net_creation": 1_000_000,
        })
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["status"], "CONFIRMED_SQUEEZE")
        self.assertTrue(result["alert_needed"])

    def test_early_absorption_without_alert(self):
        result = MacroSqueezeDetector.evaluate({
            "iv_change_pct": -4, "spot_price": 500,
            "call_flow_pct": 0.6,
        })
        self.assertEqual(result["score"], 55)
        self.assertEqual(result["status"], "EARLY_ABSORPTION")
        self.assertFalse(result["alert_needed"])

    def test_invalid_or_nonpositive_levels_cannot_trigger(self):
        result = MacroSqueezeDetector.evaluate({
            "spot_price": 500, "gamma_flip_level": 0,
            "cta_position_zscore": -2, "cta_cover_trigger": 0,
            "call_flow_pct": math.nan, "tqqq_net_creation": math.inf,
        })
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["signals"]["gamma_flip"])
        self.assertFalse(result["signals"]["cta_reversal"])
        self.assertFalse(result["signals"]["ap_absorption"])

    def test_official_flow_and_vix_proxy_alone_do_not_confirm(self):
        result = MacroSqueezeDetector.evaluate({
            "iv_source": "VIX daily-change proxy", "iv_change_pct": -5,
            "tqqq_net_creation": 2_000_000,
        })
        self.assertEqual(result["score"], 45)
        self.assertFalse(result["alert_needed"])
        self.assertIn("VIX daily-change proxy", result["diagnostic_text"])


if __name__ == "__main__":
    unittest.main()

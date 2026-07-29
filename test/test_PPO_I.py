from __future__ import annotations

import unittest

from environment.class_public import Config


class ConfigTest(unittest.TestCase):
    def test_default_config_loads_required_sections(self):
        parameters = Config().parameters
        for section in ("warehouse", "robot", "picker", "order", "item", "ppo", "experiment", "paths"):
            self.assertIn(section, parameters)
        self.assertEqual(parameters["experiment"]["item_scenario"], 2)
        self.assertEqual(parameters["experiment"]["item_scenarios"], [2, 4, 6, 10])
        self.assertEqual(parameters["experiment"]["months"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.assertIn(parameters["experiment"]["month"], parameters["experiment"]["months"])
        self.assertIn("fixed_hybrid", parameters["experiment"])
        fixed_hybrid = parameters["experiment"]["fixed_hybrid"]
        self.assertGreaterEqual(fixed_hybrid["long_term_robots"], 1)
        self.assertEqual(
            len(fixed_hybrid["long_term_pickers_area"]),
            parameters["warehouse"]["area_num"],
        )
        self.assertTrue(all(count >= 1 for count in fixed_hybrid["long_term_pickers_area"]))
        self.assertGreater(parameters["warehouse"]["shelf_capacity"], 0)


if __name__ == "__main__":
    unittest.main()

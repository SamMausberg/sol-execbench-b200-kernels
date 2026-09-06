# SPDX-License-Identifier: Apache-2.0
"""CPU replay checks for rejecting contaminated timing evidence."""

import unittest

from qualify_gpu import audit_window, update_family


def sample(second, contexts=(), error=None):
    return {"timestamp": f"2026-09-06T18:00:{second:06.3f}+00:00",
            "finished_at": f"2026-09-06T18:00:{second + 0.01:06.3f}+00:00",
            "contexts": list(contexts), "error": error}


class QualificationReplay(unittest.TestCase):
    def test_foreign_during_lock_wait_does_not_reject_clean_trial(self):
        foreign = {"pid": 900, "campaign_descendant": False}
        own = {"pid": 101, "campaign_descendant": True}
        samples = [sample(0, [foreign]), sample(0.5), sample(1, [own]), sample(1.5, [own]), sample(2)]
        result = audit_window(samples, sample(0.6)["timestamp"], sample(1.9)["timestamp"], 1.75)
        self.assertTrue(result["clean"])

    def test_foreign_context_within_trial_is_rejected(self):
        foreign = {"pid": 900, "campaign_descendant": False}
        samples = [sample(0), sample(0.5, [foreign]), sample(1)]
        result = audit_window(samples, sample(0.1)["timestamp"], sample(0.9)["timestamp"], 1.75)
        self.assertFalse(result["clean"])
        self.assertEqual(result["foreign_context_observations"][0]["pid"], 900)

    def test_missing_monitor_data_does_not_count_as_clean(self):
        result = audit_window([sample(0), sample(5)], sample(0.1)["timestamp"], sample(4.9)["timestamp"], 1.75)
        self.assertFalse(result["clean"])
        result = audit_window([sample(0), sample(0.5, error="query failed"), sample(1)],
                              sample(0.1)["timestamp"], sample(0.9)["timestamp"], 1.75)
        self.assertFalse(result["clean"])

    def test_descendant_reparenting_and_pid_reuse(self):
        family = {}
        update_family({100: (1, 10), 101: (100, 11), 102: (101, 12), 900: (1, 90)}, 100, family)
        self.assertEqual(family, {100: 10, 101: 11, 102: 12})
        update_family({101: (1, 11), 102: (101, 120), 103: (102, 130)}, 100, family)
        self.assertEqual(family[101], 11)
        self.assertEqual(family[102], 120)
        self.assertEqual(family[103], 130)
        unrelated_reuse = {100: 10, 101: 11}
        update_family({101: (1, 111), 103: (101, 130)}, 100, unrelated_reuse)
        self.assertNotIn(103, unrelated_reuse)
        self.assertNotEqual(unrelated_reuse[101], 111)


if __name__ == "__main__":
    unittest.main()

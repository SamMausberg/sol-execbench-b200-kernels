#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused integrity checks for optional model-ID normalization during staging."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from sol_execbench.core import Definition

import campaign


class DefinitionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.definition = {
            "name": "metadata_integrity_example", "hf_id": "",
            "axes": {"M": {"type": "var"}},
            "inputs": {"A": {"shape": ["M"], "dtype": "float32"}},
            "outputs": {"C": {"shape": ["M"], "dtype": "float32"}},
            "reference": "def run(A):\n    return A\n",
        }
        self.original = json.dumps(self.definition, indent=3).encode() + b"\n\n"
        self.adapter = campaign.stage_definition(self.directory, self.original)
        self.run = {
            "definition_adapter": self.adapter,
            "evaluation_definition_file": "definition.native.json",
            "trials": [{"command": ["python", "--definition", "definition.native.json"]}],
        }

    def test_only_optional_metadata_changes_and_original_bytes_survive(self):
        self.assertEqual((self.directory / "definition.json").read_bytes(), self.original)
        native = campaign.read_json(self.directory / "definition.native.json")
        self.assertEqual(native, {**self.definition, "hf_id": None})
        with self.assertRaises(ValueError):
            Definition(**self.definition)
        self.assertIsNone(Definition(**native).hf_id)
        campaign.validate_definition_adapter(self.directory, self.run, self.definition)

    def test_valid_or_missing_metadata_has_no_adapter(self):
        for value in (None, "model/example"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                original = campaign.encoded({**self.definition, "hf_id": value})
                self.assertIsNone(campaign.stage_definition(path, original))
                self.assertEqual((path / "definition.json").read_bytes(), original)
                self.assertFalse((path / "definition.native.json").exists())
                campaign.validate_definition_adapter(path, {}, json.loads(original))

    def test_rehashed_reference_change_is_rejected(self):
        native = {**self.definition, "hf_id": None, "reference": "def run(A): return A * 2"}
        content = campaign.encoded(native)
        campaign.write(self.directory / "definition.native.json", content)
        self.adapter["native_definition_sha256"] = campaign.sha(content)
        with self.assertRaisesRegex(campaign.CampaignError, "beyond optional hf_id"):
            campaign.validate_definition_adapter(self.directory, self.run, self.definition)

    def test_original_byte_tampering_is_rejected(self):
        campaign.write(self.directory / "definition.json", self.original + b" ")
        with self.assertRaisesRegex(campaign.CampaignError, "hash differs"):
            campaign.validate_definition_adapter(self.directory, self.run, self.definition)

    def test_native_byte_tampering_is_rejected(self):
        path = self.directory / "definition.native.json"
        campaign.write(path, path.read_bytes() + b" ")
        with self.assertRaisesRegex(campaign.CampaignError, "hash differs"):
            campaign.validate_definition_adapter(self.directory, self.run, self.definition)

    def test_wrong_command_definition_is_rejected(self):
        wrong = copy.deepcopy(self.run)
        wrong["trials"][0]["command"][-1] = "definition.json"
        with self.assertRaisesRegex(campaign.CampaignError, "different evaluation definition"):
            campaign.validate_definition_adapter(self.directory, wrong, self.definition)

    def test_unrecorded_adapter_is_rejected(self):
        wrong = copy.deepcopy(self.run)
        del wrong["definition_adapter"]
        with self.assertRaisesRegex(campaign.CampaignError, "disagrees with metadata adapter"):
            campaign.validate_definition_adapter(self.directory, wrong, self.definition)


if __name__ == "__main__":
    unittest.main()

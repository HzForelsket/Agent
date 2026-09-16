# Copyright (c) Microsoft. All rights reserved.

"""CPU-only checks for the offline single-turn sharing analysis."""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_2wikimqa_sharing.py"
SPEC = importlib.util.spec_from_file_location("analyze_2wikimqa_sharing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class SharingAnalysisTests(unittest.TestCase):
    """Check branch visibility and strict input handling with hand-counted costs."""

    def rows(self):
        return [
            {
                "sample_id": "question",
                "rollout_index": i,
                "finish_reason": "stop",
                "prompt_tokens": 2,
                "response_tokens": 2,
                "prompt_token_ids": [1, 2],
                "response_token_ids": [3, last],
            }
            for i, last in enumerate((4, 5))
        ]

    def read(self, rows, assume=False):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "responses.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return analysis.read_groups(
                argparse.Namespace(input=source, group_size=2, sharing="prompt", assume_identical_prompts=assume)
            )

    def test_branch_pairs(self):
        prompt = analysis.analyze_group("question", self.rows(), "prompt")
        tree = analysis.analyze_group("question", self.rows(), "tree")
        self.assertEqual((prompt["baseline_tokens"], prompt["baseline_causal_pairs"]), (8, 20))
        self.assertEqual((prompt["shared_tokens"], prompt["shared_causal_pairs"]), (6, 17))
        # Shared chain depths 1,2,3; each of two leaves contributes 4, not 5.
        self.assertEqual((tree["shared_tokens"], tree["shared_causal_pairs"]), (5, 14))

    def test_duplicate_and_prefix_sequences(self):
        rows = self.rows()
        rows[1]["response_token_ids"] = [3]
        rows[1]["response_tokens"] = 1
        result = analysis.analyze_group("question", rows, "tree")
        self.assertEqual((result["shared_tokens"], result["shared_causal_pairs"]), (4, 10))
        result = analysis.analyze_group("question", [rows[0], rows[0]], "tree")
        self.assertEqual((result["shared_tokens"], result["shared_causal_pairs"]), (4, 10))

    def test_reject_duplicate_indices_and_bad_counts(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read([rows[0], rows[0]])
        rows[0]["response_tokens"] = 3
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.read(rows)

    def test_reject_different_prompts(self):
        rows = self.rows()
        rows[1]["prompt_token_ids"] = [1, 9]
        with self.assertRaisesRegex(ValueError, "prompt token IDs differ"):
            analysis.analyze_group("question", rows, "prompt")

    def test_count_only_requires_explicit_assumption(self):
        rows = self.rows()
        for row in rows:
            del row["prompt_token_ids"], row["response_token_ids"]
        with self.assertRaisesRegex(ValueError, "Exact token IDs required"):
            self.read(rows)
        groups = self.read(rows, assume=True)
        result = analysis.analyze_group("question", list(groups["question"].values()), "prompt")
        self.assertEqual(result["count_only_sequences"], 2)

    def test_cli_reports_coverage_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "responses.jsonl"
            output = Path(directory) / "analysis"
            rows = self.rows()
            rows[1]["finish_reason"] = "length"
            rows.append({**rows[0], "sample_id": "incomplete"})
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            command = [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(source),
                "--output",
                str(output),
                "--group-size",
                "2",
                "--sharing",
                "tree",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["complete_groups"], 1)
            self.assertEqual(summary["excluded_groups"][0]["sample_id"], "incomplete")
            self.assertEqual(summary["shared_causal_pairs"], 14)
            self.assertEqual(summary["length_terminated"], 1)
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "per_task.csv").is_file())
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("already exists", retry.stderr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


BENCHSCRIBE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHSCRIBE_DIR))

from characterize import characterize
from cli import main
from model import GroupKey, MetricName, Status
from render import (
    phase_overlap,
    phase_rows,
    render_csv,
    render_fit_csv,
    render_fit_json,
    render_fit_markdown,
    render_json,
    render_markdown,
    render_phase_csv,
    render_phase_markdown,
)
from scan import parse_report_line, scan_results
from summary import SummaryTable


class BenchscribeTest(unittest.TestCase):
    def scan_lines(self, *lines: str):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "1n2g"
            output_dir.mkdir()
            (output_dir / "run-stdout.txt").write_text("\n".join(lines))
            return scan_results(Path(directory))

    def test_moe_hidden_widths_remain_separate_through_render_and_characterization(self):
        measurements = self.scan_lines(
            "cuda_mpi_moe n=8 case=uniform hidden=256 usec=10 bytes=1024 gbytes_per_s=0.1 validation=PASS status=OK",
            "cuda_mpi_moe n=8 case=uniform hidden=512 usec=20 bytes=2048 gbytes_per_s=0.05 validation=PASS status=OK",
        )

        table = SummaryTable.from_measurements(measurements)

        cases = {"uniform,hidden=256", "uniform,hidden=512"}
        self.assertEqual({measurement.case for measurement in measurements}, cases)
        self.assertIn(GroupKey("moe", "1n2g", 8, "uniform,hidden=256"), table.groups)
        self.assertIn(GroupKey("moe", "1n2g", 8, "uniform,hidden=512"), table.groups)
        self.assertEqual(table.rows_for("moe", "1n2g", 8, "uniform,hidden=256")[0].value, 10.0)
        self.assertEqual(table.rows_for("moe", "1n2g", 8, "uniform,hidden=512")[0].value, 20.0)
        characterizations = characterize(table)
        self.assertEqual({item.case for item in characterizations}, cases)

        markdown = io.StringIO()
        render_markdown(table, markdown)
        self.assertIn("### Case: uniform,hidden=256", markdown.getvalue())
        self.assertIn("### Case: uniform,hidden=512", markdown.getvalue())

        csv_output = io.StringIO()
        render_csv(table, csv_output)
        rows = list(csv.DictReader(io.StringIO(csv_output.getvalue())))
        self.assertEqual({row["case"] for row in rows}, cases)

        fit_markdown = io.StringIO()
        render_fit_markdown(characterizations, fit_markdown)
        self.assertIn("uniform,hidden=256", fit_markdown.getvalue())
        self.assertIn("uniform,hidden=512", fit_markdown.getvalue())

        fit_csv = io.StringIO()
        render_fit_csv(characterizations, fit_csv)
        fit_rows = list(csv.DictReader(io.StringIO(fit_csv.getvalue())))
        self.assertEqual({row["case"] for row in fit_rows}, cases)

    def test_not_implemented_backend_is_retained_but_excluded_from_metrics(self):
        measurements = self.scan_lines(
            "cuda_mpi_moe n=8 case=decode usec=10 bytes=1024 gbytes_per_s=0.1 validation=PASS status=OK",
            "cuda_mpi_moe n=8 case=decode usec=100 bytes=2048 gbytes_per_s=8 validation=SKIP status=NOT_IMPLEMENTED",
            "sycl_oneccl_moe n=8 case=decode usec=1 bytes=1024 gbytes_per_s=9 validation=SKIP status=NOT_IMPLEMENTED",
            "sycl_oneccl_moe n=8 case=decode usec=2 bytes=2048 gbytes_per_s=10 validation=SKIP status=NOT_IMPLEMENTED",
        )
        table = SummaryTable.from_measurements(measurements)
        rows = {row.backend: row for row in table.rows_for("moe", "1n2g", 8, "decode")}
        unsupported = rows["sycl_oneccl"]

        self.assertEqual(rows["cuda_mpi"].value, 10.0)
        self.assertEqual(rows["cuda_mpi"].nbytes, 1024)
        self.assertEqual(rows["cuda_mpi"].trials, 1)
        self.assertEqual(unsupported.status, Status.NOT_IMPLEMENTED)
        self.assertTrue(unsupported.valid_all)
        self.assertEqual(unsupported.trials, 2)
        self.assertIsNone(unsupported.value)
        self.assertIsNone(unsupported.bandwidth)
        self.assertIsNone(unsupported.nbytes)
        self.assertIsNone(unsupported.delta_pct_vs_base)
        self.assertIsNone(unsupported.speedup_vs_base)
        self.assertNotIn("sycl_oneccl", {item.backend for item in characterize(table)})

        markdown = io.StringIO()
        render_markdown(table, markdown)
        unsupported_line = next(
            line for line in markdown.getvalue().splitlines() if "`sycl_oneccl`" in line
        )
        self.assertIn("N/I", unsupported_line)
        self.assertNotIn("FAIL", unsupported_line)

        csv_output = io.StringIO()
        render_csv(table, csv_output)
        csv_rows = {
            row["backend"]: row for row in csv.DictReader(io.StringIO(csv_output.getvalue()))
        }
        self.assertEqual(csv_rows["sycl_oneccl"]["status"], "NOT_IMPLEMENTED")
        self.assertEqual(csv_rows["sycl_oneccl"]["valid"], "N/I")
        self.assertEqual(csv_rows["sycl_oneccl"]["value_mean"], "")

    def test_failed_records_are_excluded_from_all_numeric_results(self):
        measurements = self.scan_lines(
            "cuda_mpi_halo_1d n=1 time_per_iter_s=10 bytes=16 gbytes_per_s=1 validation=PASS status=OK",
            "cuda_mpi_halo_1d n=1 usec=1000 time_per_iter_s=100 bytes=160 gbytes_per_s=10 validation=FAIL status=ERROR",
            "cuda_nccl_halo_1d n=1 time_per_iter_s=5 bytes=16 gbytes_per_s=2 validation=PASS status=OK",
            "oshmpi_halo_1d n=1 usec=1 time_per_iter_s=1 bytes=8 gbytes_per_s=9 validation=FAIL status=OK",
        )

        self.assertEqual(measurements[-1].status, Status.ERROR)
        self.assertFalse(measurements[-1].valid)

        table = SummaryTable.from_measurements(measurements)
        rows = {row.backend: row for row in table.rows()}
        baseline = rows["cuda_mpi"]
        failed = rows["oshmpi"]

        self.assertEqual(table.metric_by_benchmark["halo_1d"].name, MetricName.TIME_PER_ITER_S)
        self.assertEqual(baseline.value, 10.0)
        self.assertEqual(baseline.bandwidth, 1.0)
        self.assertEqual(baseline.nbytes, 16)
        self.assertEqual(baseline.trials, 1)
        self.assertEqual(rows["cuda_nccl"].delta_pct_vs_base, -50.0)
        self.assertEqual(rows["cuda_nccl"].speedup_vs_base, 2.0)
        self.assertEqual(failed.status, Status.ERROR)
        self.assertIsNone(failed.value)
        self.assertIsNone(failed.bandwidth)
        self.assertIsNone(failed.nbytes)
        self.assertIsNone(failed.delta_pct_vs_base)
        self.assertIsNone(failed.speedup_vs_base)
        self.assertNotIn("oshmpi", {item.backend for item in characterize(table)})

    def test_halo_batch_cases_remain_separate(self):
        measurements = self.scan_lines(
            "cuda_mpi_halo_1d n=8 case=isolated batch_iters=1 usec=5 validation=PASS status=OK",
            "cuda_mpi_halo_1d n=8 case=steady batch_iters=100 usec=2 validation=PASS status=OK",
        )

        table = SummaryTable.from_measurements(measurements)

        self.assertEqual({measurement.case for measurement in measurements}, {"isolated", "steady"})
        self.assertEqual(table.rows_for("halo_1d", "1n2g", 8, "isolated")[0].value, 5.0)
        self.assertEqual(table.rows_for("halo_1d", "1n2g", 8, "steady")[0].value, 2.0)

    def test_characterize_uses_small_message_median_for_alpha(self):
        measurements = self.scan_lines(
            "cuda_mpi_pingpong n=1 usec=15 bytes=4 gbytes_per_s=0.001 validation=PASS status=OK",
            "cuda_mpi_pingpong n=16 usec=14 bytes=64 gbytes_per_s=0.005 validation=PASS status=OK",
            "cuda_mpi_pingpong n=1024 usec=16 bytes=4096 gbytes_per_s=0.25 validation=PASS status=OK",
            "cuda_mpi_pingpong n=4096 usec=9 bytes=16384 gbytes_per_s=1.8 validation=PASS status=OK",
            "cuda_mpi_pingpong n=262144 usec=100 bytes=1048576 gbytes_per_s=10 validation=PASS status=OK",
        )

        result = characterize(SummaryTable.from_measurements(measurements))[0]

        self.assertEqual(result.alpha, 15.0)
        self.assertEqual(result.binf_gbs, 10.0)
        self.assertEqual(result.nhalf_bytes, 150000.0)

    def test_old_format_infers_status_and_keeps_default_case_output(self):
        measurements = self.scan_lines(
            "cuda_mpi_allreduce n=4 usec=3 validation=PASS",
            "cuda_nccl_allreduce n=4 usec=2 validation=FAIL",
        )

        self.assertEqual(measurements[0].case, "")
        self.assertEqual(measurements[0].status, Status.OK)
        self.assertEqual(measurements[1].status, Status.ERROR)
        table = SummaryTable.from_measurements(measurements)
        rows = {row.backend: row for row in table.rows()}
        self.assertTrue(rows["cuda_mpi"].valid_all)
        self.assertFalse(rows["cuda_nccl"].valid_all)

        markdown = io.StringIO()
        render_markdown(table, markdown)
        self.assertNotIn("Case:", markdown.getvalue())
        self.assertIn("PASS", markdown.getvalue())
        self.assertIn("FAIL", markdown.getvalue())

    def test_skip_status_parses_and_malformed_lines_are_rejected(self):
        parsed = parse_report_line(
            "sycl_oneccl_moe suite_version=0.1.0 source_revision=0.1.0 "
            "validation=SKIP status=NOT_IMPLEMENTED case=decode"
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.fields["suite_version"], "0.1.0")
        self.assertEqual(parsed.fields["source_revision"], "0.1.0")
        self.assertIsNone(parse_report_line("sycl_oneccl_moe status=NOT_IMPLEMENTED"))
        self.assertIsNone(parse_report_line("sycl_oneccl_moe validation=SKIP status=UNKNOWN"))
        self.assertIsNone(parse_report_line('sycl_oneccl_moe validation="SKIP'))
        self.assertIsNone(parse_report_line("validation=PASS"))

    def test_not_implemented_requires_skip_validation(self):
        measurements = self.scan_lines(
            "sycl_oneccl_moe n=8 case=decode usec=1 validation=SKIP status=NOT_IMPLEMENTED",
            "cuda_nccl_moe n=8 case=decode usec=2 validation=PASS status=NOT_IMPLEMENTED",
            "oshmpi_moe n=8 case=decode usec=3 validation=FAIL status=NOT_IMPLEMENTED",
        )

        self.assertEqual(
            [measurement.status for measurement in measurements],
            [Status.NOT_IMPLEMENTED, Status.ERROR, Status.ERROR],
        )
        table = SummaryTable.from_measurements(measurements)
        rows = {row.backend: row for row in table.rows_for("moe", "1n2g", 8, "decode")}
        self.assertEqual(rows["sycl_oneccl"].status, Status.NOT_IMPLEMENTED)
        for backend in ("cuda_nccl", "oshmpi"):
            self.assertEqual(rows[backend].status, Status.ERROR)
            self.assertIsNone(rows[backend].value)
            self.assertFalse(rows[backend].valid_all)

    def test_cli_benchmark_filter_still_applies(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "1n1g"
            output_dir.mkdir()
            (output_dir / "run-stdout.txt").write_text(
                "cuda_mpi_moe n=4 case=decode usec=1 validation=PASS status=OK\n"
                "cuda_mpi_allreduce n=4 usec=2 validation=PASS status=OK\n"
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main([directory, "--benchmark", "moe"])

        self.assertEqual(result, 0)
        self.assertIn("## moe", stdout.getvalue())
        self.assertNotIn("## allreduce", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_fit_json_is_versioned_and_carries_every_characterization(self):
        measurements = self.scan_lines(
            "cuda_mpi_halo_1d n=16 case=steady usec=5 bytes=256 gbytes_per_s=0.05 validation=PASS status=OK",
            "cuda_mpi_halo_1d n=65536 case=steady usec=50 bytes=1048576 gbytes_per_s=21 validation=PASS status=OK",
            "cuda_nccl_halo_1d n=16 case=steady usec=9 bytes=256 gbytes_per_s=0.03 validation=PASS status=OK",
            "cuda_nccl_halo_1d n=65536 case=steady usec=45 bytes=1048576 gbytes_per_s=23 validation=PASS status=OK",
        )
        characterizations = characterize(SummaryTable.from_measurements(measurements))

        stream = io.StringIO()
        render_fit_json(characterizations, stream)
        payload = json.loads(stream.getvalue())

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["alpha_max_bytes"], 4 * 1024)
        self.assertEqual(len(payload["fits"]), len(characterizations))

        by_backend = {fit["backend"]: fit for fit in payload["fits"]}
        self.assertEqual(set(by_backend), {"cuda_mpi", "cuda_nccl"})
        # The plotter reads these keys by name; a rename is a schema break.
        self.assertLessEqual(
            {
                "benchmark",
                "case",
                "topology",
                "backend",
                "unit",
                "alpha",
                "binf_gbs",
                "peak_bytes",
                "nhalf_bytes",
                "tail_gbs",
                "points",
            },
            set(by_backend["cuda_mpi"]),
        )
        self.assertEqual(by_backend["cuda_mpi"]["alpha"], 5.0)
        self.assertEqual(by_backend["cuda_nccl"]["binf_gbs"], 23.0)


    def _phase_table(self):
        """A cg_step point with a breakdown, beside one without."""
        line = (
            "cuda_mpi_cg_step n=512 ranks=16 bytes=4096 iters=50 warmup=10 "
            "time_per_iter_s=0.000111 usec=111.0 min_usec=110.0 max_usec=115.0 "
            "gbytes_per_s=0.037 phase_pack_usec=13.0 phase_halo_usec=22.0 "
            "phase_compute_usec=24.0 phase_reduce_usec=59.0 phase_sum_usec=118.0 "
            "validation=PASS"
        )
        bare = (
            "sycl_mpi_cg_step n=512 ranks=16 bytes=4096 iters=50 warmup=10 "
            "time_per_iter_s=0.000164 usec=164.0 min_usec=161.0 max_usec=170.0 "
            "gbytes_per_s=0.025 validation=PASS"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cg-step-cuda-mpi-4n4g" / "cg_step"
            root.mkdir(parents=True)
            (root / "job-1-1-stdout.txt").write_text(line + "\n" + bare + "\n")
            return SummaryTable.from_measurements(scan_results(Path(tmp)))

    def test_phase_rows_only_include_points_that_carry_a_breakdown(self):
        table = self._phase_table()
        rows = phase_rows(table)
        self.assertEqual([row.backend for row in rows], ["cuda_mpi"])
        # A record measured without the pass has no breakdown, which is not the
        # same as a breakdown of zero.
        bare = [row for row in table.rows() if row.backend == "sycl_mpi"]
        self.assertEqual(len(bare), 1)
        self.assertIsNone(bare[0].phases)

    def test_phase_overlap_is_the_sum_less_the_reported_time(self):
        row = phase_rows(self._phase_table())[0]
        self.assertAlmostEqual(row.phases.pack, 13.0)
        self.assertAlmostEqual(row.phases.total, 118.0)
        # 118 serialized against 111 reported: the unsplit loop overlapped 7 us.
        self.assertAlmostEqual(phase_overlap(row), 7.0)

    def test_phase_views_render_every_phase(self):
        table = self._phase_table()
        out = io.StringIO()
        render_phase_markdown(table, out)
        text = out.getvalue()
        for phase in ("pack", "halo", "compute", "reduce"):
            self.assertIn(phase, text)
        self.assertIn("`cuda_mpi`", text)
        self.assertNotIn("`sycl_mpi`", text)

        out = io.StringIO()
        render_phase_csv(table, out)
        rows = list(csv.DictReader(io.StringIO(out.getvalue())))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["compute"], "24")
        self.assertEqual(rows[0]["overlap"], "7")

    def test_points_json_carries_the_breakdown_and_null_without_it(self):
        table = self._phase_table()
        out = io.StringIO()
        render_json(table, out)
        points = {point["backend"]: point for point in json.loads(out.getvalue())["points"]}
        self.assertEqual(points["cuda_mpi"]["phases"]["compute"], 24.0)
        self.assertIsNone(points["sycl_mpi"]["phases"])


if __name__ == "__main__":
    unittest.main()

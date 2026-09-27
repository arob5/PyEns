"""Tests for pyens.backends.gridengine, using the fake scheduler."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import threading

import pytest

from pyens.backends import (
    GridEngineBackend,
    GridEngineError,
    RemoteError,
    TaskFailedError,
)
from pyens.backends import gridengine
from pyens.backends._batch import BatchDir
from pyens.backends.gridengine import (
    _OWNED_OPTIONS,
    _signals_ignored,
    _signals_raise_system_exit,
    parse_qacct,
    parse_qstat_xml,
    parse_submit_output,
    directive_options,
    parse_task_ids,
    task_ranges,
)
from tests import _models


def _backend(fake, **overrides) -> GridEngineBackend:
    options = dict(
        walltime="00:05:00", work_dir=fake.work_dir, n_jobs=2,
        poll_interval=0.05, missing_grace=0.3,
    )
    options.update(overrides)
    return GridEngineBackend(**options)


def _submitted_script(fake) -> str:
    [script] = fake.scripts()
    return script


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_requires_exactly_one_of_n_jobs_and_runs_per_job(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one"):
            GridEngineBackend(walltime=60, work_dir=tmp_path)
        with pytest.raises(ValueError, match="exactly one"):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=2, runs_per_job=3)

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_counts_must_be_positive_integers(self, tmp_path, value):
        with pytest.raises(ValueError):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=value)
        with pytest.raises(ValueError):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1, slots=value)

    @pytest.mark.parametrize("value, expected", [
        (90, "00:01:30"), (3600 * 30, "30:00:00"), ("1:00:00", "1:00:00"),
        ("100:05:09", "100:05:09"),
    ])
    def test_walltime_normalized(self, tmp_path, value, expected):
        backend = GridEngineBackend(walltime=value, work_dir=tmp_path, n_jobs=1)
        assert backend.walltime == expected

    @pytest.mark.parametrize("value", ["1h", "00:60:00", "", 0, "1:2:3", "00:00:00"])
    def test_walltime_invalid(self, tmp_path, value):
        with pytest.raises(ValueError, match="walltime"):
            GridEngineBackend(walltime=value, work_dir=tmp_path, n_jobs=1)

    @pytest.mark.parametrize("directive", [
        "-t 1-5", "-tc 3", "-o /x", "-e /x", "-N name", "-pe omp 4", "-cwd",
        "-l h_rt=01:00:00", "-l mem_per_core=4G,h_rt=1:00:00", "-sync y",
        "P myproject", "", "-P a\n-P b",
    ])
    def test_rejected_directives(self, tmp_path, directive):
        with pytest.raises(ValueError):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1,
                              directives=[directive])

    def test_accepted_directives_stored_as_tuple(self, tmp_path):
        backend = GridEngineBackend(
            walltime=60, work_dir=tmp_path, n_jobs=1,
            directives=["-P myproject", "-l mem_free=4G", "-v OMP_NUM_THREADS"],
        )
        assert backend.directives == ("-P myproject", "-l mem_free=4G", "-v OMP_NUM_THREADS")

    @pytest.mark.parametrize("name", ["directives", "setup"])
    @pytest.mark.parametrize("value", [
        "-P myproject", {"-P myproject"}, frozenset({"-P myproject"}),
        (d for d in ["-P myproject"]), ["-P myproject", 3], [None],
    ])
    def test_directives_and_setup_must_be_lists_of_strings(self, tmp_path, name, value):
        with pytest.raises(TypeError, match=name):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1, **{name: value})

    def test_directive_order_is_kept(self, tmp_path):
        directives = ["-q b.q", "-P myproject", "-q a.q"]
        backend = GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1,
                                    directives=directives)
        assert backend.directives == tuple(directives)
        assert dataclasses.replace(backend).directives == tuple(directives)

    @pytest.mark.parametrize("name", ["1job", "a b", "a/b", ""])
    def test_invalid_job_name(self, tmp_path, name):
        with pytest.raises(ValueError, match="job_name"):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1, job_name=name)

    def test_work_dir_whitespace_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="whitespace"):
            GridEngineBackend(walltime=60, work_dir=tmp_path / "a b", n_jobs=1)

    def test_work_dir_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        backend = GridEngineBackend(walltime=60, work_dir="batches", n_jobs=1)
        assert backend.work_dir == str(tmp_path / "batches")

    @pytest.mark.parametrize("overrides", [
        {"poll_interval": 0}, {"timeout": 0}, {"missing_grace": -1},
        {"missing_grace": float("nan")}, {"poll_interval": float("nan")},
        {"keep_batch_dir": "sometimes"},
    ])
    def test_other_invalid_values(self, tmp_path, overrides):
        with pytest.raises(ValueError):
            GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1, **overrides)

    def test_replace_makes_validated_variant(self, tmp_path):
        backend = GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=4)
        variant = dataclasses.replace(backend, n_jobs=10)
        assert variant.n_jobs == 10 and backend.n_jobs == 4
        with pytest.raises(ValueError):
            dataclasses.replace(backend, runs_per_job=3)


# ---------------------------------------------------------------------------
# Job script
# ---------------------------------------------------------------------------


_OWNED_ARGS = {
    "-t": "1-10", "-tc": "5", "-o": "/x", "-e": "/x", "-j": "y", "-N": "name",
    "-pe": "omp 4", "-wd": "/x", "-cwd": None, "-S": "/bin/sh", "-sync": "y",
    "-terse": None, "-b": "y", "-now": "y",
}


def _spellings(option: str) -> list[str]:
    arg = _OWNED_ARGS[option]
    if arg is None:
        return [option]
    return [f"{option} {arg}", f"{option}{arg}"]


def _check(tmp_path, directive: str) -> GridEngineBackend:
    return GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1,
                             directives=[directive])


class TestDirectives:
    def test_owned_args_cover_every_owned_option(self):
        assert set(_OWNED_ARGS) == set(_OWNED_OPTIONS)

    @pytest.mark.parametrize("spelled", [
        s for option in _OWNED_ARGS for s in _spellings(option)
    ])
    @pytest.mark.parametrize("template", ["{}", "-m ea {} -M me@example.org",
                                          "-m ea -M me@example.org {}"])
    def test_owned_option_refused_in_any_position(self, tmp_path, spelled, template):
        directive = template.format(spelled)
        option = directive_options(spelled)[0][0]
        with pytest.raises(ValueError) as info:
            _check(tmp_path, directive)
        message = str(info.value)
        assert repr(directive) in message
        assert f"sets {option}" in message
        assert _OWNED_OPTIONS[option] in message

    @pytest.mark.parametrize("directive", [
        "-l h_rt=01:00:00", "-l s_rt=01:00:00", "-hard -l h_rt=1:00:00",
        "-soft -l s_rt=1:00:00", "-l H_RT=1:00:00", "-l mem_free=4G,h_rt=1:00:00",
        "-l mem_free=4G h_rt=1:00:00", "-l 'mem_free=4G, h_rt=1:00:00'",
        "-lh_rt=1:00:00", "-m ea -l h_rt=2:00:00", "-l h_rt = 1:00:00",
        "-q all.q -hard -l mem_free=4G,S_RT=1:00:00 -m ea",
        "-hard h_rt=1:00:00", "-soft s_rt=1:00:00", "-masterl h_rt=1:00:00",
    ])
    def test_run_time_limit_refused(self, tmp_path, directive):
        with pytest.raises(ValueError, match="walltime=") as info:
            _check(tmp_path, directive)
        assert repr(directive) in str(info.value)

    @pytest.mark.parametrize("directive", [
        "-clear", "-m ea -clear", "-@ opts.txt", "-@opts.txt",
        "-adds l_hard h_rt 10:00:00", "-mods l_hard h_vmem 4G", "-clearp l_hard",
        "-clears l_hard h_rt", "-h", "-m ea -h", "-verify", "-help",
        "-w v", "-w p", "-m ea -wv",
    ])
    def test_refused_options(self, tmp_path, directive):
        with pytest.raises(ValueError) as info:
            _check(tmp_path, directive)
        assert repr(directive) in str(info.value)

    @pytest.mark.parametrize("directive", ["-N a#b", "-q a.q # -P myproject",
                                           "-ac 'k=a # b'", "-P myproject #"])
    def test_comment_character_refused(self, tmp_path, directive):
        with pytest.raises(ValueError, match="comment"):
            _check(tmp_path, directive)
        with pytest.raises(ValueError, match="comment"):
            directive_options(directive)

    @pytest.mark.parametrize("directive, word", [("-t1-10", "-t1-10"),
                                                 ("-m ea -exec foo", "-exec")])
    def test_message_names_the_word_as_written(self, tmp_path, directive, word):
        with pytest.raises(ValueError, match=f"read from '{word}'"):
            _check(tmp_path, directive)

    @pytest.mark.parametrize("directive", [
        "-m ea -M me@example.org", "-l mem_per_core=4G", "-q long.q -l mem_per_core=4G",
        "-p -100", "-binding linear:1", "-js 10", "-jsv /x/jsv.sh", "-tcon y",
        "-hard -l h_vmem=4G", "-notify", "-v A=1,B=2 -V", "-ac note='a b'",
        "-hold_jid prep", "-shell y", "-w e", "-w w", "-w n", "-hard -q long.q",
    ])
    def test_allowed_directives(self, tmp_path, directive):
        assert _check(tmp_path, directive).directives == (directive,)

    @pytest.mark.parametrize("directive", ["-m 'ea", '-N "x', "-m ea -M 'me"])
    def test_unbalanced_quote(self, tmp_path, directive):
        with pytest.raises(ValueError) as info:
            _check(tmp_path, directive)
        assert repr(directive) in str(info.value)
        with pytest.raises(ValueError):
            directive_options(directive)

    @pytest.mark.parametrize("directive, expected", [
        ("-P myproject", [("-P", ["myproject"])]),
        ("-Pmyproject", [("-P", ["myproject"])]),
        ("-t 1-10", [("-t", ["1-10"])]),
        ("-t1-10", [("-t", ["1-10"])]),
        ("-tc5", [("-tc", ["5"])]),
        ("-l a=b", [("-l", ["a=b"])]),
        ("-la=b", [("-l", ["a=b"])]),
        ("-l a=b c=d", [("-l", ["a=b", "c=d"])]),
        ("-pe omp 4", [("-pe", ["omp", "4"])]),
        ("-peomp 4", [("-pe", ["omp", "4"])]),
        ("-js10", [("-js", ["10"])]),
        ("-jsv /x", [("-jsv", ["/x"])]),
        ("-p -100", [("-p", ["-100"])]),
        ("-m ea -M me@example.org", [("-m", ["ea"]), ("-M", ["me@example.org"])]),
        ("-hard -l h_vmem=4G -soft -q a.q",
         [("-hard", []), ("-l", ["h_vmem=4G"]), ("-soft", []), ("-q", ["a.q"])]),
        ("-ac 'note=a b'", [("-ac", ["note=a b"])]),
        ("-cwd", [("-cwd", [])]),
        ("-Vx", [("-Vx", [])]),
        ("-q -t 1-10", [("-q", []), ("-t", ["1-10"])]),
    ])
    def test_directive_options(self, directive, expected):
        assert directive_options(directive) == expected

    @pytest.mark.parametrize("directive", ["", "   ", "P myproject", "-P a\n-P b",
                                           "-P a\r"])
    def test_directive_options_rejects_malformed(self, directive):
        with pytest.raises(ValueError):
            directive_options(directive)

    def test_directive_options_rejects_non_string(self):
        with pytest.raises(TypeError):
            directive_options(["-P", "x"])  # type: ignore[arg-type]


class TestScript:
    def test_all_options_rendered(self, tmp_path):
        backend = GridEngineBackend(
            walltime="02:00:00", work_dir=tmp_path, n_jobs=1, slots=4,
            max_concurrent=5, directives=["-P myproject", "-l mem_free=4G"],
            setup=["module load gcc", "export OMP_NUM_THREADS=1"],
            python="/opt/my env/python", job_name="eki_iter",
        )
        batch = BatchDir.create(tmp_path, "x")
        lines = backend.render_script(batch, 7).splitlines()
        assert lines[0] == "#!/bin/bash"
        for expected in [
            "#$ -N eki_iter", "#$ -t 1-7", "#$ -tc 5", "#$ -l h_rt=02:00:00",
            "#$ -pe omp 4", f"#$ -wd {os.getcwd()}", "#$ -j y",
            f"#$ -o {batch.logs_dir}/task-$TASK_ID.log",
            "#$ -P myproject", "#$ -l mem_free=4G",
        ]:
            assert expected in lines
        assert lines.index("module load gcc") < lines.index("export OMP_NUM_THREADS=1")
        assert lines[-1] == (
            f"exec '/opt/my env/python' -m pyens.backends._gridengine_worker "
            f'{batch.root} "$SGE_TASK_ID"'
        )

    def test_single_slot_has_no_pe_or_tc(self, tmp_path):
        backend = GridEngineBackend(walltime=60, work_dir=tmp_path, n_jobs=1)
        script = backend.render_script(BatchDir.create(tmp_path, "x"), 2)
        assert "-pe" not in script
        assert "-tc" not in script


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_QSTAT_XML = """<?xml version='1.0'?>
<job_info xmlns:xsd="http://example">
  <queue_info>
    <job_list state="running">
      <JB_job_number>555</JB_job_number><state>r</state><tasks>1</tasks>
    </job_list>
    <job_list state="running">
      <JB_job_number>999</JB_job_number><state>r</state><tasks>2</tasks>
    </job_list>
  </queue_info>
  <job_info>
    <job_list state="pending">
      <JB_job_number>555</JB_job_number><state>qw</state><tasks>3-7:2</tasks>
    </job_list>
    <job_list state="pending">
      <JB_job_number>555</JB_job_number><state>Eqw</state><tasks>2</tasks>
    </job_list>
  </job_info>
</job_info>
"""

_QACCT = """==============================================================
qname        all.q
jobnumber    555
taskid       1
failed       0
exit_status  0
==============================================================
qname        all.q
jobnumber    555
taskid       2
failed       37  : qmaster enforced h_rt, h_cpu, or h_vmem limit
exit_status  137
maxvmem      1.2G
"""


class TestParsers:
    def test_submit_output(self):
        assert parse_submit_output("12345.1-10:1\n") == "12345"
        assert parse_submit_output("12345\n") == "12345"
        assert parse_submit_output("Your job 1 has been submitted\n") is None

    def test_task_ids(self):
        assert parse_task_ids("3-7:2") == [3, 5, 7]
        assert parse_task_ids("1-3,9") == [1, 2, 3, 9]
        assert parse_task_ids("") == []
        assert parse_task_ids("junk") == []

    def test_qstat_filters_job_and_expands_ranges(self):
        assert parse_qstat_xml(_QSTAT_XML, "555") == {
            1: "r", 2: "Eqw", 3: "qw", 5: "qw", 7: "qw",
        }

    def test_qstat_tolerates_control_characters(self):
        xml = _QSTAT_XML.replace("<JB_job_number>999", "<JB_name>bad\x01name</JB_name>"
                                 "<JB_job_number>999")
        assert parse_qstat_xml(xml, "555")[1] == "r"

    def test_task_ranges(self):
        assert task_ranges([7, 1, 2, 3, 5, 3]) == ["1-3", "5", "7"]
        assert task_ranges([]) == []

    def test_qstat_empty_and_invalid(self):
        assert parse_qstat_xml("<job_info><queue_info/><job_info/></job_info>", "1") == {}
        assert parse_qstat_xml("error: not xml", "1") is None

    def test_qacct(self):
        records = parse_qacct(_QACCT)
        assert records[1]["exit_status"] == "0"
        assert records[2]["exit_status"] == "137"
        assert records[2]["failed"].startswith("37")
        assert records[2]["maxvmem"] == "1.2G"


# ---------------------------------------------------------------------------
# Running against the fake scheduler
# ---------------------------------------------------------------------------


class TestMap:
    def test_runs_per_job_sets_task_count(self, fake_ge):
        backend = _backend(fake_ge, n_jobs=None, runs_per_job=2)
        runs = [{"x": i, "y": 100} for i in range(5)]
        assert backend.map(_models.add, runs) == [100, 101, 102, 103, 104]
        assert "#$ -t 1-3" in _submitted_script(fake_ge)

    def test_more_jobs_than_runs(self, fake_ge):
        backend = _backend(fake_ge, n_jobs=10)
        assert backend.map(_models.add, [{"x": 1, "y": 1}, {"x": 2, "y": 2}]) == [2, 4]
        assert "#$ -t 1-2" in _submitted_script(fake_ge)

    def test_empty_runs_submit_nothing(self, fake_ge):
        assert _backend(fake_ge).map(_models.add, []) == []
        assert fake_ge.calls() == []

    def test_multiple_processes_per_task(self, fake_ge):
        backend = _backend(fake_ge, n_jobs=1, slots=2)
        runs = [{"x": i, "y": 0, "delay": 0.2} for i in range(6)]
        assert backend.map(_models.slow_add, runs) == list(range(6))
        assert "#$ -pe omp 2" in _submitted_script(fake_ge)

    def test_multiple_processes_capture_exceptions(self, fake_ge):
        backend = _backend(fake_ge, n_jobs=1, slots=2)
        results = backend.map(_models.kw_only_crash, [{"x": i} for i in range(4)])
        assert results[0] == 0 and results[2:] == [2, 3]
        assert isinstance(results[1], RemoteError)

    def test_remote_traceback_attached(self, fake_ge):
        [err] = _backend(fake_ge).map(_models.sometimes_fails, [{"x": -1, "y": 0}])
        assert type(err) is ValueError
        assert "sometimes_fails" in str(err.__cause__)

    def test_unpicklable_output(self, fake_ge):
        results = _backend(fake_ge).map(_models.unpicklable_output, [{"x": 0}, {"x": 1}])
        assert results[0] == 0
        assert isinstance(results[1], RemoteError)
        assert "cannot be pickled" in results[1].message

    def test_output_that_cannot_be_unpickled_on_driver(self, fake_ge):
        [out] = _backend(fake_ge).map(_models.return_kw_only_error, [{"x": 1}])
        assert isinstance(out, RemoteError)
        assert "could not be unpickled on the driver" in out.message

    def test_env_forwarded_only_when_requested(self, fake_ge, monkeypatch):
        monkeypatch.setenv("PYENS_TEST_VAR", "forwarded")
        backend = _backend(fake_ge, n_jobs=1, directives=["-v PYENS_TEST_VAR"],
                           setup=["export PYENS_SETUP_VAR=from_setup"])
        [env] = backend.map(_models.getenv_pair, [{}])
        assert env == ("forwarded", "from_setup")


class TestPreSubmissionErrors:
    def test_main_module_model_rejected(self, fake_ge):
        def model(x):
            return x
        model.__module__ = "__main__"
        with pytest.raises(ValueError, match="__main__"):
            _backend(fake_ge).map(model, [{"x": 1}])
        assert fake_ge.calls() == []

    def test_unpicklable_model_raises_before_submission(self, fake_ge):
        with pytest.raises(Exception) as info:
            _backend(fake_ge).map(lambda x: x, [{"x": 1}])
        assert "could not pickle the model" in "".join(info.value.__notes__)
        assert fake_ge.calls() == []
        assert fake_ge.batch_dirs() == []

    def test_unpicklable_input_raises_before_submission(self, fake_ge):
        with pytest.raises(TypeError) as info:
            _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1},
                                                {"x": threading.Lock(), "y": 1}])
        assert "runs 1..1" in "".join(info.value.__notes__)
        assert fake_ge.calls() == []
        assert fake_ge.batch_dirs() == []

    def test_too_many_tasks_rejected_before_writing(self, fake_ge):
        backend = _backend(fake_ge, n_jobs=None, runs_per_job=1)
        with pytest.raises(ValueError, match="max_aj_tasks"):
            backend.map(_models.add, [{"x": 1, "y": 1}] * 75_001)
        assert fake_ge.batch_dirs() == []

    def test_qsub_timeout_keeps_batch(self, fake_ge, monkeypatch):
        real_run = gridengine._run

        def slow_qsub(cmd):
            if cmd[0] == "qsub":
                raise gridengine.subprocess.TimeoutExpired(cmd, 120)
            return real_run(cmd)

        monkeypatch.setattr(gridengine, "_run", slow_qsub)
        with pytest.raises(GridEngineError, match="may have been submitted"):
            _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}])
        assert len(fake_ge.batch_dirs()) == 1

    def test_qsub_failure_raises(self, fake_ge):
        fake_ge.set_faults(qsub_fail="project myproject does not exist")
        with pytest.raises(GridEngineError, match="does not exist"):
            _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}])
        assert fake_ge.batch_dirs() == []


class TestTaskFailures:
    def test_task_killed_mid_chunk_keeps_finished_runs(self, fake_ge):
        fake_ge.set_faults(kill_after={"1": 0.9})
        backend = _backend(fake_ge, n_jobs=2)
        runs = [{"x": i, "y": 0, "delay": 0.4} for i in range(8)]
        results = backend.map(_models.slow_add, runs)
        # Task 1 (runs 0-3) is killed after about two runs; task 2 finishes.
        assert results[4:] == [4, 5, 6, 7]
        done = [r for r in results[:4] if not isinstance(r, BaseException)]
        lost = [r for r in results[:4] if isinstance(r, BaseException)]
        assert done == list(range(len(done))) and 1 <= len(done) <= 3
        for err in lost:
            assert isinstance(err, TaskFailedError)
            assert err.kind == "died"
            assert err.task_id == 1
            assert "exit_status=137" in err.reason
            assert "h_rt=00:05:00" in err.reason
            assert f"{len(done)} of 4 runs" in err.reason
            assert err.log_path.endswith("logs/task-1.log")

    def test_dead_task_deleted_as_safety_net(self, fake_ge):
        fake_ge.set_faults(kill_after={"1": 0.3})
        runs = [{"x": i, "y": 0, "delay": 1.0} for i in range(2)]
        results = _backend(fake_ge, n_jobs=2).map(_models.slow_add, runs)
        assert isinstance(results[0], TaskFailedError) and results[1] == 1
        assert ["qdel", "1000", "-t", "1"] in fake_ge.calls("qdel")

    def test_model_calling_sys_exit_fails_only_its_run(self, fake_ge):
        results = _backend(fake_ge, n_jobs=1).map(
            _models.exits_at, [{"x": i, "at": 1} for i in range(3)])
        assert results[0] == 0 and results[2] == 2
        assert isinstance(results[1], SystemExit)

    def test_listing_error_tolerated(self, fake_ge, monkeypatch):
        real = BatchDir.finished_tasks
        calls = []

        def flaky(self):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("stale file handle")
            return real(self)

        monkeypatch.setattr(BatchDir, "finished_tasks", flaky)
        assert _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}]) == [2]

    def test_worker_process_exit(self, fake_ge):
        backend = _backend(fake_ge, n_jobs=2)
        results = backend.map(_models.die_at, [{"x": i, "die": 1} for i in range(4)])
        assert results[0] == 0
        assert isinstance(results[1], TaskFailedError) and results[1].kind == "died"
        assert results[2:] == [2, 3]

    def test_error_state_task_deleted(self, fake_ge):
        fake_ge.set_faults(eqw_tasks=[2])
        results = _backend(fake_ge, n_jobs=2).map(
            _models.add, [{"x": i, "y": 0} for i in range(4)])
        assert results[:2] == [0, 1]
        for err in results[2:]:
            assert isinstance(err, TaskFailedError)
            assert err.kind == "error_state"
            assert "can't make directory" in err.reason
        assert ["qdel", "1000", "-t", "2"] in fake_ge.calls("qdel")
        assert fake_ge.queued() == []

    def test_timeout_deletes_job(self, fake_ge):
        fake_ge.set_faults(pending_forever=True)
        results = _backend(fake_ge, timeout=0.5).map(
            _models.add, [{"x": i, "y": 0} for i in range(3)])
        assert all(isinstance(r, TaskFailedError) and r.kind == "timeout" for r in results)
        assert ["qdel", "1000"] in fake_ge.calls("qdel")
        assert fake_ge.queued() == []

    def test_worker_setup_error(self, fake_ge):
        results = _backend(fake_ge, n_jobs=1).map(_models.BrokenOnLoad(), [{"x": 1}])
        [err] = results
        assert isinstance(err, TaskFailedError)
        assert err.kind == "worker_error"
        assert "simulated missing dependency" in err.reason
        assert isinstance(err.__cause__, ImportError)

    def test_qstat_failures_tolerated(self, fake_ge):
        fake_ge.set_faults(qstat_fail=True)
        assert _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}]) == [2]

    def test_result_file_wins_over_qstat(self, fake_ge):
        fake_ge.set_faults(qstat_hide=[1, 2])
        backend = _backend(fake_ge, missing_grace=5.0)
        assert backend.map(_models.add, [{"x": 1, "y": 1}, {"x": 2, "y": 2}]) == [2, 4]


class TestInterruption:
    def test_keyboard_interrupt_deletes_job_and_keeps_batch(self, fake_ge, monkeypatch):
        fake_ge.set_faults(pending_forever=True)
        calls = []

        def interrupt(seconds):
            calls.append(seconds)
            if len(calls) == 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(gridengine, "_sleep", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}])
        assert ["qdel", "1000"] in fake_ge.calls("qdel")
        assert fake_ge.queued() == []
        [batch] = fake_ge.batch_dirs()
        assert (batch / "job_id").read_text().strip() == "1000"

    def test_interrupt_during_qsub_keeps_batch(self, fake_ge, monkeypatch):
        def interrupted_submit(script):
            raise KeyboardInterrupt

        monkeypatch.setattr(gridengine._SCHEDULER, "submit", interrupted_submit)
        with pytest.raises(KeyboardInterrupt):
            _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}])
        assert len(fake_ge.batch_dirs()) == 1

    def test_sigterm_becomes_system_exit(self):
        before = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as info:
            with _signals_raise_system_exit():
                os.kill(os.getpid(), signal.SIGTERM)
                for _ in range(1000):
                    pass
        assert info.value.code == 128 + signal.SIGTERM
        assert signal.getsignal(signal.SIGTERM) == before

    def test_sigusr2_becomes_system_exit(self):
        with pytest.raises(SystemExit) as info:
            with _signals_raise_system_exit():
                os.kill(os.getpid(), signal.SIGUSR2)
                for _ in range(1000):
                    pass
        assert info.value.code == 128 + signal.SIGUSR2

    def test_signals_ignored_while_deleting(self):
        with _signals_ignored():
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGINT)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    def test_ignored_signal_left_alone(self):
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            with _signals_raise_system_exit():
                assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        finally:
            signal.signal(signal.SIGHUP, previous)


class TestBatchDirectory:
    def test_removed_on_success_by_default(self, fake_ge):
        _backend(fake_ge).map(_models.add, [{"x": 1, "y": 1}])
        assert fake_ge.batch_dirs() == []

    def test_kept_on_failure_by_default(self, fake_ge):
        _backend(fake_ge).map(_models.always_raises, [{"x": 1}])
        [batch] = fake_ge.batch_dirs()
        manifest = json.loads((batch / "manifest.json").read_text())
        assert manifest["n_runs"] == 1
        assert (batch / "results" / "task-1.pkl").exists()
        assert "task 1: done" in (batch / "logs" / "task-1.log").read_text()

    def test_never_keeps(self, fake_ge):
        _backend(fake_ge, keep_batch_dir="never").map(_models.always_raises, [{"x": 1}])
        assert fake_ge.batch_dirs() == []

    def test_always_keeps(self, fake_ge):
        _backend(fake_ge, keep_batch_dir="always").map(_models.add, [{"x": 1, "y": 1}])
        assert len(fake_ge.batch_dirs()) == 1

    def test_separate_directory_per_map_call(self, fake_ge):
        backend = _backend(fake_ge, keep_batch_dir="always")
        backend.map(_models.add, [{"x": 1, "y": 1}])
        backend.map(_models.add, [{"x": 1, "y": 1}])
        assert len(fake_ge.batch_dirs()) == 2

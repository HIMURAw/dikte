"""Level metering, the WAV writer, and what the sound system is asked for.

The device list is where a platform port lands first, so the parsing is pinned
here: on Linux a source that is not a monitor is an input, one that is belongs
to the speakers, and neither list may go missing when pactl is absent. macOS
answers the same three questions out of one ffmpeg listing.

Each class says which machine it is standing on, so both halves run on either
one: nothing here reaches a real sound server.
"""

import array
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
import wave
from unittest import mock

import audio
from tests.support import (
    DikteTest,
    FakeCompleted,
    only_these_tools,
    pcm,
    silence,
    stereo,
    tone,
)


class OnLinux:
    """A test that runs as if the machine ran PulseAudio or PipeWire."""

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(sys, "platform", "linux"))


class OnMacOS:
    """A test that runs as if the machine were a Mac."""

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(sys, "platform", "darwin"))


class ChunkLevels(unittest.TestCase):
    def test_silence(self):
        self.assertEqual(audio.chunk_levels(silence(0.1)), (0.0, 0.0))

    def test_nothing_at_all(self):
        self.assertEqual(audio.chunk_levels(b""), (0.0, 0.0))

    def test_half_a_sample_is_not_a_sample(self):
        self.assertEqual(audio.chunk_levels(b"\x00"), (0.0, 0.0))

    def test_an_odd_trailing_byte_is_ignored_rather_than_fatal(self):
        peak, _ = audio.chunk_levels(pcm([16384, 16384]) + b"\x7f")
        self.assertAlmostEqual(peak, 0.5, places=3)

    def test_the_peak_is_the_loudest_sample_either_way(self):
        peak, _ = audio.chunk_levels(pcm([0, 0, -32768, 100]))
        self.assertEqual(peak, 1.0)

    def test_the_rms_of_a_constant_signal_is_that_constant(self):
        _, rms = audio.chunk_levels(pcm([16384] * 100))
        self.assertAlmostEqual(rms, 0.5, places=3)

    def test_the_rms_sits_below_the_peak_for_a_tone(self):
        peak, rms = audio.chunk_levels(tone(0.1, amplitude=16384))
        self.assertLess(rms, peak)
        self.assertGreater(rms, 0.0)

    def test_neither_number_ever_passes_one(self):
        peak, rms = audio.chunk_levels(pcm([-32768] * 100))
        self.assertEqual(peak, 1.0)
        self.assertEqual(rms, 1.0)


class StereoLevels(unittest.TestCase):
    def test_the_channels_are_read_apart(self):
        left, right = audio.stereo_levels(stereo(pcm([16384] * 50),
                                                 pcm([0] * 50)))
        self.assertAlmostEqual(left, 0.5, places=3)
        self.assertEqual(right, 0.0)

    def test_nothing_at_all(self):
        self.assertEqual(audio.stereo_levels(b""), (0.0, 0.0))

    def test_a_partial_frame_is_ignored(self):
        self.assertEqual(audio.stereo_levels(b"\x00\x01\x00"), (0.0, 0.0))

    def test_a_meeting_with_both_sides_talking(self):
        left, right = audio.stereo_levels(stereo(pcm([8192] * 50),
                                                 pcm([-16384] * 50)))
        self.assertAlmostEqual(left, 0.25, places=3)
        self.assertAlmostEqual(right, 0.5, places=3)


class WriteWav(DikteTest):
    def test_the_header_says_what_the_recorder_captured(self):
        path = audio.write_wav(silence(0.5))
        self.addCleanup(os.unlink, path)
        with contextlib.closing(wave.open(path, "rb")) as wav:
            self.assertEqual(wav.getnchannels(), audio.CHANNELS)
            self.assertEqual(wav.getsampwidth(), audio.SAMPLE_WIDTH)
            self.assertEqual(wav.getframerate(), audio.RATE)
            self.assertEqual(wav.getnframes(), int(audio.RATE * 0.5))

    def test_the_samples_survive(self):
        path = audio.write_wav(pcm([1000, -1000, 2000]))
        self.addCleanup(os.unlink, path)
        with contextlib.closing(wave.open(path, "rb")) as wav:
            samples = array.array("h")
            samples.frombytes(wav.readframes(3))
        self.assertEqual(list(samples), [1000, -1000, 2000])

    def test_a_meeting_is_written_at_two_channels(self):
        path = audio.write_wav(stereo(silence(0.1), silence(0.1)), channels=2)
        self.addCleanup(os.unlink, path)
        with contextlib.closing(wave.open(path, "rb")) as wav:
            self.assertEqual(wav.getnchannels(), 2)


SOURCES = [
    {"name": "alsa_input.pci-0000_00_1f.3.analog-stereo",
     "description": "Built-in Audio Analog Stereo"},
    {"name": "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
     "description": "Monitor of Built-in Audio"},
    {"name": "bluez_input.AA_BB.headset", "description": ""},
]


class Devices(OnLinux, DikteTest):
    @contextlib.contextmanager
    def pactl(self, sources=None, sink=None, tools=("pactl",)):
        payloads = {
            "list": FakeCompleted(stdout=json.dumps(
                SOURCES if sources is None else sources)),
            "get-default-sink": FakeCompleted(stdout=(sink or "") + "\n"),
        }

        def run(cmd, **kwargs):
            return payloads["get-default-sink" if "get-default-sink" in cmd
                            else "list"]

        with only_these_tools(*tools), \
                mock.patch.object(subprocess, "run", side_effect=run):
            yield

    def test_no_pactl_installed(self):
        with only_these_tools():
            self.assertEqual(audio.list_sources(), [])
            self.assertEqual(audio.list_monitors(), [])
            self.assertEqual(audio.default_monitor(), "")

    def test_inputs_leave_the_monitors_out(self):
        with self.pactl():
            names = [name for name, _ in audio.list_sources()]
        self.assertEqual(names, [SOURCES[0]["name"], SOURCES[2]["name"]])

    def test_monitors_are_the_other_half(self):
        with self.pactl():
            self.assertEqual([name for name, _ in audio.list_monitors()],
                             [SOURCES[1]["name"]])

    def test_a_device_with_no_description_is_shown_by_its_name(self):
        with self.pactl():
            sources = dict(audio.list_sources())
        self.assertEqual(sources[SOURCES[2]["name"]], SOURCES[2]["name"])

    def test_pactl_output_that_is_not_json(self):
        with only_these_tools("pactl"), \
                mock.patch.object(subprocess, "run",
                                  return_value=FakeCompleted(stdout="not json")):
            self.assertEqual(audio.list_sources(), [])

    def test_pactl_that_will_not_run(self):
        with only_these_tools("pactl"), \
                mock.patch.object(subprocess, "run", side_effect=OSError("nope")):
            self.assertEqual(audio.list_sources(), [])

    def test_pactl_that_exits_non_zero(self):
        with only_these_tools("pactl"), \
                mock.patch.object(subprocess, "run",
                                  side_effect=subprocess.CalledProcessError(1, "pactl")):
            self.assertEqual(audio.list_sources(), [])

    def test_the_default_output_is_found_by_its_monitor(self):
        with self.pactl(sink="alsa_output.pci-0000_00_1f.3.analog-stereo"):
            self.assertEqual(audio.default_monitor(),
                             SOURCES[1]["name"])

    def test_a_default_sink_with_no_monitor_of_its_own(self):
        with self.pactl(sink="alsa_output.usb-something"):
            self.assertEqual(audio.default_monitor(), "")

    def test_no_default_sink_at_all(self):
        with self.pactl(sink=""):
            self.assertEqual(audio.default_monitor(), "")

    def test_a_monitor_is_trusted_when_the_list_is_empty(self):
        """pactl answered about the sink but not about the sources."""
        with self.pactl(sources=[], sink="alsa_output.usb-something"):
            self.assertEqual(audio.default_monitor(),
                             "alsa_output.usb-something.monitor")


class FakeProcess:
    """A pw-record that hands over a fixed buffer and then ends."""

    def __init__(self, data):
        self.stdout = io.BytesIO(data)
        self.stderr = io.BytesIO(b"")
        self.signals = []
        self.returncode = 0
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False


class _HookedStdout:
    """A stream that says how much has been handed over, as it hands it over."""

    def __init__(self, data, hook):
        self._buffer = io.BytesIO(data)
        self._hook = hook
        self._given = 0

    def read(self, size):
        chunk = self._buffer.read(size)
        self._given += len(chunk)
        self._hook(self._given)
        return chunk


class PausingProcess(FakeProcess):
    """A recorder that pauses part way through what it hands over.

    Flipped from inside read(), which runs on the pump thread, so the pause
    lands between two chunks rather than wherever another thread reached.
    """

    def __init__(self, data, recorder, pause_at, resume_at=None):
        super().__init__(data)
        self.stdout = _HookedStdout(data, self._at)
        self._recorder = recorder
        self._pause_at = pause_at
        self._resume_at = resume_at

    def _at(self, given):
        if self._pause_at is not None and given >= self._pause_at:
            self._recorder.pause()
            self._pause_at = None
        elif self._resume_at is not None and given >= self._resume_at:
            self._recorder.resume()
            self._resume_at = None


class RecordingCommand(OnLinux, DikteTest):
    """Which program captures the microphone, and how it is asked to."""

    def setUp(self):
        super().setUp()
        # Whether pw-record takes --raw is read off the installed binary, and
        # what is being tested here is the command rather than the machine the
        # test is running on. PwRecordRawOption covers the reading itself.
        self.enterContext(mock.patch.object(
            audio, "_pw_record_raw_option", return_value=["--raw"]))

    def test_parec_is_preferred(self):
        """It speaks to PulseAudio and to PipeWire's compatibility service, so
        it is the one that works on both desktops."""
        with only_these_tools("parec", "pw-record"):
            self.assertEqual(audio.recording_command()[0], "parec")

    def test_pw_record_is_the_fallback(self):
        with only_these_tools("pw-record"):
            self.assertEqual(audio.recording_command()[0], "pw-record")

    def test_neither_is_installed(self):
        with only_these_tools():
            self.assertEqual(audio.recording_command(), [])

    def test_both_capture_the_format_the_rest_of_the_code_expects(self):
        for tool in ("parec", "pw-record"):
            with self.subTest(tool=tool), only_these_tools(tool):
                cmd = audio.recording_command()
                joined = " ".join(cmd)
                self.assertIn(str(audio.RATE), joined)
                self.assertIn(str(audio.CHANNELS), joined)
                self.assertIn("s16", joined)

    def test_parec_is_asked_for_the_level_meter_s_own_chunk(self):
        """Left alone it buffers about two seconds, which the waveform shows as
        a still bar that jumps once a second, and which can cost the tail of a
        recording when the process is asked to stop."""
        with only_these_tools("parec"):
            self.assertIn(f"--latency-msec={audio.CHUNK_LATENCY_MS}",
                          audio.recording_command())

    def test_the_latency_asked_for_is_the_chunk_the_meter_reads(self):
        self.assertEqual(audio.CHUNK_LATENCY_MS,
                         round(audio.CHUNK_FRAMES / audio.RATE * 1000))

    def test_a_chosen_microphone_reaches_either_one(self):
        with only_these_tools("parec"):
            self.assertIn("--device=alsa_input.usb", audio.recording_command(
                "alsa_input.usb"))
        with only_these_tools("pw-record"):
            self.assertIn("--target=alsa_input.usb", audio.recording_command(
                "alsa_input.usb"))

    def test_no_microphone_named_means_no_device_flag(self):
        for tool, flag in (("parec", "--device="), ("pw-record", "--target=")):
            with self.subTest(tool=tool), only_these_tools(tool):
                self.assertFalse([arg for arg in audio.recording_command()
                                  if arg.startswith(flag)])


class PwRecordRawOption(DikteTest):
    """Two pw-record generations want opposite commands for the same stream.

    PipeWire 1.4 added --raw and stopped treating a filename of "-" as raw on
    its own, so the option is refused by everything older and needed by
    everything newer. The help text is the only thing that tells them apart.
    """

    def option(self, **run):
        with mock.patch.object(audio.subprocess, "run", **run):
            return audio._pw_record_raw_option()

    def test_a_version_that_offers_raw_is_asked_for_it(self):
        self.assertEqual(["--raw"], self.option(
            return_value=FakeCompleted(stdout="  -a, --raw   RAW mode\n")))

    def test_a_version_without_it_is_not(self):
        self.assertEqual([], self.option(
            return_value=FakeCompleted(stdout="  --rate  Sample rate\n")))

    def test_help_that_could_not_be_read_keeps_the_option(self):
        """Whatever is installed, the command that worked before this check
        existed is the safer guess."""
        self.assertEqual(["--raw"], self.option(side_effect=OSError))
        self.assertEqual(["--raw"], self.option(
            side_effect=subprocess.TimeoutExpired("pw-record", 2)))

    def test_help_that_said_nothing_keeps_it_too(self):
        self.assertEqual(["--raw"], self.option(return_value=FakeCompleted()))


class RecorderChain(OnLinux, DikteTest):
    """Start to WAV, with pw-record faked out."""

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(
            audio, "_pw_record_raw_option", return_value=["--raw"]))

    def record(self, data, target="", max_seconds=300):
        recorder = audio.Recorder()
        results = []
        failures = []
        recorder.stopped.connect(lambda *args: results.append(args))
        recorder.failed.connect(failures.append)
        proc = FakeProcess(data)
        with only_these_tools("pw-record"), \
                mock.patch.object(subprocess, "Popen", return_value=proc) as popen:
            recorder.start(target=target, max_seconds=max_seconds)
            recorder._thread.join(timeout=5)
            recorder.stop()
        return recorder, results, failures, popen

    def test_the_capture_format_is_what_the_rest_of_the_code_expects(self):
        _, _, _, popen = self.record(silence(1.0))
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[0], "pw-record")
        self.assertIn(f"--rate={audio.RATE}", cmd)
        self.assertIn(f"--channels={audio.CHANNELS}", cmd)
        self.assertIn("--format=s16", cmd)
        self.assertEqual(cmd[-1], "-")

    def test_no_target_means_no_target_flag(self):
        _, _, _, popen = self.record(silence(0.5))
        self.assertFalse([arg for arg in popen.call_args.args[0]
                          if arg.startswith("--target=")])

    def test_a_chosen_microphone_is_passed_on(self):
        _, _, _, popen = self.record(silence(0.5), target="alsa_input.usb")
        self.assertIn("--target=alsa_input.usb", popen.call_args.args[0])

    def test_a_recording_ends_as_a_wav_with_its_duration_and_levels(self):
        _, results, failures, _ = self.record(tone(1.0))
        self.assertEqual(failures, [])
        path, duration, rms = results[0]
        self.addCleanup(os.unlink, path)
        self.assertAlmostEqual(duration, 1.0, places=2)
        self.assertTrue(rms)
        self.assertGreater(max(rms), 0.0)
        with contextlib.closing(wave.open(path, "rb")) as wav:
            self.assertEqual(wav.getnframes(), audio.RATE)

    def test_a_stray_keypress_is_not_a_recording(self):
        _, results, failures, _ = self.record(silence(0.1))
        self.assertEqual(results, [])
        self.assertIn("0.3", failures[0])

    def test_a_cancelled_recording_produces_nothing(self):
        recorder = audio.Recorder()
        results = []
        recorder.stopped.connect(lambda *args: results.append(args))
        proc = FakeProcess(tone(1.0))
        with only_these_tools("pw-record"), \
                mock.patch.object(subprocess, "Popen", return_value=proc):
            recorder.start()
            recorder._thread.join(timeout=5)
            recorder.cancel()
            recorder.stop()
        self.assertEqual(results, [])

    def paused_run(self, seconds, pause_at, resume_at=None):
        """Record `seconds` of tone, pausing and resuming at those offsets."""
        per_second = audio.RATE * audio.SAMPLE_WIDTH * audio.CHANNELS
        recorder = audio.Recorder()
        results = []
        recorder.stopped.connect(lambda *args: results.append(args))
        proc = PausingProcess(
            tone(seconds), recorder, int(pause_at * per_second),
            None if resume_at is None else int(resume_at * per_second))
        with only_these_tools("pw-record"), \
                mock.patch.object(subprocess, "Popen", return_value=proc):
            recorder.start()
            recorder._thread.join(timeout=5)
            recorder.stop()
        return recorder, results

    # The pause takes effect between two chunks, so where it falls is exact
    # to one of them and no closer. Written as the chunk rather than as a
    # number, so that changing CHUNK_BYTES does not turn this red.
    CHUNK_SECONDS = audio.CHUNK_BYTES / (audio.RATE * audio.SAMPLE_WIDTH
                                         * audio.CHANNELS)

    def test_what_is_recorded_while_paused_is_not_kept(self):
        """Half a second in, and the second half of it never reaches the file."""
        _, results = self.paused_run(1.0, pause_at=0.5)
        path, duration, _ = results[0]
        self.addCleanup(os.unlink, path)
        self.assertAlmostEqual(duration, 0.5, delta=2 * self.CHUNK_SECONDS)

    def test_the_two_halves_either_side_of_a_pause_are_one_recording(self):
        _, results = self.paused_run(1.0, pause_at=0.25, resume_at=0.75)
        path, duration, _ = results[0]
        self.addCleanup(os.unlink, path)
        self.assertAlmostEqual(duration, 0.5, delta=2 * self.CHUNK_SECONDS)

    def test_resuming_leaves_it_recording_again(self):
        recorder, _ = self.paused_run(1.0, pause_at=0.25, resume_at=0.75)
        self.assertFalse(recorder.paused)

    def test_there_is_nothing_to_pause_before_it_starts(self):
        recorder = audio.Recorder()
        self.assertFalse(recorder.pause())
        self.assertFalse(recorder.resume())
        self.assertFalse(recorder.paused)

    def test_a_recording_that_runs_past_the_limit_is_cut_off(self):
        _, results, _, _ = self.record(tone(3.0), max_seconds=1)
        path, duration, _ = results[0]
        self.addCleanup(os.unlink, path)
        self.assertLessEqual(duration, 1.1)

    def test_a_recorder_that_is_not_installed_at_all(self):
        recorder = audio.Recorder()
        failures = []
        recorder.failed.connect(failures.append)
        with only_these_tools():
            recorder.start()
        self.assertEqual(len(failures), 1)
        self.assertIn("pulseaudio-utils", failures[0])

    def pump(self, data=b"", stderr=b"", stopping=False, cancelled=False):
        """Run the pump in this thread, where a queued signal would need an
        event loop nobody is running here."""
        recorder = audio.Recorder()
        failures = []
        recorder.failed.connect(failures.append)
        proc = FakeProcess(data)
        proc.stderr = io.BytesIO(stderr)
        proc._alive = False
        recorder._proc = proc
        recorder._max_bytes = 10 ** 9
        recorder._stopping = stopping
        recorder._cancelled = cancelled
        recorder._pump()
        return failures

    def test_a_recorder_that_died_on_its_own_says_so(self):
        """parec refused the device, or the sound server went away."""
        failures = self.pump(stderr=b"connection refused\n")
        self.assertEqual(len(failures), 1)
        self.assertIn("connection refused", failures[0])

    def test_a_death_with_nothing_on_stderr_still_names_the_exit_code(self):
        failures = self.pump()
        self.assertIn("exit code", failures[0])

    def test_a_recording_we_ended_ourselves_is_not_a_death(self):
        """Otherwise a stray keypress produces two errors, and the first one
        sends the user looking for a broken sound server."""
        self.assertEqual(self.pump(stopping=True), [])

    def test_a_cancelled_recording_is_not_a_death(self):
        self.assertEqual(self.pump(cancelled=True), [])

    def test_a_recorder_that_captured_something_first_is_not_a_death(self):
        self.assertEqual(self.pump(data=silence(0.5)), [])

    def test_a_short_recording_reports_only_that(self):
        _, results, failures, _ = self.record(silence(0.1))
        self.assertEqual(results, [])
        self.assertEqual(len(failures), 1)
        self.assertIn("0.3", failures[0])

    def test_a_recorder_that_could_not_start(self):
        recorder = audio.Recorder()
        failures = []
        recorder.failed.connect(failures.append)
        with only_these_tools("pw-record"), \
                mock.patch.object(subprocess, "Popen", side_effect=OSError("nope")):
            recorder.start()
        self.assertEqual(len(failures), 1)
        self.assertFalse(recorder.active)


class MeetingCommand(unittest.TestCase):
    """One process reading both devices, because two would drift apart."""

    def command(self, platform, mic="", system="them"):
        with mock.patch.object(sys, "platform", platform):
            return audio.meeting_command(mic, system)

    def test_linux_reads_both_through_pulse(self):
        cmd = self.command("linux", mic="mine")
        self.assertEqual(cmd.count("pulse"), 2)
        self.assertEqual(cmd[cmd.index("mine") - 1], "-i")
        self.assertEqual(cmd[cmd.index("them") - 1], "-i")

    def test_a_mac_reads_both_through_avfoundation(self):
        cmd = self.command("darwin", mic="1")
        self.assertEqual(cmd.count("avfoundation"), 2)
        self.assertIn(":1", cmd)
        self.assertIn(":them", cmd)

    def test_no_microphone_named_means_the_default_one(self):
        self.assertIn("default", self.command("linux"))
        self.assertIn(":default", self.command("darwin"))

    def test_both_merge_the_two_into_one_stereo_stream(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                cmd = self.command(platform)
                self.assertIn(audio.MERGE_FILTER, cmd)
                self.assertEqual(cmd[cmd.index("-map") + 1], "[out]")
                self.assertEqual(cmd[cmd.index("-f", cmd.index("-map")) + 1], "s16le")

    def test_neither_lets_ffmpeg_read_the_terminal(self):
        """It shares stdin with Dikte, and would eat a keypress meant for it."""
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                self.assertIn("-nostdin", self.command(platform))


class MacDevices(OnMacOS, DikteTest):
    """The one ffmpeg listing all three device questions are answered from."""

    LISTING = (
        "[AVFoundation indev @ 0x7fb] AVFoundation video devices:\n"
        "[AVFoundation indev @ 0x7fb] [0] FaceTime HD Camera\n"
        "[AVFoundation indev @ 0x7fb] [1] Capture screen 0\n"
        "[AVFoundation indev @ 0x7fb] AVFoundation audio devices:\n"
        "[AVFoundation indev @ 0x7fb] [0] MacBook Pro Microphone\n"
        "[AVFoundation indev @ 0x7fb] [1] BlackHole 2ch\n"
        ": Input/output error\n"
    )

    @contextlib.contextmanager
    def listing(self, stderr=None, tools=("ffmpeg",)):
        completed = FakeCompleted(
            returncode=1, stderr=self.LISTING if stderr is None else stderr)
        with only_these_tools(*tools), \
                mock.patch.object(subprocess, "run", return_value=completed):
            yield

    def test_the_audio_half_of_the_listing_is_the_only_half_read(self):
        with self.listing():
            self.assertEqual(audio.list_sources(),
                             [("0", "MacBook Pro Microphone"), ("1", "BlackHole 2ch")])

    def test_the_index_is_what_ffmpeg_is_given_and_the_name_what_is_shown(self):
        with self.listing():
            name, description = audio.list_sources()[1]
        self.assertEqual(name, "1")
        self.assertIn("BlackHole", description)

    def test_no_ffmpeg_installed(self):
        with only_these_tools():
            self.assertEqual(audio.list_sources(), [])
            self.assertEqual(audio.list_monitors(), [])
            self.assertEqual(audio.default_monitor(), "")

    def test_an_ffmpeg_that_will_not_run(self):
        with only_these_tools("ffmpeg"), \
                mock.patch.object(subprocess, "run", side_effect=OSError("nope")):
            self.assertEqual(audio.list_sources(), [])

    def test_a_listing_with_no_audio_section(self):
        with self.listing(stderr="[AVFoundation indev @ 0x7fb] [0] FaceTime\n"):
            self.assertEqual(audio.list_sources(), [])

    def test_the_far_side_of_a_meeting_is_offered_the_same_devices(self):
        """macOS calls none of them an output, so the loopback one is in here."""
        with self.listing():
            self.assertEqual(audio.list_monitors(), audio.list_sources())

    def test_the_loopback_driver_is_picked_out_by_name(self):
        with self.listing():
            self.assertEqual(audio.default_monitor(), "1")

    def test_the_other_two_drivers_people_install(self):
        for name in ("Loopback Audio", "Soundflower (2ch)"):
            with self.subTest(name=name):
                listing = ("AVFoundation audio devices:\n"
                           f"[0] Built-in Microphone\n[1] {name}\n")
                with self.listing(stderr=listing):
                    self.assertEqual(audio.default_monitor(), "1")

    def test_a_mac_with_nothing_to_record_the_far_side_from(self):
        listing = "AVFoundation audio devices:\n[0] MacBook Pro Microphone\n"
        with self.listing(stderr=listing):
            self.assertEqual(audio.default_monitor(), "")


class MacRecordingCommand(OnMacOS, DikteTest):
    def test_the_microphone_is_read_through_avfoundation(self):
        with only_these_tools("ffmpeg"):
            cmd = audio.recording_command()
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertEqual(cmd[cmd.index("-f") + 1], "avfoundation")

    def test_the_empty_half_in_front_of_the_colon_is_the_missing_picture(self):
        with only_these_tools("ffmpeg"):
            self.assertIn(":default", audio.recording_command())
            self.assertIn(":2", audio.recording_command("2"))

    def test_it_captures_the_format_the_rest_of_the_code_expects(self):
        with only_these_tools("ffmpeg"):
            cmd = audio.recording_command()
        self.assertEqual(cmd[cmd.index("-ar") + 1], str(audio.RATE))
        self.assertEqual(cmd[cmd.index("-ac") + 1], str(audio.CHANNELS))
        self.assertEqual(cmd[-2:], ["s16le", "-"])

    def test_no_ffmpeg_installed(self):
        with only_these_tools():
            self.assertEqual(audio.recording_command(), [])

    def test_what_a_mac_is_told_to_install(self):
        recorder = audio.Recorder()
        failures = []
        recorder.failed.connect(failures.append)
        with only_these_tools():
            recorder.start()
        self.assertIn("brew install ffmpeg", failures[0])
        self.assertFalse(recorder.active)


if __name__ == "__main__":
    unittest.main()

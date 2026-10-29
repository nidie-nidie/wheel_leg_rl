import importlib.util
import pathlib
import tempfile
import unittest
import struct

from helpers_v2 import header, stage, sample, footer, finish, stream, decoded_fixture
from pace_raw.decoder import decode_bytes, decode_file
from pace_raw.quality import require_fit_quality, quality_report
from pace_raw.scanner import FrameScanner
from pace_raw.normalize import normalize_session
from pace_raw.session_metadata import write_session_metadata
from tools.capture_session import capture
from fit.pace_optimizer_bridge import export_pace_dataset
from fit.fit_leg_dm import fit_leg_dm
from fit.fit_wheel_lk import fit_wheel_lk
from fit.model_manifest import create_model_manifest
from pace_raw.session_metadata import build_session_metadata,manifest_snapshot,content_hash
from pace_raw.manifest import load_firmware_manifest


class ProtocolV2Test(unittest.TestCase):
    def test_stage_fields_quality_and_fixed_snapshot(self):
        decoded=decoded_fixture()
        require_fit_quality(decoded)
        config=decoded.stage_configs[1]
        self.assertEqual((1,1,50,1),(config.experiment_role,config.segment_id,config.action_update_rate_hz,config.command_hold_method))
        dataset=normalize_session(decoded)
        self.assertEqual("FIT",dataset.rows[0]["role"])
        self.assertEqual(1,dataset.rows[0]["segment_id"])
        self.assertEqual("user_confirmed_existing_code",decoded.metadata["calibration_snapshot"]["direction_source"])

    def test_closed_session_refuses_following_sample_and_bytes(self):
        for tail in (sample(4,8000),b"x",b"P"):
            decoded=decode_bytes(stream()+tail)
            self.assertEqual(1,len(decoded.samples))
            self.assertIn("post_footer_data",[issue.code for issue in decoded.issues])

    def test_zero_rate_reports_issue_without_exception(self):
        decoded=decode_bytes(header(0)+stage()+sample()+sample(3,8000)+footer(4))
        self.assertIn("header_rate",[issue.code for issue in decoded.issues])

    def test_false_magic_and_crc_recover_real_footer(self):
        prefix=struct.pack("<2sBBH",b"PA",2,2,126)+b"xx"
        scanner=FrameScanner()
        output=scanner.feed(prefix+footer(0))+scanner.finish()
        self.assertEqual(1,len(output))
        corrupted=bytearray(sample());corrupted[40]^=1
        decoded=decode_bytes(header()+stage()+bytes(corrupted)+footer(3))
        self.assertIsNotNone(decoded.footer)
        self.assertIn("crc_error",[issue.code for issue in decoded.issues])

    def test_incremental_all_split_boundaries_equal_offline(self):
        raw=stream()
        for cut in range(len(raw)+1):
            scanner=FrameScanner()
            found=scanner.feed(raw[:cut])+scanner.feed(raw[cut:])+scanner.finish()
            self.assertEqual([0,1,2,3],[frame.header.sequence for _,frame in found])
            self.assertFalse(scanner.issues)

    def test_nested_crc_valid_frame_in_payload_never_changes_chunk_interpretation(self):
        from helpers_v2 import frame
        nested=finish(frame(5,30,99))
        raw=bytearray(sample())
        raw[26:56]=nested
        raw=finish(raw)
        expected=[(offset,type(frame).__name__) for offset,frame in FrameScanner().feed(raw)]
        for cut in range(len(raw)+1):
            scanner=FrameScanner()
            found=scanner.feed(raw[:cut])+scanner.feed(raw[cut:])+scanner.finish()
            self.assertEqual(expected,[(offset,type(frame).__name__) for offset,frame in found])
            self.assertFalse(scanner.issues)

    def test_invalid_gain_active_group_and_stale_config_are_rejected(self):
        bad=bytearray(stage());struct.pack_into("<H",bad,30,5000)
        decoded=decode_bytes(header()+finish(bad)+sample()+footer())
        self.assertIn("invalid_controller_gain",[issue.code for issue in decoded.issues])
        decoded=decode_bytes(header()+stage()+sample(mask=3)+footer())
        self.assertIn("active_group_mismatch",[issue.code for issue in decoded.issues])
        decoded=decode_bytes(header()+stage()+stage(sequence=2,config=2,segment=2)+sample(sequence=3)+footer(4))
        self.assertIn("stale_stage_config",[issue.code for issue in decoded.issues])

    def test_sidecar_tampering_and_raw_tampering_block_fit(self):
        decoded=decoded_fixture();decoded.metadata["calibration_snapshot"]["motors"][0]["position_sign"]=-1
        with self.assertRaisesRegex(ValueError,"hash"):
            require_fit_quality(decoded)
        decoded=decoded_fixture();decoded.raw_stream+=b"x"
        with self.assertRaisesRegex(ValueError,"SHA"):
            require_fit_quality(decoded)

    def test_formal_export_refuses_ineligible_sample(self):
        dataset=normalize_session(decoded_fixture())
        dataset.rows[0]["m0_fit_eligible"]=False
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"invalid"):
                export_pace_dataset(dataset,pathlib.Path(directory)/"out.npz")
            self.assertFalse((pathlib.Path(directory)/"out.npz").exists())

    def test_all_public_identification_entries_use_quality_gate(self):
        dataset=normalize_session(decoded_fixture())
        dataset.decoded.raw_stream+=b"tamper"
        for entry in (fit_leg_dm,fit_wheel_lk,create_model_manifest):
            with self.subTest(entry=entry.__name__):
                with self.assertRaisesRegex(ValueError,"SHA"):
                    entry(dataset)

    def test_snapshot_override_and_sample_fault_cannot_pass_quality(self):
        import dataclasses
        dataset=normalize_session(decoded_fixture())
        motor=dataset.manifest.motors[0]
        dataset.manifest.motors[0]=dataclasses.replace(motor,position_sign=-motor.position_sign)
        with self.assertRaisesRegex(ValueError,"immutable capture snapshot"):
            require_fit_quality(dataset)
        decoded=decoded_fixture()
        decoded.samples[0].safety_flags=1
        with self.assertRaisesRegex(ValueError,"safety/CAN fault"):
            require_fit_quality(decoded)

    def test_expected_plan_conflicts_are_rejected(self):
        snapshot=manifest_snapshot(load_firmware_manifest())
        base={"session_type":"commissioning","collection_contract":{"sample_rate_hz":250,"control_rate_hz":500,"action_rate_hz":50,"command_hold":"zero_order_hold","dm_kp":20.,"dm_kd":.6}}
        decoded=decoded_fixture()
        decoded.metadata=build_session_metadata(decoded,snapshot,plan=base)
        require_fit_quality(decoded)
        declared={**base,"collection_contract":{**base["collection_contract"],"fit_and_independent_validation":True}}
        decoded.metadata=build_session_metadata(decoded,snapshot,plan=declared)
        with self.assertRaisesRegex(ValueError,"coverage missing"):
            require_fit_quality(decoded)
        for key,value in (("sample_rate_hz",500),("dm_kp",120.),("action_rate_hz",100)):
            plan={**base,"collection_contract":{**base["collection_contract"],key:value}}
            decoded.metadata=build_session_metadata(decoded,snapshot,plan=plan)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"planned"):
                require_fit_quality(decoded)

    def test_binary_identity_requires_actual_file_hash(self):
        import hashlib
        snapshot=manifest_snapshot(load_firmware_manifest())
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/"firmware.bin"
            path.write_bytes(b"real build artifact")
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            decoded=decoded_fixture()
            decoded.metadata=build_session_metadata(decoded,snapshot,plan={"firmware_binary":{"path":str(path),"sha256":actual}})
            require_fit_quality(decoded)
            self.assertEqual(actual,decoded.metadata["firmware_binary_sha256"])
            path.write_bytes(b"changed")
            decoded.metadata=build_session_metadata(decoded,snapshot,plan={"firmware_binary":{"path":str(path),"sha256":actual}})
            self.assertIsNone(decoded.metadata["firmware_binary_sha256"])
            with self.assertRaisesRegex(ValueError,"binary differs"):
                require_fit_quality(decoded)

    def test_nonfinite_stage_still_retains_diagnostic_metadata(self):
        invalid=bytearray(stage())
        struct.pack_into("<f",invalid,80,float("nan"))
        decoded=decode_bytes(header()+finish(invalid)+sample()+footer())
        decoded.metadata=build_session_metadata(decoded,manifest_snapshot(load_firmware_manifest()))
        self.assertEqual("nan",decoded.metadata["online"]["stages"][0]["phase_rad"])
        self.assertFalse(quality_report(decoded)["fit_quality_passed"])

    def test_stop_unconfirmed_never_passes_quality(self):
        decoded=decode_bytes(header()+stage()+sample()+footer(3,9,0))
        from pace_raw.session_metadata import build_session_metadata,manifest_snapshot
        from pace_raw.manifest import load_firmware_manifest
        decoded.metadata=build_session_metadata(decoded,manifest_snapshot(load_firmware_manifest()))
        with self.assertRaisesRegex(ValueError,"footer"):
            require_fit_quality(decoded)


class FakeSerial:
    def __init__(self,chunks):
        self.chunks=list(chunks)
        self.writes=[]
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,size):
        chunk=self.chunks.pop(0) if self.chunks else b""
        if isinstance(chunk,BaseException): raise chunk
        return chunk
    def write(self,data): self.writes.append(data)
    def flush(self): pass
    def reset_input_buffer(self): raise AssertionError("capture must preserve queued header")


class CaptureV2Test(unittest.TestCase):
    def test_passive_preserves_prebuffered_header_and_sidecar(self):
        device=FakeSerial([stream()[:3],stream()[3:]])
        with tempfile.TemporaryDirectory() as directory:
            output=capture("FAKE",pathlib.Path(directory)/"session.raw",serial_factory=lambda **kwargs:device)
            self.assertEqual(stream(),output.read_bytes())
            self.assertTrue(output.with_suffix(".quality.json").exists())
            decoded=decode_file(output)
            require_fit_quality(decoded)
            self.assertFalse(device.writes)

    def test_passive_waits_ten_second_autostart(self):
        device=FakeSerial([b""]*10+[stream()])
        ticks=iter(range(30))
        with tempfile.TemporaryDirectory() as directory:
            output=capture("FAKE",pathlib.Path(directory)/"session.raw",serial_factory=lambda **kwargs:device,clock=lambda:next(ticks))
            require_fit_quality(decode_file(output))

    def test_incomplete_false_candidate_recovers_footer_at_idle_end(self):
        prefix=struct.pack("<2sBBH",b"PA",2,2,126)+b"xx"
        raw=header()+stage()+prefix+footer(2)
        device=FakeSerial([raw]+[b""]*10)
        ticks=iter(range(30))
        with tempfile.TemporaryDirectory() as directory:
            output=capture("FAKE",pathlib.Path(directory)/"session.raw",serial_factory=lambda **kwargs:device,clock=lambda:next(ticks))
            decoded=decode_file(output)
            self.assertIsNotNone(decoded.footer)
            self.assertFalse(quality_report(decoded)["fit_quality_passed"])

    def test_native_c_fixtures_capture_both_modes(self):
        build=pathlib.Path(__file__).resolve().parents[2]/"host_debug"/"build"
        paths=[build/"fixture_passive.raw",build/"fixture_active.raw"]
        if not all(path.exists() for path in paths):
            self.skipTest("run host_debug/run_checks.py to generate real C fixtures")
        for active,path in zip((False,True),paths):
            with self.subTest(active=active),tempfile.TemporaryDirectory() as directory:
                raw=path.read_bytes()
                widths=(1,3,17,127,4096,16384)
                chunks=[]
                offset=0
                while offset<len(raw):
                    size=widths[len(chunks)%len(widths)]
                    chunks.append(raw[offset:offset+size]);offset+=size
                device=FakeSerial(chunks)
                output=capture("FAKE",pathlib.Path(directory)/"session.raw",passive=not active,
                               serial_factory=lambda **kwargs:device,sleeper=lambda _:None)
                self.assertEqual(raw,output.read_bytes())
                decoded=decode_file(output)
                require_fit_quality(decoded)
                self.assertEqual(26,len(decoded.stage_configs))
                self.assertEqual(52750,len(decoded.samples))
                self.assertEqual(active,bool(device.writes))

    def test_interrupt_retains_incomplete_raw_and_failed_sidecar(self):
        device=FakeSerial([header(),KeyboardInterrupt()])
        with tempfile.TemporaryDirectory() as directory:
            output=pathlib.Path(directory)/"session.raw"
            with self.assertRaisesRegex(RuntimeError,"without a footer"):
                capture("FAKE",output,serial_factory=lambda **kwargs:device)
            decoded=decode_file(output)
            self.assertEqual("interrupted",decoded.metadata["capture_state"])
            self.assertFalse(quality_report(decoded)["fit_quality_passed"])


if __name__ == "__main__":
    unittest.main()

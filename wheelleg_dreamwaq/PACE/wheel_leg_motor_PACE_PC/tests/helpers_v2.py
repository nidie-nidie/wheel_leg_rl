import pathlib
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pace_raw.decoder import decode_bytes
from pace_raw.manifest import load_firmware_manifest
from pace_raw.session_metadata import build_session_metadata, manifest_snapshot


def frame(kind, length, sequence, version=2):
    result = bytearray(length)
    struct.pack_into("<2sBBHI", result, 0, b"PA", version, kind, length, sequence)
    return result


def finish(result):
    struct.pack_into("<I", result, len(result)-4, zlib.crc32(result[:-4]) & 0xffffffff)
    return bytes(result)


def header(rate=250, sequence=0):
    result = frame(1,58,sequence)
    result[10:14] = bytes((1,1,1,1))
    struct.pack_into("<IHHIIHH",result,14,7,rate,10,1000,0x50414331,126,500)
    struct.pack_into("<IIIII",result,34,921600,1000000,0x4D4F5431,0x20260928,0x574C5031)
    return finish(result)


def stage(sequence=1, config=1, segment=1, role=1, stage_id=2, group=15, mode=5):
    result = frame(3,128,sequence)
    result[10:18]=bytes((config,stage_id,1,1,1,1,mode,mode))
    struct.pack_into("<6H",result,18,500,500,500,500,100,100)
    struct.pack_into("<4H",result,30,*([163]*4))
    struct.pack_into("<4H",result,38,*([491]*4))
    result[46]=1
    struct.pack_into("<fffIH",result,48,.15,.2,5.,12000,1)
    struct.pack_into("<I",result,68,0x50414331)
    struct.pack_into("<BBHHBB",result,72,role,group,segment,50,1,1)
    struct.pack_into("<11f",result,80,.47,0.,0.,0.,0.,1.,-1.,-1.,1.,1.,-1.)
    return finish(result)


def sample(sequence=2,time=4000,config=1,stage_id=2,mask=15):
    result = frame(2,126,sequence)
    struct.pack_into("<IBBBBBBBBHH",result,10,time,stage_id,config,mask,63,63,63,0,0,0,0)
    for index in range(4):
        struct.pack_into("<8H",result,26+16*index,32768,2048,2048,32768,2048,2048,100,200)
    for index in range(2):
        struct.pack_into("<iHhhhHH",result,90+16*index,0,32768,0,0,0,100,200)
    return finish(result)


def footer(sequence=3,stop_reason=0,complete=1):
    result=frame(6,34,sequence)
    struct.pack_into("<IIIBBHI",result,10,8000,sequence+1,0,0,complete,stop_reason,0x50414331)
    return finish(result)


def stream():
    return header()+stage()+sample()+footer()


def decoded_fixture(rate=250):
    result=decode_bytes(header(rate)+stage()+sample()+footer())
    result.metadata=build_session_metadata(result,manifest_snapshot(load_firmware_manifest()))
    return result

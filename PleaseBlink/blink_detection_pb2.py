# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: blink_detection.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15\x62link_detection.proto\x12\x0f\x62link_detection\"\x1e\n\rGetEARRequest\x12\r\n\x05image\x18\x01 \x01(\t\"B\n\x0eGetEARResponse\x12\x0b\n\x03\x65\x61r\x18\x01 \x01(\x02\x12\x10\n\x08left_eye\x18\x02 \x03(\x05\x12\x11\n\tright_eye\x18\x03 \x03(\x05\x32[\n\x0e\x42linkDetection\x12I\n\x06GetEAR\x12\x1e.blink_detection.GetEARRequest\x1a\x1f.blink_detection.GetEARResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'blink_detection_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_GETEARREQUEST']._serialized_start=42
  _globals['_GETEARREQUEST']._serialized_end=72
  _globals['_GETEARRESPONSE']._serialized_start=74
  _globals['_GETEARRESPONSE']._serialized_end=140
  _globals['_BLINKDETECTION']._serialized_start=142
  _globals['_BLINKDETECTION']._serialized_end=233
# @@protoc_insertion_point(module_scope)

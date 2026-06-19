# Schema provenance

`schema.fbs` is the TensorFlow Lite FlatBuffers schema. It is committed to this
repository so the build is reproducible and the binding generation is pinned. We
do not fetch it at build time.

- Source: https://github.com/tensorflow/tensorflow, path
  `tensorflow/compiler/mlir/lite/schema/schema.fbs`
- `file_identifier`: `TFL3`
- `root_type`: `Model`

The committed file is the pin. Updating it is a deliberate, reviewed act:
replace the file, run `npm run gen:schema`, review the regenerated bindings under
`src/serialize/schema`, and run the full test suite before committing. The
generator uses the `flatc` compiler; install it with `brew install flatbuffers`
or from https://github.com/google/flatbuffers/releases.

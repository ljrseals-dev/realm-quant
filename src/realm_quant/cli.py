"""Command-line interface for the Realm of Quant Python workbench."""
from __future__ import annotations

import argparse
import sys

from .quantization import quantize_model
from .runtime import (\n    compare_models,\n    decode_codes,\n    encode_audio,\n    inspect_model,\n    roundtrip_audio,\n    write_json,\n)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realm-quant",
        description="Quantize ONNX weights and execute neural audio codec models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect ONNX Runtime input/output contracts")
    inspect.add_argument("model")

    quantize = sub.add_parser("quantize", help="create fp16, dynamic-int8, or ternary-weight model")
    quantize.add_argument("model")
    quantize.add_argument("--mode", choices=("fp16", "int8", "ternary"), required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--threshold-ratio", type=float, default=0.7)
    quantize.add_argument("--op-types", default="MatMul,Gemm", help="comma-separated ONNX op types for INT8")

    compare = sub.add_parser("compare", help="compare two models with identical NPZ input feeds")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument("--feeds", required=True)
    compare.add_argument("--threads", type=int, default=1)
    compare.add_argument("--report")

    encode = sub.add_parser("encode", help="encode a WAV file into DAC codes")
    encode.add_argument("encoder")
    encode.add_argument("audio")
    encode.add_argument("--codes", required=True)
    encode.add_argument("--threads", type=int, default=1)

    decode = sub.add_parser("decode", help="decode a saved DAC code tensor into WAV")
    decode.add_argument("decoder")
    decode.add_argument("codes")
    decode.add_argument("--output", required=True)
    decode.add_argument("--threads", type=int, default=1)

    roundtrip = sub.add_parser("roundtrip", help="encode then decode a WAV through the codec")
    roundtrip.add_argument("encoder")
    roundtrip.add_argument("decoder")
    roundtrip.add_argument("audio")
    roundtrip.add_argument("--output", required=True)
    roundtrip.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_model(args.model)
        elif args.command == "quantize":
            result = quantize_model(
                args.model,
                args.output,
                args.mode,
                args.threshold_ratio,
                tuple(x.strip() for x in args.op_types.split(",") if x.strip()),
            )
        elif args.command == "compare":
            result = compare_models(args.reference, args.candidate, args.feeds, args.threads)
            write_json(result, args.report)
            return 0
        elif args.command == "encode":
            result = encode_audio(args.encoder, args.audio, args.codes, args.threads)
        elif args.command == "decode":
            result = decode_codes(args.decoder, args.codes, args.output, args.threads)
        elif args.command == "roundtrip":
            result = roundtrip_audio(args.encoder, args.decoder, args.audio, args.output, args.threads)
        else:
            raise AssertionError(args.command)
        write_json(result)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"realm-quant: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

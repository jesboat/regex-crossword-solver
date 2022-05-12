#!/bin/sh

mydir="$(python3 -c 'import pathlib; import sys; print(pathlib.Path(sys.argv[1]).resolve().parent)' "$0")"


in_venv() {
    (
        . "$mydir/venv/bin/activate"
        "$@"
        exit $?
    )
    return $?
}


make_venv() {
    python3 -m venv "$mydir/venv"
    in_venv pip3 install --upgrade pip wheel
    in_venv pip3 install pyre-check z3-solver
}


pyre() {
    in_venv command pyre "$@"
    return $?
}


run() {
    in_venv python3 "$mydir/src/solver.py" "$@"
    return $?
}


case "${1:-}" in
    in_venv)    shift; in_venv "$@"; exit $?;;
    pyre)       shift; pyre "$@"; exit $?;;
    make_venv)  shift; make_venv "$@"; exit $?;;
    run)        shift; run "$@"; exit $?;;
    *)
        echo >&2 "Usage: $0 in_venv CMD ARG..."
        echo >&2 "   or: $0 pyre ARG..."
        echo >&2 "   or: $0 make_venv ..."
        echo >&2 "   or: $0 run_solver {--help | ARG ...}"
        exit 1
        ;;
esac


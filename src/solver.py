# pyre-strict

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import json
import pathlib
import pprint
import re
import subprocess
import sys
import time

import z3

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    NoReturn,
    Optional,
    Pattern,
    Tuple,
    Type,
    TypeVar,
    Union,
)


#
# ============================================================
# Some basic utilities
# ============================================================
#


T = TypeVar('T')


UUID_RX: re.Pattern[str] = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.I)

DEBUG: bool = False


class UserError(Exception):
    pass

class InternalError(Exception):
    pass

class InputDataError(Exception):
    pass


def fail(exn: Union[Exception, Callable[[], Exception]]) -> NoReturn:
    if DEBUG:
        import pdb
        pdb.set_trace()
    raise (exn if isinstance(exn, Exception) else exn())


def check(
        cond: object,
        exn: Union[Exception, Callable[[], Exception]],
) -> None:
    if not cond:
        fail(exn)


def checked_cast(
        obj: object,
        typ: Type[T],
        exn: Union[Exception, Callable[[], Exception]],
) -> T:
    if isinstance(obj, typ):
        return obj
    else:
        fail(exn)


def checked_castL(
        obj: object,
        elemtype: Type[T],
        exn: Union[Exception, Callable[[], Exception]],
) -> list[T]:
    if isinstance(obj, list) and all(isinstance(elem, elemtype) for elem in obj):
        return obj
    else:
        fail(exn)


def checked_castD(
        obj: object,
        valtype: Type[T],
        exn: Union[Exception, Callable[[], Exception]],
) -> dict[str, T]:
    if isinstance(obj, dict) and all(isinstance(key, str) and isinstance(val, valtype) for key, val in obj.items()):
        return obj
    else:
        fail(exn)


#
# ============================================================
# Modeling clues, puzzles, groups 
# ============================================================
#


@dataclasses.dataclass
class Group():
    id: str
    name: str
    puzzles: list[Puzzle]


@dataclasses.dataclass
class Clue():
    regex: str
    cells: List[Tuple[int, ...]]

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclasses.dataclass
class Puzzle():
    id: str
    name: str

    @property
    def is_hexagonal(self) -> bool:
        raise AssertionError('abstract')

    def iter_clues(self) -> Iterable[Clue]:
        raise AssertionError('abstract')


@dataclasses.dataclass
class RectangularPuzzle(Puzzle):
    patternsX: List[Tuple[str, Optional[str]]]
    patternsY: List[Tuple[str, Optional[str]]]

    def iter_clues(self) -> Iterable[Clue]:
        nrows = len(self.patternsY)
        ncols = len(self.patternsX)

        for colidx, (clueF, clueR) in enumerate(self.patternsX):
            cells=[(rowidx, colidx) for rowidx in range(nrows)]
            yield Clue(regex=clueF, cells=cells)
            if clueR is not None:
                yield Clue(regex=clueR, cells=cells)

        for rowidx, (clueF, clueR) in enumerate(self.patternsY):
            cells=[(rowidx, colidx) for colidx in range(ncols)]
            yield Clue(regex=clueF, cells=cells)
            if clueR is not None:
                yield Clue(regex=clueR, cells=cells)

    @property
    def is_hexagonal(self) -> bool:
        return False


HexPoint = Tuple[int, int, int]


@dataclasses.dataclass
class HexagonalPuzzle(Puzzle):
    patternsP: List[str]
    patternsQ: List[str]
    patternsR: List[str]

    @property
    def is_hexagonal(self) -> bool:
        return True

    def iter_clues(self) -> Iterable[Clue]:
        patsP, patsQ, patsR = self.patternsP, self.patternsQ, self.patternsR
        coord_chart = self._gen_coords(len(patsP), len(patsQ), len(patsR))
        
        for pats, axis in ((patsP, 'p'), (patsQ, 'q'), (patsR, 'r')):
            for pat, cells in zip(pats, coord_chart[axis], strict=True):
                yield Clue(regex=pat, cells=list(cells))

    @staticmethod
    @functools.lru_cache()
    def _gen_coords(lenP: int, lenQ: int, lenR: int) -> dict[str, list[list[HexPoint]]]:
        Pm, Qm, Rm = lenP - 1, lenQ - 1, lenR - 1
        S = (Pm + Qm + Rm) // 2

        alpha   = (Pm, S - Pm, 0)
        beta    = (S - Qm, Qm, 0)
        gamma   = (0, Qm, S - Qm)
        delta   = (0, S - Rm, Rm)
        epsilon = (S - Rm, 0, Rm)
        zeta    = (Pm, 0, S - Pm)

        def traverse(start: HexPoint, end: HexPoint, *, inc_end: bool) -> Iterable[HexPoint]:
            assert(any(s == e for s, e in zip(start, end)))
            assert(0 == sum(s - e for s, e in zip(start, end)))
            dp, dq, dr = (min(+1, max(-1, e - s)) for s, e in zip(start, end))
            p, q, r = start
            while (p, q, r) != end:
                yield (p, q, r)
                p, q, r = p + dp, q + dq, r + dr
            if inc_end:
                yield (p, q, r)

        def do_axis(
                scoord1: HexPoint,
                scoord2: HexPoint,
                scoord3: HexPoint,
                ecoord1: HexPoint,
                ecoord2: HexPoint,
                ecoord3: HexPoint
        ) -> list[list[HexPoint]]:
            starts = (
                list(traverse(scoord1, scoord2, inc_end=False)) +
                list(traverse(scoord2, scoord3, inc_end=True)))
            ends = (
                list(traverse(ecoord1, ecoord2, inc_end=False)) +
                list(traverse(ecoord2, ecoord3, inc_end=True)))
            return [
                list(traverse(start, end, inc_end=True))
                for start, end in zip(starts, ends)
            ]

        return {
            'p': do_axis(delta, epsilon, zeta, gamma, beta, alpha),
            'q': do_axis(zeta, alpha, beta, epsilon, delta, gamma),
            'r': do_axis(beta, gamma, delta, alpha, zeta, epsilon),
        }


#
# ============================================================
# Decoding `challenges.json` input file
# ============================================================
#


def decode_root(obj: object) -> list[Group]:
    obj = checked_cast(obj, list, InputDataError)
    return [decode_group(sub) for sub in obj]


def decode_group(obj: object) -> Group:
    obj = checked_castD(obj, object, InputDataError)

    id = checked_cast(obj.get('id', None), str, InputDataError)
    check(id, InputDataError)

    name = checked_cast(obj.get('name', None), str, InputDataError)
    check(name, InputDataError)

    puzzle_objs = checked_cast(obj.get('puzzles', None), list, InputDataError)
    puzzles = [decode_puzzle(p) for p in puzzle_objs]

    return Group(id=id, name=name, puzzles=puzzles)


def decode_puzzle(obj: object) -> Puzzle:
    obj = checked_castD(obj, object, InputDataError)

    id = checked_cast(obj.get('id', None), str, InputDataError)
    check(UUID_RX.fullmatch(id), InputDataError)

    name = checked_cast(obj.get('name', None), str, InputDataError)
    check(name, InputDataError)

    pX = obj.get('patternsX', None)
    patternsX = decode_patternlist(pX)

    pY = obj.get('patternsY', None)
    patternsY = decode_patternlist(pY)

    pZ = obj.get('patternsZ', None)

    if pZ is not None:
        patternsZ = decode_patternlist(pZ)

        check(
            all(line[1] is None
                for ps in (patternsX, patternsY, patternsZ)
                    for line in ps
            ),
            InputDataError)

        return HexagonalPuzzle(
            id=id,
            name=name,
            patternsP=[line[0] for line in reversed(patternsX)],
            patternsQ=[line[0] for line in patternsY],
            patternsR=[line[0] for line in patternsZ],
        )
    else:
        return RectangularPuzzle(
            id=id,
            name=name,
            patternsX=patternsX,
            patternsY=patternsY,
        )


def decode_patternlist(obj: object) -> List[Tuple[str, Optional[str]]]:
    obj = checked_cast(obj, list, InputDataError)
    for row in obj:
        row = checked_castL(row, str, InputDataError)
        check(len(row) in (1, 2), InputDataError)
        check(len(row[0]), InputDataError)
    return [(row[0], (row[1] if len(row) == 2 and row[1] else None))
            for row in obj]


#
# ============================================================
# Modeling and parsing symbolic constraint representation
# ============================================================
#


@dataclasses.dataclass
class SBTrue(): pass

@dataclasses.dataclass
class SBFalse(): pass

@dataclasses.dataclass
class SBAnd():
    a: SBBool
    b: SBBool

@dataclasses.dataclass
class SBOr():
    a: SBBool
    b: SBBool

@dataclasses.dataclass
class SBNot():
    a: SBBool

@dataclasses.dataclass
class SBCond():
    cond: SBBool
    ifT: SBBool
    ifF: SBBool

@dataclasses.dataclass
class SBCharIn():
    idx: int
    charlist: str

@dataclasses.dataclass
class SBCharEq():
    idx1: int
    idx2: int

class SBBool: pass

SBBool = Union[
    SBTrue,
    SBFalse,
    SBAnd,
    SBOr,
    SBNot,
    SBCond,
    SBCharIn,
    SBCharEq,
]


WORDPAT: Pattern[str] = re.compile(r'([a-z][a-zA-Z0-9_-]*)(?!\S)')
STRPAT: Pattern[str]  = re.compile(r'"(([a-zA-Z0-9]+|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})*)"(?!\S)')
NUMPAT: Pattern[str]  = re.compile(r'(0|[1-9][0-9]*)(?!\S)')
NULLPAT: Pattern[str] = re.compile(r'NULL(?!\S)')
WSPAT: Pattern[str]   = re.compile(r'\s+')

@dataclasses.dataclass
class WordToken():
    val: str

@dataclasses.dataclass
class StringToken():
    val: str

@dataclasses.dataclass
class NumberToken():
    val: int

@dataclasses.dataclass
class NullToken():
    pass

@dataclasses.dataclass
class EofToken():
    pass

Token = Union[WordToken, StringToken, NumberToken, NullToken, EofToken]


class ConstraintParser(object):
    text: str
    pos: int

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _fail(self) -> NoReturn:
        fail(InternalError("failed to parse regex constraint data"))

    def _token(self) -> Union[WordToken, StringToken, NumberToken, NullToken, EofToken]:
        if self.pos == len(self.text):
            return EofToken()

        if self.pos:
            m = WSPAT.match(self.text, self.pos)
            m or self._fail()
            assert(m and m.start() == self.pos)
            self.pos = m.end()

        if self.pos == len(self.text):
            return EofToken()

        if m := WORDPAT.match(self.text, self.pos):
            assert(m.start() == self.pos)
            self.pos = m.end()
            return WordToken(m.group())

        elif m := STRPAT.match(self.text, self.pos):
            assert(m.start() == self.pos)
            self.pos = m.end()
            return StringToken(re.sub(
                r'\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4})',
                lambda m2: chr(int(m2.group(1)[1:], 16)),
                m.group(1),
            ))

        elif m := NUMPAT.match(self.text, self.pos):
            assert(m.start() == self.pos)
            self.pos = m.end()
            return NumberToken(int(m.group(1)))

        elif m := NULLPAT.match(self.text, self.pos):
            assert(m.start() == self.pos)
            self.pos = m.end()
            return NullToken()

        else:
            self._fail()

    def parse_RegexSynbolicCondition(self) -> Union[SBCharIn, SBCharEq]:
        match self._token():
            case WordToken("in"):
                return SBCharIn(self.get_num(), self.get_str())
            case WordToken('eq'):
                return SBCharEq(self.get_num(), self.get_num())
            case _:
                self._fail()

    def parse_SymbolicBoolOver(self, parse_embed: Callable[[], SBBool]) -> SBBool:
        match self._token():
            case WordToken('true'):
                return SBTrue()
            case WordToken('false'):
                return SBFalse()
            case WordToken('and'):
                return SBAnd(*(self.parse_SymbolicBoolOver(parse_embed) for _ in range(2)))
            case WordToken('or'):
                return SBOr(*(self.parse_SymbolicBoolOver(parse_embed) for _ in range(2)))
            case WordToken('cond'):
                return SBCond(*(self.parse_SymbolicBoolOver(parse_embed) for _ in range(3)))
                #return SBCond(
                #    self.parse_SymbolicBoolOver(parse_embed),
                #    self.parse_SymbolicBoolOver(parse_embed),
                #    self.parse_SymbolicBoolOver(parse_embed))
            case WordToken('not'):
                return SBNot(self.parse_SymbolicBoolOver(parse_embed))
            case WordToken('embed'):
                return parse_embed()
            case _:
                self._fail()

    def get_num(self) -> int:
        match self._token():
            case NumberToken(val):  return val
            case _:                 self._fail()

    def get_str(self) -> str:
        match self._token():
            case StringToken(val):  return val
            case _:                 self._fail()

    def parse_constraints(self) -> SBBool:
        ret = self.parse_SymbolicBoolOver(self.parse_RegexSynbolicCondition)
        match self._token():
            case EofToken():        return ret
            case _:                 self._fail()


#
# ============================================================
# Computing symbolic representations (aka shelling out to Haskell)
# ============================================================
#


class RegexSymbolifier:
    _decoded_table: dict[Tuple[str, int], SBBool]

    def __init__(self) -> None:
        self._decoded_table = {}

    @staticmethod
    @functools.lru_cache()
    def _get_haskell_runner() -> Callable[[str], str]:
        project_dir: pathlib.Path = pathlib.Path(sys.argv[0]).resolve().parent.parent

        # Check if stack is available
        proc = subprocess.run(
            args=['sh', '-c', 'command -v stack'],
            stdout=subprocess.PIPE,
        )

        if proc.returncode == 0 and proc.stdout:
            return(lambda haskell_input:
                subprocess.run(
                    args=[
                        'stack', 'exec',
                        'regex-crossword-solver-tool',
                        'generate-constraints',
                    ],
                    cwd=project_dir,
                    input=haskell_input,
                    stdout=subprocess.PIPE,
                    encoding='utf-8',
                    check=True,
                ).stdout
            )

        # No stack. Can we use runhaskell?
        proc = subprocess.run(
            args=['sh', '-c', 'command -v runhaskell'],
            stdout=subprocess.PIPE,
        )

        if proc.returncode == 0 and proc.stdout:
            return(lambda haskell_input:
                subprocess.run(
                    args=[
                        'runhaskell',
                        project_dir / 'src' / 'Regex.hs',
                        'generate-constraints',
                    ],
                    stdout=subprocess.PIPE,
                    encoding='utf-8',
                    check=True,
                    input=haskell_input,
                ).stdout
            )

        # No way to run Haskell :(
        raise InternalError(
                "Unable to run Haskell code: neither 'stack' nor 'runhaskell' is installed")

    def process(self, queries: Iterable[Tuple[str, int]]) -> None:
        pending_queries = list(set(q for q in queries if q not in self._decoded_table))

        haskell_input = ""
        for pat, strlen in pending_queries:
            check(" " not in pat, lambda: InternalError("implementation limit"))
            check("\n" not in pat, lambda: InternalError("implementation limit"))
            haskell_input += "%d %s\n" % (strlen, pat)

        haskell_output = self._get_haskell_runner()(haskell_input).splitlines()
        check(len(haskell_output) == len(pending_queries) * 2, InternalError)

        lineno, querynum = 0, 0
        while lineno < len(haskell_output):
            pat, strlen = pending_queries[querynum]
            input_line_expected = "%d %s" % (strlen, pat)
            input_line_actual = haskell_output[lineno]
            constraints_line = haskell_output[lineno + 1]
            check(input_line_expected == input_line_actual, InternalError)
            self._decoded_table[(pat, strlen)] = (
                ConstraintParser(constraints_line).parse_constraints()
            )
            lineno += 2
            querynum += 1

    def query(self, pat: str, strlen: int) -> SBBool:
        return self._decoded_table[(pat, strlen)]


def build_symbolifier_for(puzzles: Iterable[Puzzle]) -> RegexSymbolifier:
    symbolifier = RegexSymbolifier()
    symbolifier.process(
        (c.regex, len(c.cells))
        for puzzle in puzzles
            for c in puzzle.iter_clues())
    return symbolifier


#
# ============================================================
# Hexagon leftovers
# ============================================================
#


#def gen_hexagon_coords(patsX, patsY, patsZ):
#    patsP = list(reversed(patsX))
#    patsQ = patsY
#    patsR = patsZ
#
#    Pm, Qm, Rm = len(patsP) - 1, len(patsQ) - 1, len(patsR) - 1
#    S = (Pm + Qm + Rm) // 2
#
#    alpha   = (Pm, S - Pm, 0)
#    beta    = (S - Qm, Qm, 0)
#    gamma   = (0, Qm, S - Qm)
#    delta   = (0, S - Rm, Rm)
#    epsilon = (S - Rm, 0, Rm)
#    zeta    = (Pm, 0, S - Pm)
#
#    def traverse(start, end, *, inc_end):
#        assert(any(s == e for s, e in zip(start, end)))
#        dp, dq, dr = (min(+1, max(-1, e - s)) for s, e in zip(start, end))
#        p, q, r = start
#        while (p, q, r) != end:
#            yield (p, q, r)
#            p, q, r = p + dp, q + dq, r + dr
#        if inc_end:
#            yield (p, q, r)
#
#    def do_axis(scoord1, scoord2, scoord3, ecoord1, ecoord2, ecoord3):
#        starts = (
#            list(traverse(scoord1, scoord2, inc_end=False)) +
#            list(traverse(scoord2, scoord3, inc_end=True)))
#        ends = (
#            list(traverse(ecoord1, ecoord2, inc_end=False)) +
#            list(traverse(ecoord2, ecoord3, inc_end=True)))
#        return [
#            list(traverse(start, end, inc_end=True))
#            for start, end in zip(starts, ends)
#        ]
#
#    return {
#        'p': do_axis(delta, epsilon, zeta, gamma, beta, alpha),
#        'q': do_axis(zeta, alpha, beta, epsilon, delta, gamma),
#        'r': do_axis(beta, gamma, delta, alpha, zeta, epsilon),
#    }


#
# ============================================================
# Modeling solutions
# ============================================================
#


@dataclasses.dataclass
class PuzzleSolution:
    model:      dict[str, str]              # var_name => str(model[var])
    cell_ords:  dict[Tuple[int, ...], int]  # coords => ord(character)
    cell_chars: dict[Tuple[int, ...], str]  # coords => character


@dataclasses.dataclass
class SolvedPuzzle(PuzzleSolution):
    z3_time: float
    z3_input: List[str]

    unique: bool
    alt_solution: Optional[Union[PuzzleSolution, str]]

    @property
    def is_solved(self) -> bool:
        return True


@dataclasses.dataclass
class UnsolvedPuzzle():
    z3_time: float
    z3_input: List[str]
    z3_result: str

    @property
    def is_solved(self) -> bool:
        return False


@dataclasses.dataclass
class PuzzleResult():
    puzzle: Puzzle
    solution: Union[SolvedPuzzle, UnsolvedPuzzle]
    verified: bool
    verification: Optional[list[Tuple[str, str, bool]]]


#
# ============================================================
# Bridging to/from Z3
# ============================================================
#


CellMap = dict[Tuple[int, ...], z3.ExprRef]


def puzzle_to_constraints(
        puzzle: Puzzle,
        symbolifier: RegexSymbolifier,
) -> Tuple[CellMap, List[z3.ExprRef]]:
    cell_map = {
        cell: z3.Int('c_' + '_'.join(str(coord) for coord in cell))
            for clue in puzzle.iter_clues()
                for cell in clue.cells
    }
    
    all_constraints = [
        sb_to_z3(symbolifier.query(clue.regex, len(clue.cells)), clue.cells, cell_map)
        for clue in puzzle.iter_clues()
    ]
    
    return (cell_map, all_constraints)


def sb_to_z3(
        sb: SBBool,
        clue_cells: list[Tuple[int, ...]],
        cell_map: CellMap,
) -> z3.ExprRef:
    def extract_conjuncts(sb: SBBool) -> Iterable[SBBool]:
        match sb:
            case SBAnd(a, b):
                yield from extract_conjuncts(a)
                yield from extract_conjuncts(b)
            case _:
                yield sb
                
    def extract_disjuncts(sb: SBBool) -> Iterable[SBBool]:
        match sb:
            case SBOr(a, b):
                yield from extract_disjuncts(a)
                yield from extract_disjuncts(b)
            case _:
                yield sb
    
    def map0(sb: SBBool) -> z3.ExprRef:
        match sb:
            case SBTrue(): return z3.BoolVal(True)
            case SBFalse(): return z3.BoolVal(False)
            case SBAnd(a, b): return z3.And(*(map(map0, extract_conjuncts(sb))))
            case SBOr(a, b): return z3.Or(*(map(map0, extract_disjuncts(sb))))
            case SBNot(a): return z3.Not(map0(a))
            case SBCond(c, ifT, ifF): return z3.If(map0(c), map0(ifT), map0(ifF))
            
            case SBCharIn(idx, charlist):
                return z3.Or(*(cell_map[clue_cells[idx]] == ord(ch) for ch in charlist))
            case SBCharEq(idx1, idx2):
                return cell_map[clue_cells[idx1]] == cell_map[clue_cells[idx2]]
            
            case _:
                raise InternalError()
                
    return map0(sb)


def interpret_model(
        cell_map: dict[Tuple[int, ...], z3.ExprRef],
        model: z3.ModelRef,
) -> PuzzleSolution:
    return PuzzleSolution(
        model={
            str(d.name()): str(model[d])
            for d in model.decls()
        },
        cell_ords={
            cell: model[var].as_long()
            for cell, var in cell_map.items()
        },
        cell_chars={
            cell: chr(model[var].as_long())
            for cell, var in cell_map.items()
        },
    )


def solve_constraints(
        cell_map: dict[Tuple[int, ...], z3.ExprRef],
        constraints: List[z3.ExprRef],
) -> Union[SolvedPuzzle, UnsolvedPuzzle]:
    s = z3.Solver()
    s.add(z3.And(*(var != ord('\t') for var in cell_map.values())))
    s.add(z3.And(*constraints))
    t_begin = time.time()
    is_sat = s.check()
    if is_sat == z3.sat:
        model = s.model()
        t_end = time.time()
        r1 = interpret_model(cell_map, model)
        
        s.add(z3.Or(*(var != model[var].as_long() for var in cell_map.values())))
        is_sat = s.check()
        if is_sat == z3.unsat:
            alt_solution = None
        else:
            if is_sat == z3.sat:
                alt_solution = interpret_model(cell_map, s.model())
            else:
                alt_solution = str(is_sat)

        return SolvedPuzzle(
            model=r1.model,
            cell_ords=r1.cell_ords,
            cell_chars=r1.cell_chars,
            z3_input=list(map(str, constraints)),
            z3_time=(t_end - t_begin),
            unique=(alt_solution is None),
            alt_solution=alt_solution,
        )
    else:
        t_end = time.time()
        return UnsolvedPuzzle(
            z3_time=(t_end - t_begin),
            z3_input=list(map(str, constraints)),
            z3_result=str(is_sat),
        )


def solve_puzzle(
        puzzle: Puzzle,
        symbolifier: RegexSymbolifier,
) -> PuzzleResult:
    cell_map, constraints = puzzle_to_constraints(puzzle, symbolifier)
    solve_result = solve_constraints(cell_map, constraints)
    
    if solve_result.is_solved:
        assert isinstance(solve_result, SolvedPuzzle)
        solved_clues = []
        for clue in puzzle.iter_clues():
            regex, cells = clue.regex, clue.cells
            string = "".join(solve_result.cell_chars[c] for c in cells)
            matched = bool(re.fullmatch(regex, string))
            solved_clues.append((regex, string, matched))

        verified = all(matched for _, _, matched in solved_clues)

        return PuzzleResult(
            puzzle=puzzle,
            solution=solve_result,
            verified=verified,
            verification=solved_clues,
        )
    else:
        return PuzzleResult(
            puzzle=puzzle,
            solution=solve_result,
            verified=False,
            verification=None,
        )


#
# I can has driver?
#


def load_puzzles_from_file(
        path: str,
        only_groups: set[str] = set(),
        only_puzzles: set[str] = set(),
) -> list[Group]:
    with open(path, 'r') as fh:
        obj = json.load(fh)
        groups = decode_root(obj)

    if only_puzzles:
        existing_puzzles = set(p.id for g in groups for p in g.puzzles)
        if extra_puzzles := only_puzzles - existing_puzzles:
            raise UserError(
                    "only-puzzles was specified with puzzle IDs %r"
                    " which do not exist"
                    % (list(extra_puzzles), ))

        for index, group in enumerate(groups):
            groups[index] = dataclasses.replace(group,
                puzzles=[p for p in group.puzzles if p.id in only_puzzles],
            )

    if only_groups:
        known_groups = set(g.id for g in groups)
        if (extra_groups := only_groups - known_groups):
            raise UserError(
                    "only_groups was specified including groups %r"
                    " which do not exist"
                    % (list(extra_groups)))
        groups = [g for g in groups if g.id in only_groups]

    if only_groups or only_puzzles:
        groups = [g for g in groups if g.puzzles]

    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file',
        type=str,
        required=True,
        metavar='PATH',
    )
    ap.add_argument('--debug',
        action='store_true',
        help='Drop into python debugger on various failures',
    )

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--inspect',
        action='store_true',
        help='Print a human-readable representation of parsed puzzles',
        default=False,
    )
    grp.add_argument('--dump-regexes',
        action='store_true',
        help='Print regexes in "expected-length regex-string"-line format',
        default=False,
    )
    grp.add_argument('--generate-constraints',
        action='store_true',
        help='Parse regexes and print constraints',
        default=False,
    )
    grp.add_argument('--dump-solutions',
        action='store_true',
        help='Compute and output solutions in a detailed format',
        default=False,
    )

    ap.add_argument('--only-groups',
        action='append',
        help='Limit to groups with the specified ID (e.g. "tutorial"). Maybe be specified more than once.',
        metavar='ID',
    )
    ap.add_argument('--only-puzzles',
        action='append',
        help='Limit to only puzzles with the specified ID (a UUID). Maybe be specified more than once.',
        metavar='ID',
    )

    ns = ap.parse_args()

    if ns.debug:
        global DEBUG
        DEBUG = True

    groups = load_puzzles_from_file(
        path=ns.input_file,
        only_groups=(set(ns.only_groups) if ns.only_groups else set()),
        only_puzzles=(set(ns.only_puzzles) if ns.only_puzzles else set()),
    )

    # do stuff

    if ns.inspect:
        pprint.pp(groups)

    elif ns.dump_regexes:
        for group in groups:
            for puz in group.puzzles:
                for clue in puz.iter_clues():
                    pat, strlen = clue.regex, len(clue.cells)
                    check(" " not in pat, lambda: InternalError("implementation limit"))
                    check("\n" not in pat, lambda: InternalError("implementation limit"))
                    print("%d %s" % (strlen, pat))

    elif ns.generate_constraints:
        symbolifier = build_symbolifier_for(
            puz for group in groups for puz in group.puzzles
        )

        for group in groups:
            for puz in group.puzzles:
                for clue in puz.iter_clues():
                    pat, strlen = clue.regex, len(clue.cells)
                    constraints = symbolifier.query(pat, strlen)
                    print("%d %s %r" % (strlen, pat, constraints))

    elif ns.dump_solutions:
        symbolifier = build_symbolifier_for(
            puz for group in groups for puz in group.puzzles
        )

        n_solved = 0
        n_failed = 0

        tbegin = time.time()

        for group in groups:
            print()
            print()
            print("Group: %s" % (group.name, ))
            
            for puzzle in group.puzzles:
                print()
                puzresult = solve_puzzle(puzzle, symbolifier)
                solution = puzresult.solution
                if isinstance(solution, SolvedPuzzle) and solution.unique and puzresult.verified:
                    n_solved += 1
                else:
                    n_failed += 1
                pprint.pp(puzresult, width=100)

        tend = time.time()
                
        print()
        print()
        print("Summary: solved %d, failed to solve %d, total %d puzzles. Wall time: %f seconds"
                % (n_solved, n_failed, n_solved + n_failed, tend - tbegin))

    else:
        fail(InternalError)


if __name__ == '__main__':
    main()

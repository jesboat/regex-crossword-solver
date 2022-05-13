# Regex Crossword Solver


## Regex Crosswords?

This is a solver for Regex Crossword puzzles, as found on
[regexcrossword.com](https://regexcrossword.com/). Each puzzle consists of a
grid of unknowns and a series of clues; each clue is a regular expression,
which should match the letters in (specific) parts of the grid.

For example, a simple puzzle might be:

```
                /A./    /CY|DZ/
    /[AB]C/     ?1      ?2
    /X[YZ]/     ?3      ?4
```

which requires

```
    /[AB]C/     =~ 12
    /X[YZ]/     =~ 34
    /A./        =~ 13
    /CY|DZ/     =~ 24
```

The (unique) solution to the puzzle is:

```
                /A./    /CY|DZ/
    /[AB]C/     A       C
    /X[YZ]/     X       Y
```

Which this solver promptly finds:

```python
PuzzleResult(
    puzzle=RectangularPuzzle(
        id='9ddc357b-c925-4d47-8889-4b0f10ee46de',
        name='The first example in the documentation',
        patternsX=[('A.', None), ('CY|DZ', None)],
        patternsY=[('[AB]C', None), ('X[YZ]', None)]),
    solution=SolvedPuzzle(
        z3_input=['Or(c_0_0 == 65)',
                  'Or(And(Or(c_0_1 == 67), Or(c_1_1 == 89)),\n'
                  '   And(Or(c_0_1 == 68), Or(c_1_1 == 90)))',
                  'And(Or(c_0_0 == 65, c_0_0 == 66), Or(c_0_1 == 67))',
                  'And(Or(c_1_0 == 88), Or(c_1_1 == 89, c_1_1 == 90))'],
        z3_time=0.0026848316192626953,
        model={'c_1_0': '88',
               'c_0_1': '67',
               'c_1_1': '89',
               'c_0_0': '65'},
        cell_ords={(0, 0): 65,
                   (1, 0): 88,
                   (0, 1): 67,
                   (1, 1): 89},
        cell_chars={(0, 0): 'A',
                    (1, 0): 'X',
                    (0, 1): 'C',
                    (1, 1): 'Y'},
        unique=True,
        alt_solution=None),
    verified=True,
    verification=[('A.',    'AX', True),
                  ('CY|DZ', 'CY', True),
                  ('[AB]C', 'AC', True),
                  ('X[YZ]', 'XY', True)])
```

Or, on a real-world puzzle:

```python
PuzzleResult(
    puzzle=HexagonalPuzzle(
        id='3d870dd4-9c37-4e82-bbce-033d38acf36f',
        name='Buzz buzz',
        patternsP=['(VE|IL?|\\sS)*[DIES]?',
                   '[HELP]*(Y|QU|I)?'],
        patternsQ=['[HAIR]*(D|O)*',
                   '(\\W|\\w)?[CLVST3R]*',
                   '[HAV3N]?[REA]*'],
        patternsR=['[EVA]?[MASONIC\\s]?',
                   '[STARV3]*(O|H)?']),
    solution=SolvedPuzzle(
        z3_input=['Or(And(Or(c_0_1_1 == 86), Or(c_0_2_0 == 69)),\n'
                  '   And(Or(c_0_1_1 == 73),\n'
                  '       Or(Or(c_0_2_0 == 76),\n'
                  '          Or(c_0_2_0 == 73),\n'
                  '          Or(c_0_2_0 == 68,\n'
                  ... 212 total lines ...
                  '       Or(c_1_0_1 == 79),\n'
                  '       Or(c_1_0_1 == 72)))'],
        z3_time=0.005857706069946289,
        model={'c_0_1_1': '86',
               'c_1_1_0': '73',
               'c_0_2_0': '69',
               'c_1_0_1': '72'},
        cell_ords={(0, 1, 1): 86,
                   (0, 2, 0): 69,
                   (1, 0, 1): 72,
                   (1, 1, 0): 73},
        cell_chars={(0, 1, 1): 'V',
                    (0, 2, 0): 'E',
                    (1, 0, 1): 'H',
                    (1, 1, 0): 'I'},
        unique=True,
        alt_solution=None),
    verified=True,
    verification=[('(VE|IL?|\\sS)*[DIES]?', 'VE', True),
                  ('[HELP]*(Y|QU|I)?',      'HI', True),
                  ('[HAIR]*(D|O)*',         'H',  True),
                  ('(\\W|\\w)?[CLVST3R]*',  'IV', True),
                  ('[HAV3N]?[REA]*',        'E',  True),
                  ('[EVA]?[MASONIC\\s]?',   'EI', True),
                  ('[STARV3]*(O|H)?',       'VH', True)])
```

## Usage

First, install dependencies. The project uses both Python and Haskell:

- Python 3.10 or newer is required. We require `z3-solver` at runtime and
  Pyre as an optional typechecker. Running `./util.sh make_venv` will
  install both of them in a venv.

- The Haskell side can be built using [Stack](https://haskellstack.org/).

    1. Running `stack build` will download/build GHC and the Haskell
       dependencies fully automatically.

    2. If you prefer to use the system's GHC (ignoring the version check),
       point the `STACK_YAML` environment variable at `stack-sysghc.yaml`, e.g.
       with `STACK_YAML="$PWD/stack-sysghc.yaml" stack build`. Set
       `STACK_YAML` in the same way when running the solver.

    3. Alternatively, if you don't have Stack available, the code will fall
       back to using `runhaskell` on your `PATH`. You're responsible for
       ensuring that dependencies are installed.

Second, download the challenges file from
`https://regexcrossword.com/data/challenges.json`

You can run the solver like so:

```sh
    ./util.sh run \
                --input-file live/challenges.json \
                --dump-solutions \
            > solutions.txt
```

The solutions are in an unspecified human-readable format. We verify that
(i) each puzzle has a solution, according to the constraints we generate;
(ii) the solution is unique, according to the constraints we generate; and
(iii) each regex matches the corresponding part of the puzzle (according to
Python's `re` module.)


## Caveats

The official Regex Crossword implementation validates solutions by checking
the (proposed) solution against the regexes using the native browser's
JavaScript implementation. This project implements a decent chunk of
JavaScript-compatible regexes (most of E262-5.1, with a bit of ES12 thrown
in), including enough to correctly interpret every regex actually used in
the online puzzles (at time of writing.) The code is intended to be largely
fail-safe, in that any regex which contains features we don't handle should
result in an error. There are some known differences (see comments in
`Regex.hs`), but we do not expect to hit them in practice.


## Implementation

Python code validates the JSON and builds a model of each puzzle, treating the
regexes as opaque. Then, Haskell code converts each regex (along with the
length of the string it's intended to match) into a boolean formula over
two primitives, "the character at position `i` must be in `{c1, c2, ... cn}`"
and "the characters at positions `i1` and `i2` must be equal". The result gets
passed back to Python, which tosses all formulae for a single puzzle's clues
into Z3, which produces the solution. The solution is then printed by Python.


### Stage 1: interpreting puzzles

The solver's input is the (undocumented) JSON format used by
`regexcrossword.com` at time of writing. We parse it into a typed model.
The following example shows three puzzles, each from a separate group:

```
$ ./util.sh run \
            --input-file live/challenges.json \
            --inspect \
            --only-puzzle 272901bb-0855-4157-9b45-272935da8c93 \
            --only-puzzle f0f06b00-ec0a-4572-935d-7459e2a13064 \
            --only-puzzle 3d870dd4-9c37-4e82-bbce-033d38acf36f

[Group(id='tutorial',
       name='Tutorial',
       puzzles=[RectangularPuzzle(id='272901bb-0855-4157-9b45-272935da8c93',
                                  name='The OR symbol',
                                  patternsX=[('A|B', None)],
                                  patternsY=[('A|Z', None)])]),
 Group(id='doublecross',
       name='Double Cross',
       puzzles=[RectangularPuzzle(id='f0f06b00-ec0a-4572-935d-7459e2a13064',
                                  name='Telekinesis',
                                  patternsX=[('[D-HJ-M]', '[^F-KM-Z]'),
                                             ('[^A-RU-Z]', '[A-KS-V]')],
                                  patternsY=[('[A-GN-Z]+', '[^A-DI-S]+')])]),
 Group(id='hexagonal',
       name='Hexagonal',
       puzzles=[HexagonalPuzzle(id='3d870dd4-9c37-4e82-bbce-033d38acf36f',
                                name='Buzz buzz',
                                patternsP=['(VE|IL?|\\sS)*[DIES]?',
                                           '[HELP]*(Y|QU|I)?'],
                                patternsQ=['[HAIR]*(D|O)*',
                                           '(\\W|\\w)?[CLVST3R]*',
                                           '[HAV3N]?[REA]*'],
                                patternsR=['[EVA]?[MASONIC\\s]?',
                                           '[STARV3]*(O|H)?'])])]
$
```

Most puzzles have a rectangular grid, and are represented as instances of
RectangularPuzzle. The naming convention matches the `challenges.json` file:

```
    patternsX[x] = (clueT, clueB)
    patternsY[y] = (clueL, clueR)

    clueT should match [(x, 0) .. (x, maxY)]
    clueB should match [(x, 0) .. (x, maxY)]
    clueL should match [(0, y) .. (maxX, y)]
    clueR should match [(0, y) .. (maxX, y)]
```

where

- `clueT` is printed on the top of the puzzle, applies to one column (read from top to bottom), and is always present
- `clueB` is printed on the bottom of the puzzle, applies to one column (read from top to bottom), and may be `None`
- `clueL` is printed on the left of the puzzle, applies to one row (read from left to right), and is always present
- `clueR` is printed on the right of the puzzle, applies to one row (read from left to right), and may be `None`

Some puzzles are a hexagonal segment of a hexagonal grid.  The representation
and algorithm for handling hexagonal puzzles were suggested by
[Justin Pombrio](https://justinpombrio.net/).

<details>
<summary>Hexagon details</summary>
<p>Want an overly formal definition of "segment of a hexagonal grid?" Formally:
fix `T` as some tiling of the plane with regular hexagons. Let `H` be some
hexagon on the plane such that (a) `H` is equiangular, (b) each vertex of `H`
is the center of one tile of `T`, and (c) each side of `H` is parallel to one
axis of `T`. The puzzle is the subset of tiles of `T` which overlap the
interior of `H`.</p>
</details>


### Stage 2: parsing regexes


Python is not particularly well-suited for nontrivial parsing, ASTs, or
borderline compiler work, so we switch to Haskell.

Regexes are parsed with Parsec with a parser loosely inspired by the grammar in
ES3 and ES5.1. We do not aim to implement a perfect parser for JavaScript
regexes, but we do parse every regex used in the crosswords and, in most cases,
if the parser accepts a regex, then the regex is a valid ES3/ES5.1/ES12 regex
for which the parser constructed a semantically correct representation.

The parser imposes the following restrictions relative to ES3/ES5:

- Word boundary assertions (`/\b/` and `/\B/`) are unsupported and rejected
- Lookaheads (both positive and negative) are unsupported and rejected
- Characters which can occur syntactically unescaped are more limited
- We reject certain insane character class constructs like `/[--]/`

We support two constructs which are technically forbidden by ES3 and ES5, but
used in the puzzles, widely supported in practice, and permitted by ES12:

- Octal escape sequences are supported in character classes
- We allow `/\$/` to match a literal dollar-sign

There are three final quirks:

- Our implementation operates on Unicode strings (i.e. sequences of Unicode
  code points), instead of JavaScript's traditional strings (i.e. sequences
  of UTF-16 code units)
- We treat character classes `\w`, `\s`, and `\d` as only matching ASCII
  characters
- We treat `.` as able to match any character (whereas JS excludes newlines)


### Stage 3: processing regexes


Regexes are first parsed to an AST relatively close to the surface syntax
(although we do some normalization during parse time). We desugar to a simpler
form, and do some very simple simplification.

For example, the regex `(U|O|I)*T[FRO]+` parses initially to

```haskell
    SConcat
      (SConcat
        (SRepeated SRepeatStar
          (SCapture ()
            (SAlternative (SAlternative (SLiteral 'U')
                                        (SLiteral 'O'))
                          (SLiteral 'I'))))
        (SLiteral 'T'))
      (SRepeated SRepeatPlus
        (SCharClass {negated = False, ranges = [
                SCharClassSingle 'F',SCharClassSingle 'R',SCharClassSingle 'O']}))
```

but gets desugared (approximately) to

```haskell
    DConcat [
        DRepeat 0 Nothing (
            DChoice [
                DChar (MSetPos "U"),
                DChar (MSetPos "O"),
                DChar (MSetPos "I")]),
        DChar (MSetPos "T"),
        DRepeat 1 Nothing (DChar (MSetPos "FRO"))]
```

(Note that the representation of qualifiers changed, the unreferenced capture
group was inlined, and the associative operators `Choice` and `Concat` are no
longer binary.)

In principle, we could do more complex analyses on the regexes to try to
simplify them further. In practice, the constraints we generate are already
simple enough for Z3 to solve quickly (0.15 seconds worst-case to find a
solution; average 0.25 seconds to find solution and verify uniqueness), and the
increasingly complex code wouldn't increase the utility of the project.

We implemented a matcher based largely on the formal algorithm which defines
the semantics of JavaScript regexes. It was surprisingly simple to write the
code in a way which was fully abstract over any operations which were directly
or indirectly on the input string's characters (although not its length.)

As an example, consider the code for matching a single character against a
character set.

The concrete code would be:

```haskell
    if   charIn (inputString !! currentPos) charSet
    then restOfRegexMatcher (matchState { currentPos = currentPos + 1 })
    else mzero
```

The abstracted code is:

```haskell
    abstractConditional
        (abstractCharIn input currentPos charSet))
        (restOfRegexMatcher (matchState { currentPos = currentPos + 1 }))
        abstractFail
```

Instantiating the matching code with definitions like

```haskell
    abstractConditional c t f =
        if c then t else f
    abstractCharIn string index charset =
        charIn (string !! index) charset
    abstractFail =
        mzero
```

yields a concrete matcher; instantiating it with definitions like

```haskell
    abstractCharIn c t f =
        SymbolicConditional c t f
    abstractCharIn () index charset =
        SymbolicCharInCharset index charset
    abstractFail =
        SymbolicFalse
```

yields a matcher whose result is a symbolic representation of the conditions
necessary to match a string of the specified length.

<details>
<summary>Detailed match code example</summary>

The example above is slightly simplified. We ignore three main factors: (a) the
actual code uses different variable names, largely to match the variable names
used in the ES3/ES5 spec; (b) the match code supports both finite charsets and
infinite charsets, which are represented as the negation of a finite set; (c)
the ES3/ES5 does does Unicode canonicalization, which we do not. The relevant
chunk of the spec (abstract operation `CharacterSetMatcher`, as in E262 5.1
section 15.10.2.8) reads:

> 1. Let `e` be `x`'s endIndex.
> 2. If `e == InputLength`, return `failure`.
> 3. Let `ch` be the character `Input[e]`.
> 4. Let `cc` be the result of `Canonicalize(ch)`.
> 5. If `invert` is false, then
>     1) If there does not exist a member `a` of set `A` such that `Canonicalize(a) == cc`, return `failure`.
> 6. Else `invert` is true,
>     1) If there exists a member `a` of set `A` such that `Canonicalize(a) == cc`, return `failure`.
> 7. Let `cap` be `x`'s captures internal array.
> 8. Let `y` be the State `(e+1, cap)`.
> 9. Call `c(y)` and return its result.

The Haskell code, written in the concrete style, is:

```haskell
let MatchState {msEndIndex, msInput, msStrlen} = x in
if msEndIndex < msStrlen
then if   ((if neg then not else id) $ elem (msInput !! msEndIndex) chars)
     then (c (x { msEndIndex = msEndIndex + 1 }))
     else mzero
else mzero
```

and, once abstracted, is:

```haskell
let MatchState {msEndIndex, msInput, msStrlen} = x in
if msEndIndex < msStrlen
then
    amrCond ((if neg then ambNot else id) $ amiCharIn msInput msEndIndex chars)
            (c (x { msEndIndex = msEndIndex + 1 }))
            amrFail
else amrFail
```

---
</details>

The symbolic representation is isomorphic to the type

```haskell
    data Symbolic =
            SymbolicTrue
        |   SymbolicFalse
        |   SymbolicAnd Symbolic Symbolic
        |   SymbolicOr  Symbolic Symbolic
        |   SymbolicNot Symbolic
        |   SymbolicConditional  { cond :: Symbolic, ifT :: Symbolic, ifF :: Symbolic }
        |   SymbolicCharInSet    { indexInInputString :: Int, chars :: [Char] }
        |   SymbolicCharsEqual   { indexInInputString1 :: Int, indexInInputString2 :: Int }
```

We do some trivial simplifications (which only really serve to make the
intermediate form more human-readable).

#### Examples

Here are some examples. Each shows a regex and target length, the surface-level
parse, a (slightly simplified) version of its desugared and simplified form,
and a (slightly simplified) version of the conditions we compute for the regex
to match against a string of the specified length.

```haskell
-- 2 A?B?
SConcat (SRepeated SRepeatQuestion (SLiteral 'A'))
        (SRepeated SRepeatQuestion (SLiteral 'B'))
DConcat [
    DRepeat 0 (Just 1) (DChar (MSetPos "A")),
    DRepeat 0 (Just 1) (DChar (MSetPos "B"))]
SBAnd (SBCharIn 0 "A") (SBCharIn 1 "B")


-- 2 (A)\1
SConcat (SCapture () (SLiteral 'A'))
        (SBackReference 1)
DConcat [
    DCapture 1 (DChar (MSetPos "A")),
    DBackReference 1]
SBAnd (SBCharIn 0 "A") (SBCharEq 1 0)


-- 2 .*M?O.*
SConcat (SConcat (SConcat (SRepeated SRepeatStar SAnyChar)
                          (SRepeated SRepeatQuestion (SLiteral 'M')))
                 (SLiteral 'O'))
        (SRepeated SRepeatStar SAnyChar)
DConcat [
    DRepeat 0 Nothing (DChar (MSetNeg "")),
    DRepeat 0 (Just 1) (DChar (MSetPos "M")),
    DChar (MSetPos "O"),
    DRepeat 0 Nothing (DChar (MSetNeg ""))]
SBOr (SBCharIn 1 "O")
     (SBOr (SBAnd (SBCharIn 0 "M") (SBCharIn 1 "O"))
           (SBCharIn 0 "O"))


-- 3 RY|TY\-
SAlternative (SConcat (SLiteral 'R') (SLiteral 'Y'))
             (SConcat (SConcat (SLiteral 'T') (SLiteral 'Y')) (SLiteral '-'))
DChoice [
    DConcat [
        DChar (MSetPos "R"),
        DChar (MSetPos "Y")]
    DConcat [
        DChar (MSetPos "T"),
        DChar (MSetPos "Y"),
        DChar (MSetPos "-")]]
SBAnd (SBCharIn 0 "T")
      (SBAnd (SBCharIn 1 "Y")
             (SBCharIn 2 "-"))


-- 5 (.).*\1N\1
SConcat (SConcat (SConcat (SConcat (SCapture () SAnyChar)
                                   (SRepeated SRepeatStar SAnyChar))
                          (SBackReference 1))
                 (SLiteral 'N'))
        (SBackReference 1)
DConcat [
    DCapture 1 (DChar (MSetNeg "")),
    DRepeat 0 Nothing (DChar (MSetNeg "")),
    DBackReference 1,
    DChar (MSetPos "N"),
    DBackReference 1]
SBAnd (SBCharEq 2 0)
      (SBAnd (SBCharIn 3 "N")
             (SBCharEq 4 0))


5 (SOD|DO|GE)*
SRepeated SRepeatStar (SCapture ()
    (SAlternative (SAlternative
                        (SConcat (SConcat (SLiteral 'S') (SLiteral 'O')) (SLiteral 'D'))
                        (SConcat (SLiteral 'D') (SLiteral 'O')))
                  (SConcat (SLiteral 'G') (SLiteral 'E'))))
DRepeat 0 Nothing (DChoice [
    DConcat [
        DChar (MSetPos "S"), DChar (MSetPos "O"), DChar (MSetPos "D")],
    DConcat [
        DChar (MSetPos "D"), DChar (MSetPos "O")],
    DConcat [
        DChar (MSetPos "G"), DChar (MSetPos "E")]])
SBOr (SBAnd
        (SBAnd (SBCharIn 0 "S") (SBAnd (SBCharIn 1 "O") (SBCharIn 2 "D")))
        (SBOr (SBAnd (SBCharIn 3 "D") (SBCharIn 4 "O"))
              (SBAnd (SBCharIn 3 "G") (SBCharIn 4 "E"))))
     (SBOr
        (SBAnd (SBCharIn 0 "D")
               (SBAnd (SBCharIn 1 "O")
                      (SBAnd (SBCharIn 2 "S")
                             (SBAnd (SBCharIn 3 "O") (SBCharIn 4 "D")))))
        (SBAnd (SBCharIn 0 "G")
               (SBAnd (SBCharIn 1 "E")
                      (SBAnd (SBCharIn 2 "S")
                             (SBAnd (SBCharIn 3 "O") (SBCharIn 4 "D"))))))
```

An example where this approach is much less elegant is:

```haskell
-- 8 (SR|LF|C.)+[END]+
SConcat (SRepeated SRepeatPlus
            (SCapture ()
                (SAlternative (SAlternative (SConcat (SLiteral 'S') (SLiteral 'R'))
                                            (SConcat (SLiteral 'L') (SLiteral 'F')))
                              (SConcat (SLiteral 'C') SAnyChar))))
        (SRepeated SRepeatPlus
            (SCharClass {negated = False, ranges = [
                SCharClassSingle 'E',SCharClassSingle 'N',SCharClassSingle 'D']}))
DConcat [
    DRepeat 1 Nothing (DChoice [
        DConcat [DChar (MSetPos "S"),DChar (MSetPos "R")],
        DConcat [DChar (MSetPos "L"),DChar (MSetPos "F")],
        DConcat [DChar (MSetPos "C"),DChar (MSetNeg "")]]),
    DRepeat 1 Nothing (DChar (MSetPos "END"))]
```

where the generated constraints (when printed) more than fill a screen.

### Stage 4: solving with Z3


At this point, we switch back to Python to make use of its (remarkably fluent)
API for Z3.

Each unknown in the puzzle gets a Z3 variable of sort `Int` (we represent
characters by their `ord()`). Translating a symbolic constraint to a Z3
formula is essentially trivial. If `z3vars` is an array of the variables
corresponding to the unknowns which form the string the current regex is
supposed to match, then the translation is:

```python
    SymbolicTrue        =>  True
    SymbolicFalse       =>  False
    SymbolicAnd a b     =>  And(translate(a), translate(b))
    SymbolicOr a b      =>  Or(translate(a), translate(b))
    SymbolicNot a       =>  Not(translate(a))
    SymbolicCond c t e  =>  If(translate(c), translate(t), translate(e))

    SymbolicCharInSeq idx chars
        => Or(*(z3vars[idx] == ord(ch) for ch in chars))
    SymbolicCharsEqual idx1 idx2
        => (z3vars[idx1] == z3vars[idx2])
```

We dump all the constraints for a puzzle together, and let Z3 solve. Once Z3
provides us with a solution, we can verify the uniqueness by confirming that
adding one additional constraint to reject exactly the first solution makes
the system unsatisfiable.


### Finishing up


We can audit the solution by checking whether each regex matches the string
we fund with Python's `re` module. This is not technically correct, because
JS and Python can interpret regexes differently. (Even among the subset we
support, Python will treat `/^(a?)*b\1$/` as matching "ab", while JS won't.)
But, in practice, no puzzles appear to have regexes whose behavior does
differ between JS and Python.

The output format is not machine-readable, and also not a fancy rendered grid
which matches the layout of the original puzzle. The solver was written for
fun, not as a way to enable a human to solve puzzles which they would not
otherwise be able to solve by hand.


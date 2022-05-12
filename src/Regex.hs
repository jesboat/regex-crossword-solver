-- Syntactic sugar for records
{-# LANGUAGE NamedFieldPuns, DuplicateRecordFields, RecordWildCards #-}
-- Some other sugar
{-# LANGUAGE ViewPatterns, MultiWayIf, LambdaCase #-}
-- Fancy stuff to make the abstract matching code nice
{-# LANGUAGE TypeFamilies, ImplicitParams, ConstrainedClassMethods #-}


{-# OPTIONS_GHC
        -Wall -Wextra
        -Wno-name-shadowing
        -Wno-unused-do-bind
        -Wno-unused-matches
        -Wno-type-defaults
        -Wno-unused-top-binds
    #-}

module Regex where

import Control.Monad (liftM, liftM2, mplus, when)
import Control.Monad.Trans.State.Lazy (evalState, get, modify)
import Data.Char (chr, toUpper, ord, intToDigit, digitToInt)
import Data.Function ((&))
import Data.Functor ((<&>), ($>))
import Data.List (sort, union, intersect, (\\), group, intercalate)
import System.Environment (getArgs, getProgName)
import System.IO (stderr, stdout, hPutStrLn)

import Text.ParserCombinators.Parsec


-- ----------------------------------------
-- Some miscelaneous stuff
-- ----------------------------------------


data Phantom t = Phantom


numToHex :: Int -> String
numToHex n | n < 0   = undefined
           | n == 0  = "0"
           | True    = loop n
    where
        loop 0 = ""
        loop ((`quotRem` 16) -> (q, r)) = loop q ++ [toUpper $ intToDigit r]


hexToNum :: String -> Int
hexToNum [] = undefined
hexToNum str = foldl (\n c -> n * 16 + digitToInt c) 0 str


leftpad :: Int -> a -> [a] -> [a]
leftpad goal pad lst@(length -> len) | len >= goal  = lst
                                     | otherwise    = (take (goal - len) (repeat pad)) ++ lst


charToEscapedHex :: Char -> String
charToEscapedHex (ord -> o) | o < 256   = "\\x" ++ leftpad 2 '0' (numToHex o)
                            | otherwise = "\\u" ++ leftpad 4 '0' (numToHex o)


orElse :: Maybe a -> a -> a
(Just a) `orElse` _ = a
Nothing `orElse` a = a


-- Like `nub`, but (i) also sorts the list, (ii) requires Ord and not just
-- Eq, and (iii) isn't O(n^2).
dedup :: (Eq a, Ord a) => [a] -> [a]
dedup = map head . group . sort


-- MSet is a set in the mathematical sense, which means it might be infinite.
-- In particular, we support sets which are either finite or cofinite.
data MSet t = MSetPos [t] | MSetNeg [t]
        deriving (Eq, Read, Show)

type CharSet = MSet Char

msetEmpty :: MSet e
msetEmpty = MSetPos []

msetUniversal :: MSet e
msetUniversal = MSetNeg []

msetUnion :: (Eq e, Ord e) => MSet e -> MSet e -> MSet e
msetUnion (MSetPos as) (MSetPos bs) = MSetPos (as `union` bs)
msetUnion (MSetNeg as) (MSetNeg bs) = MSetNeg (as `intersect` bs)
msetUnion (MSetPos as) (MSetNeg bs) = MSetNeg (bs \\ as)
msetUnion (MSetNeg as) (MSetPos bs) = MSetNeg (as \\ bs)

msetInverse :: MSet e -> MSet e
msetInverse (MSetPos cs) = MSetNeg cs
msetInverse (MSetNeg cs) = MSetPos cs

msetIntersect :: (Eq e, Ord e) => MSet e -> MSet e -> MSet e
msetIntersect as bs = msetInverse (msetInverse as `msetUnion` msetInverse bs)

msetFromList :: (Eq e, Ord e) => [e] -> MSet e
msetFromList = MSetPos . dedup

msetSingleton :: e -> MSet e
msetSingleton = MSetPos . (:[])


-- ----------------------------------------
-- Surface syntax for regexes
-- ----------------------------------------


data SCharType =
        SCharTypeSpace
    |   SCharTypeWord
    |   SCharTypeDigit
    deriving (Eq, Read, Show)

data SCharClassElement =
        SCharClassSingle Char
    |   SCharClassRange Char Char
    |   SCharClassCharType { negated :: Bool, charType :: SCharType }
    deriving (Eq, Read, Show)

data SRepeatQualifier =
        SRepeatStar
    |   SRepeatPlus
    |   SRepeatQuestion
    |   SRepeatCount { repeatCountMin :: Maybe Int, repeatCountMax :: Maybe Int}
    deriving (Eq, Read, Show)

data SRegex ann =
        SLiteral Char
    |   SCharClass { negated :: Bool, ranges :: [SCharClassElement] }
    |   SBackReference Int
    |   SCharType { negated :: Bool, charType :: SCharType }
    |   SAnyChar
    |   SEmpty
            -- either: entire regex is empty; top level is an `|`
            -- with (at least) one empty alternative; some capture
            -- group is empty; some capture group has an empty
            -- alternative
    |   SBeginString
    |   SEndString
    |   SRepeated SRepeatQualifier (SRegex ann)
    |   SAlternative (SRegex ann) (SRegex ann)
    |   SCapture ann (SRegex ann)
    |   SConcat (SRegex ann) (SRegex ann)
    deriving (Read, Show)


assignGroups :: SRegex _irrelevant -> SRegex Int
assignGroups = flip evalState 0 . f
    where
        alloc = modify (+1) >> get
        f r@(SLiteral c)            = return $ SLiteral c
        f r@(SCharClass neg cces)   = return $ SCharClass neg cces
        f r@(SBackReference i)      = return $ SBackReference i
        f r@(SCharType neg ct)      = return $ SCharType neg ct
        f r@(SAnyChar)              = return $ SAnyChar
        f r@(SEmpty)                = return $ SEmpty
        f r@(SBeginString)          = return $ SBeginString
        f r@(SEndString)            = return $ SEndString
        f r@(SRepeated q r1)        = SRepeated q <$> f r1
        f r@(SAlternative r1 r2)    = SAlternative <$> f r1 <*> f r2
        f r@(SCapture _ r1)         = SCapture <$> alloc <*> f r1
        f r@(SConcat r1 r2)         = SConcat <$> f r1 <*> f r2


-- The intent is for renderS to render a regex which (1) can be parsed with our
-- parser or any ES3/ES5/ES12 parser, (2) has identical semantics, and (3) in
-- most cases, matches the surface syntax of the originally parsed rx.
-- (The only known exception to 1 is that we output `SLiteral '$'` as "\$",
-- but ES3/ES5 technically require "\x24".)
renderS :: SRegex _irrelevant -> String
renderS = f 0
    where
        f :: Int -> SRegex _irrelevant -> String
        f _ (SLiteral ch)                   = renderLiteral ch
        f _ (SCharClass neg ranges)         = renderCharClass neg ranges
        f _ (SBackReference i)              = "\\" ++ show i
        f _ (SCharType neg cht)             = "\\" ++ [charTypeToLetter neg cht]
        f _ (SAnyChar)                      = "."
        f _ (SEmpty)                        = ""
        f _ (SBeginString)                  = "^"
        f _ (SEndString)                    = "$"
        f p (SRepeated q rx)       | p < 3  = f 3 rx ++ renderRepQualifier q
                                   | True   = "(?:" ++ f 0 rx ++ ")" ++ renderRepQualifier q
        f p (SAlternative rx1 rx2) | p < 1  = f 0 rx1 ++ "|" ++ f 0 rx2
                                   | True   = "(?:" ++ f 0 (SAlternative rx1 rx2) ++ ")"
        f p (SCapture _ rx)                 = "(" ++ f 0 rx ++ ")"
        f p (SConcat rx1 rx2)      | p < 2  = f 1 rx1 ++ f 1 rx2
                                   | True   = "(?:" ++ f 0 (SConcat rx1 rx2) ++ ")"

        renderLiteral :: Char -> String
        renderLiteral ch =
            if | ch `elem` concat [['a'..'z'], ['A'..'Z'], ['0'..'9'], "`~!@#%&-_=]:;'\",<>/"]
                    -> [ch]
               | ch `elem` "^$\\.*+?()[]{}|"
                   -> "\\" ++ [ch]
               | otherwise
                   -> charToEscapedHex ch

        renderCharClass neg ranges =
                "[" ++ (if neg then "^" else "")
                ++ concatMap renderCCRange ranges
                ++ "]"
        renderCCRange (SCharClassSingle ch)             = renderCCChar ch
        renderCCRange (SCharClassRange min max)         = renderCCChar min ++ "-" ++ renderCCChar max
        renderCCRange (SCharClassCharType neg cht)      = "\\" ++ [charTypeToLetter neg cht]
        renderCCChar ch@(ord -> o) | ch `elem` "-]\\"   = "\\" ++ [ch]
                                   | 32 <= o && o < 127 = [ch]
                                   | otherwise          = charToEscapedHex ch

        charTypeToLetter False SCharTypeSpace = 's'
        charTypeToLetter False SCharTypeWord  = 'w'
        charTypeToLetter False SCharTypeDigit = 'd'
        charTypeToLetter True cht             = toUpper (charTypeToLetter False cht)

        renderRepQualifier SRepeatStar              = "*"
        renderRepQualifier SRepeatPlus              = "+"
        renderRepQualifier SRepeatQuestion          = "?"
        renderRepQualifier (SRepeatCount mmin mmax) =
            case (mmin, mmax) of
                (Nothing, Nothing)                  -> undefined
                (Nothing, Just max)                 -> "{0," ++ show max ++ "}"
                (Just min, Nothing)                 -> "{" ++ show min ++ ",}"
                (Just min, Just max) | min == max   -> "{" ++ show min ++ "}"
                                     | otherwise    -> "{" ++ show min ++ "," ++ show max ++ "}"


-- ----------------------------------------
-- Desugared forms, and simple manipulation of them
-- ----------------------------------------


data DRegex ann =
        DChar ann CharSet
    |   DBackReference ann Int
    |   DBeginString ann
    |   DEndString ann
    |   DRepeat ann Int (Maybe Int) (DRegex ann)
    |   DCapture ann Int (DRegex ann)
    |   DChoice ann [DRegex ann]
    |   DConcat ann [DRegex ann]
    deriving (Read, Show)

annOf :: DRegex ann -> ann
annOf (DChar ann cs)                    = ann
annOf (DBackReference ann n)            = ann
annOf (DBeginString ann)                = ann
annOf (DEndString ann)                  = ann
annOf (DRepeat ann min mmax ar)         = ann
annOf (DCapture ann n ar)               = ann
annOf (DChoice ann ars)                 = ann
annOf (DConcat ann ars)                 = ann

fixupPrecedence :: DRegex e -> DRegex e
fixupPrecedence = f where
    f r@(DChar _ _)                                         = r
    f r@(DBackReference _ _)                                = r
    f r@(DBeginString _)                                    = r
    f r@(DEndString _)                                      = r
    f (DRepeat ann mn mx (f -> r))                          = DRepeat ann mn mx r
    f (DCapture ann i (f -> r))                             = DCapture ann i r
    f (DChoice ann (map f -> choices))                      = rewrap $ concatMap unwrap choices where
        unwrap (DChoice _ choices) = choices
        unwrap r = [r]
        rewrap [x] = x
        rewrap choices = DChoice ann choices
    f (DConcat ann (map f -> concats))                      = rewrap $ concatMap unwrap concats where
        unwrap (DConcat _ concats) = concats
        unwrap r = [r]
        rewrap [x] = x
        rewrap concats = DConcat ann concats


-- This is almost reparseable as a regex with the correct semantics (either
-- using an off-the-shelf regex engine's X flag or after stripping spaces.)
-- However, we use parentheses to mean *non-capturing* groups only, and
-- indicate capturing groups with made-up `(?<n> bleh)` syntax.
renderD :: DRegex _ign -> String
renderD = f 0 where
    f :: Int -> DRegex _ign -> String
    f _ (DChar _ (MSetPos []))      = "(?!)"
    f _ (DChar _ (MSetPos [ch]))    = renderChar ch
    f _ (DChar _ (MSetNeg []))      = "."
    f _ (DChar _ (MSetPos chars))   = "[" ++ concatMap renderChar chars ++ "]"
    f _ (DChar _ (MSetNeg chars))   = "[^" ++ concatMap renderChar chars ++ "]"
    f _ (DBackReference _ i)        = "\\" ++ show i
    f _ (DBeginString _)            = "^"
    f _ (DEndString _)              = "$"
    f _ (DRepeat _ min mmax rx)     = f 2 rx ++ "{" ++ show min ++ "," ++ (maybe "" show mmax) ++ "}"
    f _ (DCapture _ i rx)           = "(?<" ++ show i ++ "> " ++ f 0 rx ++ ")"
    f _ (DChoice _ [])              = "(?!)"
    f p (DChoice _ choices)         = intercalate " | " (map (f 0) choices)   & wrapPrec p 0
    f _ (DConcat _ [])              = "()"
    f p (DConcat _ concats)         = intercalate " " (map (f 1) concats)     & wrapPrec p 1

    wrapPrec available needed str =
        if available <= needed then str else "(" ++ str ++ ")"

    renderChar ch =
        if | ch `elem` concat [['a'..'z'], ['A'..'Z'], ['0'..'9'], "`~!@#%&_=:;'\",<>/"]
                -> [ch]
           | ch `elem` "^$\\.*+?()[]{}|-"
                -> "\\" ++ [ch]
           | otherwise
                -> charToEscapedHex ch

-- ----------------------------------------
-- Desugaring
-- ----------------------------------------

desugar :: SRegex Int -> DRegex ()
desugar (SLiteral ch)                       = DChar () (msetSingleton ch)
desugar (SCharClass { negated, ranges })    = desugarCharClass negated ranges
desugar (SBackReference num)                = DBackReference () num
desugar (SCharType { negated, charType })   = desugarCharClass negated [SCharClassCharType False charType]
desugar SAnyChar                            = DChar () msetUniversal
desugar SEmpty                              = DConcat () []
desugar SBeginString                        = DBeginString ()
desugar SEndString                          = DEndString ()
desugar (SRepeated qual rx)                 = desugarRepeated qual (desugar rx)
desugar (SAlternative rx1 rx2)              = DChoice () [desugar rx1, desugar rx2]
desugar (SCapture num rx)                   = DCapture () num (desugar rx)
desugar (SConcat rx1 rx2)                   = DConcat () [desugar rx1, desugar rx2]

expandCharType :: SCharType -> CharSet
expandCharType SCharTypeSpace   = msetFromList $ [' ', '\t']
expandCharType SCharTypeWord    = msetFromList $ ['a' .. 'z'] ++ ['A' .. 'Z'] ++ ['0' .. '9'] ++ ['_']
expandCharType SCharTypeDigit   = msetFromList $ ['0' .. '9']

desugarCharClass :: Bool -> [SCharClassElement] -> DRegex ()
desugarCharClass negated ccelems =
    DChar () $
        (if negated then msetInverse else id)
        (foldl msetUnion msetEmpty $ map expandCharClassElement ccelems)

expandCharClassElement :: SCharClassElement -> MSet Char
expandCharClassElement (SCharClassSingle ch) = msetSingleton ch
expandCharClassElement (SCharClassRange low high) = msetFromList [low .. high]
expandCharClassElement (SCharClassCharType { negated, charType }) =
    (if negated then msetInverse else id) (expandCharType charType)

desugarRepeated :: SRepeatQualifier -> DRegex () -> DRegex ()
desugarRepeated SRepeatStar drx             = DRepeat () 0 Nothing drx
desugarRepeated SRepeatPlus drx             = DRepeat () 1 Nothing drx
desugarRepeated SRepeatQuestion drx         = DRepeat () 0 (Just 1) drx
desugarRepeated (SRepeatCount min max) drx  = DRepeat () (min `orElse` 0) max drx


-- ----------------------------------------
-- Capture analysis.
-- ----------------------------------------
--
-- Used to remove unreferenced capture groups. This can
-- end up simplifying the desugared/evaluated representation
-- somewhat significantly.


data CaptureAnalysis = CaptureAnalysis {
        inscopeDefs :: [Int],
        subtreeDefs :: [(Int, DRegex CaptureAnalysis)],
        subtreeRefs :: [Int]
    }
    deriving (Show)

caBlank :: CaptureAnalysis
caBlank = CaptureAnalysis [] [] []

analyzeCapturesWithScope :: [Int] -> DRegex _ign -> DRegex CaptureAnalysis
analyzeCapturesWithScope = f where
    f :: [Int] -> DRegex _ign -> DRegex CaptureAnalysis
    f ctx = \case
        (DChar _ cs)                                        -> DChar (boring ctx) cs
        (DBackReference _ n)                                -> DBackReference (boring ctx){subtreeRefs = [n]} n
        (DBeginString _)                                    -> DBeginString (boring ctx)
        (DEndString _)                                      -> DEndString (boring ctx)
        (DRepeat _ min mmax (f ctx -> arx@(annOf -> arxAnn))) -> DRepeat arxAnn min mmax arx
        (DCapture _ n (f ctx -> arx@(annOf -> arxAnn)))     -> DCapture arxAnn{subtreeDefs = [(n, arx)]} n arx
        (DChoice _ (doChoices ctx -> (arxs, arxsAnn)))      -> DChoice arxsAnn arxs
        (DConcat _ (doConcats ctx -> (arxs, arxsAnn)))      -> DConcat arxsAnn arxs

    boring ctx = caBlank{inscopeDefs = ctx}

    doChoices = doChain False
    doConcats = doChain True

    doChain :: Bool -> [Int] -> [DRegex _ign] -> ([DRegex CaptureAnalysis], CaptureAnalysis)
    doChain cumulative ctx [] = ([], boring ctx)
    doChain cumulative ctx ((f ctx -> arx@(annOf -> arxAnn)):rxsRest) =
        let nextctx = if cumulative then (map fst $ subtreeDefs arxAnn) ++ ctx else ctx in
        let (arxsRest, annRest) = doChain cumulative nextctx rxsRest in
        (arx:arxsRest,
         CaptureAnalysis{
                inscopeDefs = ctx,
                subtreeDefs = (subtreeDefs arxAnn) ++ (subtreeDefs annRest),
                subtreeRefs = (subtreeRefs arxAnn) ++ (subtreeRefs annRest)})

analyzeCaptures :: DRegex _ign -> DRegex CaptureAnalysis
analyzeCaptures = analyzeCapturesWithScope []

reduceCaptures :: DRegex e -> DRegex e
reduceCaptures rx0 = f rx0 where
    CaptureAnalysis{subtreeDefs = (map fst -> allDefs),
                    subtreeRefs = allRefs} =
            annOf (analyzeCaptures rx0)
    f :: DRegex e -> DRegex e
    f r@(DChar _ _)                                     = r
    f r@(DBackReference _ n) | n `elem` allDefs         = r
                             | otherwise                = error "undefined backrefs"
    f r@(DBeginString _)                                = r
    f r@(DEndString _)                                  = r
    f r@(DRepeat ann mn mx (f -> rx))                   = DRepeat ann mn mx rx
    f r@(DCapture ann i (f -> rx)) | i `elem` allRefs   = DCapture ann i rx
                                   | otherwise          = rx
    f r@(DChoice ann (map f -> choices))                = DChoice ann choices
    f r@(DConcat ann (map f -> concats))                = DConcat ann concats


-- ----------------------------------------
-- Positional analysis.
-- ----------------------------------------
--
-- Some code to attempt to make inferences based on min/max lengths of
-- various chunks of the regex. The analysis appears to work, but we're
-- not actually using it to simplify the regexes.

data PositionalAnalysis = PositionalAnalysis {
        minWidth :: Int,
        maxWidth :: Maybe Int,
        hasAssertion :: Maybe Bool,
        requiresBeginning :: Maybe Bool,
        requiresEnd :: Maybe Bool
    }
    deriving (Show)


doPositionalAnalysis :: DRegex CaptureAnalysis -> DRegex PositionalAnalysis
doPositionalAnalysis rx0 = f rx0 where
    CaptureAnalysis{subtreeDefs = allDefs} = annOf rx0

    f :: DRegex CaptureAnalysis -> DRegex PositionalAnalysis
    f r@(DChar _ cset)                                  = DChar pa cset where
            pa = PositionalAnalysis {
                    minWidth = 1,
                    maxWidth = return 1,
                    hasAssertion = return False, 
                    requiresBeginning = return False,
                    requiresEnd = return False }

    f r@(DBackReference ca n)                           = DBackReference pa n where
            pa = case (n `elem` inscopeDefs ca, lookup n allDefs) of
                    (True, Just def) -> PositionalAnalysis {
                            minWidth = minWidth (annOf $ f def),
                            maxWidth = maxWidth (annOf $ f def),
                            hasAssertion = return False,
                            requiresBeginning = return False,
                            requiresEnd = return False }
                    (True, Nothing) -> error "assertion failure"
                    (False, _) -> PositionalAnalysis {
                            minWidth = 0,
                            maxWidth = Just 0,
                            hasAssertion = return False,
                            requiresBeginning = return False,
                            requiresEnd = return False }

    f r@(DBeginString _)                                = DBeginString pa where
            pa = PositionalAnalysis {
                    minWidth = 0,
                    maxWidth = return 0,
                    hasAssertion = return True, 
                    requiresBeginning = return True,
                    requiresEnd = return False }

    f r@(DEndString _)                                  = DEndString pa where
            pa = PositionalAnalysis {
                    minWidth = 0,
                    maxWidth = Just 0,
                    hasAssertion = return True, 
                    requiresBeginning = return False,
                    requiresEnd = return True }

    f r@(DRepeat ca mn mx (f -> rx@(annOf -> rxa)))     = DRepeat pa mn mx rx where
            pa = PositionalAnalysis {
                    minWidth = mn * minWidth rxa,
                    maxWidth = (*) <$> mx <*> maxWidth rxa,
                    hasAssertion = hasAssertion rxa,
                    requiresBeginning = if mn == 0 then Just False else requiresBeginning rxa,
                    requiresEnd = if mn == 0 then Just False else requiresEnd rxa }

    f r@(DCapture ca i (f -> rx@(annOf -> rxa)))        = DCapture rxa i rx

    f r@(DChoice _ [])                                  = DChoice pa [] where
            pa = PositionalAnalysis {
                    minWidth = 0,
                    maxWidth = return 0,
                    hasAssertion = return False,
                    requiresBeginning = return False,
                    requiresEnd = return False }

    f r@(DChoice ca (map f -> choices))                 = DChoice pa choices where
            pa = foldl1 merge $ map annOf choices
            merge pa1 pa2 = PositionalAnalysis {
                    minWidth = min (minWidth pa1) (minWidth pa2),
                    maxWidth = liftM2 max (maxWidth pa1) (maxWidth pa2),
                    hasAssertion = (
                        case (hasAssertion pa1, hasAssertion pa2) of
                            (Just True, _) -> Just True
                            (_, Just True) -> Just True
                            (Just False, b) -> b
                            (Nothing, _) -> Nothing
                    ),
                    requiresBeginning = (
                        case (requiresBeginning pa1, requiresBeginning pa2) of
                            (_, Just False) -> Just False
                            (Just False, _) -> Just False
                            (Just True, b) -> b
                            (Nothing, _) -> Nothing
                    ),
                    requiresEnd = (
                        case (requiresEnd pa1, requiresEnd pa2) of
                            (_, Just False) -> Just False
                            (Just False, _) -> Just False
                            (Just True, b) -> b
                            (Nothing, _) -> Nothing
                    )}

    f r@(DConcat _ [])                                  = DConcat pa [] where
            pa = PositionalAnalysis {
                    minWidth = 0,
                    maxWidth = return 0,
                    hasAssertion = return False,
                    requiresBeginning = return False,
                    requiresEnd = return False }

    f r@(DConcat ca (map f -> concats))                 = DConcat pa concats where
            pa = foldl1 merge $ map annOf concats
            merge pa1 pa2 = PositionalAnalysis {
                    minWidth = (+) (minWidth pa1) (minWidth pa2),
                    maxWidth = liftM2 (+) (maxWidth pa1) (maxWidth pa2),
                    hasAssertion = (
                        case (hasAssertion pa1, hasAssertion pa2) of
                            (Just True, _) -> Just True
                            (_, Just True) -> Just True
                            (Just False, b) -> b
                            (Nothing, _) -> Nothing
                    ),
                    requiresBeginning = requiresBeginning pa1,
                    requiresEnd = requiresEnd pa2 }


-- ----------------------------------------
-- Parsing
-- ----------------------------------------

{-

    The goal of the parser is not to be a perfect ES3/ES5/ES12/whatever regex
    parser. Rather, our goal is to be conservative: (a) if we parse a string as
    a regex, we parse it into a semantically correct representation, and (b) if
    a string should not parse as a regex, we reject it. This means we behave
    quite safely: the result of a parse attempt is either "that was a valid
    regex and here is a semantically correct representation" or "iunno". It
    wouldn't actually be too difficult to modify the parser into something
    fully complete and correct; the main known deficiencies relate to
    rejecting some regexes which contain non-ASCII characters.

    Specific known differences are:

    Non-greedy qualifiers are unsupported, and rejected with an informative error.

    Word boundary assertions (`/\b/` and `/\B/`) are unsupported, and rejected with
    an informative error.

    Lookaheads (both positive and negative) are unsupported, and rejected with an
    informative error.

    The characters which can occur unescaped in a regex are limited to alnums and
    whitelisted punctuation. (In ES, everything except blacklisted punctuation
    can be used.) The characters which can occur escaped with an identity-escape
    are also limited to ASCII.

    The ES3/ES5 rules for parsing character classes are insane, inscrutable, and
    allow all sorts of insane constructs like `/[-]/`. We throw them out with
    extreme prejudice and instead parse with

        CharacterClass  :=  '['  ('^')? ('-')?  CCElement*  ('-')?  ']'
        CCElement       :=  '\\'  CharacterClassEscape
                         |  CCSingleChar
                         |  CCSingleChar  '-'  CCSingleChar
        CCSingleChar    :=  '\\'  CharacterEscape
                         |  '\\'  'b'
                         |  '\\'  '0'
                         |  '\\'  OctalEscapeSequence
                         |  boring unescaped character

    (I think this is conservative relative to ES3/ES5 (except for the octal
    escape sequences (see below)), but given how insane the spec is, I'm not
    making any promises.)

    We parse octal escape sequences in character classes.  While verboten in
    ES3 and ES5, ES12 B.1.4 permits them for legacy support in web browsers.
    Sadly, the puzzles actually use one in a character class, so we support it
    there. (But still reject octal escape sequences elsewhere.)

    We allow out-of-range backrefs at parse time (which ES3/ES5 doesn't). Note,
    however, that we reject them in `reduceCaptures`.

    We support `/\$/` as a way to match the literal dollar-sign, which ES3 and
    ES5 managed to accidentally ban (but it was fixed by ES12.)
 -}


type ParseTo t = GenParser Char () t
type ParseR = ParseTo (SRegex ())

optionBool :: ParseTo _ignored -> ParseTo Bool
optionBool p = option False (True <$ p)


-- Parse ES3 rule `Pattern`
parseFullRegex :: ParseR
parseFullRegex = parseRegex <* eof

-- Parse ES3 rule `Disjunction`
parseRegex :: ParseR
parseRegex = chainl1 parseAlternative (char '|' $> SAlternative)

-- Parse ES3 rule `Alternative`
parseAlternative :: ParseR
parseAlternative = do
    many parseTerm <&> \case
                [] ->  SEmpty
                [x] -> x
                lst -> foldl1 SConcat lst

-- Parse ES3 rule `Term` (nested: `Qualifier`)
parseTerm :: ParseR
parseTerm = parseAssertion <|> do
    u <- parseAtom
    r <- optionMaybe (
                (char '*' $> SRepeatStar)
            <|> (char '+' $> SRepeatPlus)
            <|> (char '?' $> SRepeatQuestion)
            <|> (char '{' >> (
                    try (read <$> many1 digit <* char '}'
                        <&> \n -> SRepeatCount (Just n) (Just n))
                <|> choice [
                        try $ do
                            min <- if hasmin then (Just <$> read <$> many1 digit) else return Nothing
                            char ','
                            max <- if hasmax then (Just <$> read <$> many1 digit) else return Nothing
                            char '}'
                            return $ SRepeatCount min max
                        | hasmin <- [True]
                        , hasmax <- [False, True]]))
        )
    isNonGreedy <- optionBool (char '?')
    when isNonGreedy $ fail "non-greedy is unsupported"
    return $ maybe id SRepeated r u

-- Parse ES3 rule `Assertion`
parseAssertion :: ParseR
parseAssertion = (
            (char '^' $> SBeginString)
        <|> (char '$' $> SEndString)
        <|> (try (string "\\b") >> fail "word boundary assertions are unsupported")
        <|> (try (string "\\B") >> fail "word boundary assertions are unsupported")
    )

-- Parse ES3 rule `Atom`
parseAtom :: ParseR
parseAtom = (
            parseCharClass                                                          -- CharacterClass
        <|> parseCaptureGroup                                                       -- (foo), (?:foo), (?=foo), (?!foo)
        <|> (char '.' $> SAnyChar)                                                  -- .
        <|> (choice [letter, digit, oneOf "`~!@#%&-_=]:;'\",<>/"] <&> (SLiteral))   -- PatternCharacter, ish
        <|> try parseBackReference                                                  -- \ (AtomEscape -> DecimalEscape)
        <|> try parseCharType                                                       -- \ (AtomEscape -> CharacterClassEscape)
        <|> try (char '\\' >> parseCharacterEscape <&> SLiteral)                    -- \ (AtomEscape -> CharacterEscape)
    )

-- Parse ES3 rule `CharacterClass`
parseCharClass :: ParseR
parseCharClass = do
    char '['
    negated      <- optionBool (char '^')
    leadingDash  <- optionBool (char '-')
    elements     <- many parseCharClassRange
    trailingDash <- optionBool (char '-')
    char ']'
    return $ SCharClass negated $
        if | leadingDash    -> (SCharClassSingle '-'):elements
           | trailingDash   -> elements ++ [SCharClassSingle '-']
           | otherwise      -> elements

parseCharClassRange :: ParseTo SCharClassElement
parseCharClassRange =
    try (char '\\' >> parseCharTypeLetter <&> uncurry SCharClassCharType)
    <|> do
            sc <- single
            ecMaybe <- try (lookAhead $ string "-]" $> Nothing)
                   <|> (char '-' >> Just <$> single)
                   <|> return Nothing
            case ecMaybe of
                Nothing             -> return $ SCharClassSingle sc
                Just ec | sc <= ec  -> return $ SCharClassRange sc ec
                        | otherwise -> fail "character class range: start>end"
    where
        single = noneOf "\\-]"
             <|> try (char '\\' >> parseCharacterEscape)
             <|> try (string "\\b" $> '\b')
             <|> (char '\\' >> chr <$> octalEscape)

        -- E262-12 B.1.4 contains modifications to the regex grammar which are
        -- only supposed to be used in web browsers (for compatibility with
        -- in-the-wild code which violates the rest of the spec.) It contains
        --
        --      ClassAtomNoDash :: \ ClassEscape
        --      ClassEscape :: CharacterEscape
        --      CharacterEscape :: LegacyOctalEscapeSequence  [non-unicode]
        --
        -- which is necessary to parse something of the form `[\2]` which
        -- is present in puzzle 2e9955c8-a87a-4e7d-8e7d-fae2e4bdbde2 (the
        -- hexagonal group, named "Winne"). If the regex was parsed in a
        -- E262-3 implementation, E262-5 implementation, non-browser E262-12
        -- implementation, or any E262-12 implementation with the unicode flag,
        -- it would be a syntax error. (Note that this does *not* apply to
        -- `[\0]` which is always a null character and valid everywhere.)
        --
        -- This is why we can't have nice things.
        --
        -- Proof of equivalence of this parser to E262-12 B.1.2 is left as
        -- an excercise to the reader.
        octalEscape :: ParseTo Int
        octalEscape = liftM interpOctal $ choice $ map (try . sequence) [
                [oneOf "0123", octDigit, octDigit],
                [octDigit, octDigit],
                [octDigit]
            ]
        interpOctal = foldl (\num digit -> (num * 8) + (digitToInt digit)) 0

-- Parse ES3 productions `Atom := ( anything )`
parseCaptureGroup :: ParseR
parseCaptureGroup = char '(' *> inside <* char ')' where
    inside = (try (string "?=") >> fail "lookaheads unsupported")
         <|> (try (string "?!") >> fail "lookaheads unsupported")
         <|> string "?:" *> parseRegex
         <|> (parseRegex <&> SCapture ())

-- Parse ES3 production `Atom := \ AtomEscape := \ DecimalEscape`  (except for "\0")
parseBackReference :: ParseR
parseBackReference = do
    char '\\'
    notFollowedBy (char '0')
    i <- read <$> many1 digit
    return $ SBackReference i

-- Parse ES3 production `Atom := \ AtomEscape := \ CharacterClassEscape`
parseCharType :: ParseR
parseCharType = do
    char '\\'
    (neg, cht) <- parseCharTypeLetter
    return $ SCharType neg cht

-- Parse ES3 rule `CharacterEscape`  (plus "\0")
parseCharacterEscape :: ParseTo Char
parseCharacterEscape =
        choice [
            -- CharacterEscape := ControlEscape
            (char 'f' $> '\f'),
            (char 'n' $> '\n'),
            (char 'r' $> '\r'),
            (char 't' $> '\t'),
            (char 'v' $> '\v'),
            -- CharacterEscape := 'c' ControlLetter
            (char 'c' >> oneOf (['A' .. 'Z'] ++ ['a' .. 'z'])
                <&> \c -> chr (ord c `mod` 32)),
            -- CharacterEscape := HexEscapeSequence and UnicodeEscapeSequence
            (char 'x' >> mapM (const hexDigit) [0..1] <&> (chr . hexToNum)),
            (char 'u' >> mapM (const hexDigit) [0..3] <&> (chr . hexToNum)),
            -- DecimalEscape := '0'
            (char '0' $> chr 0),
            -- SourceCharacter but not IdentifierPart  (plus dollarsign)
            --  (in ASCII, printable ASCII except alnum, underscore)
            oneOf " !\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~"  -- ASCII order
        ]

-- Parse ES3 rule `CharacterClassEscape`
parseCharTypeLetter :: ParseTo (Bool, SCharType)
parseCharTypeLetter = choice [
        char 's' $> (False, SCharTypeSpace),
        char 'S' $> (True, SCharTypeSpace),
        char 'w' $> (False, SCharTypeWord),
        char 'W' $> (True, SCharTypeWord),
        char 'd' $> (False, SCharTypeDigit),
        char 'D' $> (True, SCharTypeDigit)
    ]


-- ----------------------------------------
-- Symbolic booleans
-- ----------------------------------------


data SymbolicBoolOver embed =
        SBTrue
    |   SBFalse
    |   SBAnd (SymbolicBoolOver embed) (SymbolicBoolOver embed)
    |   SBOr (SymbolicBoolOver embed) (SymbolicBoolOver embed)
    |   SBNot (SymbolicBoolOver embed)
    |   SBCond (SymbolicBoolOver embed) (SymbolicBoolOver embed) (SymbolicBoolOver embed)
    |   SBEmbed embed
    deriving (Show)

simplifySB :: (SymbolicBoolOver embed) -> (SymbolicBoolOver embed)
simplifySB sb@(SBTrue) = sb
simplifySB sb@(SBFalse) = sb
simplifySB (SBAnd (simplifySB -> a) (simplifySB -> b)) =
    case (a, b) of
        (SBFalse, _)    -> SBFalse
        (SBTrue, _)     -> b
        (_, SBFalse)    -> SBFalse
        (a, SBTrue)     -> a
        (_, _)          -> SBAnd a b
simplifySB (SBOr (simplifySB -> a) (simplifySB -> b)) =
    case (a, b) of
        (SBFalse, b)    -> b
        (SBTrue, _)     -> SBTrue
        (a, SBFalse)    -> a
        (_, SBTrue)     -> SBTrue
        (a, b)          -> SBOr a b
simplifySB (SBNot (simplifySB -> a)) =
    case a of
        SBTrue          -> SBFalse
        SBFalse         -> SBTrue
        SBNot x         -> x
        SBAnd x y       -> simplifySB (SBOr (SBNot x) (SBNot y))
        SBOr x y        -> simplifySB (SBAnd (SBNot x) (SBNot y))
        SBCond c t f    -> simplifySB (SBCond c (SBNot t) (SBNot f))
        a               -> SBNot a
simplifySB (SBCond (SBNot cn) t f) =
        SBCond cn f t
simplifySB (SBCond (simplifySB -> c) (simplifySB -> t) (simplifySB -> e)) =
    case (c, t, e) of
        (SBTrue, t, _)  -> t
        (SBFalse, _, f) -> f
        (c, SBTrue, f)  -> simplifySB (SBOr c f)
        (c, SBFalse, f) -> simplifySB (SBAnd (SBNot c) f)
        (c, t, SBTrue)  -> simplifySB (SBOr (SBNot c) t)
        (c, t, SBFalse) -> simplifySB (SBAnd c t)
        (c, t, f)       -> SBCond c t f
simplifySB sb@(SBEmbed _) = sb


-- ----------------------------------------
-- Matching
-- ----------------------------------------

{-
    The matching implementation tries to match the regex against the start of a
    fixed-length string. The code is implemented at a frighteningly high level
    of abstraction; specifically, it's abstracted over two primitive operations
    (amiCharIn, amiCharEq) and **all the boolean operators necessary to define
    the match in terms of those operations**.

    This means we can evaluate the match on an actual string. In this case, the
    abstract operations are instantiated with the obvious things: the primitives
    look at an actual string and return a Bool, the operations on Bool are the normal,
    boring ones, etc.

    But, we can also evaluate the match on a purely symbolic representation.
    Instead of having the abstract operations return a Bool, they return the
    check they would have done, reified into a symbolic representation, SBBool.
    The boolean connectives build up a more complex symbolic representation.
    Eventually, all the logic necessary to evaluate the match gets reified into
    one giant SBBool representation, which we can then simplify and eventually
    pass to Z3.
-}



-- All the operations necessary to evaluate a match, bundled into a typeclass.
--
-- The `key` type parameter is here to identify which instance of the typeclass
-- we're supposed to be using. We pass around a value of type `Phantom key`
-- to disambiguate between instances, and do so as an implicit argument because
-- its actual value is irrelevant and we really do care about it only as part
-- of the context.

class AbstractMatchOps key where
    -- Type of a lifted boolean. It should behave kinda like a functor which
    -- was restricted to only working on Bool. (Fun fact: actually, it should
    -- be isomorphic to a functor restricted to non-bottom Bool, with
    -- `fmap fn amb = ambCond amb (fn True) (fn False)`)
    type AMBool key

    -- Type of the input to the match, e.g. a concrete or symbolic string.
    type AMInput key

    -- Type of the result of the match. It's kinda like a MondaPlus and
    -- MonadChoice, except not a Monad because you can't lift arbitrary
    -- types or values into it.
    type AMResult key

    ambPure :: (?key :: Phantom key) =>
        Bool -> AMBool key
    ambCond :: (?key :: Phantom key) =>
        AMBool key -> AMBool key -> AMBool key -> AMBool key

    ambNot :: (?key :: Phantom key) =>
        AMBool key -> AMBool key
    ambNot b = ambCond b (ambPure False) (ambPure True)

    amiCharIn :: (?key :: Phantom key) =>
        AMInput key -> Int -> [Char] -> AMBool key
    amiCharEq :: (?key :: Phantom key) =>
        AMInput key -> Int -> Int -> AMBool key

    amrFail :: (?key :: Phantom key) =>
        AMResult key
    amrFirst :: (?key :: Phantom key) =>
        AMResult key -> AMResult key -> AMResult key
    amrCond :: (?key :: Phantom key) =>
        AMBool key -> AMResult key -> AMResult key -> AMResult key


-- Matching against an actual string. This is boring, and probably most
-- useful as a mental model of "when I see the abstract one used in the
-- match code, what concrete thing was it before it got abstracted over?"

data ConcreteMatch = ConcreteMatch

instance AbstractMatchOps ConcreteMatch where
    type AMBool ConcreteMatch = Bool
    type AMInput ConcreteMatch = String
    type AMResult ConcreteMatch = Maybe (MatchState String)

    ambPure b = b
    ambCond b t f = if b then t else f

    amiCharIn str pos chars = (str !! pos) `elem` chars
    amiCharEq str pos1 pos2 = (str !! pos1) == (str !! pos2)

    amrFail = Nothing
    amrFirst = mplus
    amrCond b t f = if b then t else f


-- Symbolic matching, on the other hand, just constructs a giant reified
-- `SymbolicBoolOver RegexSymbolicCondition`.

data SymbolicMatch = SymbolicMatch

data RegexSymbolicCondition = SBCharIn Int [Char] | SBCharEq Int Int
    deriving (Show)

instance AbstractMatchOps SymbolicMatch where
    type AMBool SymbolicMatch = SymbolicBoolOver RegexSymbolicCondition
    type AMInput SymbolicMatch = ()
    type AMResult SymbolicMatch = SymbolicBoolOver RegexSymbolicCondition

    ambPure b = if b then SBTrue else SBFalse
    ambCond b t f = SBCond b t f

    amiCharIn () pos []    = SBFalse
    amiCharIn () pos chars = SBEmbed (SBCharIn pos chars)

    amiCharEq () pos1 pos2 = SBEmbed (SBCharEq pos1 pos2)

    amrFail = SBFalse
    amrFirst a b = (SBOr a b)
    amrCond b t f = (SBCond b t f)

-- Actual matching.
-- This is quite close to a transliteration of the semantics given in ES5.

data MatchState input = MatchState {
        msInput    :: input,
        msStrlen   :: Int,
        msEndIndex :: Int,
        msCaptures :: [(Int, (Int, Int))] }
    deriving (Eq, Read, Show)

type MatchContinuation key = MatchState (AMInput key) -> AMResult key

type Matcher key = MatchState (AMInput key) -> MatchContinuation key -> AMResult key

matcherFor :: (AbstractMatchOps key, ?key :: Phantom key) => DRegex _ign -> Matcher key
matcherFor = f where
    f :: (AbstractMatchOps key, ?key :: Phantom key) => DRegex _ign -> Matcher key
    f (DChar _ cset) =
        \x c ->
            let MatchState {msEndIndex, msInput, msStrlen} = x in
            if msEndIndex < msStrlen
            then
                --(if neg then flip else id)
                amrCond
                    ((if neg then ambNot else id) $ amiCharIn msInput msEndIndex chars)
                    (c (x { msEndIndex = msEndIndex + 1 }))
                    amrFail
            else amrFail
        where
            (chars, neg) = case cset of
                MSetPos chars -> (chars, False)
                MSetNeg chars -> (chars, True)
    f (DBackReference _ n) =
        -- ES5 15.10.2.9
        \x c ->
            case lookup n (msCaptures x) of
                Nothing -> c x
                Just (spos, epos) ->
                    let MatchState {msEndIndex, msInput, msStrlen} = x in
                    let len = epos - spos in
                    if (msEndIndex + len <= msStrlen)
                    then
                        let conds =
                                [amiCharEq msInput (msEndIndex + i) (spos + i)
                                | i <- [0 .. len - 1]] in
                        foldr (\cond ifok -> amrCond cond ifok amrFail)
                              (c (x { msEndIndex = msEndIndex + len }))
                              conds
                    else amrFail
    f (DBeginString _) =
        -- ES5 15.10.2.5, 15.10.2.6
        \x c ->
            let MatchState {msEndIndex} = x in
            if msEndIndex == 0 then c x else amrFail
    f (DEndString _) =
        -- ES5 15.10.2.5, 15.10.2.6
        \x c ->
            let MatchState {msEndIndex, msStrlen} = x in
            if msEndIndex == msStrlen then c x else amrFail
    f (DRepeat _ min mmax rx) =
        -- ES5 15.10.2.5
        let m = f rx in
        \x c -> repeatMatcher rx m min mmax x c
    f (DCapture _ n rx) =
        -- ES5 15.10.2.8
        \x c -> (f rx) x $
            \y -> c (y { msCaptures = (n, (msEndIndex x, msEndIndex y)):(msCaptures x) })
    f (DChoice _ choices) =
        foldr (\m1 ms -> \x c -> amrFirst (m1 x c) (ms x c)) (\x c -> amrFail) $ map f choices
    f (DConcat _ concats) =
        foldr (\m1 ms -> \x c -> m1 x (\y -> ms y c)) (\x c -> c x) $ map f concats

repeatMatcher :: (AbstractMatchOps key, ?key :: Phantom key) =>
    DRegex _ign -> Matcher key -> Int -> Maybe Int -> Matcher key
repeatMatcher rx rxmatcher min mmax state0 finalc = loop min mmax state0 where
    -- ES5 15.10.2.5, heavily rephrased
    loop _ (Just 0) state = finalc state
    loop min mmax state =
        -- termination condition: on each iteration of loop, either
        -- min decreases ior state.msEndIndex increases
        amrFirst (rxmatcher
                        (state0 { msEndIndex = msEndIndex state })
                        ifsucceeded)
                     (amrCond (ambPure $ min > 0) amrFail $ finalc state)
        where ifsucceeded y =
                if | min == 0 && msEndIndex state == msEndIndex y
                        -- The internal regex matched an empty
                        -- string once, which means it would
                        -- continue to match the empty string
                        -- forever. Terminate loop. But see note
                        -- below.
                        -> amrFail
                   | min == 0 && msEndIndex y > msStrlen y
                        -- /rx/ matches a full string iff /rx(.|$)/
                        -- matches the full string; since msEndIndex
                        -- now exceeds msStrlen, we know /(.|$)/
                        -- could not match in our future, and therefore
                        -- whatever's after us couldn't either. This
                        -- is sufficient for guaranteeing termination
                        -- of symbolic evaluation; I'm not sure if it's
                        -- necessary.
                        -> amrFail
                   | otherwise
                        -> loop (if min == 0 then 0 else min - 1)
                                (mmax <&> subtract 1)
                                y
{- Note from above:

    If the regex inside the repeated group matched the empty string, we know it
    could continue to match the empty string infinitely often. Stopping the
    loop is necessary to guarantee termination (both for concrete and abstract
    evaluation.)

    The specific behavior of failing is mandated by ES5 (see 15.10.2.6 Note 4).
    Implementation-wise, this means that greedy matches are not actually as
    greedy as possible. In the absence of capture groups, there is no
    observable difference between failing here or continuing to match an
    arbitrary (or conceptually infinite) number of times.

    This changes in the presence of capture groups. Consider matching `"a"`
    against `/^(a?)+$/`. The greediest behavior is that the `+` matches an
    infinite number of times; the first iteration grabs the `"a"` (since the
    `?` is also greedy), and future iterations match empty. The resulting value
    of the capture group is the empty string. However, with ES5's behavior, the
    second repetition of the group fails, so the `+` matches only one time, and
    the value of the capture group is `"a"`. This is observable both in which
    strings match and, for strings which do match, when looking at the final values
    of capture groups. Contrast JS (Chrome 100.0.4896.75), Perl (5.30.3),
    Python (3.10.2), Ruby (2.6.8p205), and libpcre (8.45)

        JS:     /^(a?)*$/.exec("a")[1]
        result: "a"

        Perl:   "a" =~ /^(a?)*$/; print "$1\n";
        result: ""

        Python: re.match(r'^(a?)*$', 'a').group(1)
        result: ""

        Ruby:   puts "a".match(/^(a?)+$/)[1]
        result: ""

        pcre:   (echo '/^(a?)+$/'; echo a) | pcretest
        result: ""


        JS:     /^(a?)*b\1$/.exec("ab")
        result: no match

        Perl:   print("ab" =~ /^(a?)*b\1$/ ? "match" : "no match")
        result: match

        Python: re.match(r'^(a?)*b\1$', 'ab')
        result: match

        Ruby:   puts "ab".match(/^(a?)+b\1$/)
        result: match

        pcre:   (echo '/^(a?)*b\1$/'; echo ab) | pcretest 
        result: match

    Changing this to `finalc state` would be observationally equivalent to the
    *actually* greediest behavior (in which the group is treated as matching
    infinitely often) while still guaranteeing termination.
-}


-- ----------------------------------------
-- Dumpable
-- ----------------------------------------
--
-- Outputting symbolic expressions in a very boring format.


class Dumpable t where
    dump :: t -> String

instance (Dumpable embed) => Dumpable (SymbolicBoolOver embed) where
    dump (SBTrue)           = "true"
    dump (SBFalse)          = "false"
    dump (SBAnd a b)        = "and "    ++ dump a ++ " " ++ dump b
    dump (SBOr a b)         = "or "     ++ dump a ++ " " ++ dump b
    dump (SBNot a)          = "not "    ++ dump a
    dump (SBCond a b c)     = "cond "   ++ dump a ++ " " ++ dump b ++ " " ++ dump c
    dump (SBEmbed embed)    = "embed "  ++ dump embed

dumpQuotedString :: String -> String
dumpQuotedString str = "\"" ++ concatMap dumpChar str ++ "\"" where
    dumpChar ch@(ord -> o) =
        if | ch `elem` concat [['a'..'z'], ['A'..'Z'], ['0'..'9']]
                -> [ch]
           | otherwise
                -> charToEscapedHex ch

instance Dumpable RegexSymbolicCondition where
    dump (SBCharIn idx charlist) = "in " ++ show idx ++ " " ++ dumpQuotedString charlist
    dump (SBCharEq idx1 idx2)    = "eq " ++ show idx1 ++ " " ++ show idx2

instance Dumpable (DRegex any) where
    dump (DChar _ charset)              = "charset " ++ kind ++ " " ++ dumpQuotedString chars
        where (chars, kind) = case charset of
                MSetPos chars -> (chars, "setContaining")
                MSetNeg chars -> (chars, "setNotContaining")
    dump (DBackReference _ id)          = "backref " ++ show id
    dump (DBeginString _)               = "beginString"
    dump (DEndString _)                 = "endString"
    dump (DRepeat _ min mmax inside)    = "repeat " ++ show min ++ " " ++ maybe "NULL" show mmax ++ " " ++ dump inside
    dump (DCapture _ id inside)         = "capture " ++ show id ++ " " ++ dump inside
    dump (DChoice _ choices)            = "choice " ++ show (length choices) ++ " " ++ intercalate " " (map dump choices)
    dump (DConcat _ concats)            = "concat " ++ show (length concats) ++ " " ++ intercalate " " (map dump concats)




-- ----------------------------------------
-- Main
-- ----------------------------------------


regexToSymbolic :: DRegex _ignored -> Int -> SymbolicBoolOver RegexSymbolicCondition
regexToSymbolic rx strlen = simplifySB matchResult where
    state0 = MatchState { msInput = (), msStrlen = strlen, msEndIndex = 0, msCaptures = [] }
    cont0 st = SBTrue
    matchResult =
        let ?key = (Phantom :: Phantom SymbolicMatch) in
        let noAnn = undefined `asTypeOf` annOf rx in
        matcherFor (DConcat noAnn [rx, DEndString noAnn]) state0 cont0

data RunMode = GenerateConstraints | Inspect
    deriving (Show)

main :: IO ()
main = getArgs >>= \case
        ["generate-constraints"] -> process GenerateConstraints
        ["inspect"] -> process Inspect
        ["help"] -> help stdout
        _ -> help stderr >> fail "Usage error"
    where
        process mode = interact $ unlines . concatMap (doLine mode) . zip [1 ..] . lines
        doLine mode (lineno, linestr) =
            let
                [read -> expectedLength, rxString] = words linestr
                Right rx = parse parseFullRegex ("stdin line " ++ show lineno) rxString
                desugared = (desugar . assignGroups) rx
                simplified = (fixupPrecedence . reduceCaptures) desugared
                symbolic = regexToSymbolic simplified expectedLength
            in case mode of
                GenerateConstraints -> [linestr, dump symbolic]
                Inspect -> [
                    linestr,
                    show rx,
                    show desugared,
                    show simplified,
                    show symbolic,
                    show $ doPositionalAnalysis $ analyzeCaptures simplified,
                    dump symbolic,
                    ""]
        help handle = do
            progName <- getProgName
            mapM_ (hPutStrLn handle) [
                    "Usage: " ++ progName ++ " " ++ "{generate-constraints | inspect}",
                    "",
                    "generate-constraints:",
                    "  read \"strlen regex\" lines from stdin, and output the symbolic",
                    "  representation to stdout on a single machine-readable line",
                    "",
                    "inspect:",
                    "  read \"strlen regex\" lines from stdin, and output various",
                    "  information about the regex and symolic representation,",
                    "  with information for different regexes separated by blanks",
                    ""
                ]


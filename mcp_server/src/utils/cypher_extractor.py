"""AST-based Cypher element extraction using antlr4-cypher.

Parses Cypher queries into an AST and extracts structural elements:
labels, relationship types with direction, property accesses, and
variable-to-label bindings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
from antlr4_cypher import CypherLexer, CypherParser, CypherParserListener


@dataclass
class PropertyAccess:
    """A property access expression like ``n.date``."""

    variable: str
    property_name: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropertyAccess):
            return NotImplemented
        return self.variable == other.variable and self.property_name == other.property_name

    def __hash__(self) -> int:
        return hash((self.variable, self.property_name))


@dataclass(frozen=True)
class PropertyChain:
    """A full dotted property path like ``n.attributes.edad``.

    ``PropertyAccess`` flattens a nested path into one entry per segment, with no
    record that one was written through the other. That distinction is decisive
    for a backend whose node properties are FLAT — there ``n.attributes.edad``
    reads a map that does not exist and yields null for every row, while
    ``n.edad`` is the correct form — so the path is recorded alongside rather
    than by widening ``PropertyAccess``, whose identity (variable + single name)
    other call sites already depend on.

    ``path`` is the segment tuple after the variable: ``('edad',)`` for a
    one-level read, ``('attributes', 'edad')`` for a nested one. The LEAF is
    always ``path[-1]``.
    """

    variable: str
    path: tuple[str, ...]


@dataclass
class RelPattern:
    """A relationship pattern with source, target, type and direction."""

    source_var: str | None
    rel_type: str | None
    target_var: str | None
    direction: str  # "right" | "left" | "undirected"


@dataclass
class CypherElements:
    """Structural elements extracted from a Cypher query."""

    labels: list[str] = field(default_factory=list)
    rel_types: list[str] = field(default_factory=list)
    properties: list[PropertyAccess] = field(default_factory=list)
    # The same accesses with their PATH intact — see PropertyChain. Additive:
    # `properties` keeps its historic flattened shape for existing consumers.
    property_chains: list[PropertyChain] = field(default_factory=list)
    var_labels: dict[str, str] = field(default_factory=dict)
    rel_patterns: list[RelPattern] = field(default_factory=list)
    parse_errors: int = 0


def _strip_backticks(name: str) -> str:
    """Strip backtick quoting from an identifier."""
    if name.startswith('`') and name.endswith('`'):
        return name[1:-1]
    return name


def _normalize_names(names_result: object) -> list[str]:
    """Normalize the result of a ``.name()`` call to a list of text strings.

    ``antlr4-cypher`` sometimes returns a list, sometimes a single context.
    """
    if isinstance(names_result, list):
        return [_strip_backticks(n.getText()) for n in names_result]
    if names_result is not None:
        return [_strip_backticks(names_result.getText())]
    return []


class _ElementListener(CypherParserListener):
    """ANTLR4 listener that collects structural elements from the parse tree."""

    def __init__(self) -> None:
        self._labels: list[str] = []
        self._rel_types: list[str] = []
        self._properties: list[PropertyAccess] = []
        self._property_chains: list[PropertyChain] = []
        self._var_labels: dict[str, str] = {}
        self._rel_patterns: list[RelPattern] = []
        # Track the previous node variable in a pattern element chain
        # so multi-hop paths get correct source variables.
        self._chain_prev_var: str | None = None

    # -- Node patterns ---------------------------------------------------------

    def enterNodePattern(self, ctx: CypherParser.NodePatternContext) -> None:
        sym = ctx.symbol()
        var_name = sym.getText() if sym else None

        labels_ctx = ctx.nodeLabels()
        if labels_ctx:
            label_names = _normalize_names(labels_ctx.name())
            for label in label_names:
                if label not in self._labels:
                    self._labels.append(label)
                if var_name and var_name not in self._var_labels:
                    self._var_labels[var_name] = label

    # -- Pattern element chains (multi-hop source tracking) --------------------

    def enterPatternElem(self, ctx: CypherParser.PatternElemContext) -> None:
        """Initialize chain tracking with the head node's variable."""
        np = ctx.nodePattern()
        if np:
            sym = np.symbol()
            self._chain_prev_var = sym.getText() if sym else None

    def enterPatternElemChain(self, ctx: CypherParser.PatternElemChainContext) -> None:
        """Extract a relationship pattern from a chain element.

        The source comes from ``_chain_prev_var`` (the previous node in the
        chain), not from the parent ``patternElem``'s head node.
        """
        rp_ctx = ctx.relationshipPattern()
        np_ctx = ctx.nodePattern()

        # Target variable
        target_var: str | None = None
        if np_ctx:
            sym = np_ctx.symbol()
            target_var = sym.getText() if sym else None

        # Relationship type
        rel_type: str | None = None
        if rp_ctx:
            detail = rp_ctx.relationDetail()
            if detail:
                rt = detail.relationshipTypes()
                if rt:
                    type_names = _normalize_names(rt.name())
                    if type_names:
                        rel_type = type_names[0]
                        for t in type_names:
                            if t not in self._rel_types:
                                self._rel_types.append(t)

        # Direction
        direction = 'undirected'
        if rp_ctx:
            has_lt = rp_ctx.LT() is not None
            has_gt = rp_ctx.GT() is not None
            if has_lt:
                direction = 'left'
            elif has_gt:
                direction = 'right'

        self._rel_patterns.append(RelPattern(
            source_var=self._chain_prev_var,
            rel_type=rel_type,
            target_var=target_var,
            direction=direction,
        ))

        # Update chain tracking: the target of this chain becomes
        # the source of the next chain element.
        self._chain_prev_var = target_var

    # -- Relationship details (for rel_types collection in non-chain context) --

    def enterRelationDetail(self, ctx: CypherParser.RelationDetailContext) -> None:
        """Collect relationship types from standalone relationship details.

        This handles cases where relationship types appear outside of a
        patternElemChain (though rare). The chain handler above also
        collects types; duplicates are prevented by the ``not in`` check.
        """
        rt = ctx.relationshipTypes()
        if rt:
            type_names = _normalize_names(rt.name())
            for t in type_names:
                if t not in self._rel_types:
                    self._rel_types.append(t)

    # -- Property expressions --------------------------------------------------

    def enterPropertyExpression(self, ctx: CypherParser.PropertyExpressionContext) -> None:
        """Extract property accesses like ``n.name``.

        String literals like ``'2024'`` also match PropertyExpressionContext
        but with only 1 child — filter by requiring at least 3 children
        (atom + dot + name).
        """
        if ctx.getChildCount() < 3:
            return

        atom = ctx.atom()
        if atom is None:
            return

        variable = atom.getText()
        prop_names = _normalize_names(ctx.name())

        for prop_name in prop_names:
            pa = PropertyAccess(variable=variable, property_name=prop_name)
            if pa not in self._properties:
                self._properties.append(pa)

        # The same access with its path kept whole. Guarded on `prop_names` so a
        # bare atom never produces an empty chain.
        if prop_names:
            chain = PropertyChain(variable=variable, path=tuple(prop_names))
            if chain not in self._property_chains:
                self._property_chains.append(chain)

    # -- Build result ----------------------------------------------------------

    def build(self, parse_errors: int) -> CypherElements:
        return CypherElements(
            labels=self._labels,
            rel_types=self._rel_types,
            properties=self._properties,
            property_chains=self._property_chains,
            var_labels=self._var_labels,
            rel_patterns=self._rel_patterns,
            parse_errors=parse_errors,
        )


def extract_elements(query: str) -> CypherElements:
    """Parse a Cypher query and extract its structural elements.

    Returns a ``CypherElements`` dataclass with labels, relationship types,
    property accesses, variable-label bindings, directional relationship
    patterns, and a count of parse errors (0 = clean parse).
    """
    input_stream = InputStream(query)
    lexer = CypherLexer(input_stream)
    tokens = CommonTokenStream(lexer)
    parser = CypherParser(tokens)

    # Suppress ANTLR's default stderr error reporting
    parser.removeErrorListeners()

    tree = parser.query()
    parse_errors = parser.getNumberOfSyntaxErrors()

    listener = _ElementListener()
    walker = ParseTreeWalker()
    walker.walk(listener, tree)

    return listener.build(parse_errors)

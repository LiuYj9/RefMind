# Request-local extractive evidence views

The optional enhanced path calls build_extractive_views after ranking and before evidence review. It returns copies in the original order. A view has the same paper/chunk identity as its source; it is not a new indexed chunk. There is no vector-store write, approximate chunk deduplication, or legacy character-budget truncation in this module.

## Interface

build_extractive_views(query, documents, scorer=None, task_kind="method", max_blocks=12, max_sentences=24) returns ExtractiveViewResult with documents, records, coverage_limit, scorer and score_type. audit_dict() provides a JSON-compatible audit without Document instances.

scorer(query, sentence_texts) returns one finite score in [-1, 1] per sentence. Set its score_type attribute to identify a deterministic fixture or model-backed scorer. Scores are only selection heuristics, never answer confidence or evidence sufficiency. The default is explicitly offline_lexical_overlap and performs no service call. The production caller must explicitly pass make_embedding_scorer(embeddings). That factory makes no request until invoked, caches this request's query vector, and preserves service errors as whole-block fallbacks. Constructing the embedding client outside the factory can itself raise; callers must handle that setup error too.

Only ordinary prose (content_type absent/text/paragraph, non-atomic, no detected table/equation/code markup) is eligible. At most 12 blocks are selected for processing and at most 24 sentences are scored in a block; caller values cannot increase these caps. Over-limit blocks remain complete. Comparison and summary tasks allocate processing slots by source round-robin; returned documents keep their retrieval order. These caps do not prove that all required sources or conditions have been covered.

## Conservative selection and fail-closed behavior

A strong sentence seed is combined with the introductory sentence, recognized numerical/unit/negation/object/condition sentences, adjacent sentences, and preceding context for local anaphora. The guards cover common Chinese/English scientific language and HTS terminology. They are deliberately conservative lexical rules, not a proof of semantic entailment or a universal language understanding system. Weak relevance, ambiguous omitted relevance/references, small savings, invalid scores, service failure and invalid span validation retain the complete supplied block. Different-condition blocks are never merged or removed.

Candidate thresholds are MIN_SEED_SCORE=0.35 and MAX_OMITTED_SCORE=0.25; a seed is within 0.12 of the best score and above the floor. A view must save at least 40 characters and 10% of source length after explicit omission markers. These are selection defaults, not probabilities or calibrated correctness thresholds. Real scientific paragraphs can appropriately produce unchanged_no_gain when every sentence carries necessary scope.

## Exact source contract

Offsets are Python Unicode code-point indices with half-open [start, end) intervals, not UTF-8 byte positions or JavaScript UTF-16 positions. Do not normalize whitespace, typography, hyphens, accents, Unicode or line breaks before checking offsets. source_chunk_hash is SHA256 of the supplied original page_content encoded as UTF-8.

Each selected_spans/omitted_spans entry includes start, end and an exact quote. Both lists are ordered and non-overlapping, together partition the original text, and are checked against the hash and original slice. Consecutive selected sentences are merged. Every omitted gap, including a beginning or trailing gap, is represented by a visible omission marker; separate spans are never silently concatenated into a fabricated continuous quotation.

Document.metadata.extractive_view contains the audit record and Document.metadata.source_text retains the complete original text only for internal source positioning/fallback. The existing HTTP sanitization must continue to exclude source_text. The canonical Document and its nested metadata are never mutated. Source page/bbox metadata are retained, but extracting text does not invent a more precise PDF rectangle: map verified quotations to the page or fall back to the original block/page.

## Integration and offline verification

Do not pass these views through compression.compress_context or any character-slice truncator afterward. Send the same views to evidence review, draft generation and answer review. If a later whole-prompt budget is needed, admit/exclude whole evidence views while reporting lost coverage.

Run:
D:\Projects\RefMind\.venv\Scripts\python.exe -m unittest tests.test_extractive_views -v

The focused tests cover exact spans and UTF-8 hashes, Unicode/whitespace, identity and immutability, protected conditions/negation/units/numbers and adjacent context, no cross-condition deduplication, non-prose exclusion, service/validation/ambiguity/no-gain fallback, hard processing caps, source round-robin and opt-in embedding calls. They are offline mechanism tests and are not retrieval-quality evaluation or a model-run claim.

Only extracted results expose evidence_view="extractive_text", deterministic view_id, source_chunk_hash, view_hash, selected_spans, omitted_spans, compression_status and coverage_limit at metadata top level for the HTTP contract. Span text is the exact source slice; quote is an equal audit alias. Fallbacks are not labelled extractive_text.

Top-level coverage_limit is a natural-language boundary for the generation prompt; the audit record retains structured processing caps. Top-level offset_unit="unicode_code_point" and interval="half_open" must survive safe HTTP serialization.

After a scoring error or malformed score result, the remainder of that request retains full source blocks without further service attempts. This avoids repeating the same failing query/API call across twelve blocks.

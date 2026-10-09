# Data-feasibility audit: results

| repo | window issues / PRs | linked chains (auto) | adjudicated | genuine | auto-link precision | est. genuine chains | review themes (#PRs) |
|---|---|---:|---:|---:|---:|---:|---|
| pip | 111 / 336 | 15 | 15 | 5 | 0.33 | ~4.0 | naming_style 14, docs 11, edge_case 10, compat 9 |
| pydantic | 184 / 692 | 46 | 15 | 7 | 0.47 | ~13.0 | docs 5, typing 5, naming_style 3, performance 1 |
| pytest | 144 / 628 | 39 | 22 | 6 | 0.27 | ~13.4 | docs 23, tests 17, compat 15, typing 14 |

## Precision of automatic linking, by method (adjudicated sample)

| repo | method | judged | yes | no | unclear |
|---|---|---:|---:|---:|---:|
| pip | followup | 4 | 3 | 1 | 0 |
| pip | main_break | 6 | 0 | 6 | 0 |
| pip | regression/SZZ | 1 | 1 | 0 | 0 |
| pip | regression/unlinked | 4 | 1 | 2 | 1 |
| pydantic | followup | 5 | 5 | 0 | 0 |
| pydantic | main_break | 5 | 0 | 4 | 1 |
| pydantic | regression/SZZ | 1 | 0 | 1 | 0 |
| pydantic | regression/explicit-reference | 1 | 1 | 0 | 0 |
| pydantic | regression/unlinked | 2 | 1 | 0 | 1 |
| pydantic | revert | 1 | 0 | 0 | 1 |
| pytest | followup | 8 | 3 | 4 | 1 |
| pytest | main_break | 3 | 0 | 2 | 1 |
| pytest | regression/SZZ | 5 | 2 | 2 | 1 |
| pytest | regression/unlinked | 6 | 1 | 1 | 4 |

## Genuine chains: where was the evidence?

- **pip**: stage {'review': 5}; evidence {'yes': 1, 'partial': 4}; problem {'functional_regression': 1, 'packaging': 1, 'test_gap': 2, 'performance': 1}; generalizes {'recurring': 4, 'one_off': 1}; false-positive risk {'medium': 4, 'low': 1}
- **pydantic**: stage {'review': 5, 'pr_creation': 1, 'none': 1}; evidence {'partial': 2, 'yes': 4, 'no': 1}; problem {'functional_regression': 5, 'compatibility': 1, 'docs': 1}; generalizes {'recurring': 5, 'one_off': 2}; false-positive risk {'low': 5, 'medium': 2}
- **pytest**: stage {'review': 4, 'pr_creation': 1, 'pre_merge_ci': 1}; evidence {'yes': 4, 'partial': 2}; problem {'functional_regression': 5, 'ci_infra': 1}; generalizes {'recurring': 5, 'one_off': 1}; false-positive risk {'low': 4, 'medium': 2}

## Candidate interventions from genuine chains

| repo | episode | stage | intervention | generalizes | FP risk |
|---|---|---|---|---|---|
| pip | followup:14199 | review | When a PR wires one finder option into a command, ask or grep whether sibling finder options in the same constructor call are also missing, and either fix them or file an issue | recurring | medium |
| pip | followup:14148 | review | Reviewer check that the lock-generation command applies the same binary/hash policy as the install-from-lock step | one_off | medium |
| pip | followup:14161 | review | Reviewer question: do the new tests fail on versions or behaviour that lack the documented feature, and do they cover nested, circular, upgrade and normalization cases? | recurring | medium |
| pip | regression:14160 | review | Reviewer question on #14157: do these tests exercise nested, circular, upgrade and normalization scenarios, and would they fail on a broken implementation? | recurring | medium |
| pip | regression_fix:14054 | review | Extend the import-time test (e.g. assert pylock and urllib.request are not imported for pip --help) or ask reviewers whether new top-level imports in cli/ modules are lazy | recurring | low |
| pydantic | followup:13122 | review | Round-trip/property test: for generated Decimal values (incl. very small/large magnitudes) assert model_dump_json output validates against the serialization-mode schema pattern | recurring | low |
| pydantic | regression:13950 | review | Add a test with a GenerateJsonSchema subclass overriding generate_definitions() used via TypeAdapter.json_schemas(), and require an explicit maintainer reply/resolution to automated review findings on public-API behavior before merge | recurring | medium |
| pydantic | followup:13673 | review | When a bulk URL rewrite touches lines mentioning a version (e.g. 1.10/V1), have reviewer or script flag them for manual check; or link-target check that link text version matches target | one_off | medium |
| pydantic | followup:13537 | pr_creation | Add equality tests for SecretStr with non-ASCII/unicode values (or encode to bytes before compare_digest) when introducing a stdlib comparison primitive with input restrictions | one_off | low |
| pydantic | followup:13090 | review | Round-trip test validating model_dump_json() output (including values like Decimal('0.0000001') -> '1E-7') against the serialization-mode schema | recurring | low |
| pydantic | regression:13729 | none | Differential test that smart-union selection is identical for validate_python and validate_json inputs (with and without before validators) | recurring | low |
| pydantic | followup:13690 | review | Table-driven equivalence test asserting every type in the removed LIST_TYPES/SET_TYPES/etc. lists yields the same core schema before and after a dispatch refactor | recurring | low |
| pytest | regression:14474 | review | For a refactor that claims to preserve behaviour, diff old and new semantics and add a mixed case: backslash in an identifier plus a string literal in the same expression | recurring | low |
| pytest | followup:14616 | review | Exercise the changed workflow on a fork or branch via workflow_dispatch, or check each gh subcommand against `gh --help`, before merging | recurring | medium |
| pytest | regression:14483 | pr_creation | When replacing computed character ranges with a literal, add a test that includes boundary code points (an emoji at U+1F600) or a lint for \u escapes followed by extra hex digits | recurring | low |
| pytest | followup:14505 | pre_merge_ci | Require a fresh CI run on a rebased or up-to-date head (merge queue or update-branch) before merging PRs that sat long since last CI | recurring | low |
| pytest | regression:15064 | review | Review question or test: what happens if logger.propagate changes after capture starts? Add a caplog test toggling propagate in both directions | one_off | medium |
| pytest | followup:15136 | review | Block merge until a CHANGES_REQUESTED regression report is addressed, and require a test for custom __setattr__, metaclass or slots targets | recurring | low |

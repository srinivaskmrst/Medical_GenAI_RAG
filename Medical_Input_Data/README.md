# Medical AI Agent — RAG Input Test Data (PDF / DOCX / JSON)

**108 input files**, exclusively in the three formats requested: **PDF,
DOCX, and JSON**. Every PDF and every DOCX contains at least one real,
extractable **table** and one real, embedded **image** — so your ingestion
code's table-extraction and image-extraction paths both get exercised on
every non-JSON file, not just a sample of them.

All content is synthetic and clearly labeled as such. No real patient
data (PHI) anywhere in this corpus.

## File counts

| Format | Count | Contains tables | Contains images |
|---|---|---|---|
| JSON | 24 | — (structured data instead) | — |
| PDF | 49 | Yes, all 49 | Yes, all 49 |
| DOCX | 35 | Yes, all 35 | Yes, all 35 |
| **Total** | **108** | **84 / 84 non-JSON files** | **84 / 84 non-JSON files** |

## Folder structure

```
rag_corpus_v2/
├── json/    24 files — structured screen definitions (fields, preconditions, validations)
├── pdf/     49 files — screen instructions, guidelines, terminology, case examples
├── docx/    35 files — workflows, SOPs, app docs, tool instructions, agent procedures
├── assets/images/   the 29 source images embedded into the PDFs/DOCX above
│   ├── screen_mockups/      24 PNGs, one per screen — embedded into matching screen-instruction PDFs
│   └── clinical_visuals/    5 PNGs (ECG waveform, lab chart, vitals trend, imaging placeholder, medication label)
└── manifest.json    all 108 documents with full metadata + has_table / has_image flags
```

## What's in each format

**JSON (24 files)** — one structured screen definition per screen
(SCREEN_01–SCREEN_24): fields, preconditions, postconditions, validation
rules, common errors, UI elements. Pure structured data — the natural
format for this content, no tables/images needed since nothing here is
tabular or visual by nature.

**PDF (49 files)**, each with real tables + a real embedded image:
| Category | Count | Tables | Embedded Image |
|---|---|---|---|
| Screen instructions (one per screen) | 24 | Fields table, Verifications table, Common Errors table (3 tables/file) | The matching screen's mockup |
| Medical guidelines (10 domains) | 10 | Recommendations table | Domain-relevant clinical visual |
| Medical terminology glossaries | 5 | Term/definition table (~15 rows) | Domain-relevant clinical visual |
| Synthetic case walkthroughs | 10 | Vitals table + screen-sequence table | First screen's mockup for that workflow |

**DOCX (35 files)**, each with a real table + a real embedded image:
| Category | Count | Table | Embedded Image |
|---|---|---|---|
| Workflow overviews (5 workflows) | 5 | Screen-sequence table | Domain-relevant clinical visual |
| Hospital SOPs | 10 | Procedure-steps table | Domain-relevant clinical visual |
| Application documentation | 5 | Reference table (module/rule/error-class specific) | Vitals trend chart |
| Agent tool instructions (OCR, Vision, Click, Type, Zoom, Segment, Check, Save, Verify, Screen-Detection) | 10 | Usage-reference table | ECG waveform |
| Agent operating procedures | 5 | Procedure-steps table | ECG waveform |

## Metadata

`manifest.json` at the corpus root lists all 108 documents with:
`document_id, document_name, document_type, version, workflow_id,
screen_id, medical_domain, effective_date, source, access_level, file,
format, has_table, has_image`.

Every PDF and DOCX also prints its `document_id`, `type`, `domain`, and
(where applicable) `workflow`/`screen` directly in the document body, just
below the synthetic-data disclaimer banner — so metadata is recoverable
even without the manifest, by parsing the first block of text.

## Workflow / screen map

| Workflow | Domain | Screens |
|---|---|---|
| WF_001 — Radiology Imaging Order & Review | radiology | SCREEN_01–06 |
| WF_002 — Cardiology Consult & ECG Review | cardiology | SCREEN_07–10 |
| WF_003 — Medication Administration | pharmacy | SCREEN_11–15 |
| WF_004 — Emergency Triage & Intake | emergency | SCREEN_16–20 |
| WF_005 — Lab Specimen Collection & Results | laboratory | SCREEN_21–24 |

## Safety note

All clinical-sounding content (guidelines, SOPs, terminology) is
explicitly synthetic and labeled as such in a visible banner on every
document. Recommendations are generic and illustrative, not real clinical
guidance. Case examples use fabricated patient references (e.g.
`SYN-1000`) and fabricated vitals/lab values only.

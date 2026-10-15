Your previous call to the `record_extraction` tool did not match the
required schema. The following paths were malformed or missing:

{{violations}}

Call `record_extraction` again. Every leaf in the schema -- each
top-level field, each line item's `description`, `quantity`,
`unit_price`, and `total`, and each tax line's `label`, `rate`, `base`,
and `amount` -- must be an object of the exact form
`{"value": ..., "confidence": ..., "evidence": ...}`, where `confidence` is
a number.
Fix ONLY the structure of the leaves listed above so they conform to
that shape. Do not re-read the document, and do not change the value,
confidence, or evidence of any field you already got right -- keep
everything else, evidence included, exactly as it was in your previous call.
